"""Pure PyTorch reproduction of Score Centering on Qwen3-0.6B + Countdown."""
from __future__ import annotations

import argparse
import gc
import json
import random
import time
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.countdown import CountdownProblem, build_dataset, compute_reward, format_messages
from src.losses import METHODS, compute_advantages, make_weight_fn, score_centering_loss


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_prompt_ids_list(
    problems: list[CountdownProblem],
    tokenizer: AutoTokenizer,
) -> list[list[int]]:
    """Format and tokenize Countdown problems into raw prompt token-ID lists."""
    encoded: list[list[int]] = []
    for prob in problems:
        msgs = format_messages(prob)
        try:
            ids = tokenizer.apply_chat_template(
                msgs,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=False,
                enable_thinking=False,
            )
        except TypeError:
            ids = tokenizer.apply_chat_template(
                msgs,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=False,
            )
        encoded.append(list(ids))
    return encoded


def encode_prompts(
    problems: list[CountdownProblem],
    tokenizer: AutoTokenizer,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Tokenize and left-pad Countdown prompts into [B, P] tensors."""
    encoded = build_prompt_ids_list(problems, tokenizer)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    max_len = max(len(x) for x in encoded)
    input_ids = torch.full((len(encoded), max_len), pad_id, dtype=torch.long, device=device)
    attn_mask = torch.zeros((len(encoded), max_len), dtype=torch.long, device=device)
    for i, seq in enumerate(encoded):
        input_ids[i, max_len - len(seq) :] = torch.tensor(seq, dtype=torch.long, device=device)
        attn_mask[i, max_len - len(seq) :] = 1
    return input_ids, attn_mask


def save_json_checkpoint(path: Path, payload: dict[str, object]) -> None:
    """Write experiment results JSON atomically so live dashboards never read truncated files."""
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


@torch.no_grad()
def sample_rollouts(
    sampler_model: AutoModelForCausalLM,
    prompt_ids: Tensor,
    prompt_mask: Tensor,
    tokenizer: AutoTokenizer,
    max_new_tokens: int,
    topk: int,
    temperature: float = 1.0,
) -> dict[str, Tensor | list[str]]:
    """Generate completions with KV cache and record sampled + Top-k sampler logprobs."""
    device = prompt_ids.device
    batch_size = prompt_ids.size(0)
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    pos_ids = (prompt_mask.cumsum(dim=-1) - 1).clamp_min(0)
    outputs = sampler_model(
        input_ids=prompt_ids,
        attention_mask=prompt_mask,
        position_ids=pos_ids,
        use_cache=True,
    )
    logits = outputs.logits[:, -1, :]
    past_kv = outputs.past_key_values

    curr_mask = prompt_mask
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    gen_tokens: list[Tensor] = []
    samp_tok_logps: list[Tensor] = []
    samp_topk_logps: list[Tensor] = []
    topk_indices: list[Tensor] = []
    loss_masks: list[Tensor] = []

    for _ in range(max_new_tokens):
        scaled_logits = logits.float() / max(temperature, 1e-5)
        logp = F.log_softmax(scaled_logits, dim=-1)
        probs = logp.exp()
        next_tok = torch.multinomial(probs, num_samples=1)  # [B, 1]
        tok_logp = logp.gather(-1, next_tok).squeeze(-1)  # [B]
        tk_logp, tk_ids = torch.topk(logp, k=topk, dim=-1)  # [B, k]

        active = ~finished
        tok_1d = torch.where(active, next_tok.squeeze(-1), torch.full_like(next_tok.squeeze(-1), pad_id))

        gen_tokens.append(tok_1d)
        samp_tok_logps.append(tok_logp)
        samp_topk_logps.append(tk_logp)
        topk_indices.append(tk_ids)
        loss_masks.append(active)

        finished = finished | (next_tok.squeeze(-1) == eos_id)
        if finished.all():
            break

        next_pos = curr_mask.sum(dim=-1, keepdim=True)
        curr_mask = torch.cat(
            [curr_mask, torch.ones((batch_size, 1), dtype=curr_mask.dtype, device=device)],
            dim=1,
        )
        outputs = sampler_model(
            input_ids=tok_1d.unsqueeze(-1),
            attention_mask=curr_mask,
            position_ids=next_pos,
            past_key_values=past_kv,
            use_cache=True,
        )
        logits = outputs.logits[:, -1, :]
        past_kv = outputs.past_key_values

    gen_tensor = torch.stack(gen_tokens, dim=1)  # [B, T]
    mask_tensor = torch.stack(loss_masks, dim=1)  # [B, T]
    texts = [
        tokenizer.decode(gen_tensor[i][mask_tensor[i]], skip_special_tokens=True)
        for i in range(batch_size)
    ]
    return {
        "prompt_ids": prompt_ids,
        "prompt_mask": prompt_mask,
        "gen_tokens": gen_tensor,
        "loss_mask": mask_tensor,
        "samp_token_logp": torch.stack(samp_tok_logps, dim=1),
        "samp_topk_logp": torch.stack(samp_topk_logps, dim=1),
        "topk_ids": torch.stack(topk_indices, dim=1),
        "texts": texts,
    }


def make_fixed_noise(model: AutoModelForCausalLM, scale: float, seed: int) -> dict[str, Tensor]:
    """Sample a fixed relative Gaussian weight perturbation matching the paper's setup."""
    if scale <= 0.0:
        return {}
    gen = torch.Generator(device=next(model.parameters()).device).manual_seed(seed)
    noise: dict[str, Tensor] = {}
    for name, param in model.named_parameters():
        if param.ndim > 1 and "norm" not in name.lower():
            delta = torch.randn(param.shape, generator=gen, device=param.device, dtype=param.dtype)
            noise[name] = (scale * delta * param.detach()).to(param.dtype)
    return noise


@torch.no_grad()
def sync_sampler(
    sampler_model: AutoModelForCausalLM,
    trainer_model: AutoModelForCausalLM,
    fixed_noise: dict[str, Tensor],
) -> None:
    """Copy trainer fp32 master weights into bf16 sampler and inject fixed TIM noise."""
    trainer_params = dict(trainer_model.named_parameters())
    for name, s_param in sampler_model.named_parameters():
        t_bf16 = trainer_params[name].detach().to(s_param.dtype)
        if name in fixed_noise:
            s_param.copy_(t_bf16 + fixed_noise[name])
        else:
            s_param.copy_(t_bf16)


@torch.no_grad()
def evaluate_policy(
    model: AutoModelForCausalLM,
    eval_problems: list[CountdownProblem],
    tokenizer: AutoTokenizer,
    max_new_tokens: int,
    batch_size: int = 32,
    return_samples: bool = False,
) -> float | tuple[float, list[dict[str, object]]]:
    """Evaluate clean bfloat16 trainer policy deterministically on held-out problems."""
    device = next(model.parameters()).device
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    rewards: list[float] = []
    samples: list[dict[str, object]] = []
    try:
        torch.manual_seed(9999)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(9999)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            for start in range(0, len(eval_problems), batch_size):
                chunk = eval_problems[start : start + batch_size]
                p_ids, p_mask = encode_prompts(chunk, tokenizer, device)
                roll = sample_rollouts(model, p_ids, p_mask, tokenizer, max_new_tokens=max_new_tokens, topk=16)
                for text, prob in zip(roll["texts"], chunk):
                    r = compute_reward(text, prob)
                    rewards.append(r)
                    if return_samples and len(samples) < 4:
                        samples.append({
                            "numbers": prob.numbers,
                            "target": prob.target,
                            "solution": prob.solution,
                            "completion": text,
                            "reward": r,
                        })
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state)
    acc = float(np.mean(rewards))
    return (acc, samples) if return_samples else acc


def run_single_method(
    method_key: str,
    base_state: dict[str, Tensor],
    tokenizer: AutoTokenizer,
    train_problems: list[CountdownProblem],
    eval_problems: list[CountdownProblem],
    args: argparse.Namespace,
    device: torch.device,
    checkpoint_cb: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    """Train Qwen3-0.6B with one RL method under TIM and return tracked metrics."""
    spec = METHODS[method_key]
    weight_fn = make_weight_fn(spec.weight_kind, tis_max=args.tis_max)
    set_seed(args.seed)

    trainer = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.float32,
        attn_implementation="eager",
    ).to(device)
    trainer.load_state_dict(base_state)
    trainer.train()

    sampler = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device).eval()
    sync_sampler(sampler, trainer, {})
    fixed_noise = make_fixed_noise(sampler, scale=args.noise_scale, seed=args.seed + 77)
    optimizer = torch.optim.SGD(trainer.parameters(), lr=args.lr)

    history: list[dict[str, float]] = []
    sample_completions: dict[str, list[dict[str, object]]] = {}
    t0 = time.time()

    for step in range(args.steps):
        if step % (args.staleness + 1) == 0:
            sync_sampler(sampler, trainer, fixed_noise)

        eval_reward = None
        if step % args.eval_every == 0 or step == args.steps - 1:
            trainer.eval()
            capture = step == 0 or step == args.steps - 1
            res_eval = evaluate_policy(
                trainer,
                eval_problems,
                tokenizer,
                max_new_tokens=args.max_new_tokens,
                batch_size=min(32, len(eval_problems)),
                return_samples=capture,
            )
            if capture:
                eval_reward, samp_list = res_eval  # type: ignore[misc]
                sample_completions[f"step_{step}"] = samp_list
            else:
                eval_reward = float(res_eval)  # type: ignore[arg-type]
            trainer.train()

        # Sample a batch of Countdown prompts and repeat group_size times
        batch_probs = [
            train_problems[(step * args.num_prompts + i) % len(train_problems)]
            for i in range(args.num_prompts)
        ]
        grouped_probs = [p for p in batch_probs for _ in range(args.group_size)]

        # Rollout in micro-chunks if needed to keep KV-cache memory bounded
        p_ids, p_mask = encode_prompts(grouped_probs, tokenizer, device)
        roll = sample_rollouts(
            sampler,
            p_ids,
            p_mask,
            tokenizer,
            max_new_tokens=args.max_new_tokens,
            topk=args.topk,
        )
        rewards_list = [compute_reward(txt, prob) for txt, prob in zip(roll["texts"], grouped_probs)]
        rewards = torch.tensor(rewards_list, dtype=torch.float32, device=device)
        advantages = compute_advantages(rewards, args.group_size, args.reward_mode).unsqueeze(-1)  # [B, 1]

        gen_tokens: Tensor = roll["gen_tokens"]  # type: ignore[assignment]
        loss_mask: Tensor = roll["loss_mask"]  # type: ignore[assignment]
        samp_token_logp: Tensor = roll["samp_token_logp"]  # type: ignore[assignment]
        samp_topk_logp: Tensor = roll["samp_topk_logp"]  # type: ignore[assignment]
        topk_ids: Tensor = roll["topk_ids"]  # type: ignore[assignment]

        prompt_len = p_ids.size(1)
        gen_len = gen_tokens.size(1)
        full_ids = torch.cat([p_ids, gen_tokens], dim=1)
        full_mask = torch.cat(
            [p_mask, torch.ones((p_ids.size(0), gen_len), dtype=p_mask.dtype, device=device)],
            dim=1,
        )
        full_pos = (full_mask.cumsum(dim=-1) - 1).clamp_min(0)
        total_tokens = loss_mask.sum().clamp_min(1.0)

        optimizer.zero_grad(set_to_none=True)
        total_loss_val = 0.0
        total_kl_val = 0.0
        max_ratio_val = 1.0

        mb = args.micro_batch_size
        for s_idx in range(0, full_ids.size(0), mb):
            e_idx = min(s_idx + mb, full_ids.size(0))
            mb_bool = loss_mask[s_idx:e_idx]
            if mb_bool.sum().item() == 0:
                continue

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = trainer(
                    input_ids=full_ids[s_idx:e_idx],
                    attention_mask=full_mask[s_idx:e_idx],
                    position_ids=full_pos[s_idx:e_idx],
                    use_cache=False,
                )
            gen_logits = out.logits[:, prompt_len - 1 : prompt_len - 1 + gen_len, :].float()
            train_logp = F.log_softmax(gen_logits, dim=-1)

            mb_tok = gen_tokens[s_idx:e_idx]
            mb_samp_tok_logp = samp_token_logp[s_idx:e_idx]
            mb_loss = score_centering_loss(
                train_logp=train_logp,
                samp_logp=samp_topk_logp[s_idx:e_idx],
                topk_ids=topk_ids[s_idx:e_idx],
                sampled_token=mb_tok,
                samp_token_logp=mb_samp_tok_logp,
                advantage=advantages[s_idx:e_idx],
                weight_fn=weight_fn,
                use_sc=spec.use_sc,
            )
            scaled_loss = torch.where(mb_bool, mb_loss, 0.0).sum() / total_tokens
            scaled_loss.backward()

            with torch.no_grad():
                mb_train_tok_logp = train_logp.gather(-1, mb_tok.unsqueeze(-1)).squeeze(-1)
                mb_kl = torch.where(mb_bool, mb_samp_tok_logp - mb_train_tok_logp, 0.0).sum()
                mb_ratios = (mb_train_tok_logp - mb_samp_tok_logp).clamp(-20.0, 20.0).exp()
                mb_max_r = torch.where(mb_bool, mb_ratios, 0.0).max().item()
                total_loss_val += scaled_loss.item()
                total_kl_val += mb_kl.item()
                max_ratio_val = max(max_ratio_val, mb_max_r)

        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(trainer.parameters(), max_norm=args.max_grad_norm).item()
        )
        optimizer.step()

        kl_mean = total_kl_val / float(total_tokens.item())
        rec = {
            "step": float(step),
            "train_reward": float(rewards.mean().item()),
            "loss": total_loss_val,
            "kl_samp_train": kl_mean,
            "grad_norm": grad_norm,
            "max_is_ratio": max_ratio_val,
            "mean_len": float(loss_mask.float().sum(dim=1).mean().item()),
        }
        if eval_reward is not None:
            rec["eval_reward"] = eval_reward
        history.append(rec)

        if eval_reward is not None or step % 10 == 0:
            ev_str = f" | eval={eval_reward * 100:5.1f}%" if eval_reward is not None else ""
            print(
                f"[{spec.name:<22}] step={step:3d}/{args.steps} | "
                f"train={rec['train_reward'] * 100:5.1f}%{ev_str} | "
                f"kl={kl_mean:+.4f} | gnorm={grad_norm:.3f} | max_r={max_ratio_val:.2f}",
                flush=True,
            )
        if checkpoint_cb is not None and (eval_reward is not None or step == args.steps - 1):
            checkpoint_cb({
                "method": method_key,
                "name": spec.name,
                "elapsed_sec": time.time() - t0,
                "history": list(history),
                "sample_completions": dict(sample_completions),
            })

    elapsed = time.time() - t0
    del trainer, sampler, fixed_noise, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "method": method_key,
        "name": spec.name,
        "elapsed_sec": elapsed,
        "history": history,
        "sample_completions": sample_completions,
    }


def plot_results(
    all_results: list[dict[str, object]],
    out_path: Path,
    title_prefix: str = "Qwen3-0.6B Countdown",
) -> None:
    """Plot eval reward and KL drift curves across the 4 methods."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=150)
    colors = {
        "pg": "#ef4444",
        "is": "#f59e0b",
        "tis": "#3b82f6",
        "tis_sc": "#10b981",
    }

    for res in all_results:
        key = str(res["method"])
        name = str(res["name"])
        hist = res["history"]  # type: ignore[assignment]
        color = colors.get(key, "#6b7280")

        eval_pts = [(h["step"], h["eval_reward"] * 100.0) for h in hist if "eval_reward" in h]
        if eval_pts:
            xs, ys = zip(*eval_pts)
            axes[0].plot(xs, ys, marker="o", markersize=4, linewidth=2.0, label=name, color=color)

        steps = [h["step"] for h in hist]
        kls = [abs(h["kl_samp_train"]) for h in hist]
        axes[1].plot(steps, kls, linewidth=1.8, alpha=0.9, label=name, color=color)

    axes[0].set_title(f"{title_prefix}: Eval Solve Rate")
    axes[0].set_xlabel("Training Step")
    axes[0].set_ylabel("Solve Rate (%)")
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend(frameon=True)

    axes[1].set_title(f"{title_prefix}: Sampler-Trainer KL Drift |KL(q || p)|")
    axes[1].set_xlabel("Training Step")
    axes[1].set_ylabel("|log q - log p|")
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].legend(frameon=True)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the Score Centering reproduction run."""
    parser = argparse.ArgumentParser(description="Reproduce Score Centering on Qwen3-0.6B + Countdown.")
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--methods", nargs="+", default=["pg", "is", "tis", "tis_sc"])
    parser.add_argument("--engine", default="vllm", choices=["vllm", "hf"])
    parser.add_argument("--vllm-dtype", default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.18)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--eval-prompts", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--num-operands", type=int, default=3)
    parser.add_argument("--max-number", type=int, default=25)
    parser.add_argument("--max-target", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.006)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--noise-scale", type=float, default=0.03)
    parser.add_argument("--staleness", type=int, default=1)
    parser.add_argument("--topk", type=int, default=128)
    parser.add_argument("--tis-max", type=float, default=1.5)
    parser.add_argument("--reward-mode", default="group_centered")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--calibrate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run calibration or the full 4-method Qwen3-0.6B Countdown reproduction."""
    args = parse_args()
    if args.engine == "vllm" and not args.calibrate_only:
        from src.vllm_suite import run_vllm_suite

        run_vllm_suite(
            args=args,
            results_filename="results.json",
            plot_filename="plot.png",
            title_prefix="Controlled Weight-Noise TIM",
            engine_label=f"vLLM V1 ({args.vllm_dtype}, noise_scale={args.noise_scale}, staleness={args.staleness})",
            log_prefix="Controlled TIM",
            fp8_quant=False,
            noise_scale=args.noise_scale,
        )
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | model={args.model_id}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    eval_problems = build_dataset(
        args.eval_prompts,
        seed=args.seed + 1000,
        num_operands=args.num_operands,
        max_number=args.max_number,
        max_target=args.max_target,
    )
    train_problems = build_dataset(
        max(512, args.steps * args.num_prompts),
        seed=args.seed,
        num_operands=args.num_operands,
        max_number=args.max_number,
        max_target=args.max_target,
    )

    base_model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.float32)
    base_state = {k: v.cpu().clone() for k, v in base_model.state_dict().items()}

    if args.calibrate_only:
        model = base_model.to(device).eval()
        t0 = time.time()
        init_acc = evaluate_policy(model, eval_problems, tokenizer, args.max_new_tokens, batch_size=24)
        print(f"Calibration initial eval solve rate ({len(eval_problems)} problems): "
              f"{init_acc * 100:.1f}% in {time.time() - t0:.1f}s")

        # Verify KV-cache vs full-forward alignment (zero noise)
        p_ids, p_mask = encode_prompts(eval_problems[:4], tokenizer, device)
        roll = sample_rollouts(model, p_ids, p_mask, tokenizer, max_new_tokens=args.max_new_tokens, topk=args.topk)
        gen_tok = roll["gen_tokens"]
        l_mask = roll["loss_mask"]
        samp_lp = roll["samp_token_logp"]
        full_ids = torch.cat([p_ids, gen_tok], dim=1)
        full_mask = torch.cat([p_mask, torch.ones_like(gen_tok)], dim=1)
        full_pos = (full_mask.cumsum(dim=-1) - 1).clamp_min(0)
        with torch.no_grad():
            logits = model(input_ids=full_ids, attention_mask=full_mask, position_ids=full_pos).logits
            g_lp = F.log_softmax(logits[:, p_ids.size(1) - 1 : p_ids.size(1) - 1 + gen_tok.size(1), :].float(), dim=-1)
            tr_lp = g_lp.gather(-1, gen_tok.unsqueeze(-1)).squeeze(-1)
            diff = (tr_lp - samp_lp).abs()[l_mask].max().item()
        print(f"KV-cache vs full-forward max logp diff: {diff:.6f}")
        print("Sample completion:", repr(roll["texts"][0][:180]))
        return

    del base_model
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_path = out_dir / "results.json"
    plot_path = out_dir / "plot.png"
    results: list[dict[str, object]] = []
    for key in args.methods:
        def on_ckpt(partial: dict[str, object]) -> None:
            save_json_checkpoint(
                results_path,
                {"status": "running", "args": vars(args), "results": results + [partial]},
            )

        res = run_single_method(
            key, base_state, tokenizer, train_problems, eval_problems, args, device, checkpoint_cb=on_ckpt
        )
        results.append(res)
        save_json_checkpoint(
            results_path,
            {"status": "running", "args": vars(args), "results": results},
        )

    save_json_checkpoint(results_path, {"status": "completed", "args": vars(args), "results": results})
    plot_results(results, plot_path, title_prefix="Controlled Weight-Noise TIM")
    print(f"Saved {results_path} and {plot_path}", flush=True)


if __name__ == "__main__":
    main()
