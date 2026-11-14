"""Real vLLM inference engine vs HuggingFace trainer TIM runner on Qwen3-0.6B + Countdown."""
from __future__ import annotations

import os

# Must be set before importing vllm so LLM.apply_model executes in-process on Colab.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from src.countdown import CountdownProblem, build_dataset, compute_reward
from src.engine import (
    build_prompt_ids_list,
    make_fixed_noise,
    plot_results,
    save_json_checkpoint,
    set_seed,
)
from src.losses import METHODS, compute_advantages, make_weight_fn, score_centering_loss

FP8_E4M3_MAX = 448.0


def quantize_fp8_weight(weight: Tensor, out_dtype: torch.dtype = torch.bfloat16) -> Tensor:
    """Apply real per-row E4M3 FP8 quantization and cast back to sampler dtype."""
    w_f32 = weight.detach().float()
    scale = (w_f32.abs().amax(dim=-1, keepdim=True) / FP8_E4M3_MAX).clamp_min(1e-8)
    w_fp8 = (w_f32 / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return (w_fp8.float() * scale).to(out_dtype)


def _pad_prompt_ids(
    prompt_id_lists: list[list[int]],
    pad_id: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Left-pad already-tokenized prompt ID lists into [B, P] tensors."""
    max_len = max(len(x) for x in prompt_id_lists)
    input_ids = torch.full((len(prompt_id_lists), max_len), pad_id, dtype=torch.long, device=device)
    attn_mask = torch.zeros((len(prompt_id_lists), max_len), dtype=torch.long, device=device)
    for i, seq in enumerate(prompt_id_lists):
        input_ids[i, max_len - len(seq) :] = torch.tensor(seq, dtype=torch.long, device=device)
        attn_mask[i, max_len - len(seq) :] = 1
    return input_ids, attn_mask


def build_sampler_weights(
    trainer_model: AutoModelForCausalLM,
    target_dtype: torch.dtype,
    fixed_noise: dict[str, Tensor] | None = None,
    fp8_quant: bool = False,
) -> list[tuple[str, Tensor]]:
    """Build detached sampler weight list with optional FP8 quantization or noise."""
    weights: list[tuple[str, Tensor]] = []
    for name, param in trainer_model.state_dict().items():
        tensor = param.detach()
        is_linear = tensor.ndim > 1 and "norm" not in name.lower() and "embed" not in name.lower()
        if fp8_quant and is_linear and "lm_head" not in name.lower():
            w_out = quantize_fp8_weight(tensor, out_dtype=target_dtype)
        else:
            w_out = tensor.to(target_dtype)
        if fixed_noise and name in fixed_noise:
            w_out = (w_out.float() + fixed_noise[name].float()).to(target_dtype)
        weights.append((name, w_out.cpu()))
    return weights


def _load_cpu_or_gpu_weights(model: torch.nn.Module, weights: list[tuple[str, Tensor]]) -> None:
    """Move staged CPU sampler weights onto the vLLM worker device and load them."""
    dev = next(model.parameters()).device
    model.load_weights([(name, w.to(dev, non_blocking=True)) for name, w in weights])


def load_weights_into_vllm(llm: LLM, weights: list[tuple[str, Tensor]]) -> None:
    """Push a prepared (name, tensor) weight list into the in-process vLLM engine."""
    llm.apply_model(lambda model: _load_cpu_or_gpu_weights(model, weights))
    if hasattr(llm, "reset_prefix_cache"):
        llm.reset_prefix_cache()


def vllm_sample_rollouts(
    llm: LLM,
    problems: list[CountdownProblem],
    tokenizer: AutoTokenizer,
    max_new_tokens: int,
    topk: int,
    device: torch.device,
    temperature: float = 1.0,
    seed: int | None = None,
) -> dict[str, Tensor | list[str]]:
    """Generate rollouts through real vLLM PagedAttention/FlashInfer and unpack Top-k logprobs."""
    prompt_id_lists = build_prompt_ids_list(problems, tokenizer)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    p_ids, p_mask = _pad_prompt_ids(prompt_id_lists, pad_id, device)

    sp = SamplingParams(
        temperature=temperature,
        top_p=1.0,
        top_k=-1,
        max_tokens=max_new_tokens,
        logprobs=topk,
        seed=seed,
    )
    outputs = llm.generate(
        [{"prompt_token_ids": seq} for seq in prompt_id_lists],
        sampling_params=sp,
        use_tqdm=False,
    )

    tok_seqs = [list(req.outputs[0].token_ids) for req in outputs]
    lp_seqs = [req.outputs[0].logprobs or [] for req in outputs]
    max_gen_len = max((len(s) for s in tok_seqs), default=1)
    max_gen_len = max(max_gen_len, 1)
    batch_size = len(problems)

    gen_tokens = torch.full((batch_size, max_gen_len), pad_id, dtype=torch.long, device=device)
    loss_mask = torch.zeros((batch_size, max_gen_len), dtype=torch.bool, device=device)
    samp_token_logp = torch.zeros((batch_size, max_gen_len), dtype=torch.float32, device=device)
    samp_topk_logp = torch.full((batch_size, max_gen_len, topk), -30.0, dtype=torch.float32, device=device)
    topk_ids = torch.zeros((batch_size, max_gen_len, topk), dtype=torch.long, device=device)
    texts: list[str] = []

    for i, (toks, lp_list) in enumerate(zip(tok_seqs, lp_seqs)):
        texts.append(tokenizer.decode(toks, skip_special_tokens=True))
        for t, tid in enumerate(toks):
            gen_tokens[i, t] = tid
            loss_mask[i, t] = True
            if t >= len(lp_list) or not lp_list[t]:
                continue
            step_dict = lp_list[t]
            if tid in step_dict:
                samp_token_logp[i, t] = float(step_dict[tid].logprob)
            else:
                samp_token_logp[i, t] = float(min(v.logprob for v in step_dict.values()))

            sorted_items = sorted(step_dict.items(), key=lambda kv: kv[1].logprob, reverse=True)[:topk]
            m = len(sorted_items)
            if m > 0:
                ids_t = torch.tensor([int(k) for k, _ in sorted_items], dtype=torch.long, device=device)
                lps_t = torch.tensor([float(v.logprob) for _, v in sorted_items], dtype=torch.float32, device=device)
                topk_ids[i, t, :m] = ids_t
                samp_topk_logp[i, t, :m] = lps_t
                if m < topk:
                    topk_ids[i, t, m:] = ids_t[-1]
                    samp_topk_logp[i, t, m:] = -30.0

    return {
        "prompt_ids": p_ids,
        "prompt_mask": p_mask,
        "gen_tokens": gen_tokens,
        "loss_mask": loss_mask,
        "samp_token_logp": samp_token_logp,
        "samp_topk_logp": samp_topk_logp,
        "topk_ids": topk_ids,
        "texts": texts,
    }


@torch.no_grad()
def vllm_evaluate_policy(
    llm: LLM,
    trainer_model: AutoModelForCausalLM,
    vllm_dtype: torch.dtype,
    eval_problems: list[CountdownProblem],
    eval_extra_problems: list[CountdownProblem],
    tokenizer: AutoTokenizer,
    max_new_tokens: int,
    return_samples: bool = False,
    target_per_bucket: int = 10,
) -> float | tuple[float, dict[str, list[dict[str, object]]]]:
    """Evaluate clean trainer weights via vLLM and optionally collect 10 success + 10 failure samples."""
    clean_weights = build_sampler_weights(trainer_model, vllm_dtype, fixed_noise=None, fp8_quant=False)
    load_weights_into_vllm(llm, clean_weights)

    sp = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=max_new_tokens,
        seed=9999,
    )

    def _run_problems(probs: list[CountdownProblem]) -> tuple[list[float], list[str]]:
        if not probs:
            return [], []
        prompt_id_lists = build_prompt_ids_list(probs, tokenizer)
        outputs = llm.generate(
            [{"prompt_token_ids": seq} for seq in prompt_id_lists],
            sampling_params=sp,
            use_tqdm=False,
        )
        texts = [tokenizer.decode(list(req.outputs[0].token_ids), skip_special_tokens=True) for req in outputs]
        rewards = [compute_reward(txt, prob) for txt, prob in zip(texts, probs)]
        return rewards, texts

    eval_rewards, eval_texts = _run_problems(eval_problems)
    acc = float(np.mean(eval_rewards))
    if not return_samples:
        return acc

    successes: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    def _collect(probs: list[CountdownProblem], rewards: list[float], texts: list[str], source: str) -> None:
        for prob, r, txt in zip(probs, rewards, texts):
            item = {
                "numbers": prob.numbers,
                "target": prob.target,
                "solution": prob.solution,
                "completion": txt,
                "reward": r,
                "source": source,
            }
            if r > 0.5 and len(successes) < target_per_bucket:
                successes.append(item)
            elif r <= 0.5 and len(failures) < target_per_bucket:
                failures.append(item)

    _collect(eval_problems, eval_rewards, eval_texts, "eval_main")

    # Supplement from held-out extra eval pool only to fill 10 success / 10 failure showcases.
    if (len(successes) < target_per_bucket or len(failures) < target_per_bucket) and eval_extra_problems:
        extra_rewards, extra_texts = _run_problems(eval_extra_problems)
        _collect(eval_extra_problems, extra_rewards, extra_texts, "eval_supplement")

    return acc, {"success": successes, "failure": failures}


@torch.no_grad()
def probe_vllm_drift(
    trainer: AutoModelForCausalLM,
    llm: LLM,
    vllm_dtype: torch.dtype,
    problems: list[CountdownProblem],
    tokenizer: AutoTokenizer,
    max_new_tokens: int,
    topk: int,
    device: torch.device,
    fixed_noise: dict[str, Tensor] | None = None,
    fp8_quant: bool = False,
    engine_label: str = "",
) -> dict[str, float | bool | str]:
    """Measure zero-step logprob discrepancy between the configured vLLM sampler and HF trainer."""
    weights = build_sampler_weights(trainer, vllm_dtype, fixed_noise=fixed_noise, fp8_quant=fp8_quant)
    load_weights_into_vllm(llm, weights)
    roll = vllm_sample_rollouts(
        llm, problems[:12], tokenizer, max_new_tokens=min(max_new_tokens, 96), topk=topk, device=device, seed=1234
    )
    p_ids: Tensor = roll["prompt_ids"]  # type: ignore[assignment]
    p_mask: Tensor = roll["prompt_mask"]  # type: ignore[assignment]
    gen_tokens: Tensor = roll["gen_tokens"]  # type: ignore[assignment]
    loss_mask: Tensor = roll["loss_mask"]  # type: ignore[assignment]
    samp_lp: Tensor = roll["samp_token_logp"]  # type: ignore[assignment]

    full_ids = torch.cat([p_ids, gen_tokens], dim=1)
    full_mask = torch.cat([p_mask, torch.ones_like(gen_tokens)], dim=1)
    full_pos = (full_mask.cumsum(dim=-1) - 1).clamp_min(0)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits = trainer(
            input_ids=full_ids, attention_mask=full_mask, position_ids=full_pos, use_cache=False
        ).logits
    gen_logits = logits[:, p_ids.size(1) - 1 : p_ids.size(1) - 1 + gen_tokens.size(1), :].float()
    train_lp = F.log_softmax(gen_logits, dim=-1).gather(-1, gen_tokens.unsqueeze(-1)).squeeze(-1)

    abs_diff = (train_lp - samp_lp).abs()[loss_mask]
    kl_val = (samp_lp - train_lp)[loss_mask].mean().item()
    max_r = (train_lp - samp_lp).clamp(-20.0, 20.0).exp()[loss_mask].max().item()
    return {
        "sampler_engine": engine_label
        or f"vLLM V1 ({str(vllm_dtype).replace('torch.', '')}, PagedAttention+FlashInfer)",
        "trainer_engine": "HF FP32 master + BF16 eager parallel forward",
        "mean_abs_logp_diff": float(abs_diff.mean().item()),
        "max_abs_logp_diff": float(abs_diff.max().item()),
        "initial_kl_samp_train": float(kl_val),
        "initial_max_is_ratio": float(max_r),
    }


def run_single_vllm_method(
    method_key: str,
    base_state: dict[str, Tensor],
    llm: LLM,
    vllm_dtype: torch.dtype,
    tokenizer: AutoTokenizer,
    train_problems: list[CountdownProblem],
    eval_problems: list[CountdownProblem],
    eval_extra_problems: list[CountdownProblem],
    args: argparse.Namespace,
    device: torch.device,
    fp8_quant: bool = False,
    noise_scale: float = 0.0,
    log_prefix: str = "vLLM TIM",
    checkpoint_cb: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    """Train one RL method with the shared vLLM sampler and HF trainer."""
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

    fixed_noise = (
        make_fixed_noise(trainer, scale=noise_scale, seed=args.seed + 77)
        if noise_scale > 0.0
        else None
    )
    optimizer = torch.optim.SGD(trainer.parameters(), lr=args.lr)
    history: list[dict[str, float]] = []
    eval_samples: dict[str, dict[str, list[dict[str, object]]]] = {}
    cached_sampler_weights: list[tuple[str, Tensor]] | None = None
    t0 = time.time()

    for step in range(args.steps):
        need_sync = step % (args.staleness + 1) == 0 or cached_sampler_weights is None
        if need_sync:
            cached_sampler_weights = build_sampler_weights(
                trainer, vllm_dtype, fixed_noise=fixed_noise, fp8_quant=fp8_quant
            )

        eval_reward = None
        if step % args.eval_every == 0 or step == args.steps - 1:
            trainer.eval()
            capture = step == 0 or step == args.steps - 1
            res_eval = vllm_evaluate_policy(
                llm=llm,
                trainer_model=trainer,
                vllm_dtype=vllm_dtype,
                eval_problems=eval_problems,
                eval_extra_problems=eval_extra_problems,
                tokenizer=tokenizer,
                max_new_tokens=args.max_new_tokens,
                return_samples=capture,
                target_per_bucket=10,
            )
            if capture:
                eval_reward, samp_buckets = res_eval  # type: ignore[misc]
                eval_samples[f"step_{step}"] = samp_buckets
            else:
                eval_reward = float(res_eval)  # type: ignore[arg-type]
            trainer.train()
            # Restore stale/noisy sampler weights because evaluation loaded clean trainer weights.
            load_weights_into_vllm(llm, cached_sampler_weights)
        elif need_sync:
            load_weights_into_vllm(llm, cached_sampler_weights)

        batch_probs = [
            train_problems[(step * args.num_prompts + i) % len(train_problems)]
            for i in range(args.num_prompts)
        ]
        grouped_probs = [p for p in batch_probs for _ in range(args.group_size)]
        roll = vllm_sample_rollouts(
            llm,
            grouped_probs,
            tokenizer,
            max_new_tokens=args.max_new_tokens,
            topk=args.topk,
            device=device,
            seed=args.seed + step * 31,
        )

        rewards_list = [compute_reward(txt, prob) for txt, prob in zip(roll["texts"], grouped_probs)]
        rewards = torch.tensor(rewards_list, dtype=torch.float32, device=device)
        advantages = compute_advantages(rewards, args.group_size, args.reward_mode).unsqueeze(-1)

        p_ids: Tensor = roll["prompt_ids"]  # type: ignore[assignment]
        p_mask: Tensor = roll["prompt_mask"]  # type: ignore[assignment]
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

        for s_idx in range(0, full_ids.size(0), args.micro_batch_size):
            e_idx = min(s_idx + args.micro_batch_size, full_ids.size(0))
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
            del out, gen_logits, train_logp, mb_loss, scaled_loss, mb_train_tok_logp, mb_kl, mb_ratios

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
                f"[{log_prefix} {spec.name:<22}] step={step:3d}/{args.steps} | "
                f"train={rec['train_reward'] * 100:5.1f}%{ev_str} | "
                f"kl={kl_mean:+.4f} | gnorm={grad_norm:.3f} | max_r={max_ratio_val:.2f} | "
                f"len={rec['mean_len']:.1f}",
                flush=True,
            )
        if checkpoint_cb is not None and (eval_reward is not None or step == args.steps - 1):
            checkpoint_cb({
                "method": method_key,
                "name": spec.name,
                "elapsed_sec": time.time() - t0,
                "history": list(history),
                "eval_samples": dict(eval_samples),
            })

    elapsed = time.time() - t0
    del trainer, fixed_noise, optimizer, cached_sampler_weights
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "method": method_key,
        "name": spec.name,
        "elapsed_sec": elapsed,
        "history": history,
        "eval_samples": eval_samples,
    }


def run_vllm_suite(
    args: argparse.Namespace,
    results_filename: str,
    plot_filename: str,
    title_prefix: str,
    engine_label: str,
    log_prefix: str,
    fp8_quant: bool = False,
    noise_scale: float = 0.0,
) -> None:
    """Shared entrypoint executing zero-step probe and 4-method vLLM RL comparison."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{log_prefix}] device={device} | model={args.model_id} | steps={args.steps}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    eval_pool = build_dataset(
        max(args.eval_prompts + 128, 192),
        seed=args.seed + 1000,
        num_operands=args.num_operands,
        max_number=args.max_number,
        max_target=args.max_target,
    )
    eval_problems = eval_pool[: args.eval_prompts]
    eval_extra_problems = eval_pool[args.eval_prompts :]
    train_problems = build_dataset(
        max(512, args.steps * args.num_prompts),
        seed=args.seed,
        num_operands=args.num_operands,
        max_number=args.max_number,
        max_target=args.max_target,
    )

    llm = LLM(
        model=args.model_id,
        dtype=args.vllm_dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_logprobs=args.topk + 4,
        enforce_eager=True,
        enable_prefix_caching=False,
        seed=args.seed,
    )
    vllm_dtype = torch.float16 if args.vllm_dtype == "float16" else torch.bfloat16

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id, dtype=torch.float32, attn_implementation="eager"
    ).to(device)
    base_state = {k: v.cpu().clone() for k, v in base_model.state_dict().items()}
    probe_noise = make_fixed_noise(base_model, scale=noise_scale, seed=args.seed + 77) if noise_scale > 0.0 else None
    probe = probe_vllm_drift(
        base_model,
        llm,
        vllm_dtype,
        eval_problems,
        tokenizer,
        args.max_new_tokens,
        args.topk,
        device,
        fixed_noise=probe_noise,
        fp8_quant=fp8_quant,
        engine_label=engine_label,
    )
    print(f"[{log_prefix} Probe] {json.dumps(probe)}", flush=True)
    del base_model, probe_noise
    gc.collect()
    torch.cuda.empty_cache()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / results_filename
    plot_path = out_dir / plot_filename

    results: list[dict[str, object]] = []
    for key in args.methods:
        def on_ckpt(partial: dict[str, object]) -> None:
            save_json_checkpoint(
                results_path,
                {
                    "status": "running",
                    "args": vars(args),
                    "engine_drift_probe": probe,
                    "results": results + [partial],
                },
            )

        res = run_single_vllm_method(
            method_key=key,
            base_state=base_state,
            llm=llm,
            vllm_dtype=vllm_dtype,
            tokenizer=tokenizer,
            train_problems=train_problems,
            eval_problems=eval_problems,
            eval_extra_problems=eval_extra_problems,
            args=args,
            device=device,
            fp8_quant=fp8_quant,
            noise_scale=noise_scale,
            log_prefix=log_prefix,
            checkpoint_cb=on_ckpt,
        )
        results.append(res)
        save_json_checkpoint(
            results_path,
            {"status": "running", "args": vars(args), "engine_drift_probe": probe, "results": results},
        )

    save_json_checkpoint(
        results_path,
        {"status": "completed", "args": vars(args), "engine_drift_probe": probe, "results": results},
    )
    plot_results(results, plot_path, title_prefix=title_prefix)
    print(f"Saved {results_path} and {plot_path}", flush=True)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the real vLLM TIM experiment."""
    parser = argparse.ArgumentParser(description="Real vLLM inference engine TIM experiment.")
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--methods", nargs="+", default=["pg", "is", "tis", "tis_sc"])
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
    parser.add_argument("--staleness", type=int, default=2)
    parser.add_argument("--weight-mismatch-sigma", type=float, default=0.015)
    parser.add_argument("--topk", type=int, default=128)
    parser.add_argument("--tis-max", type=float, default=1.4)
    parser.add_argument("--reward-mode", default="group_centered")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vllm-dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.18)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--output-dir", default=".")
    return parser.parse_args()


def main() -> None:
    """Run Experiment 2: Real vLLM FP16 PagedAttention+FlashInfer vs HF BF16 trainer."""
    args = parse_args()
    run_vllm_suite(
        args=args,
        results_filename="results.json",
        plot_filename="plot.png",
        title_prefix="Real vLLM Engine TIM (PagedAttention+FlashInfer)",
        engine_label=f"vLLM V1 ({args.vllm_dtype}, PagedAttention+FlashInfer, staleness={args.staleness}, sigma={args.weight_mismatch_sigma})",
        log_prefix="vLLM TIM",
        fp8_quant=False,
        noise_scale=args.weight_mismatch_sigma,
    )


if __name__ == "__main__":
    main()
