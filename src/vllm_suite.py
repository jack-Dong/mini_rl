"""Real vLLM inference engine vs HuggingFace trainer TIM runner on Qwen3-0.6B + Countdown."""
from __future__ import annotations

import os

# Must be set before importing vllm so LLM.apply_model executes in-process on Colab.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import argparse
import gc
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from src.countdown import (
    CountdownProblem,
    build_balanced_dataset,
    build_dataset,
    build_sft_demonstration,
    classify_problem_skill,
    compute_reward,
    compute_shaped_reward,
    extract_answer,
    format_messages,
    has_strict_single_answer_format,
    requires_muldiv,
)


@dataclass(frozen=True)
class VariantConfig:
    """Single-factor or combined RL ablation configuration over an official MethodSpec."""

    key: str
    label: str
    base_method: str = "grpo_sc"
    reward_shaping: str = "none"
    reward_mode_override: str | None = None
    stop_strings: tuple[str, ...] = ()
    kl_beta: float = 0.0
    entropy_coef: float = 0.0
    dynamic_sampling: bool = False
    muldiv_frac: float = 0.0
    sft_steps: int = 0
    optimizer_kind: str = "sgd"
    lr: float | None = None
    warmup_steps: int = 0
    group_size: int | None = None
    use_balanced_data: bool | None = None
from src.engine import (
    build_prompt_ids_list,
    make_fixed_noise,
    save_json_checkpoint,
    set_seed,
)
from src.losses import (
    METHODS,
    compute_advantages,
    make_weight_fn,
    resolve_correction,
    score_centering_loss,
)

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
    need_topk: bool = True,
    stop_strings: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Tensor | list[str]]:
    """Generate rollouts through real vLLM PagedAttention/FlashInfer and unpack logprobs.

    Uses preallocated CPU NumPy buffers and requests logprobs=1 when ``need_topk=False``
    to avoid per-token CUDA tensor allocations and 128-way Python sorting overhead.
    """
    prompt_id_lists = build_prompt_ids_list(problems, tokenizer)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    p_ids, p_mask = _pad_prompt_ids(prompt_id_lists, pad_id, device)

    # One SamplingParams per request with a distinct seed: a single shared seed makes
    # vLLM return identical completions for the group_size copies of each prompt,
    # which collapses every group to all-correct/all-wrong and zeroes the advantages.
    stop_list = list(stop_strings) if stop_strings else None
    sp_list = []
    for i in range(len(prompt_id_lists)):
        sp_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "top_p": 1.0,
            "top_k": -1,
            "max_tokens": max_new_tokens,
            "logprobs": topk if need_topk else 1,
            "seed": None if seed is None else seed * 1000 + i,
        }
        if stop_list:
            sp_kwargs["stop"] = stop_list
            sp_kwargs["include_stop_str_in_output"] = True
        sp_list.append(SamplingParams(**sp_kwargs))
    outputs = llm.generate(
        [{"prompt_token_ids": seq} for seq in prompt_id_lists],
        sampling_params=sp_list,
        use_tqdm=False,
    )

    tok_seqs = [list(req.outputs[0].token_ids) for req in outputs]
    lp_seqs = [req.outputs[0].logprobs or [] for req in outputs]
    max_gen_len = max((len(s) for s in tok_seqs), default=1)
    max_gen_len = max(max_gen_len, 1)
    batch_size = len(problems)

    gen_tokens_np = np.full((batch_size, max_gen_len), pad_id, dtype=np.int64)
    loss_mask_np = np.zeros((batch_size, max_gen_len), dtype=bool)
    samp_token_logp_np = np.zeros((batch_size, max_gen_len), dtype=np.float32)
    head_width = topk if need_topk else 1
    samp_topk_logp_np = np.full((batch_size, max_gen_len, head_width), -30.0, dtype=np.float32)
    topk_ids_np = np.zeros((batch_size, max_gen_len, head_width), dtype=np.int64)
    texts: list[str] = []

    for i, (toks, lp_list) in enumerate(zip(tok_seqs, lp_seqs)):
        texts.append(tokenizer.decode(toks, skip_special_tokens=True))
        row_len = len(toks)
        if row_len == 0:
            continue
        gen_tokens_np[i, :row_len] = toks
        loss_mask_np[i, :row_len] = True
        for t, tid in enumerate(toks):
            if t >= len(lp_list) or not lp_list[t]:
                continue
            step_dict = lp_list[t]
            tok_entry = step_dict.get(tid)
            if tok_entry is not None:
                samp_token_logp_np[i, t] = float(tok_entry.logprob)
            else:
                samp_token_logp_np[i, t] = float(min(v.logprob for v in step_dict.values()))

            if not need_topk:
                continue

            if len(step_dict) <= topk:
                items = list(step_dict.items())
            else:
                items = sorted(step_dict.items(), key=lambda kv: kv[1].logprob, reverse=True)[:topk]
            m = len(items)
            if m > 0:
                ids_arr = [int(k) for k, _ in items]
                lps_arr = [float(v.logprob) for _, v in items]
                topk_ids_np[i, t, :m] = ids_arr
                samp_topk_logp_np[i, t, :m] = lps_arr
                if m < topk:
                    topk_ids_np[i, t, m:] = ids_arr[-1]
                    samp_topk_logp_np[i, t, m:] = -30.0

    return {
        "prompt_ids": p_ids,
        "prompt_mask": p_mask,
        "gen_tokens": torch.from_numpy(gen_tokens_np).to(device),
        "loss_mask": torch.from_numpy(loss_mask_np).to(device),
        "samp_token_logp": torch.from_numpy(samp_token_logp_np).to(device),
        "samp_topk_logp": torch.from_numpy(samp_topk_logp_np).to(device),
        "topk_ids": torch.from_numpy(topk_ids_np).to(device),
        "texts": texts,
    }


def _summarize_eval_metrics(
    problems: list[CountdownProblem],
    rewards: list[float],
    texts: list[str],
) -> dict[str, float]:
    """Compute held-out eval breakdown across add/sub, mul, div, parens, and format."""
    by_skill: dict[str, list[float]] = {
        "addsub": [],
        "mul_flat": [],
        "div_flat": [],
        "mul_parens": [],
        "div_parens": [],
    }
    for prob, rew in zip(problems, rewards):
        by_skill[classify_problem_skill(prob)].append(rew)

    mul_r = by_skill["mul_flat"] + by_skill["mul_parens"]
    div_r = by_skill["div_flat"] + by_skill["div_parens"]
    parens_r = by_skill["mul_parens"] + by_skill["div_parens"]
    muldiv_r = mul_r + div_r

    solved_exprs = [extract_answer(t) for r, t in zip(rewards, texts) if r > 0.5]
    expr_muldiv = [
        1.0 if (expr is not None and ("*" in expr or "/" in expr)) else 0.0
        for expr in solved_exprs
    ]
    strict_fmt = [1.0 if has_strict_single_answer_format(t) else 0.0 for t in texts]
    metrics: dict[str, float] = {
        "eval_addsub_reward": float(np.mean(by_skill["addsub"])) if by_skill["addsub"] else 0.0,
        "eval_mul_reward": float(np.mean(mul_r)) if mul_r else 0.0,
        "eval_div_reward": float(np.mean(div_r)) if div_r else 0.0,
        "eval_parens_reward": float(np.mean(parens_r)) if parens_r else 0.0,
        "eval_muldiv_reward": float(np.mean(muldiv_r)) if muldiv_r else 0.0,
        "eval_expr_muldiv_frac": float(np.mean(expr_muldiv)) if expr_muldiv else 0.0,
        "eval_strict_fmt_rate": float(np.mean(strict_fmt)) if strict_fmt else 0.0,
    }
    for skill, vals in by_skill.items():
        if vals:
            metrics[f"eval_skill_{skill}"] = float(np.mean(vals))
    return metrics


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
    stop_strings: tuple[str, ...] | list[str] | None = None,
    metrics_out: dict[str, float] | None = None,
    alt_eval_problems: list[CountdownProblem] | None = None,
) -> float | tuple[float, dict[str, list[dict[str, object]]]]:
    """Evaluate clean trainer weights via vLLM and optionally collect 10 success + 10 failure samples."""
    clean_weights = build_sampler_weights(trainer_model, vllm_dtype, fixed_noise=None, fp8_quant=False)
    load_weights_into_vllm(llm, clean_weights)

    sp_kwargs: dict[str, Any] = {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "max_tokens": max_new_tokens,
        "seed": 9999,
    }
    if stop_strings:
        sp_kwargs["stop"] = list(stop_strings)
        sp_kwargs["include_stop_str_in_output"] = True
    sp = SamplingParams(**sp_kwargs)

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
    if metrics_out is not None:
        metrics_out.update(_summarize_eval_metrics(eval_problems, eval_rewards, eval_texts))
        if alt_eval_problems:
            alt_rewards, _ = _run_problems(alt_eval_problems)
            metrics_out["eval_orig_reward"] = float(np.mean(alt_rewards))
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
    trainer_attn = getattr(getattr(trainer, "config", None), "_attn_implementation", "sdpa")
    return {
        "sampler_engine": engine_label
        or f"vLLM V1 ({str(vllm_dtype).replace('torch.', '')}, PagedAttention+FlashInfer)",
        "trainer_engine": f"HF FP32 master + BF16 {trainer_attn} parallel forward",
        "mean_abs_logp_diff": float(abs_diff.mean().item()),
        "max_abs_logp_diff": float(abs_diff.max().item()),
        "initial_kl_samp_train": float(kl_val),
        "initial_max_is_ratio": float(max_r),
    }


def _select_batch_prompts(
    cursor: int,
    num_prompts: int,
    train_problems: list[CountdownProblem],
    muldiv_probs: list[CountdownProblem],
    muldiv_frac: float,
) -> list[CountdownProblem]:
    """Select training prompts for one step, optionally enforcing a */÷ curriculum fraction."""
    if muldiv_frac <= 0.0 or not muldiv_probs:
        return [train_problems[(cursor + i) % len(train_problems)] for i in range(num_prompts)]
    n_md = min(num_prompts, round(num_prompts * muldiv_frac))
    selected: list[CountdownProblem] = []
    chosen_ids: set[int] = set()
    for i in range(n_md):
        p = muldiv_probs[(cursor + i) % len(muldiv_probs)]
        selected.append(p)
        chosen_ids.add(id(p))
    scan = cursor
    while len(selected) < num_prompts:
        p = train_problems[scan % len(train_problems)]
        scan += 1
        if id(p) not in chosen_ids or scan - cursor > len(train_problems):
            selected.append(p)
            chosen_ids.add(id(p))
    return selected


def _concat_rollout_slices(
    collected: list[tuple[dict[str, Any], int, int]],
    pad_id: int,
) -> dict[str, Tensor | list[str]]:
    """Concatenate selected prompt-group slices across candidate rounds with uniform padding."""
    max_p = max(int(r["prompt_ids"].size(1)) for r, _, _ in collected)
    max_g = max(int(r["gen_tokens"].size(1)) for r, _, _ in collected)

    p_ids_list, p_mask_list = [], []
    gen_list, mask_list, tok_lp_list, topk_lp_list, topk_id_list = [], [], [], [], []
    texts: list[str] = []

    for r, s, e in collected:
        p_chunk = r["prompt_ids"][s:e]
        pm_chunk = r["prompt_mask"][s:e]
        if p_chunk.size(1) < max_p:
            pad_w = max_p - p_chunk.size(1)
            p_chunk = F.pad(p_chunk, (pad_w, 0), value=pad_id)
            pm_chunk = F.pad(pm_chunk, (pad_w, 0), value=0)
        p_ids_list.append(p_chunk)
        p_mask_list.append(pm_chunk)

        g_chunk = r["gen_tokens"][s:e]
        lm_chunk = r["loss_mask"][s:e]
        tlp_chunk = r["samp_token_logp"][s:e]
        klp_chunk = r["samp_topk_logp"][s:e]
        kid_chunk = r["topk_ids"][s:e]
        if g_chunk.size(1) < max_g:
            pad_g = max_g - g_chunk.size(1)
            g_chunk = F.pad(g_chunk, (0, pad_g), value=pad_id)
            lm_chunk = F.pad(lm_chunk, (0, pad_g), value=False)
            tlp_chunk = F.pad(tlp_chunk, (0, pad_g), value=0.0)
            klp_chunk = F.pad(klp_chunk, (0, 0, 0, pad_g), value=-30.0)
            kid_chunk = F.pad(kid_chunk, (0, 0, 0, pad_g), value=0)
        gen_list.append(g_chunk)
        mask_list.append(lm_chunk)
        tok_lp_list.append(tlp_chunk)
        topk_lp_list.append(klp_chunk)
        topk_id_list.append(kid_chunk)
        texts.extend(r["texts"][s:e])

    return {
        "prompt_ids": torch.cat(p_ids_list, dim=0),
        "prompt_mask": torch.cat(p_mask_list, dim=0),
        "gen_tokens": torch.cat(gen_list, dim=0),
        "loss_mask": torch.cat(mask_list, dim=0),
        "samp_token_logp": torch.cat(tok_lp_list, dim=0),
        "samp_topk_logp": torch.cat(topk_lp_list, dim=0),
        "topk_ids": torch.cat(topk_id_list, dim=0),
        "texts": texts,
    }


def _run_sft_warmstart(
    trainer: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    sft_steps: int,
    args: argparse.Namespace,
    log_prefix: str,
    label: str,
    batch_size: int = 8,
    lr: float = 1e-5,
    exclude_sigs: set[tuple[tuple[int, ...], int]] | None = None,
) -> float:
    """Warm-start weights on balanced minimal-parens Countdown CoT demos before RL."""
    sft_pool = build_balanced_dataset(
        max(512, sft_steps * batch_size * 2),
        seed=args.seed + 5000,
        num_operands=args.num_operands,
        max_number=args.max_number,
        max_target=args.max_target,
        exclude_sigs=exclude_sigs,
    )
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else pad_id
    opt = torch.optim.AdamW(trainer.parameters(), lr=lr, weight_decay=0.0)
    last_loss = 0.0
    trainer.train()

    for s_step in range(sft_steps):
        probs = [sft_pool[(s_step * batch_size + j) % len(sft_pool)] for j in range(batch_size)]
        seqs: list[list[int]] = []
        lbls: list[list[int]] = []
        for p in probs:
            demo = build_sft_demonstration(p) or ""
            msgs = format_messages(p)
            p_ids = build_prompt_ids_list([p], tokenizer)[0]
            try:
                full = list(
                    tokenizer.apply_chat_template(
                        msgs + [{"role": "assistant", "content": demo}],
                        add_generation_prompt=False,
                        tokenize=True,
                        return_dict=False,
                        enable_thinking=False,
                    )
                )
            except TypeError:
                full = list(
                    tokenizer.apply_chat_template(
                        msgs + [{"role": "assistant", "content": demo}],
                        add_generation_prompt=False,
                        tokenize=True,
                        return_dict=False,
                    )
                )
            if full[: len(p_ids)] != p_ids:
                ans_ids = tokenizer.encode(demo, add_special_tokens=False) + [eos_id]
                full = p_ids + ans_ids
            lbl = [-100] * len(p_ids) + full[len(p_ids) :]
            seqs.append(full)
            lbls.append(lbl)

        max_len = max(len(s) for s in seqs)
        input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long, device=device)
        attn_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long, device=device)
        for i, (s, l) in enumerate(zip(seqs, lbls)):
            input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
            attn_mask[i, : len(s)] = 1
            labels[i, : len(l)] = torch.tensor(l, dtype=torch.long, device=device)
        pos_ids = (attn_mask.cumsum(dim=-1) - 1).clamp_min(0)

        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = trainer(
                input_ids=input_ids,
                attention_mask=attn_mask,
                position_ids=pos_ids,
                labels=labels,
                use_cache=False,
            )
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainer.parameters(), max_norm=args.max_grad_norm)
        opt.step()
        last_loss = float(loss.item())
        del out, loss, input_ids, attn_mask, labels, pos_ids

    del opt
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[{log_prefix} {label:<22}] SFT warmup {sft_steps} steps done | last_loss={last_loss:.4f}", flush=True)
    return last_loss


def _build_optimizer_and_scheduler(
    trainer: AutoModelForCausalLM,
    vcfg: VariantConfig,
    args: argparse.Namespace,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler | None]:
    """Create either baseline SGD or warmup+cosine AdamW optimizer for RL updates."""
    lr = vcfg.lr if vcfg.lr is not None else args.lr
    if vcfg.optimizer_kind == "adamw":
        opt: torch.optim.Optimizer = torch.optim.AdamW(trainer.parameters(), lr=lr, weight_decay=0.0)
        warmup = max(1, vcfg.warmup_steps)

        def _lr_lambda(curr_step: int) -> float:
            if curr_step < warmup:
                return float(curr_step + 1) / float(warmup)
            progress = min(1.0, float(curr_step - warmup) / float(max(1, args.steps - warmup)))
            return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_lambda)
        return opt, sched
    return torch.optim.SGD(trainer.parameters(), lr=lr), None


def _sample_rl_batch(
    llm: LLM,
    tokenizer: AutoTokenizer,
    train_problems: list[CountdownProblem],
    muldiv_probs: list[CountdownProblem],
    vcfg: VariantConfig,
    spec_use_sc: bool,
    step: int,
    eff_group_size: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Any], list[float], list[float], float, int]:
    """Sample one RL batch, optionally resampling via DAPO dynamic sampling for non-zero variance."""
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    stop_arg = vcfg.stop_strings or None
    max_rounds = (4 if eff_group_size >= 16 else 6) if vcfg.dynamic_sampling else 1
    collected: list[tuple[dict[str, Any], int, int]] = []
    collected_raw: list[float] = []
    collected_shaped: list[float] = []
    fallback: tuple[dict[str, Any], list[float], list[float]] | None = None
    init_raw_mean = 0.0
    rounds_used = 0

    for rnd in range(max_rounds):
        rounds_used = rnd + 1
        cursor = step * args.num_prompts + rnd * args.num_prompts * 7
        batch_probs = _select_batch_prompts(cursor, args.num_prompts, train_problems, muldiv_probs, vcfg.muldiv_frac)
        grouped_probs = [p for p in batch_probs for _ in range(eff_group_size)]
        roll = vllm_sample_rollouts(
            llm,
            grouped_probs,
            tokenizer,
            max_new_tokens=args.max_new_tokens,
            topk=args.topk,
            device=device,
            seed=args.seed + step * 31 + rnd * 1009,
            need_topk=spec_use_sc,
            stop_strings=stop_arg,
        )
        pairs = [compute_shaped_reward(txt, prob, vcfg.reward_shaping) for txt, prob in zip(roll["texts"], grouped_probs)]
        raw_list = [p[0] for p in pairs]
        shaped_list = [p[1] for p in pairs]
        if rnd == 0:
            init_raw_mean = float(np.mean(raw_list))
            fallback = (roll, raw_list, shaped_list)
        if not vcfg.dynamic_sampling:
            return roll, raw_list, shaped_list, init_raw_mean, 1

        for g_idx in range(len(batch_probs)):
            s = g_idx * eff_group_size
            e = s + eff_group_size
            if float(np.std(shaped_list[s:e])) > 1e-6:
                collected.append((roll, s, e))
                collected_raw.extend(raw_list[s:e])
                collected_shaped.extend(shaped_list[s:e])
                if len(collected) >= args.num_prompts:
                    break
        if len(collected) >= args.num_prompts:
            break

    if collected:
        merged_roll = _concat_rollout_slices(collected, pad_id=pad_id)
        return merged_roll, collected_raw, collected_shaped, init_raw_mean, rounds_used
    assert fallback is not None
    return fallback[0], fallback[1], fallback[2], init_raw_mean, rounds_used


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
    init_eval: tuple[float, dict[str, list[dict[str, object]]]] | None = None,
    variant: VariantConfig | None = None,
    alt_eval_problems: list[CountdownProblem] | None = None,
    init_eval_metrics: dict[str, float] | None = None,
) -> dict[str, object]:
    """Train one RL method or VariantConfig ablation with shared vLLM sampler and HF trainer."""
    vcfg = variant or VariantConfig(key=method_key, label=METHODS[method_key].name, base_method=method_key)
    spec = METHODS[vcfg.base_method]
    corr = resolve_correction(vcfg.base_method, tis_max=args.tis_max)
    weight_fn = make_weight_fn(spec.weight_kind, tis_max=args.tis_max)
    eff_group_size = vcfg.group_size or args.group_size
    set_seed(args.seed)

    attn_impl = "sdpa" if device.type == "cuda" else "eager"
    trainer = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.float32,
        attn_implementation=attn_impl,
    ).to(device)
    trainer.load_state_dict(base_state)
    trainer.train()

    eval_sigs = {
        (tuple(sorted(p.numbers)), p.target)
        for p in (eval_problems + eval_extra_problems + (alt_eval_problems or []))
    }
    if vcfg.sft_steps > 0:
        _run_sft_warmstart(
            trainer=trainer,
            tokenizer=tokenizer,
            device=device,
            sft_steps=vcfg.sft_steps,
            args=args,
            log_prefix=log_prefix,
            label=vcfg.label,
            exclude_sigs=eval_sigs,
        )

    ref_model: AutoModelForCausalLM | None = None
    if vcfg.kl_beta > 0.0:
        ref_model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            dtype=torch.bfloat16,
            attn_implementation=attn_impl,
        ).to(device)
        ref_model.load_state_dict({k: v.to(torch.bfloat16) for k, v in base_state.items()})
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)

    fixed_noise = (
        make_fixed_noise(trainer, scale=noise_scale, seed=args.seed + 77)
        if noise_scale > 0.0
        else None
    )
    optimizer, scheduler = _build_optimizer_and_scheduler(trainer, vcfg, args)
    muldiv_probs = [p for p in train_problems if requires_muldiv(p)] if vcfg.muldiv_frac > 0.0 else []
    history: list[dict[str, float]] = []
    eval_samples: dict[str, dict[str, list[dict[str, object]]]] = {}
    cached_sampler_weights: list[tuple[str, Tensor]] | None = None
    weights_dirty = True
    vllm_has_sampler_weights = False

    sample_steps_set: set[int] = {0, args.steps - 1}
    if args.steps > 300:
        sample_steps_set.add(300)
    extra_sample_steps = getattr(args, "sample_steps", None) or []
    sample_steps_set.update(int(s) for s in extra_sample_steps if 0 <= int(s) < args.steps)

    t0 = time.time()

    for step in range(args.steps):
        need_sync = step % (args.staleness + 1) == 0 or cached_sampler_weights is None
        if need_sync and weights_dirty:
            cached_sampler_weights = build_sampler_weights(
                trainer, vllm_dtype, fixed_noise=fixed_noise, fp8_quant=fp8_quant
            )
            weights_dirty = False
            vllm_has_sampler_weights = False

        eval_reward = None
        step_eval_metrics: dict[str, float] = {}
        do_eval = step % args.eval_every == 0 or step in sample_steps_set
        if do_eval:
            capture = step in sample_steps_set
            can_reuse_init = (
                step == 0
                and init_eval is not None
                and not vcfg.stop_strings
                and vcfg.sft_steps == 0
            )
            if can_reuse_init:
                assert init_eval is not None
                eval_reward = float(init_eval[0])
                if init_eval_metrics:
                    step_eval_metrics.update(init_eval_metrics)
                if capture:
                    eval_samples["step_0"] = init_eval[1]
            else:
                trainer.eval()
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
                    stop_strings=vcfg.stop_strings or None,
                    metrics_out=step_eval_metrics,
                    alt_eval_problems=alt_eval_problems,
                )
                if capture:
                    eval_reward, samp_buckets = res_eval  # type: ignore[misc]
                    eval_samples[f"step_{step}"] = samp_buckets
                else:
                    eval_reward = float(res_eval)  # type: ignore[arg-type]
                trainer.train()
                # Eval loaded clean weights into vLLM; restore sampler weights below.
                vllm_has_sampler_weights = False

        if not vllm_has_sampler_weights:
            assert cached_sampler_weights is not None
            load_weights_into_vllm(llm, cached_sampler_weights)
            vllm_has_sampler_weights = True

        roll, _raw_list, shaped_list, init_raw_mean, dyn_rounds = _sample_rl_batch(
            llm=llm,
            tokenizer=tokenizer,
            train_problems=train_problems,
            muldiv_probs=muldiv_probs,
            vcfg=vcfg,
            spec_use_sc=spec.use_sc,
            step=step,
            eff_group_size=eff_group_size,
            args=args,
            device=device,
        )
        shaped_rewards = torch.tensor(shaped_list, dtype=torch.float32, device=device)
        method_reward_mode = vcfg.reward_mode_override or spec.reward_mode_override or args.reward_mode
        advantages = compute_advantages(shaped_rewards, eff_group_size, method_reward_mode).unsqueeze(-1)
        frac_nonzero_adv = float((advantages.abs() > 1e-8).float().mean().item())
        has_signal = float(advantages.abs().max().item()) > 1e-8

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

        total_loss_val = 0.0
        total_kl_val = 0.0
        total_kl_ref_val = 0.0
        total_ent_val = 0.0
        total_masked_val = 0.0
        max_ratio_val = 1.0

        if has_signal:
            optimizer.zero_grad(set_to_none=True)
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
                mb_train_tok_logp = train_logp.gather(-1, mb_tok.unsqueeze(-1)).squeeze(-1)
                mb_loss, mb_masked = score_centering_loss(
                    train_logp=train_logp,
                    samp_logp=samp_topk_logp[s_idx:e_idx],
                    topk_ids=topk_ids[s_idx:e_idx],
                    sampled_token=mb_tok,
                    samp_token_logp=mb_samp_tok_logp,
                    advantage=advantages[s_idx:e_idx],
                    weight_fn=weight_fn,
                    use_sc=spec.use_sc,
                    correction=corr,
                    loss_mask=mb_bool,
                    return_stats=True,
                )
                if ref_model is not None and vcfg.kl_beta > 0.0:
                    with torch.no_grad():
                        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                            ref_out = ref_model(
                                input_ids=full_ids[s_idx:e_idx],
                                attention_mask=full_mask[s_idx:e_idx],
                                position_ids=full_pos[s_idx:e_idx],
                                use_cache=False,
                            )
                        ref_logits = ref_out.logits[:, prompt_len - 1 : prompt_len - 1 + gen_len, :].float()
                        ref_tok_logp = F.log_softmax(ref_logits, dim=-1).gather(-1, mb_tok.unsqueeze(-1)).squeeze(-1)
                        del ref_out, ref_logits
                    kl_diff = (mb_train_tok_logp - ref_tok_logp).clamp(-10.0, 10.0)
                    kl_ref_tok = 0.5 * kl_diff.pow(2)
                    mb_loss = mb_loss + vcfg.kl_beta * kl_ref_tok
                    total_kl_ref_val += float(torch.where(mb_bool, kl_ref_tok.detach(), 0.0).sum().item())

                if vcfg.entropy_coef > 0.0:
                    tok_ent = -(train_logp.exp() * train_logp).sum(dim=-1)
                    mb_loss = mb_loss - vcfg.entropy_coef * tok_ent
                    total_ent_val += float(torch.where(mb_bool, tok_ent.detach(), 0.0).sum().item())

                if spec.seq_norm:
                    mb_seq_lens = mb_bool.sum(dim=-1, keepdim=True).to(torch.float32).clamp_min(1.0)
                    scaled_loss = (torch.where(mb_bool, mb_loss, 0.0) / mb_seq_lens).sum() / float(full_ids.size(0))
                else:
                    scaled_loss = torch.where(mb_bool, mb_loss, 0.0).sum() / total_tokens
                scaled_loss.backward()

                with torch.no_grad():
                    mb_kl = torch.where(mb_bool, mb_samp_tok_logp - mb_train_tok_logp, 0.0).sum()
                    mb_ratios = (mb_train_tok_logp - mb_samp_tok_logp).clamp(-20.0, 20.0).exp()
                    mb_max_r = torch.where(mb_bool, mb_ratios, 0.0).max().item()
                    total_loss_val += float(scaled_loss.item())
                    total_kl_val += float(mb_kl.item())
                    total_masked_val += float((mb_masked & mb_bool).sum().item())
                    max_ratio_val = max(max_ratio_val, mb_max_r)
                del out, gen_logits, train_logp, mb_loss, mb_masked, scaled_loss, mb_train_tok_logp, mb_kl, mb_ratios

            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(trainer.parameters(), max_norm=args.max_grad_norm).item()
            )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            weights_dirty = True
        else:
            # Uniform-reward batch produces identically zero policy gradient; skip backward/step.
            with torch.no_grad():
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
                    mb_train_tok_logp = train_logp.gather(-1, mb_tok.unsqueeze(-1)).squeeze(-1)
                    mb_kl = torch.where(mb_bool, mb_samp_tok_logp - mb_train_tok_logp, 0.0).sum()
                    mb_ratios = (mb_train_tok_logp - mb_samp_tok_logp).clamp(-20.0, 20.0).exp()
                    mb_max_r = torch.where(mb_bool, mb_ratios, 0.0).max().item()
                    total_kl_val += float(mb_kl.item())
                    max_ratio_val = max(max_ratio_val, mb_max_r)
                    del out, gen_logits, train_logp, mb_train_tok_logp, mb_kl, mb_ratios
            grad_norm = 0.0

        denom_tok = float(total_tokens.item())
        kl_mean = total_kl_val / denom_tok
        frac_masked = total_masked_val / denom_tok
        rec: dict[str, float] = {
            "step": float(step),
            "train_reward": init_raw_mean,
            "loss": total_loss_val,
            "kl_samp_train": kl_mean,
            "grad_norm": grad_norm,
            "max_is_ratio": max_ratio_val,
            "frac_masked": frac_masked,
            "frac_nonzero_adv": frac_nonzero_adv,
            "mean_len": float(loss_mask.float().sum(dim=1).mean().item()),
        }
        if vcfg.kl_beta > 0.0:
            rec["kl_ref"] = total_kl_ref_val / denom_tok
        if vcfg.entropy_coef > 0.0:
            rec["entropy"] = total_ent_val / denom_tok
        if vcfg.dynamic_sampling:
            rec["dyn_rounds"] = float(dyn_rounds)
        if eval_reward is not None:
            rec["eval_reward"] = eval_reward
            rec.update(step_eval_metrics)
        history.append(rec)

        if eval_reward is not None or step % 10 == 0 or step == args.steps - 1:
            ev_str = f" | eval={eval_reward * 100:5.1f}%" if eval_reward is not None else ""
            if step_eval_metrics:
                orig_p = step_eval_metrics.get("eval_orig_reward")
                orig_s = f" orig={orig_p * 100:4.1f}%" if orig_p is not None else ""
                ev_str += (
                    f" ({orig_s.strip()} as={step_eval_metrics.get('eval_addsub_reward', 0.0)*100:4.1f}%"
                    f" mul={step_eval_metrics.get('eval_mul_reward', 0.0)*100:4.1f}%"
                    f" div={step_eval_metrics.get('eval_div_reward', 0.0)*100:4.1f}%"
                    f" par={step_eval_metrics.get('eval_parens_reward', 0.0)*100:4.1f}%)"
                )
            print(
                f"[{log_prefix} {vcfg.label:<28}] step={step:3d}/{args.steps} | "
                f"train={rec['train_reward'] * 100:5.1f}%{ev_str} | "
                f"kl={kl_mean:+.4f} | gnorm={grad_norm:.3f} | max_r={max_ratio_val:.2f} | "
                f"mask={frac_masked * 100:4.1f}% | len={rec['mean_len']:.1f}",
                flush=True,
            )
        if checkpoint_cb is not None and (eval_reward is not None or step == args.steps - 1):
            checkpoint_cb({
                "method": vcfg.key,
                "name": vcfg.label,
                "elapsed_sec": time.time() - t0,
                "history": list(history),
                "eval_samples": dict(eval_samples),
            })

    elapsed = time.time() - t0
    del trainer, fixed_noise, optimizer, scheduler, cached_sampler_weights, ref_model
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "method": vcfg.key,
        "name": vcfg.label,
        "elapsed_sec": elapsed,
        "history": history,
        "eval_samples": eval_samples,
    }


METHOD_COLORS: dict[str, str] = {
    "pg": "#ef4444",
    "is": "#f59e0b",
    "tis": "#3b82f6",
    "mis": "#ec4899",
    "ppo": "#8b5cf6",
    "dapo": "#06b6d4",
    "gspo": "#f97316",
    "topr": "#84cc16",
    "dppo": "#6366f1",
    "grpo": "#e11d48",
    "sc": "#14b8a6",
    "tis_sc": "#10b981",
    "mis_sc": "#22c55e",
    "ppo_sc": "#a855f7",
    "dapo_sc": "#0891b2",
    "grpo_sc": "#059669",
    "grpo_dapo_sc": "#0d9488",
    "base_bal": "#64748b",
    "fmt_stop_dedup": "#2563eb",
    "kl_ent_dapo": "#059669",
    "sample_var": "#d97706",
    "sft_adamw": "#7c3aed",
    "orig_data_500": "#ef4444",
    "base_bal_500": "#64748b",
    "best_combo_500": "#10b981",
}


def plot_vllm_results(
    all_results: list[dict[str, object]],
    out_path: Path,
    title_prefix: str = "Qwen3-0.6B Countdown",
) -> None:
    """Plot eval reward and KL drift curves across arbitrary official RL methods."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), dpi=150)

    for res in all_results:
        key = str(res["method"])
        name = str(res["name"])
        hist = res["history"]  # type: ignore[assignment]
        color = METHOD_COLORS.get(key, "#6b7280")

        eval_pts = [(h["step"], h["eval_reward"] * 100.0) for h in hist if "eval_reward" in h]
        if eval_pts:
            xs, ys = zip(*eval_pts)
            axes[0].plot(xs, ys, marker="o", markersize=3.5, linewidth=2.0, label=name, color=color)

        steps = [h["step"] for h in hist]
        kls = [abs(h["kl_samp_train"]) for h in hist]
        axes[1].plot(steps, kls, linewidth=1.6, alpha=0.88, label=name, color=color)

    axes[0].set_title(f"{title_prefix}: Eval Solve Rate")
    axes[0].set_xlabel("Training Step")
    axes[0].set_ylabel("Solve Rate (%)")
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend(frameon=True, fontsize=8, ncol=2)

    axes[1].set_title(f"{title_prefix}: Sampler-Trainer KL Drift |KL(q || p)|")
    axes[1].set_xlabel("Training Step")
    axes[1].set_ylabel("|log q - log p|")
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].legend(frameon=True, fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run_vllm_suite(
    args: argparse.Namespace,
    results_filename: str,
    plot_filename: str,
    title_prefix: str,
    engine_label: str,
    log_prefix: str,
    fp8_quant: bool = False,
    noise_scale: float = 0.0,
    balanced_data: bool = False,
    variant_specs: dict[str, VariantConfig] | None = None,
    resume: bool = False,
) -> None:
    """Shared entrypoint executing zero-step probe and multi-method vLLM RL comparison."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attn_impl = "sdpa" if device.type == "cuda" else "eager"
    print(f"[{log_prefix}] device={device} | attn={attn_impl} | model={args.model_id} | steps={args.steps}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    orig_eval_pool = build_dataset(
        max(args.eval_prompts + 192, 256),
        seed=args.seed + 1000,
        num_operands=args.num_operands,
        max_number=args.max_number,
        max_target=args.max_target,
    )
    orig_eval_main = orig_eval_pool[: args.eval_prompts]
    unbalanced_train_pool = build_dataset(
        max(512, args.steps * args.num_prompts * 2),
        seed=args.seed,
        num_operands=args.num_operands,
        max_number=args.max_number,
        max_target=args.max_target,
    )

    if balanced_data:
        orig_sigs = {(tuple(sorted(p.numbers)), p.target) for p in orig_eval_main}
        eval_pool = build_balanced_dataset(
            max(args.eval_prompts + 64, 128),
            seed=args.seed + 2000,
            num_operands=args.num_operands,
            max_number=args.max_number,
            max_target=args.max_target,
            exclude_sigs=orig_sigs,
        )
        eval_problems = eval_pool[: args.eval_prompts]
        eval_extra_problems = eval_pool[args.eval_prompts :]
        alt_eval_problems: list[CountdownProblem] | None = orig_eval_main
        all_eval_sigs = {(tuple(sorted(p.numbers)), p.target) for p in (eval_pool + orig_eval_main)}
        train_problems = build_balanced_dataset(
            max(800, args.steps * args.num_prompts * 2),
            seed=args.seed,
            num_operands=args.num_operands,
            max_number=args.max_number,
            max_target=args.max_target,
            exclude_sigs=all_eval_sigs,
        )
    else:
        eval_problems = orig_eval_main
        eval_extra_problems = orig_eval_pool[args.eval_prompts :]
        alt_eval_problems = None
        train_problems = unbalanced_train_pool[: max(512, args.steps * args.num_prompts)]

    llm = LLM(
        model=args.model_id,
        dtype=args.vllm_dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_logprobs=args.topk + 4,
        enforce_eager=bool(getattr(args, "enforce_eager", True)),
        enable_prefix_caching=False,
        seed=args.seed,
    )
    vllm_dtype = torch.float16 if args.vllm_dtype == "float16" else torch.bfloat16

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id, dtype=torch.float32, attn_implementation=attn_impl
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

    init_eval: tuple[float, dict[str, list[dict[str, object]]]] | None = None
    init_eval_metrics: dict[str, float] = {}
    if args.steps > 0:
        acc0, buckets0 = vllm_evaluate_policy(
            llm=llm,
            trainer_model=base_model,
            vllm_dtype=vllm_dtype,
            eval_problems=eval_problems,
            eval_extra_problems=eval_extra_problems,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            return_samples=True,
            target_per_bucket=10,
            metrics_out=init_eval_metrics,
            alt_eval_problems=alt_eval_problems,
        )  # type: ignore[misc]
        init_eval = (float(acc0), buckets0)
        print(f"[{log_prefix} InitEval] step_0 eval={acc0 * 100:.1f}% | metrics={json.dumps(init_eval_metrics)}", flush=True)

    del base_model, probe_noise
    gc.collect()
    torch.cuda.empty_cache()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / results_filename
    plot_path = out_dir / plot_filename

    completed_by_key: dict[str, dict[str, object]] = {}
    if resume and results_path.exists():
        try:
            prev = json.loads(results_path.read_text(encoding="utf-8"))
            for r in prev.get("results", []):
                mkey = str(r.get("method", ""))
                hist = r.get("history", [])
                if mkey and isinstance(hist, list) and len(hist) >= args.steps:
                    completed_by_key[mkey] = r
                    print(f"[{log_prefix} Resume] Reusing completed {mkey} ({len(hist)} steps)", flush=True)
        except Exception:
            completed_by_key = {}

    results: list[dict[str, object]] = [completed_by_key[k] for k in args.methods if k in completed_by_key]
    for key in args.methods:
        if key in completed_by_key:
            continue
        vcfg = variant_specs.get(key) if variant_specs else None
        per_train = (
            unbalanced_train_pool[: max(512, args.steps * args.num_prompts)]
            if (vcfg is not None and vcfg.use_balanced_data is False)
            else train_problems
        )

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
            train_problems=per_train,
            eval_problems=eval_problems,
            eval_extra_problems=eval_extra_problems,
            args=args,
            device=device,
            fp8_quant=fp8_quant,
            noise_scale=noise_scale,
            log_prefix=log_prefix,
            checkpoint_cb=on_ckpt,
            init_eval=init_eval,
            variant=vcfg,
            alt_eval_problems=alt_eval_problems,
            init_eval_metrics=init_eval_metrics,
        )
        results.append(res)
        save_json_checkpoint(
            results_path,
            {"status": "running", "args": vars(args), "engine_drift_probe": probe, "results": results},
        )

    order_map = {k: i for i, k in enumerate(args.methods)}
    results.sort(key=lambda r: order_map.get(str(r.get("method", "")), 999))
    save_json_checkpoint(
        results_path,
        {"status": "completed", "args": vars(args), "engine_drift_probe": probe, "results": results},
    )
    plot_vllm_results(results, plot_path, title_prefix=title_prefix)
    print(f"Saved {results_path} and {plot_path}", flush=True)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the real vLLM TIM experiment."""
    parser = argparse.ArgumentParser(description="Real vLLM inference engine TIM experiment.")
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--methods", nargs="+", default=["pg", "is", "tis", "tis_sc"])
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--sample-steps", type=int, nargs="*", default=None)
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
    parser.add_argument("--tis-max", type=float, default=2.0)
    parser.add_argument("--reward-mode", default="group_centered")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vllm-dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.18)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", default=".")
    return parser.parse_args()


def main() -> None:
    """Run Real vLLM FP16 PagedAttention+FlashInfer vs HF BF16 trainer TIM suite."""
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
