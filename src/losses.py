"""PyTorch implementation of REINFORCE, IS, TIS, and Score Centering losses."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor


@dataclass(frozen=True)
class MethodSpec:
    """Configuration defining one off-policy RL baseline or Score Centering variant."""

    key: str
    name: str
    use_sc: bool
    weight_kind: str


METHODS: dict[str, MethodSpec] = {
    "pg": MethodSpec(key="pg", name="PG (Vanilla)", use_sc=False, weight_kind="ones"),
    "is": MethodSpec(key="is", name="Naive IS", use_sc=False, weight_kind="is"),
    "tis": MethodSpec(key="tis", name="TIS", use_sc=False, weight_kind="tis"),
    "tis_sc": MethodSpec(
        key="tis_sc",
        name="TIS + Score Centering",
        use_sc=True,
        weight_kind="tis",
    ),
}

REWARD_MODES: dict[str, tuple[float, float]] = {
    "plus_minus": (1.0, -1.0),
    "plus_zero": (1.0, 0.0),
    "zero_minus": (0.0, -1.0),
    "group_centered": (1.0, -1.0),
}


def make_weight_fn(weight_kind: str, tis_max: float = 2.0) -> Callable[[Tensor], Tensor]:
    """Return a detached ratio weighting function w(p/q)."""
    if weight_kind == "ones":
        return torch.ones_like
    if weight_kind == "is":
        return lambda ratio: ratio
    if weight_kind == "tis":
        return lambda ratio: ratio.clamp_max(tis_max)
    raise ValueError(f"Unsupported weight_kind: {weight_kind!r}")


def compute_advantages(
    rewards: Tensor,
    group_size: int,
    reward_mode: str = "group_centered",
) -> Tensor:
    """Map binary rewards into RL advantages following the paper's REWARD_MODES."""
    if reward_mode not in REWARD_MODES:
        raise ValueError(f"Unsupported reward_mode: {reward_mode!r}")
    solved, unsolved = REWARD_MODES[reward_mode]
    raw = torch.where(rewards > 0.5, solved, unsolved).to(torch.float32)
    if reward_mode == "group_centered":
        groups = raw.view(-1, group_size)
        g_centered = groups - groups.mean(dim=1, keepdim=True)
        if g_centered.abs().max().item() > 1e-6:
            return g_centered.reshape(-1)
        b_std = raw.std(unbiased=False)
        if b_std.item() > 1e-6:
            return (raw - raw.mean()).reshape(-1)
        return torch.zeros_like(raw)
    return raw


def score_centering_loss(
    train_logp: Tensor,
    samp_logp: Tensor,
    topk_ids: Tensor,
    sampled_token: Tensor,
    samp_token_logp: Tensor,
    advantage: Tensor,
    weight_fn: Callable[[Tensor], Tensor] = torch.ones_like,
    use_sc: bool = True,
    eps: float = 1e-6,
) -> Tensor:
    """Compute per-token REINFORCE loss with optional IS weighting and Top-k Score Centering.

    Args:
        train_logp: Float32 trainer log-probabilities of shape [B, T, V].
        samp_logp: Sampler Top-k log-probabilities of shape [B, T, k].
        topk_ids: Sampler Top-k token indices of shape [B, T, k] (torch.long).
        sampled_token: Sampled token IDs of shape [B, T] (torch.long).
        samp_token_logp: Sampler log-probabilities of sampled tokens [B, T].
        advantage: Scalar advantage broadcastable to [B, T].
        weight_fn: Importance ratio weight function w(r).
        use_sc: Whether to subtract the Top-k Score Centering drift correction.
        eps: Numerical floor for tail probability masses.

    Returns:
        Per-token scalar loss tensor of shape [B, T].
    """
    token_logp = train_logp.gather(-1, sampled_token.unsqueeze(-1)).squeeze(-1)
    log_ratio_tok = (token_logp - samp_token_logp).clamp(-20.0, 20.0)
    token_weight = weight_fn(log_ratio_tok.exp()).detach()

    if not use_sc:
        return -advantage.detach() * (token_weight * token_logp)

    head_logp = train_logp.gather(-1, topk_ids)
    p_head = head_logp.exp()
    q_head = samp_logp.exp()
    head_weight = weight_fn((head_logp - samp_logp).clamp(-20.0, 20.0).exp())
    weighted_q_head = q_head * head_weight
    alpha = (
        weighted_q_head.sum(dim=-1, keepdim=True)
        / p_head.sum(dim=-1, keepdim=True).clamp_min(eps)
    ).clamp(0.25, 4.0)
    residual = weighted_q_head - alpha * p_head
    correction = (residual.detach() * head_logp).sum(dim=-1)
    return -advantage.detach() * (token_weight * token_logp - correction)
