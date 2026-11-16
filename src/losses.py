"""PyTorch implementation of REINFORCE, IS, TIS, MIS, PPO, DAPO, GSPO, TOPR, DPPO, and Score Centering losses.

Aligned with the official martin-marek/score-centering (arXiv:2609.20807) losses.py framework.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable

import torch
from torch import Tensor

DUAL_CLIP = 3.0  # PPO dual-clip upper ratio bound for negative advantages (arXiv:1912.09729)

REWARD_MODES: dict[str, tuple[float, float]] = {
    "plus_minus": (1.0, -1.0),
    "plus_zero": (1.0, 0.0),
    "zero_minus": (0.0, -1.0),
    "group_centered": (1.0, -1.0),
    "group_std": (1.0, 0.0),
    "group_std_raw": (1.0, 0.0),
}


@dataclass(frozen=True)
class ISCorrection:
    """Importance-sampling correction point on the official score-centering grid."""

    level: str | None = "token"   # None | "token" | "seq_mean" | "seq_sum"
    stat: str = "ratio"           # "ratio" | "tv"
    low: float = 0.0
    high: float = math.inf
    outside: str = "drop"         # "clamp" | "drop"
    sign: str = "both"            # "both" | "neg" | "pos" | "ppo"

    def check(self) -> ISCorrection:
        choices = {
            "level": (None, "token", "seq_mean", "seq_sum"),
            "stat": ("ratio", "tv"),
            "outside": ("clamp", "drop"),
            "sign": ("both", "neg", "pos", "ppo"),
        }
        for field, allowed in choices.items():
            assert getattr(self, field) in allowed, f"ISCorrection.{field} must be in {allowed}"
        assert self.stat == "ratio" or self.level != "seq_sum", "seq_sum requires stat='ratio'"
        return self


@dataclass(frozen=True)
class MethodSpec:
    """Configuration defining one off-policy RL baseline or Score Centering variant."""

    key: str
    name: str
    use_sc: bool
    weight_kind: str = "ones"
    correction: ISCorrection | None = None
    reward_mode_override: str | None = None
    seq_norm: bool = False


METHODS: dict[str, MethodSpec] = {
    "pg": MethodSpec(
        key="pg",
        name="PG (Vanilla)",
        use_sc=False,
        weight_kind="ones",
        correction=ISCorrection(level=None).check(),
    ),
    "is": MethodSpec(
        key="is",
        name="Naive IS",
        use_sc=False,
        weight_kind="is",
        correction=ISCorrection(level="token").check(),
    ),
    "tis": MethodSpec(
        key="tis",
        name="TIS",
        use_sc=False,
        weight_kind="tis",
        correction=ISCorrection(level="token", high=2.0, outside="clamp").check(),
    ),
    "mis": MethodSpec(
        key="mis",
        name="MIS",
        use_sc=False,
        weight_kind="mis",
        correction=ISCorrection(level="token", low=0.5, high=5.0, outside="drop").check(),
    ),
    "ppo": MethodSpec(
        key="ppo",
        name="PPO",
        use_sc=False,
        weight_kind="ppo",
        correction=ISCorrection(level="token", low=0.8, high=1.2, outside="drop", sign="ppo").check(),
    ),
    "dapo": MethodSpec(
        key="dapo",
        name="DAPO",
        use_sc=False,
        weight_kind="dapo",
        correction=ISCorrection(level="token", low=0.8, high=1.28, outside="drop", sign="ppo").check(),
    ),
    "gspo": MethodSpec(
        key="gspo",
        name="GSPO",
        use_sc=False,
        weight_kind="ones",
        correction=ISCorrection(level="seq_mean", low=0.9997, high=1.0004, outside="drop", sign="ppo").check(),
    ),
    "topr": MethodSpec(
        key="topr",
        name="TOPR",
        use_sc=False,
        weight_kind="ones",
        correction=ISCorrection(level="seq_sum", high=1.0, outside="clamp", sign="neg").check(),
    ),
    "dppo": MethodSpec(
        key="dppo",
        name="DPPO",
        use_sc=False,
        weight_kind="ones",
        correction=ISCorrection(level="token", stat="tv", high=0.2, outside="drop", sign="ppo").check(),
    ),
    "grpo": MethodSpec(
        key="grpo",
        name="GRPO",
        use_sc=False,
        weight_kind="ppo",
        correction=ISCorrection(level="token", low=0.8, high=1.2, outside="drop", sign="ppo").check(),
        reward_mode_override="group_std",
        seq_norm=True,
    ),
    "sc": MethodSpec(
        key="sc",
        name="Score Centering",
        use_sc=True,
        weight_kind="ones",
        correction=ISCorrection(level=None).check(),
    ),
    "tis_sc": MethodSpec(
        key="tis_sc",
        name="TIS + Score Centering",
        use_sc=True,
        weight_kind="tis",
        correction=ISCorrection(level="token", high=2.0, outside="clamp").check(),
    ),
    "mis_sc": MethodSpec(
        key="mis_sc",
        name="MIS + Score Centering",
        use_sc=True,
        weight_kind="mis",
        correction=ISCorrection(level="token", low=0.5, high=5.0, outside="drop").check(),
    ),
    "ppo_sc": MethodSpec(
        key="ppo_sc",
        name="PPO + Score Centering",
        use_sc=True,
        weight_kind="ppo",
        correction=ISCorrection(level="token", low=0.8, high=1.2, outside="drop", sign="ppo").check(),
    ),
    "dapo_sc": MethodSpec(
        key="dapo_sc",
        name="DAPO + Score Centering",
        use_sc=True,
        weight_kind="dapo",
        correction=ISCorrection(level="token", low=0.8, high=1.28, outside="drop", sign="ppo").check(),
    ),
    "grpo_sc": MethodSpec(
        key="grpo_sc",
        name="GRPO + Score Centering",
        use_sc=True,
        weight_kind="ppo",
        correction=ISCorrection(level="token", low=0.8, high=1.2, outside="drop", sign="ppo").check(),
        reward_mode_override="group_std",
        seq_norm=True,
    ),
    "grpo_dapo_sc": MethodSpec(
        key="grpo_dapo_sc",
        name="GRPO (Clip-Higher) + Score Centering",
        use_sc=True,
        weight_kind="dapo",
        correction=ISCorrection(level="token", low=0.8, high=1.28, outside="drop", sign="ppo").check(),
        reward_mode_override="group_std",
        seq_norm=True,
    ),
}


def make_weight_fn(weight_kind: str, tis_max: float = 2.0) -> Callable[[Tensor], Tensor]:
    """Return a detached ratio weighting function w(p/q) (backward-compatible helper)."""
    if weight_kind == "ones":
        return torch.ones_like
    if weight_kind == "is":
        return lambda ratio: ratio
    if weight_kind == "tis":
        return lambda ratio: ratio.clamp_max(tis_max)
    if weight_kind == "mis":
        return lambda ratio: torch.where((ratio >= 0.5) & (ratio <= 5.0), ratio, torch.zeros_like(ratio))
    if weight_kind == "ppo":
        return lambda ratio: torch.where((ratio >= 0.8) & (ratio <= 1.2), ratio, torch.zeros_like(ratio))
    if weight_kind == "dapo":
        return lambda ratio: torch.where((ratio >= 0.8) & (ratio <= 1.28), ratio, torch.zeros_like(ratio))
    raise ValueError(f"Unsupported weight_kind: {weight_kind!r}")


def resolve_correction(method_key: str, tis_max: float | None = None) -> ISCorrection | None:
    """Resolve ISCorrection for a method, applying optional tis_max override on TIS variants."""
    corr = METHODS[method_key].correction
    if corr is not None and tis_max is not None and method_key in ("tis", "tis_sc"):
        corr = replace(corr, high=float(tis_max))
    return corr


def compute_advantages(
    rewards: Tensor,
    group_size: int,
    reward_mode: str = "group_centered",
) -> Tensor:
    """Map binary or shaped rewards into RL advantages following REWARD_MODES or GRPO z-score."""
    if reward_mode not in REWARD_MODES:
        raise ValueError(f"Unsupported reward_mode: {reward_mode!r}")
    if reward_mode == "group_std_raw":
        raw = rewards.to(torch.float32)
    else:
        solved, unsolved = REWARD_MODES[reward_mode]
        raw = torch.where(rewards > 0.5, solved, unsolved).to(torch.float32)
    if reward_mode == "group_centered":
        groups = raw.view(-1, group_size)
        return (groups - groups.mean(dim=1, keepdim=True)).reshape(-1)
    if reward_mode in ("group_std", "group_std_raw"):
        groups = raw.view(-1, group_size)
        mean = groups.mean(dim=1, keepdim=True)
        std = groups.std(dim=1, keepdim=True)
        adv = torch.where(std > 1e-6, (groups - mean) / (std + 1e-4), torch.zeros_like(groups))
        return adv.reshape(-1)
    return raw


def log_mass(
    log_p: Tensor | float,
    log_q: Tensor | float,
    positive: Tensor,
    c: ISCorrection | None,
    tv: Tensor | None = None,
) -> Tensor:
    """Compute log(q * w(p/q)) in log space following official martin-marek/score-centering."""
    log_q_t = torch.as_tensor(log_q, dtype=torch.float32, device=positive.device)
    log_p_t = torch.as_tensor(log_p, dtype=torch.float32, device=positive.device)
    if c is None or c.level is None:
        return log_q_t.expand(torch.broadcast_shapes(log_q_t.shape, positive.shape))

    lo = (log_q_t + math.log(c.low)) if c.low > 0 else torch.full_like(log_q_t, -math.inf)
    hi = (log_q_t + math.log(c.high)) if c.high < math.inf else torch.full_like(log_q_t, math.inf)

    if c.stat == "tv":
        assert tv is not None, "stat='tv' requires tv tensor"
        in_band = tv <= c.high
        if c.sign == "ppo":
            in_band = in_band | torch.where(positive, log_p_t <= log_q_t, log_p_t >= log_q_t)
    elif c.sign == "ppo":
        in_band = torch.where(
            positive,
            log_p_t <= hi,
            (log_p_t >= lo) & (log_p_t <= log_q_t + math.log(DUAL_CLIP)),
        )
    else:
        in_band = (log_p_t >= lo) & (log_p_t <= hi)

    if c.outside == "clamp":
        outside_val = torch.minimum(torch.maximum(log_p_t, lo), hi)
    else:
        outside_val = torch.full_like(log_p_t, -math.inf)

    log_m = torch.where(in_band, log_p_t, outside_val)
    if c.sign == "neg":
        log_m = torch.where(positive, log_q_t, log_m)
    elif c.sign == "pos":
        log_m = torch.where(positive, log_m, log_q_t)
    return log_m


def is_weight(
    log_ratio: Tensor,
    positive: Tensor,
    c: ISCorrection | None,
    tv: Tensor | None = None,
) -> Tensor:
    """Detached IS weight w(r) with log_ratio clipped to [-20, 20]."""
    if c is None or c.level is None:
        return torch.ones_like(log_ratio, dtype=torch.float32)
    clipped = log_ratio.clamp(-20.0, 20.0)
    return log_mass(clipped, 0.0, positive, c, tv=tv).exp().detach()


def score_centering_loss(
    train_logp: Tensor,
    samp_logp: Tensor,
    topk_ids: Tensor,
    sampled_token: Tensor,
    samp_token_logp: Tensor,
    advantage: Tensor,
    weight_fn: Callable[[Tensor], Tensor] = torch.ones_like,
    use_sc: bool = True,
    eps: float = 1e-3,
    correction: ISCorrection | None = None,
    loss_mask: Tensor | None = None,
    return_stats: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Compute per-token REINFORCE loss with official ISCorrection and Top-k Score Centering.

    Supports both full official ``ISCorrection`` (PG, IS, TIS, MIS, PPO, DAPO, GSPO,
    TOPR, DPPO, and SC compositions) and legacy ``weight_fn`` callers.
    """
    token_logp = train_logp.gather(-1, sampled_token.unsqueeze(-1)).squeeze(-1)
    log_ratio_tok = (token_logp - samp_token_logp).clamp(-20.0, 20.0)

    adv_2d = advantage.view(-1, 1) if advantage.ndim == 1 else advantage
    positive = (adv_2d >= 0)

    if correction is not None:
        token_c = correction if correction.level == "token" else None
        seq_c = correction if correction.level in ("seq_mean", "seq_sum") else None
        if use_sc:
            assert token_c is None or token_c.stat == "ratio", "Score centering composes only with stat='ratio'"

        tv_tok = (samp_token_logp.exp() - token_logp.exp()).abs().detach()
        token_weight = is_weight(log_ratio_tok, positive, token_c, tv=tv_tok)

        if seq_c is not None:
            mask_f = (
                loss_mask.to(torch.float32)
                if loss_mask is not None
                else torch.ones_like(token_logp, dtype=torch.float32)
            )
            seq_log_r = ((token_logp - samp_token_logp) * mask_f).sum(dim=-1, keepdim=True)
            seq_tv = (tv_tok * mask_f).sum(dim=-1, keepdim=True)
            if seq_c.level == "seq_mean":
                seq_tokens = mask_f.sum(dim=-1, keepdim=True).clamp_min(1.0)
                seq_log_r = seq_log_r / seq_tokens
                seq_tv = seq_tv / seq_tokens
            seq_weight = is_weight(seq_log_r, positive, seq_c, tv=seq_tv)
            eff_adv = adv_2d.detach() * seq_weight
            masked_tok = (seq_weight == 0).expand_as(token_logp)
        else:
            eff_adv = adv_2d.detach()
            masked_tok = (token_weight == 0)

        if not use_sc:
            loss = -eff_adv * (token_weight * token_logp)
            return (loss, masked_tok) if return_stats else loss

        head_logp = train_logp.gather(-1, topk_ids)
        p_head = head_logp.exp().detach()
        q_head = samp_logp.exp()
        p_tail = (1.0 - p_head.sum(dim=-1)).clamp_min(eps)
        q_tail = (1.0 - q_head.sum(dim=-1)).clamp_min(eps)
        rho = (q_tail / p_tail).clamp(1e-3, 1e3)

        alpha = log_mass(0.0, rho.log(), positive, token_c).exp().detach()
        mass = log_mass(head_logp.detach(), samp_logp, positive.unsqueeze(-1), token_c).exp().detach()
        residual = mass - alpha.unsqueeze(-1) * p_head
        correction_term = (residual * head_logp).sum(dim=-1)
        loss = -eff_adv * (token_weight * token_logp - correction_term)
        return (loss, masked_tok) if return_stats else loss

    # Legacy weight_fn fallback for backward compatibility
    token_weight = weight_fn(log_ratio_tok.exp()).detach()
    masked_tok = (token_weight == 0)
    if not use_sc:
        loss = -adv_2d.detach() * (token_weight * token_logp)
        return (loss, masked_tok) if return_stats else loss

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
    correction_term = (residual.detach() * head_logp).sum(dim=-1)
    loss = -adv_2d.detach() * (token_weight * token_logp - correction_term)
    return (loss, masked_tok) if return_stats else loss
