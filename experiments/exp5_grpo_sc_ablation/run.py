"""Experiment 5: GRPO+SC balanced-data 5-group ablation (max_new_tokens=512) & 500-step combo comparison."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXP5_STAGE1_ORDER: list[str] = [
    "base_bal",
    "fmt_stop_dedup",
    "kl_ent_dapo",
    "sample_var",
    "sft_adamw",
]
EXP5_STAGE2_ORDER: list[str] = [
    "orig_data_500",
    "base_bal_500",
    "best_combo_500",
]


def build_exp5_variants(
    variant_cls: type[Any],
    combo_features: list[str] | None = None,
) -> dict[str, Any]:
    """Build Stage 1 (5 grouped ablations) and Stage 2 (500-step combo) VariantConfig objects."""
    features = set(
        combo_features
        or ["sample_var", "stop_answer", "strict_fmt", "dapo_clip", "kl_ref_soft"]
    )
    use_fmt_dedup = "fmt_stop_dedup" in features
    use_stop = use_fmt_dedup or "stop_answer" in features
    use_strict_fmt = "strict_fmt" in features
    use_kl_ent = "kl_ent_dapo" in features
    use_dapo = use_kl_ent or "dapo_clip" in features
    use_kl_ref = use_kl_ent or "kl_ref" in features
    use_kl_soft = "kl_ref_soft" in features
    use_entropy = use_kl_ent or "entropy" in features
    use_sample = "sample_var" in features
    use_sft = "sft_adamw" in features

    return {
        "base_bal": variant_cls(
            key="base_bal",
            label="1. Base Balanced (grpo_sc)",
            base_method="grpo_sc",
            use_balanced_data=True,
        ),
        "fmt_stop_dedup": variant_cls(
            key="fmt_stop_dedup",
            label="2. StopEOS+StrictFmt+Dedup",
            base_method="grpo_sc",
            reward_shaping="fmt_dedup",
            reward_mode_override="group_std_raw",
            stop_strings=("</answer>",),
            use_balanced_data=True,
        ),
        "kl_ent_dapo": variant_cls(
            key="kl_ent_dapo",
            label="3. ClipHigher+KLRef+Entropy",
            base_method="grpo_dapo_sc",
            kl_beta=0.02,
            entropy_coef=0.01,
            use_balanced_data=True,
        ),
        "sample_var": variant_cls(
            key="sample_var",
            label="4. Group16+DynSample",
            base_method="grpo_sc",
            dynamic_sampling=True,
            group_size=16,
            use_balanced_data=True,
        ),
        "sft_adamw": variant_cls(
            key="sft_adamw",
            label="5. BalSFT50+AdamW(Cosine)",
            base_method="grpo_sc",
            sft_steps=50,
            optimizer_kind="adamw",
            lr=5e-6,
            warmup_steps=20,
            use_balanced_data=True,
        ),
        "orig_data_500": variant_cls(
            key="orig_data_500",
            label="Orig Unbalanced (grpo_sc 500)",
            base_method="grpo_sc",
            use_balanced_data=False,
        ),
        "base_bal_500": variant_cls(
            key="base_bal_500",
            label="Base Balanced (grpo_sc 500)",
            base_method="grpo_sc",
            use_balanced_data=True,
        ),
        "best_combo_500": variant_cls(
            key="best_combo_500",
            label=f"Best Combo ({'+'.join(sorted(features))})",
            base_method="grpo_dapo_sc" if use_dapo else "grpo_sc",
            reward_shaping=(
                "fmt_dedup" if use_fmt_dedup else ("strict_fmt" if use_strict_fmt else "none")
            ),
            reward_mode_override="group_std_raw" if use_fmt_dedup else None,
            stop_strings=("</answer>",) if use_stop else (),
            kl_beta=0.02 if use_kl_ref else (0.008 if use_kl_soft else 0.0),
            entropy_coef=0.01 if use_entropy else 0.0,
            dynamic_sampling=use_sample,
            group_size=16 if use_sample else None,
            sft_steps=50 if use_sft else 0,
            optimizer_kind="adamw" if use_sft else "sgd",
            lr=5e-6 if use_sft else None,
            warmup_steps=20 if use_sft else 0,
            use_balanced_data=True,
        ),
    }


def merge_shard_results(shard_paths: list[str], out_dir: Path, order: list[str]) -> None:
    """Merge single-method or partial shard JSON files into canonical Exp5 results.json and plot.png."""
    payloads = [json.loads(Path(p).read_text(encoding="utf-8")) for p in shard_paths]
    merged_by_key: dict[str, dict[str, object]] = {}
    for pay in payloads:
        for res in pay.get("results", []):
            merged_by_key[str(res["method"])] = res

    ordered = [merged_by_key[k] for k in order if k in merged_by_key]
    for k, v in merged_by_key.items():
        if k not in order:
            ordered.append(v)

    base_args = dict(payloads[0].get("args", {}))
    base_args["methods"] = [str(r["method"]) for r in ordered]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "results.json"
    out_json.write_text(
        json.dumps(
            {
                "status": "completed",
                "args": base_args,
                "engine_drift_probe": payloads[0].get("engine_drift_probe"),
                "results": ordered,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Merged {len(ordered)} variants into {out_json}", flush=True)
    try:
        from src.vllm_suite import plot_vllm_results

        plot_vllm_results(
            ordered,
            out_dir / "plot.png",
            title_prefix="Exp5: GRPO+SC Balanced 5-Group Ablation (512 tok)",
        )
        print(f"Saved plot to {out_dir / 'plot.png'}", flush=True)
    except Exception as exc:
        print(f"Plot skipped ({exc})", flush=True)


def parse_args() -> argparse.Namespace:
    """Parse Experiment 5 ablation CLI flags."""
    parser = argparse.ArgumentParser(description="Experiment 5: GRPO+SC balanced 5-group ablation.")
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--stage", default="stage1", choices=["stage1", "stage2", "custom"])
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--combo-features", nargs="+", default=None)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--sample-steps", type=int, nargs="*", default=None)
    parser.add_argument("--num-prompts", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--eval-prompts", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--num-operands", type=int, default=3)
    parser.add_argument("--max-number", type=int, default=25)
    parser.add_argument("--max-target", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=2.0)
    parser.add_argument("--staleness", type=int, default=2)
    parser.add_argument("--weight-mismatch-sigma", type=float, default=0.02)
    parser.add_argument("--topk", type=int, default=128)
    parser.add_argument("--tis-max", type=float, default=2.0)
    parser.add_argument("--reward-mode", default="group_centered")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vllm-dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.25)
    parser.add_argument("--max-model-len", type=int, default=768)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--results-filename", default="results.json")
    parser.add_argument("--plot-filename", default="plot.png")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--merge-parts", nargs="+", default=None)
    return parser.parse_args()


def main() -> None:
    """Run Exp5 training/ablation or merge sharded JSON outputs."""
    args = parse_args()
    if args.methods is None:
        args.methods = list(EXP5_STAGE1_ORDER if args.stage != "stage2" else EXP5_STAGE2_ORDER)

    if args.merge_parts:
        order = EXP5_STAGE1_ORDER + EXP5_STAGE2_ORDER
        merge_shard_results(args.merge_parts, Path(args.output_dir), order)
        return

    from src.vllm_suite import VariantConfig, run_vllm_suite

    variant_specs = build_exp5_variants(VariantConfig, combo_features=args.combo_features)
    stage_title = "Stage 1 (200 steps, Balanced 5-Group)" if args.stage == "stage1" else "Stage 2 (500-Step Flagship)"
    run_vllm_suite(
        args=args,
        results_filename=args.results_filename,
        plot_filename=args.plot_filename,
        title_prefix=f"Exp5 {stage_title}: GRPO+SC Ablation (max_new_tokens={args.max_new_tokens})",
        engine_label=(
            f"vLLM V1 ({args.vllm_dtype}, balanced-5stratum, max_tok={args.max_new_tokens}, "
            f"staleness={args.staleness}, sigma={args.weight_mismatch_sigma})"
        ),
        log_prefix="Exp5",
        fp8_quant=False,
        noise_scale=args.weight_mismatch_sigma,
        balanced_data=True,
        variant_specs=variant_specs,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
