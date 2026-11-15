"""Experiment 4 entrypoint: Official-Aligned 11-Method 500-Step Real vLLM TIM Suite.

Aligns hyperparameters with martin-marek/score-centering (arXiv:2609.20807):
group_size=8, lr=0.01 (SGD), TIS=2.0, MIS=[0.5, 5.0], PPO=[0.8, 1.2],
DAPO=[0.8, 1.28], Dual-Clip=3.0, GRPO (group z-score + seq_norm),
Top-k=128, weight_mismatch_sigma=0.02, Real vLLM FP16 (PagedAttention+FlashInfer),
evaluated at 300 and 500 steps.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

_HERE = Path(__file__).resolve()
for _cand in [_HERE.parents[1], _HERE.parents[2] if len(_HERE.parents) > 2 else _HERE.parent, Path("/content")]:
    if (_cand / "src").is_dir() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))
        break

DEFAULT_METHODS = ["pg", "is", "tis", "mis", "ppo", "dapo", "grpo", "tis_sc", "ppo_sc", "dapo_sc", "grpo_sc"]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments with Experiment 4 official-aligned defaults."""
    parser = argparse.ArgumentParser(description="Experiment 4: Official-aligned 500-step vLLM TIM suite.")
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--sample-steps", type=int, nargs="*", default=None)
    parser.add_argument("--num-prompts", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--eval-prompts", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--micro-batch-size", type=int, default=8)
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
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.22)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--merge-parts", nargs="+", default=None)
    return parser.parse_args()


def merge_shard_results(part_paths: list[Path], out_dir: Path) -> None:
    """Merge sharded part1/part2 JSON outputs into canonical Experiment 4 results.json and plot.png."""
    try:
        from src.vllm_suite import plot_vllm_results
    except ImportError:
        plot_vllm_results = None

    payloads = [json.loads(p.read_text(encoding="utf-8")) for p in part_paths if p.exists()]
    if not payloads:
        return

    by_method: dict[str, dict[str, object]] = {}
    probe = None
    merged_args: dict[str, object] = {}
    for pl in payloads:
        if probe is None and pl.get("engine_drift_probe"):
            probe = pl["engine_drift_probe"]
        if not merged_args and pl.get("args"):
            merged_args = dict(pl["args"])
        for res in pl.get("results", []):
            mkey = str(res["method"])
            by_method[mkey] = res

    ordered = [by_method[m] for m in DEFAULT_METHODS if m in by_method]
    for m, res in by_method.items():
        if m not in DEFAULT_METHODS:
            ordered.append(res)

    merged_args["methods"] = list(DEFAULT_METHODS)
    expected_steps = int(merged_args.get("steps", 500))
    all_done = len(ordered) >= len(DEFAULT_METHODS) and all(
        len(r.get("history", [])) >= expected_steps for r in ordered  # type: ignore[arg-type]
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    merged = {
        "status": "completed" if all_done else "running",
        "args": merged_args,
        "engine_drift_probe": probe,
        "results": ordered,
    }
    (out_dir / "results.json").write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    if ordered and plot_vllm_results is not None:
        plot_vllm_results(
            ordered,
            out_dir / "plot.png",
            title_prefix="Exp4 Official-Aligned vLLM TIM (11 Methods, 500 Steps)",
        )


def main() -> None:
    """Run Experiment 4 or merge distributed shards."""
    args = parse_args()
    if args.merge_parts:
        merge_shard_results([Path(p) for p in args.merge_parts], Path(args.output_dir))
        return

    from src.vllm_suite import run_vllm_suite

    run_vllm_suite(
        args=args,
        results_filename="results.json",
        plot_filename="plot.png",
        title_prefix="Exp4 Official-Aligned vLLM TIM (500 Steps)",
        engine_label=(
            f"vLLM V1 ({args.vllm_dtype}, PagedAttention+FlashInfer, "
            f"staleness={args.staleness}, sigma={args.weight_mismatch_sigma})"
        ),
        log_prefix="Exp4 TIM",
        fp8_quant=False,
        noise_scale=args.weight_mismatch_sigma,
    )


if __name__ == "__main__":
    main()
