"""Experiment 3 entrypoint: Hardware FP8 (E4M3FN) Weight Quantization TIM via vLLM."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

_HERE = Path(__file__).resolve()
for _cand in [_HERE.parents[1], _HERE.parents[2] if len(_HERE.parents) > 2 else _HERE.parent, Path("/content")]:
    if (_cand / "src").is_dir() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))
        break

from src.vllm_suite import run_vllm_suite


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the FP8 E4M3FN quantization TIM experiment."""
    parser = argparse.ArgumentParser(description="Hardware FP8 (E4M3FN) + vLLM TIM experiment.")
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
    parser.add_argument("--topk", type=int, default=128)
    parser.add_argument("--tis-max", type=float, default=1.4)
    parser.add_argument("--reward-mode", default="group_centered")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vllm-dtype", default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.18)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--output-dir", default=".")
    return parser.parse_args()


def main() -> None:
    """Run Experiment 3: Per-row E4M3FN FP8 weight quantization in vLLM vs BF16 Eager trainer."""
    args = parse_args()
    run_vllm_suite(
        args=args,
        results_filename="results.json",
        plot_filename="plot.png",
        title_prefix="Hardware FP8 (E4M3FN) + vLLM TIM",
        engine_label=f"vLLM V1 ({args.vllm_dtype}, per-row torch.float8_e4m3fn weights, staleness={args.staleness})",
        log_prefix="RealTIM FP8",
        fp8_quant=True,
        noise_scale=0.0,
    )


if __name__ == "__main__":
    main()
