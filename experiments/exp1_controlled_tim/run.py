"""Experiment 1 entrypoint: Controlled Weight-Noise TIM (vLLM BF16 + fixed noise vs BF16 Trainer)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

_HERE = Path(__file__).resolve()
for _cand in [_HERE.parents[1], _HERE.parents[2] if len(_HERE.parents) > 2 else _HERE.parent, Path("/content")]:
    if (_cand / "src").is_dir() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))
        break

from src.engine import main

if __name__ == "__main__":
    main()
