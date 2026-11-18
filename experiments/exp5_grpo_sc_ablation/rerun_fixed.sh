#!/usr/bin/env bash
# Run from a Colab A100 80GB runtime after installing the project dependencies.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/rerun_fixed}"
mkdir -p "${OUTPUT_DIR}"
export PYTHONUNBUFFERED=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTORCH_ALLOC_CONF=expandable_segments:True

python3 -u "${SCRIPT_DIR}/run.py" \
  --stage stage2 --methods best_combo_500 --steps 500 --eval-every 25 \
  --num-prompts 4 --group-size 8 --eval-prompts 64 \
  --max-new-tokens 512 --max-model-len 768 --micro-batch-size 8 \
  --no-enforce-eager --lr 0.01 --max-grad-norm 2.0 --staleness 2 \
  --weight-mismatch-sigma 0.02 --topk 128 --tis-max 2.0 \
  --vllm-dtype float16 --gpu-memory-utilization 0.25 \
  --output-dir "${OUTPUT_DIR}" --results-filename results.json \
  --plot-filename plot.png --resume "$@" 2>&1 | tee -a "${OUTPUT_DIR}/train.log"
