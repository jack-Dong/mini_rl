#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONUNBUFFERED=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0

STAGE="${STAGE:-stage1}"
STEPS="${STEPS:-200}"
EVAL_EVERY="${EVAL_EVERY:-20}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-8}"
EXTRA_ARGS=("$@")

python3 -u "${SCRIPT_DIR}/run.py" \
  --stage "${STAGE}" \
  --steps "${STEPS}" \
  --eval-every "${EVAL_EVERY}" \
  --num-prompts 4 \
  --group-size 8 \
  --eval-prompts 64 \
  --max-new-tokens 512 \
  --max-model-len 768 \
  --micro-batch-size "${MICRO_BATCH_SIZE}" \
  --no-enforce-eager \
  --lr 0.01 \
  --max-grad-norm 2.0 \
  --staleness 2 \
  --weight-mismatch-sigma 0.02 \
  --topk 128 \
  --tis-max 2.0 \
  --vllm-dtype float16 \
  --gpu-memory-utilization 0.25 \
  --resume \
  --output-dir "${SCRIPT_DIR}" \
  "${EXTRA_ARGS[@]}" 2>&1 | tee -a "${SCRIPT_DIR}/train.log"
