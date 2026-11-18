#!/usr/bin/env bash
# 实验四：基于实验二对齐官方开源参数 + 300/500 步长程训练 + 11 种 RL 方法对比（双 A100 分片并行）
set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${EXP_DIR}/../.." && pwd)"
COLAB_BIN="${COLAB_BIN:-/Users/dongshizhou/.local/bin/colab}"
PROXY_ENV="${PROXY_ENV:-HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890}"
SESSION_A="${SESSION_A:-sc-a100}"
SESSION_B="${SESSION_B:-sc-a100-real}"
STEPS="${STEPS:-500}"
EVAL_EVERY="${EVAL_EVERY:-20}"

run_colab() {
  env ${PROXY_ENV} "${COLAB_BIN}" "$@"
}

ensure_vm_ready() {
  local sess="$1"
  echo "=== 准备 Colab A100 实例 (${sess}) ==="
  run_colab new -s "${sess}" --gpu A100 || true
  run_colab exec -s "${sess}" --timeout 45 -f "${ROOT_DIR}/scripts/colab_bg_runner.py" \
    --env ACTION=launch \
    --env LOG_FILE=/content/install_vllm.log \
    --env DONE_FILE=/content/DONE_vllm \
    --env PID_FILE=/content/pid_vllm \
    --env "CMD=python3 -c 'import vllm' 2>/dev/null || uv pip install --system vllm"

  while true; do
    local inst_out
    inst_out="$(run_colab exec -s "${sess}" --timeout 30 -f "${ROOT_DIR}/scripts/colab_bg_runner.py" \
      --env ACTION=status \
      --env LOG_FILE=/content/install_vllm.log \
      --env DONE_FILE=/content/DONE_vllm \
      --env PID_FILE=/content/pid_vllm || true)"
    echo "[${sess} install] $(echo "${inst_out}" | grep STATUS || echo 'checking...')"
    if echo "${inst_out}" | grep -q "done_code=0"; then
      break
    fi
    sleep 20
  done

  run_colab exec -s "${sess}" --timeout 30 -f "${ROOT_DIR}/scripts/colab_bg_runner.py" \
    --env ACTION=launch --env LOG_FILE=/tmp/mkdir.log --env DONE_FILE=/tmp/DONE_mkdir --env PID_FILE=/tmp/pid_mkdir \
    --env "CMD=mkdir -p /content/src /content/exp4" || true
  for f in __init__.py countdown.py losses.py engine.py experiment_identity.py vllm_suite.py; do
    run_colab upload -s "${sess}" "${ROOT_DIR}/src/${f}" "/content/src/${f}"
  done
  run_colab upload -s "${sess}" "${EXP_DIR}/run.py" "/content/exp4/run.py"
}

cd "${ROOT_DIR}"
ensure_vm_ready "${SESSION_A}"
ensure_vm_ready "${SESSION_B}"

COMMON_FLAGS="--steps ${STEPS} --eval-every ${EVAL_EVERY} --eval-prompts 64 --num-prompts 4 --group-size 8 --max-new-tokens 160 --micro-batch-size 8 --lr 0.01 --max-grad-norm 2.0 --staleness 2 --weight-mismatch-sigma 0.02 --tis-max 2.0 --topk 128 --reward-mode group_centered --gpu-memory-utilization 0.22"

echo "=== 启动分片一 (${SESSION_A}): pg tis ppo grpo tis_sc ==="
run_colab exec -s "${SESSION_A}" --timeout 45 -f "${ROOT_DIR}/scripts/colab_bg_runner.py" \
  --env ACTION=launch \
  --env LOG_FILE=/content/exp4_part1/train.log \
  --env DONE_FILE=/content/exp4_part1/DONE \
  --env PID_FILE=/content/exp4_part1/pid \
  --env "CMD=mkdir -p /content/exp4_part1 && VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTORCH_ALLOC_CONF=expandable_segments:True python3 -u /content/exp4/run.py ${COMMON_FLAGS} --methods pg tis ppo grpo tis_sc --output-dir /content/exp4_part1"

echo "=== 启动分片二 (${SESSION_B}): is mis dapo ppo_sc dapo_sc grpo_sc ==="
run_colab exec -s "${SESSION_B}" --timeout 45 -f "${ROOT_DIR}/scripts/colab_bg_runner.py" \
  --env ACTION=launch \
  --env LOG_FILE=/content/exp4_part2/train.log \
  --env DONE_FILE=/content/exp4_part2/DONE \
  --env PID_FILE=/content/exp4_part2/pid \
  --env "CMD=mkdir -p /content/exp4_part2 && VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTORCH_ALLOC_CONF=expandable_segments:True python3 -u /content/exp4/run.py ${COMMON_FLAGS} --methods is mis dapo ppo_sc dapo_sc grpo_sc --output-dir /content/exp4_part2"

echo "=== 轮询双节点进度并实时同步合并报告 ==="
done_a=0
done_b=0
while [[ "${done_a}" -eq 0 || "${done_b}" -eq 0 ]]; do
  if [[ "${done_a}" -eq 0 ]]; then
    st_a="$(run_colab exec -s "${SESSION_A}" --timeout 30 -f "${ROOT_DIR}/scripts/colab_bg_runner.py" \
      --env ACTION=status --env LOG_FILE=/content/exp4_part1/train.log \
      --env DONE_FILE=/content/exp4_part1/DONE --env PID_FILE=/content/exp4_part1/pid || true)"
    echo "[${SESSION_A}] $(echo "${st_a}" | grep STATUS || echo 'checking...')"
    run_colab download -s "${SESSION_A}" /content/exp4_part1/results.json "${EXP_DIR}/results_part1.json" || true
    run_colab download -s "${SESSION_A}" /content/exp4_part1/train.log "${EXP_DIR}/train_part1.log" || true
    echo "${st_a}" | grep -q "done_code=0" && done_a=1
  fi
  if [[ "${done_b}" -eq 0 ]]; then
    st_b="$(run_colab exec -s "${SESSION_B}" --timeout 30 -f "${ROOT_DIR}/scripts/colab_bg_runner.py" \
      --env ACTION=status --env LOG_FILE=/content/exp4_part2/train.log \
      --env DONE_FILE=/content/exp4_part2/DONE --env PID_FILE=/content/exp4_part2/pid || true)"
    echo "[${SESSION_B}] $(echo "${st_b}" | grep STATUS || echo 'checking...')"
    run_colab download -s "${SESSION_B}" /content/exp4_part2/results.json "${EXP_DIR}/results_part2.json" || true
    run_colab download -s "${SESSION_B}" /content/exp4_part2/train.log "${EXP_DIR}/train_part2.log" || true
    echo "${st_b}" | grep -q "done_code=0" && done_b=1
  fi

  python3 -B "${EXP_DIR}/run.py" --merge-parts "${EXP_DIR}/results_part1.json" "${EXP_DIR}/results_part2.json" --output-dir "${EXP_DIR}" || true
  cat "${EXP_DIR}/train_part1.log" "${EXP_DIR}/train_part2.log" > "${EXP_DIR}/train.log" 2>/dev/null || true
  python3 -B "${ROOT_DIR}/scripts/generate_report.py" || true

  if [[ "${done_a}" -eq 0 || "${done_b}" -eq 0 ]]; then
    sleep 90
  fi
done

run_colab stop -s "${SESSION_A}" || true
run_colab stop -s "${SESSION_B}" || true
echo "实验四全部完成，结果已保存至 ${EXP_DIR}"
