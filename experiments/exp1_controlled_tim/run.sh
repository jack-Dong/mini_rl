#!/usr/bin/env bash
# 实验一：受控权重噪声 TIM（Controlled Weight-Noise TIM, 80 steps, A100）
set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${EXP_DIR}/../.." && pwd)"
COLAB_BIN="${COLAB_BIN:-/Users/dongshizhou/.local/bin/colab}"
PROXY_ENV="${PROXY_ENV:-HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890}"
SESSION="${SESSION:-sc-a100-real}"

run_colab() {
  env ${PROXY_ENV} "${COLAB_BIN}" "$@"
}

echo "=== 1. 准备 Colab A100 实例 (${SESSION}) ==="
run_colab new -s "${SESSION}" --gpu A100 || true

echo "=== 2. 上传 src 模块与实验一启动脚本 ==="
cd "${ROOT_DIR}"
run_colab exec -s "${SESSION}" --timeout 30 -f scripts/colab_bg_runner.py \
  --env ACTION=launch --env LOG_FILE=/tmp/mkdir.log --env DONE_FILE=/tmp/DONE_mkdir --env PID_FILE=/tmp/pid_mkdir \
  --env "CMD=mkdir -p /content/src /content/exp1" || true
for f in __init__.py countdown.py losses.py engine.py vllm_suite.py; do
  run_colab upload -s "${SESSION}" "src/${f}" "/content/src/${f}"
done
run_colab upload -s "${SESSION}" "experiments/exp1_controlled_tim/run.py" "/content/exp1/run.py"

echo "=== 3. 启动实验一（受控权重噪声 TIM，80 steps） ==="
run_colab exec -s "${SESSION}" --timeout 45 -f scripts/colab_bg_runner.py \
  --env ACTION=launch \
  --env LOG_FILE=/content/exp1/train.log \
  --env DONE_FILE=/content/exp1/DONE \
  --env PID_FILE=/content/exp1/pid \
  --env "CMD=mkdir -p /content/exp1 && (python3 -c 'import vllm' 2>/dev/null || uv pip install --system vllm) && VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTORCH_ALLOC_CONF=expandable_segments:True python3 -u /content/exp1/run.py --steps 80 --eval-every 10 --eval-prompts 64 --max-new-tokens 160 --noise-scale 0.03 --staleness 1 --tis-max 1.5 --lr 0.006 --max-grad-norm 1.0 --reward-mode group_centered --micro-batch-size 4 --gpu-memory-utilization 0.18 --output-dir /content/exp1"

echo "=== 4. 轮询实验状态直至完成 ==="
while true; do
  status_out="$(run_colab exec -s "${SESSION}" --timeout 30 -f scripts/colab_bg_runner.py \
    --env ACTION=status \
    --env LOG_FILE=/content/exp1/train.log \
    --env DONE_FILE=/content/exp1/DONE \
    --env PID_FILE=/content/exp1/pid || true)"
  echo "[${SESSION} exp1] $(echo "${status_out}" | grep STATUS || echo 'checking...')"
  if echo "${status_out}" | grep -q "done_code=0"; then
    break
  fi
  sleep 45
done

echo "=== 5. 回传实验结果与日志至 ${EXP_DIR} ==="
run_colab download -s "${SESSION}" /content/exp1/results.json "${EXP_DIR}/results.json"
run_colab download -s "${SESSION}" /content/exp1/plot.png "${EXP_DIR}/plot.png"
run_colab download -s "${SESSION}" /content/exp1/train.log "${EXP_DIR}/train.log"

run_colab stop -s "${SESSION}"
echo "实验一完成，结果已保存至 ${EXP_DIR}"
