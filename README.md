# mini_rl · Score Centering 复现

在三操作数 Countdown 算术任务上，用 vLLM V1 采样和 Hugging Face 训练器对 `Qwen/Qwen3-0.6B` 进行强化学习后训练。实验比较了 Score Centering 和重要性采样等方法在训推不匹配（TIM）下的表现，并考察数据均衡与动态采样对长程训练的影响。

- [实验报告](RL_BEST_PRACTICES_REPORT.md) 含实验配置、逐项指标和分析。
- [交互式报告](experiment_results_report.html) 和 [原理可视化](score_centering_interactive_guide.html) 可下载后在浏览器打开。
- [论文](https://arxiv.org/abs/2609.20807) · [本项目代码](https://github.com/jack-Dong/mini_rl)

## 实验概览

下表为原始实验记录。各目录包含运行入口、评估指标、曲线和日志。

| 实验 | 对比内容 | 主要结果 |
| --- | --- | --- |
| [一 · 受控权重噪声](experiments/exp1_controlled_tim/) | 80 步，PG、IS、TIS、TIS+SC | IS、TIS 和 TIS+SC 末次并列 35.9% |
| [二 · vLLM FP16](experiments/exp2_vllm_tim/) | 80 步，FP16 推理与训练侧精度差异 | IS 末次 50.0%，峰值 51.6% |
| [三 · FP8 权重量化](experiments/exp3_fp8_tim/) | 80 步，逐行 FP8 量化与异步采样 | TIS 末次 40.6%，TIS+SC 的末步梯度范数最低 |
| [四 · 长程方法对比](experiments/exp4_vllm_official_tim/) | 500 步，11 种强化学习方法 | GRPO+SC 末次 46.9%，峰值 56.2% |
| [五 · Stage 1](experiments/exp5_grpo_sc_ablation/results.json) | 200 步，均衡数据上的五组消融 | `G=16` 加动态采样末次 68.8%，峰值 79.7% |
| [五 · Stage 2](experiments/exp5_grpo_sc_ablation/results_stage2.json) | 500 步，对比原分布基线、均衡基线和 Best Combo | Best Combo 均衡集末次 85.9%，峰值 90.6% |

## 运行实验

从仓库根目录执行。实验一至四使用 Colab CLI 启动远程 A100。先设置 `COLAB_BIN` 为本机的 CLI 路径，并按网络环境设置 `PROXY_ENV`。直连时可用 `PROXY_ENV='HTTPS_PROXY= HTTP_PROXY='` 清空脚本默认的代理地址。

```bash
bash experiments/exp1_controlled_tim/run.sh
bash experiments/exp2_vllm_tim/run.sh
bash experiments/exp3_fp8_tim/run.sh
bash experiments/exp4_vllm_official_tim/run.sh
```

实验五在配好 PyTorch、Transformers 和 vLLM 的 CUDA 环境中运行。以下命令将新指标和曲线写入临时目录。

```bash
RUN_DIR="$(mktemp -d)"
bash experiments/exp5_grpo_sc_ablation/run.sh --output-dir "$RUN_DIR/stage1"
STAGE=stage2 STEPS=500 EVAL_EVERY=25 bash experiments/exp5_grpo_sc_ablation/run.sh \
  --output-dir "$RUN_DIR/stage2"
```

运行 `python3 -B scripts/generate_report.py` 可从仓库内的实验产物重新生成交互式 HTML。

## 代码入口

- [任务与奖励](src/countdown.py) 负责生成 Countdown 数据并验证模型答案。
- [训练与采样](src/vllm_suite.py) 使用 vLLM V1 进行生成和权重热更新。[训练引擎](src/engine.py) 提供共享训练逻辑。
- [损失函数](src/losses.py) 包含重要性采样、PPO、DAPO、GRPO 和 Score Centering。
- [远程执行与报告工具](scripts/) 包含 Colab 后台运行和报告生成脚本。

<!-- BEGIN FIXED RERUN RESULTS -->
## 修复与重跑

修复了奖励判定、训练与验证数据隔离、零优势批次的正则项、末步评估时点和断点恢复校验。在 Colab（NVIDIA A100-SXM4-80GB）上重跑实验五的 `best_combo_500`，完成 500 次参数更新后执行最终评估。

| 指标 | 重跑结果 |
| --- | --- |
| 均衡验证集末次准确率 | **84.4%** |
| 均衡验证集峰值 | **92.2%**（第 400 步） |
| 原分布验证集末次准确率 | **92.2%** |
| 加减 / 乘法 / 除法 / 括号 | 96.9% / 87.5% / 56.2% / 68.8% |
| 严格答案格式率 | 87.5% |

重跑支持原实验关于 Best Combo 长程稳定性的结论。两次运行的代码与依赖版本有差异，详细对比见[实验报告第 8 节](RL_BEST_PRACTICES_REPORT.md#八-修复后补充复跑best-combo-500-步)。

[指标与抽样回答](experiments/exp5_grpo_sc_ablation/rerun_fixed/results.json) · [训练曲线](experiments/exp5_grpo_sc_ablation/rerun_fixed/plot.png) · [训练日志](experiments/exp5_grpo_sc_ablation/rerun_fixed/train.log) · [环境版本](experiments/exp5_grpo_sc_ablation/rerun_fixed/environment.json)

在依赖已安装的 Colab A100 80GB 环境中，从仓库根目录启动重跑。

```bash
OUTPUT_DIR="$(mktemp -d)" bash experiments/exp5_grpo_sc_ablation/rerun_fixed.sh
```

断点恢复时，将 `OUTPUT_DIR` 指向已有运行目录，并保留其中的 `results.json` 和 `*.training.pt`。恢复需通过配置、源码与依赖版本校验。已发布的产物包含评估记录与日志，模型检查点未入库。
<!-- END FIXED RERUN RESULTS -->
