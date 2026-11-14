# mini_rl · Score Centering（arXiv:2609.20807）真实 vLLM 复现项目

本项目在 `Qwen/Qwen3-0.6B` 模型与 3 操作数 `Countdown`（数字范围 `1..25`，目标 `1..200`）推理任务上，基于真实 **vLLM V1 采样引擎** 与 **HuggingFace FP32 主权重 / BF16 Eager 训练器**，针对三种训推不匹配（Training-Inference Mismatch, TIM）场景完成了 **80 步** 对比复现实验，并自动采集每个方法在零步（`step_0`）与末步（`step_79`）保留集上的 **10 条成功样本 + 10 条失败样本**。

## 目录结构

```text
mini_rl/
├── README.md                              # 项目说明文档
├── .gitignore                             # Git 忽略规则
├── experiment_results_report.html         # 全中文交互式实验对比报告（含曲线、指标表与 10+10 样本抽样）
├── score_centering_interactive_guide.html # Score Centering 原理交互式可视化指南
├── src/                                   # 核心共享模块
│   ├── __init__.py
│   ├── countdown.py                       # Countdown 任务生成与规则奖励验证（0.0 / 1.0）
│   ├── losses.py                          # PG / IS / TIS / TIS+Score Centering 损失与优势计算
│   ├── engine.py                          # 训练器主循环（支持可控权重噪声 TIM 与 Eager/vLLM）
│   └── vllm_suite.py                      # 真实 vLLM 采样（V1 热更新、FP16/FP8 权重量化、微批反传）
├── scripts/                               # 工程工具与报告生成脚本
│   ├── colab_bg_runner.py                 # 远程 Colab 后台执行与实时进度记录脚本
│   └── generate_report.py                 # 自动聚合三组实验 JSON/日志生成中文 HTML 报告
└── experiments/                           # 分实验独立目录（脚本 + 日志 + 结果图全自洽）
    ├── exp1_controlled_tim/               # 实验一：受控权重噪声 TIM（vLLM BF16）
    │   ├── run.sh                         # 独立一键启动与环境校验脚本
    │   ├── run.py                         # 实验入口脚本
    │   ├── results.json                   # 80 步完整指标与 step_0/step_79（10+10）评测样本
    │   ├── plot.png                       # 4 算法收敛与 KL/比率对比图
    │   └── train.log                      # 完整远程运行日志
    ├── exp2_vllm_tim/                     # 实验二：真实 vLLM FP16 (PagedAttention+FlashInfer) TIM
    │   ├── run.sh / run.py / results.json / plot.png / train.log
    └── exp3_fp8_tim/                      # 实验三：逐行 FP8 (torch.float8_e4m3fn) + vLLM 异步采样 TIM
        └── run.sh / run.py / results.json / plot.png / train.log
```

## 对比方法与核心配置

- **对比算法**：
  - `pg`：原生 REINFORCE（未校正训推概率偏移）
  - `is`：朴素重要性采样（Naive IS，$w(r)=p/q$，无截断高方差）
  - `tis`：截断重要性采样（TIS，$w(r)=\min(p/q, C_{\text{tis}})$）
  - `tis_sc`：TIS + Top-$k$ Score Centering（在 Top-128 词汇子集上对齐头质量基线中心化词表梯度）
- **通用 80 步实验超参**：
  - 模型：`Qwen/Qwen3-0.6B`；最大生成长度：`160` tokens
  - 训练：`steps=80`，`eval_every=10`，`num_prompts=8`，`group_size=4`，`eval_prompts=64`
  - 优化：`lr=0.006`，`max_grad_norm=1.0`，`micro_batch_size=4`，`gpu_memory_utilization=0.18`（适配 40GB A100 无 OOM 共存）
  - 优势函数：`group_centered`（全零批次退化保护 + 批内均值归一化）
- **三组 TIM 机制差异**：
  1. **实验一（`exp1_controlled_tim`）**：`vllm_dtype=bfloat16`，`noise_scale=0.03`，`staleness=1`，`tis_max=1.5`（零步探针平均 $|\Delta \log p|=0.05568$，最大 $1.25917$）。
  2. **实验二（`exp2_vllm_tim`）**：真实 `vllm_dtype=float16`（PagedAttention + FlashInfer），`weight_mismatch_sigma=0.015`，`staleness=2`，`tis_max=1.4`（零步探针平均 $|\Delta \log p|=0.03276$，初始最大 IS 比率 $2.18\times$）。
  3. **实验三（`exp3_fp8_tim`）**：逐行 `torch.float8_e4m3fn` 量化权重加载至 `vLLM BF16`，`staleness=2`，`tis_max=1.4`（零步探针平均 $|\Delta \log p|=0.04435$，初始最大 IS 比率 $1.80\times$）。

## 80 步实验核心结果汇总

各组实验均在一致的 64 条保留验证集上从 **17.2%** 零步基线出发：

| 实验场景 | 算法 | step_79 评估 | 评估峰值 | 全程均梯度范数 (末步) | 平均 $\|\mathrm{KL}\|$ | 最大 IS 比率 (末步) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **实验一：受控权重噪声 TIM** | `pg` | 28.1% | 37.5% @20 | 14.56 (32.86) | 0.0252 | 10.53x (4.89x) |
| | `is` | **35.9%** | 39.1% @40 | **6.71** (6.75) | **0.0101** | 6.61x (1.77x) |
| | `tis` | **35.9%** | 39.1% @20 | 9.06 (18.98) | 0.0141 | 22.11x (1.38x) |
| | `tis_sc` | **35.9%** | 35.9% @20 | 10.50 (13.75) | 0.0187 | **6.59x** (1.42x) |
| **实验二：vLLM FP16 引擎 TIM** | `pg` | 35.9% | 43.8% @30 | **7.26** (11.07) | 0.0140 | 16.78x (2.96x) |
| | `is` | **50.0%** | **51.6%** @40 | 9.50 (9.91) | 0.0131 | 25.56x (1.93x) |
| | `tis` | 45.3% | 46.9% @70 | 8.12 (8.28) | 0.0152 | 20.75x (2.75x) |
| | `tis_sc` | 37.5% | 42.2% @40 | 7.39 (**6.83**) | **0.0130** | **10.86x** (4.36x) |
| **实验三：硬件级 FP8 + vLLM TIM** | `pg` | 26.6% | 40.6% @40 | 19.36 (51.83) | 0.1854 | 25.79x (7.66x) |
| | `is` | 34.4% | 40.6% @70 | 8.88 (26.82) | **0.1064** | 205.79x (24.41x) |
| | `tis` | **40.6%** | **40.6%** @60 | 6.55 (4.49) | 0.1225 | 139.56x (1.44x) |
| | `tis_sc` | 31.2% | 37.5% @50 | **4.89 (2.08)** | 0.1172 | 61.09x (2.62x) |

**关键观察**：
1. 在全部三组 TIM 机制下，重要度校正方法（`IS` / `TIS` / `TIS+SC`）在第 79 步保留集准确率上均一致超越原生 `PG`。
2. 在最严苛的硬件 FP8 TIM 场景（实验三，朴素 `IS` 全程最大 IS 比率飙升至 $205.79\times$、第 79 步仍达 $24.41\times$）下，原生 `PG` 与朴素 `IS` 末步梯度范数剧烈飙升至 `51.83` 和 `26.82`，而截断类的 `TIS`（末期 `40.6%` 最高，末步梯度 `4.49`）与 `TIS+SC`（末步梯度 `2.08`、全程均梯度 `4.89` 全场最低）显著抑制了尾部高方差冲击。

## 本地运行与报告重生成

在配置有 CUDA GPU（及 `vLLM` 环境，各 `run.sh` 内置环境自检与自动安装）的节点上可单独运行任一实验：

```bash
bash experiments/exp1_controlled_tim/run.sh
bash experiments/exp2_vllm_tim/run.sh
bash experiments/exp3_fp8_tim/run.sh
```

重新汇总三组实验结果与样本生成中文交互式报告：

```bash
python3 -B scripts/generate_report.py
```
