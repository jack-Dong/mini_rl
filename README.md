# mini_rl · Score Centering（arXiv:2609.20807）真实 vLLM 复现项目

本项目在 `Qwen/Qwen3-0.6B` 模型与 3 操作数 `Countdown`（数字范围 `1..25`，目标 `1..200`）推理任务上，基于真实 **vLLM V1 采样引擎** 与 **HuggingFace FP32 主权重 / BF16 训练器**，针对三种训推不匹配（Training-Inference Mismatch, TIM）场景完成了 **80 步 4 方法对比**，对齐开源实现 `martin-marek/score-centering` 的核心超参完成了 **500 步 11 方法同台长程评测**，并在 **实验五** 中构建了 **5 层技能均衡 Countdown 数据集与 `max_new_tokens=512` 长上下文消融框架**，完成 Stage 1（5 组 200 步消融）与 Stage 2（500 步旗舰对比），全程自动采集 `step_0`、`step_300`、`step_499` 等关键检查点样本。

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
│   ├── losses.py                          # PG/IS/TIS/MIS/PPO/DAPO/GRPO/GSPO/TOPR/DPPO 及 Score Centering 损失
│   ├── engine.py                          # 训练器主循环（支持可控权重噪声 TIM 与 Eager/vLLM）
│   └── vllm_suite.py                      # 真实 vLLM V1 采样（热更新、逐行 FP8 量化、微批反传、GRPO 序列均值）
├── scripts/                               # 工程工具与报告生成脚本
│   ├── colab_bg_runner.py                 # 远程 Colab 后台执行与实时进度记录脚本
│   └── generate_report.py                 # 自动聚合六组实验（实验一~实验五 Stage 1/2）JSON/日志生成中文 HTML 报告
└── experiments/                           # 分实验独立目录（脚本 + 日志 + 结果图全自洽）
    ├── exp1_controlled_tim/               # 实验一：受控权重噪声 TIM（vLLM BF16，80 步）
    │   ├── run.sh                         # 独立一键启动与环境校验脚本
    │   ├── run.py                         # 实验入口脚本
    │   ├── results.json                   # 80 步完整指标与 step_0/step_79（10+10）评测样本
    │   ├── plot.png                       # 4 算法收敛与 KL/比率对比图
    │   └── train.log                      # 完整远程运行日志
    ├── exp2_vllm_tim/                     # 实验二：真实 vLLM FP16 (PagedAttention+FlashInfer) TIM（80 步）
    │   └── run.sh / run.py / results.json / plot.png / train.log
    ├── exp3_fp8_tim/                      # 实验三：逐行 FP8 (torch.float8_e4m3fn) + vLLM 异步采样 TIM（80 步）
    │   └── run.sh / run.py / results.json / plot.png / train.log
    ├── exp4_vllm_official_tim/            # 实验四：官方超参对齐 + 500 步长程 11 算法对比（vLLM V1 FP16）
    │   └── run.sh / run.py / results.json / plot.png / train.log
    └── exp5_grpo_sc_ablation/             # 实验五：GRPO+SC 均衡数据集 5 组消融 (Stage 1) 与 500 步最优组合 (Stage 2)
        ├── run.sh / run.py                # 支持 stage1 (200 步) / stage2 (500 步) 与原子组合参数
        ├── results.json / plot.png / train_g5.log                   # Stage 1 结果与日志
        └── results_stage2.json / plot_stage2.png / train_stage2.log # Stage 2 结果与日志
```

## 对比方法与核心配置

- **通用模型与任务**：`Qwen/Qwen3-0.6B`，`max_new_tokens=160`，`eval_prompts=64` 固定保留验证集（零步基线 **17.2%**）。
- **实验一 ~ 实验三（80 步快速横向对比，4 方法）**：
  - 方法：`pg`（原生 REINFORCE）、`is`（朴素 IS，$w=p/q$）、`tis`（截断 IS，$w=\min(p/q, C_{\text{tis}})$）、`tis_sc`（TIS + Top-128 Score Centering）
  - 公共超参：`steps=80`，`eval_every=10`，`num_prompts=8`，`group_size=4`，`lr=0.006`，`max_grad_norm=1.0`，`micro_batch_size=4`，`reward_mode=group_centered`
  - TIM 机制差异：
    1. **实验一（`exp1_controlled_tim`）**：`vllm_dtype=bfloat16`，`noise_scale=0.03`，`staleness=1`，`tis_max=1.5`（零步平均 $|\Delta \log p|=0.05568$，最大 $1.25917$）。
    2. **实验二（`exp2_vllm_tim`）**：真实 `vllm_dtype=float16`（PagedAttention + FlashInfer），`weight_mismatch_sigma=0.015`，`staleness=2`，`tis_max=1.4`（零步平均 $|\Delta \log p|=0.03276$，初始最大 IS $2.18\times$）。
    3. **实验三（`exp3_fp8_tim`）**：逐行 `torch.float8_e4m3fn` 量化加载至 `vLLM BF16`，`staleness=2`，`tis_max=1.4`（零步平均 $|\Delta \log p|=0.04435$，初始最大 IS $1.80\times$）。
- **实验四（`exp4_vllm_official_tim`：官方对齐 500 步长程评测，11 方法，A100 80GB）**：
  - 对齐超参：`steps=500`（`eval_every=20`，重点记录 `step_300` 与 `step_499`），`num_prompts=4`，`group_size=8`，`lr=0.01`（SGD），`max_grad_norm=2.0`，`micro_batch_size=16`，`staleness=2`，`weight_mismatch_sigma=0.02`，`tis_max=2.0`，`topk=128`，`vllm_dtype=float16`（PagedAttention + FlashInfer）；零步探针平均 $|\Delta \log p|=0.03440$、最大 $0.59217$、初始 $|\mathrm{KL}|=0.00431$、初始最大 IS $1.57208\times$。
  - 11 种算法：无 SC 基线/裁剪族（`pg`, `is`, `tis`, `mis [0.5,5.0]`, `ppo [0.8,1.2] dual=3`, `dapo [0.8,1.28] dual=3`, `grpo (group_std+seq_norm)`）与 Score Centering 族（`tis_sc`, `ppo_sc`, `dapo_sc`, `grpo_sc`）。

## 80 步实验（实验一 ~ 实验三）核心结果

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

## 实验四（500 步官方对齐 · 11 算法长程真实 vLLM TIM）核心结果

| 算法 | 300 步评估 (`step_300`) | 500 步评估 (`step_499`) | 评估峰值 | 非零梯度步数 | 均梯度 (末步) | 平均 $\|\mathrm{KL}\|$ | 最大 IS 比率 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `pg` | 18.8% | 0.0% | 39.1% @400 | 70 / 500 | 1.69 (0.00) | 0.1183 | 10239.16x |
| `is` | 23.4% | 31.2% | 39.1% @200 | 60 / 500 | 1.00 (0.00) | 0.0548 | 211.84x |
| `tis` | 34.4% | 17.2% | 50.0% @400 | 210 / 500 | 1.00 (0.00) | 0.0326 | 188.46x |
| `mis` | 29.7% | 6.2% | 34.4% @220 | 194 / 500 | 1.12 (2.65) | 0.1048 | 3101.31x |
| `ppo` | 0.0% | 0.0% | 17.2% @0 | 26 / 500 | 0.09 (0.00) | 0.0180 | 46.84x |
| `dapo` | 0.0% | 0.0% | 34.4% @20 | 80 / 500 | 0.22 (0.00) | 0.0102 | 746.66x |
| `grpo` | 0.0% | 0.0% | 18.8% @40 | 53 / 500 | 0.23 (0.00) | 0.0094 | 30.26x |
| `tis_sc` | 9.4% | 12.5% | 20.3% @20 | 73 / 500 | 0.87 (0.00) | 0.0713 | 164.42x |
| `ppo_sc` | 0.0% | 17.2% | 50.0% @220 | 285 / 500 | 0.98 (1.15) | 0.0859 | 2183.96x |
| `dapo_sc` | 28.1% | **31.2%** | 48.4% @40 | 296 / 500 | 0.80 (0.74) | 0.0321 | 224.94x |
| **`grpo_sc`** | **51.6%** | **46.9%** | **56.2% @340** | **329 / 500** | 0.91 (0.00) | 0.0420 | 841.69x |

**关键观察**：
1. **80 步横向对比**：在全部 3 种 TIM 机制下，带重要性校正方法（`IS` / `TIS` / `TIS+SC`）在第 79 步均超越原生 `PG`；在硬件 FP8 严苛 TIM 下，`TIS` 与 `TIS+SC` 将末步梯度范数从 `PG` 的 `51.83` 压低至 `4.49` / `2.08`。
2. **500 步长程抗坍缩与 Score Centering 的决定性作用**：未加 Score Centering 的原生 `PG` 与三大裁剪算法（`PPO` / `DAPO` / `GRPO`）在第 500 步全部坍缩归零（`PPO`/`DAPO`/`GRPO` 在第 300 步已提前归零且后续几乎无有效梯度）。加入 Top-128 Score Centering 后全部避免归零：`DAPO+SC` 末期回稳至 `31.2%`，而结合 `group_std` 组内标准化 + `seq_norm` 序列均值的 **`GRPO+SC`** 以 **`51.6%`（300 步）/ `46.9%`（500 步）/ `56.2%`（峰值）** 与最高有效梯度密度（`329/500`）全程领跑。

## 实验五（GRPO+SC 均衡数据集 5 组机制消融 + 500 步最优组合）核心结果

为解决原始随机生成 Countdown 样本中纯加减占比过高（约 72%）、乘除与必含括号样本偏少导致的难题能力退化与后半程坍缩问题，实验五构建了 **5 层严格均衡且与验证集签名去重的数据集**（`BALANCED_SKILL_CYCLE`：`addsub=50%`, `mul_flat=12.5%`, `mul_parens=12.5%`, `div_flat=12.5%`, `div_parens=12.5%`，即加减 vs 乘除各占 50%，必含括号占总样本 25%），并将生成上限提升至 **`max_new_tokens=512`**（`max_model_len=768`, `num_prompts=4`, `staleness=2`, `weight_mismatch_sigma=0.02`, `topk=128`）。

### Stage 1：200 步均衡数据集 5 组压缩消融（`eval_every=20`）

| 组别 | 核心改进机制 | 0 步基线 | 200 步评估 (`step_199`) | 评估峰值 | 原分布 / 加减 / 乘 / 除 / 括号 (@199) | `strict_fmt` (@199) | 非零梯度步 | 平均 $\|\mathrm{KL}\|$ | 最大 IS 比率 | 末步长度 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `base_bal` | 均衡数据 + `grpo_sc` (`G=8`) | 10.9% | 31.2% | 62.5% @40 | 34.4% / 53.1% / 12.5% / 6.2% / 12.5% | 20.3% | 92.0% | 0.05 | 240.80x | 388.8 |
| `fmt_stop_dedup` | `stop="</answer>"` + `fmt_dedup` (`-0.25`) + `group_std_raw` | 10.9% | 21.9% | 39.1% @20 | 31.2% / 43.8% / 0.0% / 0.0% / 0.0% | 0.0% | 23.5% | 0.02 | 74.61x | 14.0 |
| `kl_ent_dapo` | `grpo_dapo_sc` (`[0.8,1.28]`) + `kl_beta=0.02` + `entropy=0.01` | 10.9% | 43.8% | 57.8% @140 | 56.2% / 68.8% / 18.8% / 18.8% / 25.0% | 48.4% | 99.5% | 0.02 | 401.88x | 246.0 |
| **`sample_var`** | **`group_size=16` + `dynamic_sampling=True`** | 10.9% | **68.8%** | **79.7% @140** | **71.9% / 93.8% / 37.5% / 50.0% / 25.0%** | 1.6% | **100.0%** | 0.05 | 593.47x | 415.3 |
| `sft_adamw` | 均衡 `SFT 50步` + `AdamW(5e-6, warmup=20+cosine)` | 17.2% | 48.4% | 50.0% @120 | 62.5% / 78.1% / 25.0% / 12.5% / 18.8% | 48.4% | 74.5% | **0.01** | **48.65x** | 394.8 |

### Stage 2：500 步终局对比（`eval_every=25`）

基于 Stage 1 归因诊断，Stage 2 将 `best_combo_500` 定为 **`sample_var` (`G=16 + DynSample`) + `stop_answer` (`"</answer>"`) + `strict_fmt`（纯 `{0,1}` 门控，剔除 `-0.25` 重复负惩罚与 `group_std_raw`）+ `dapo_clip` (`[0.8, 1.28]`) + `kl_ref_soft` (`kl_beta=0.008`，剔除导致乱码的 `entropy_coef=0.01` 与抑制乘除探索的 `SFT50`)**：

| 组别 | 数据与配置 | 0 步基线 | 300 步 (`step_300`) | **500 步 (`step_499`)** | **评估峰值** | 原分布 / 加减 / 乘 / 除 / 括号 (@499) | `strict_fmt` (@499) | 非零梯度步 | 平均 $\|\mathrm{KL}\|$ | 最大 IS 比率 (末步) | 末步长度 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `orig_data_500` | 原始非均衡 + `grpo_sc` (`G=8`) | 10.9% | 46.9% | 70.3% | 81.2% @150 | 73.4% / 96.9% / 37.5% / 50.0% / 31.2% | 29.7% | 88.4% | 0.02 | 580.61x (41.42x) | 183.0 |
| `base_bal_500` | 5 层均衡 + `grpo_sc` (`G=8`) | 10.9% | 0.0% | 0.0% | 59.4% @175 | 0.0% / 0.0% / 0.0% / 0.0% / 0.0% | 1.6% | 48.8% | 0.21 | 9074.15x (331.52x) | 505.2 |
| **`best_combo_500`** | **5 层均衡 + 最优 5 因子组合 (`G=16`)** | 10.9% | **73.4%** | **85.9%** | **90.6% @450** | **85.9% / 93.8% / 81.2% / 75.0% / 68.8%** | **90.6%** | **100.0%** | 0.03 | 1433.52x (**31.77x**) | **270.2** |

**实验五关键结论**：
1. **单靠数据均衡不足以支撑长程 GRPO+SC**：`base_bal_500` 在难题占比提高后，因 `group_size=8` 频繁遭遇全错/全对零方差批次（仅 `48.8%` 步有非零梯度）且无 KL 锚定，IS 比率飙升至 `9074.15x`，于第 250 步彻底坍缩至 `0.0%`；而未均衡 `orig_data_500` 在第 300 步同样发生阶段性退化（乘/除/括号全归 `0.0%`、生成死循环至 `512` token），500 步末期乘（`37.5%`）、除（`50.0%`）、括号（`31.2%`）及严格格式率（`29.7%`）均受限。
2. **五因子协同突破 90% 大关并消除长程退化**：`best_combo_500` 实现 **500/500 步 `100.0%` 有效梯度更新**，第 450 步平衡验证峰值达 **`90.6%`**（原分布 **`92.2%`**），第 499 步终局稳定在 **`85.9%`**（较 `orig_data_500` 提升 **`+15.6 pp`**），其中乘法 **`81.2%`**（`+43.7 pp`）、除法 **`75.0%`**（`+25.0 pp`）、必含括号 **`68.8%`**（`+37.6 pp`）、严格单标签格式率 **`90.6%`**（`+60.9 pp`），参考模型 KL 稳定收敛在 `0.15~0.23`。

## 本地运行与报告重生成

在配置有 CUDA GPU（及 `vLLM` 环境，各 `run.sh` 内置环境自检与自动安装）的节点上可单独运行任一实验：

```bash
bash experiments/exp1_controlled_tim/run.sh
bash experiments/exp2_vllm_tim/run.sh
bash experiments/exp3_fp8_tim/run.sh
bash experiments/exp4_vllm_official_tim/run.sh
bash experiments/exp5_grpo_sc_ablation/run.sh                                          # Stage 1 (200 步 5 组消融)
STAGE=stage2 STEPS=500 EVAL_EVERY=25 bash experiments/exp5_grpo_sc_ablation/run.sh \
  --results-filename results_stage2.json --plot-filename plot_stage2.png               # Stage 2 (500 步最优组合对比)
```

重新汇总全部六组实验结果与样本生成中文交互式报告：

```bash
python3 -B scripts/generate_report.py
```
