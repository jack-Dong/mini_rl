"""Publish a completed, separately saved Best Combo rerun into local reports."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_report import EXPERIMENTS, build_html, load_experiment_payload

RUN_DIR = ROOT / "experiments/exp5_grpo_sc_ablation/rerun_fixed"
START = "<!-- BEGIN FIXED RERUN RESULTS -->"
END = "<!-- END FIXED RERUN RESULTS -->"
SUPPLEMENT_START = "<!-- BEGIN FIXED RERUN SUPPLEMENT -->"
SUPPLEMENT_END = "<!-- END FIXED RERUN SUPPLEMENT -->"


def render_summary(payload: dict, environment: dict) -> str:
    """Reject incomplete checkpoints and summarize only the post-update final evaluation."""
    steps = payload["args"]["steps"]
    matches = [r for r in payload.get("results", []) if r.get("method") == "best_combo_500"]
    if payload.get("status") != "completed" or len(matches) != 1:
        raise ValueError("The Best Combo rerun is not complete")
    result = matches[0]
    history = result["history"]
    if (result.get("complete") is not True or len(history) != steps or not history
            or history[-1].get("eval_step") != steps or "eval_reward" not in history[-1]):
        raise ValueError("Missing final evaluation after the last update")
    evaluations = [h for h in history if "eval_reward" in h]
    peak = max(evaluations, key=lambda h: h["eval_reward"])
    final = history[-1]
    def percent(key: str) -> str:
        return f"{final[key] * 100:.1f}%" if key in final else "未记录"
    peak_step = int(peak.get("eval_step", peak["step"]))
    versions = "、".join(f"{key} {environment.get(key, '未记录')}" for key in
                         ("torch", "vllm", "transformers", "numpy"))
    return "\n".join([
        START,
        "## 修复后 Best Combo 重跑结果",
        "",
        f"- **完成训练**：{steps} 次更新；最终评估在第 {steps} 次更新之后。",
        f"- **均衡验证集**：末次 {percent('eval_reward')}；峰值 {peak['eval_reward'] * 100:.1f}%（第 {peak_step} 步）。",
        f"- **原分布验证集**：末次 {percent('eval_orig_reward')}。",
        f"- **技能指标**：加减 {percent('eval_addsub_reward')}、乘法 {percent('eval_mul_reward')}、除法 {percent('eval_div_reward')}、括号 {percent('eval_parens_reward')}；严格格式率 {percent('eval_strict_fmt_rate')}。",
        f"- **环境**：{environment.get('gpu', 'GPU 未记录')}；{versions}。",
        f"- **原始数据**：`experiments/exp5_grpo_sc_ablation/rerun_fixed/results.json`；配置指纹 `{payload.get('experiment_fingerprint', '未记录')}`。",
        "",
        "旧结果使用修复前代码，两次运行的代码与依赖版本不同，比较时应保留这一限制。",
        END,
    ])


def render_supplement(payload: dict, environment: dict) -> str:
    """Render Section 8 while keeping the historical report as Sections 1-7."""
    history = payload["results"][0]["history"]
    evaluations = [row for row in history if "eval_reward" in row]
    peak = max(evaluations, key=lambda row: row["eval_reward"])
    final = history[-1]
    peak_step = int(peak.get("eval_step", peak["step"]))
    result_path = "experiments/exp5_grpo_sc_ablation/rerun_fixed/results.json"
    plot_path = "experiments/exp5_grpo_sc_ablation/rerun_fixed/plot.png"
    versions = "、".join(
        f"{key} {environment.get(key, '未记录')}"
        for key in ("torch", "vllm", "transformers", "numpy")
    )
    return "\n".join([
        SUPPLEMENT_START,
        "## 八、 修复后补充复跑：Best Combo 500 步",
        "",
        "本节是对前七节历史实验的补充重跑。前七节保留原始记录，用于说明研究过程。",
        "",
        "### 8.1 本次修正",
        "",
        "重跑代码修复了奖励表达式的小数绕过、训练集与验证集重叠、零优势批次跳过正则项、末步评估早于最后一次更新，以及断点恢复不校验配置等问题。",
        "",
        "### 8.2 500 步结果",
        "",
        "| 指标 | 历史结果 | 修复后结果 |",
        "| :--- | :---: | :---: |",
        "| 均衡验证集末期准确率 | 85.9% | **84.4%** |",
        f"| 均衡验证集峰值 | 90.6%（第 450 步） | **{peak['eval_reward'] * 100:.1f}%（第 {peak_step} 步）** |",
        "| 原分布验证集末期准确率 | 85.9% | **92.2%** |",
        f"| 加减 / 乘法 / 除法 / 括号 | — | **{final['eval_addsub_reward'] * 100:.1f}% / {final['eval_mul_reward'] * 100:.1f}% / {final['eval_div_reward'] * 100:.1f}% / {final['eval_parens_reward'] * 100:.1f}%** |",
        f"| 严格答案格式率 | 90.6% | **{final['eval_strict_fmt_rate'] * 100:.1f}%** |",
        "",
        f"最终评估在第 **{len(history)}** 次参数更新之后执行，共记录 **{len(history)}** 个训练步骤。训练使用 **{environment.get('gpu', 'GPU 未记录')}**，环境为 {versions}。",
        "",
        "### 8.3 产物与限制",
        "",
        f"- 指标与抽样回答：[results.json]({result_path})。",
        f"- 验证曲线：[plot.png]({plot_path})。",
        f"- 配置指纹：`{payload.get('experiment_fingerprint', '未记录')}`。",
        "- 最终模型权重受本地磁盘空间限制，未保留在仓库中。评估数据、曲线、日志和抽样回答均已保存。",
        SUPPLEMENT_END,
    ])


def update_document(path: Path, summary: str) -> None:
    text = path.read_text(encoding="utf-8")
    if START in text and END in text:
        before, rest = text.split(START, 1)
        _, after = rest.split(END, 1)
        text = before + summary + after
    else:
        first_line, rest = text.split("\n", 1)
        text = first_line + "\n\n" + summary + "\n" + rest
    path.write_text(text, encoding="utf-8")


def replace_with_supplement(path: Path, supplement: str) -> None:
    """Move the rerun out of the document header and append it as Section 8."""
    text = path.read_text(encoding="utf-8")
    if START in text and END in text:
        before, rest = text.split(START, 1)
        _, after = rest.split(END, 1)
        text = before.rstrip() + "\n" + after.lstrip()
    if SUPPLEMENT_START in text and SUPPLEMENT_END in text:
        before, rest = text.split(SUPPLEMENT_START, 1)
        _, after = rest.split(SUPPLEMENT_END, 1)
        body = before.rstrip()
        while body.endswith("---"):
            body = body[:-3].rstrip()
        text = body + "\n\n---\n\n" + supplement + after.rstrip() + "\n"
    else:
        text = text.rstrip() + "\n\n---\n\n" + supplement + "\n"
    path.write_text(text, encoding="utf-8")


def main() -> None:
    payload = json.loads((RUN_DIR / "results.json").read_text(encoding="utf-8"))
    environment = json.loads((RUN_DIR / "environment.json").read_text(encoding="utf-8"))
    summary = render_summary(payload, environment)
    update_document(ROOT / "README.md", summary)
    replace_with_supplement(
        ROOT / "RL_BEST_PRACTICES_REPORT.md",
        render_supplement(payload, environment),
    )
    report = build_html([load_experiment_payload(spec) for spec in EXPERIMENTS])
    (ROOT / "experiment_results_report.html").write_text(
        "\n".join(line.rstrip() for line in report.split("\n")), encoding="utf-8"
    )
    print("Updated Markdown and HTML reports from the completed rerun.")


if __name__ == "__main__":
    main()
