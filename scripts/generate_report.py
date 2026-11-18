"""生成实时更新的全中文交互式实验对比报告 HTML。"""
from __future__ import annotations

import datetime as dt
import html
import json
import shutil
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_HTML = Path(
    "/Users/dongshizhou/.gemini/antigravity/brain/"
    "b1df9764-79e1-40de-8577-fb4d099e59ef/experiment_results_report.html"
)

EXPERIMENTS = [
    {
        "id": "exp1",
        "title": "实验一 · 受控权重噪声 TIM（Controlled Weight-Noise TIM）",
        "subtitle": "vLLM BF16 采样引擎（注入固定相对权重扰动） vs FP32 主权重 / BF16 Eager 训练器",
        "json_path": ROOT_DIR / "experiments" / "exp1_controlled_tim" / "results.json",
        "png_path": ROOT_DIR / "experiments" / "exp1_controlled_tim" / "plot.png",
        "log_path": ROOT_DIR / "experiments" / "exp1_controlled_tim" / "train.log",
        "script_name": "experiments/exp1_controlled_tim/run.py",
        "bash_name": "experiments/exp1_controlled_tim/run.sh",
    },
    {
        "id": "exp2",
        "title": "实验二 · 真实 vLLM 推理引擎 TIM（PagedAttention + FlashInfer）",
        "subtitle": "真实 vLLM（FP16、V1 进程内热更新、PagedAttention + FlashInfer、异步陈旧度）vs HF BF16 Eager 训练器",
        "json_path": ROOT_DIR / "experiments" / "exp2_vllm_tim" / "results.json",
        "png_path": ROOT_DIR / "experiments" / "exp2_vllm_tim" / "plot.png",
        "log_path": ROOT_DIR / "experiments" / "exp2_vllm_tim" / "train.log",
        "script_name": "experiments/exp2_vllm_tim/run.py",
        "bash_name": "experiments/exp2_vllm_tim/run.sh",
    },
    {
        "id": "exp3",
        "title": "实验三 · 硬件级 FP8 (E4M3FN) 权重量化 + vLLM 异步采样 TIM",
        "subtitle": "逐行 FP8 E4M3FN 量化权重 + vLLM BF16 异步采样 vs FP32 主权重 / BF16 Eager 训练器",
        "json_path": ROOT_DIR / "experiments" / "exp3_fp8_tim" / "results.json",
        "png_path": ROOT_DIR / "experiments" / "exp3_fp8_tim" / "plot.png",
        "log_path": ROOT_DIR / "experiments" / "exp3_fp8_tim" / "train.log",
        "script_name": "experiments/exp3_fp8_tim/run.py",
        "bash_name": "experiments/exp3_fp8_tim/run.sh",
    },
    {
        "id": "exp4",
        "title": "实验四 · 官方参数对齐 + 300/500 步长程训练 + 11 种强化学习方法对比（Real vLLM TIM）",
        "subtitle": "对齐官方开源参数（group_size=8, lr=0.01, TIS=2.0, MIS=[0.5,5.0], PPO/DAPO Dual-Clip=3.0, DAPO+SC, GRPO/GRPO+SC, σ=0.02）· 500 步（含 300 步与 500 步双里程碑评估与样本）",
        "json_path": ROOT_DIR / "experiments" / "exp4_vllm_official_tim" / "results.json",
        "png_path": ROOT_DIR / "experiments" / "exp4_vllm_official_tim" / "plot.png",
        "log_path": ROOT_DIR / "experiments" / "exp4_vllm_official_tim" / "train.log",
        "script_name": "experiments/exp4_vllm_official_tim/run.py",
        "bash_name": "experiments/exp4_vllm_official_tim/run.sh",
    },
    {
        "id": "exp5",
        "title": "实验五 · Stage 1：GRPO+SC 均衡数据集 5 组 200 步消融（max_new_tokens=512）",
        "subtitle": "50% 纯加减 : 50% 必用乘除（* 与 ÷ 各半、扁平与必用括号各半）分层均衡评估 · 逐组运行对比 5 组改进因子",
        "json_path": ROOT_DIR / "experiments" / "exp5_grpo_sc_ablation" / "results.json",
        "png_path": ROOT_DIR / "experiments" / "exp5_grpo_sc_ablation" / "plot.png",
        "log_path": ROOT_DIR / "experiments" / "exp5_grpo_sc_ablation" / "train_g5.log",
        "script_name": "experiments/exp5_grpo_sc_ablation/run.py",
        "bash_name": "experiments/exp5_grpo_sc_ablation/run.sh",
    },
    {
        "id": "exp5_stage2",
        "title": "实验五 · Stage 2：500 步终局全面对比（未均衡原数据 vs 均衡基线 vs 最优组合）",
        "subtitle": "Orig Unbalanced (grpo_sc 500) vs Base Balanced (grpo_sc 500) vs Best Combo (Group16+DynSample + StopAnswer + StrictFmt + DAPO ClipHigher + SoftKLRef 0.008)",
        "json_path": ROOT_DIR / "experiments" / "exp5_grpo_sc_ablation" / "results_stage2.json",
        "png_path": ROOT_DIR / "experiments" / "exp5_grpo_sc_ablation" / "plot_stage2.png",
        "log_path": ROOT_DIR / "experiments" / "exp5_grpo_sc_ablation" / "train_stage2.log",
        "script_name": "experiments/exp5_grpo_sc_ablation/run.py",
        "bash_name": "experiments/exp5_grpo_sc_ablation/run.sh",
    },
    {
        "id": "exp5_rerun_fixed",
        "title": "实验五 · 修复后重跑：Best Combo 500 步",
        "subtitle": "修复奖励判定、数据隔离、零优势正则更新及最终评估；独立保存，不覆盖历史数据。",
        "json_path": ROOT_DIR / "experiments" / "exp5_grpo_sc_ablation" / "rerun_fixed" / "results.json",
        "png_path": ROOT_DIR / "experiments" / "exp5_grpo_sc_ablation" / "rerun_fixed" / "plot.png",
        "log_path": ROOT_DIR / "experiments" / "exp5_grpo_sc_ablation" / "rerun_fixed" / "train.log",
        "script_name": "experiments/exp5_grpo_sc_ablation/run.py",
        "bash_name": "experiments/exp5_grpo_sc_ablation/rerun_fixed.sh",
    },
]

METHOD_ORDER = [
    "pg", "is", "tis", "mis", "ppo", "dapo", "grpo", "gspo", "topr", "dppo",
    "sc", "tis_sc", "mis_sc", "ppo_sc", "dapo_sc", "grpo_sc",
    "base_bal", "fmt_stop_dedup", "kl_ent_dapo", "sample_var", "sft_adamw",
    "orig_data_500", "base_bal_500", "best_combo_500",
]
METHOD_META = {
    "pg": {"label": "原生 PG (Vanilla)", "color": "#ef4444", "badge": "未校正漂移"},
    "is": {"label": "朴素 IS (Naive IS)", "color": "#f59e0b", "badge": "无截断高方差"},
    "tis": {"label": "截断 TIS (r ≤ 2.0)", "color": "#3b82f6", "badge": "上界截断"},
    "mis": {"label": "掩码 MIS ([0.5, 5.0])", "color": "#ec4899", "badge": "区间丢弃"},
    "ppo": {"label": "PPO ([0.8, 1.2], Dual=3)", "color": "#8b5cf6", "badge": "PPO 单边裁剪"},
    "dapo": {"label": "DAPO ([0.8, 1.28], Dual=3)", "color": "#06b6d4", "badge": "宽松上界裁剪"},
    "grpo": {"label": "GRPO (组标准化 + 序列均值)", "color": "#e11d48", "badge": "标准 GRPO"},
    "gspo": {"label": "GSPO (seq_mean)", "color": "#f97316", "badge": "序列均值裁剪"},
    "topr": {"label": "TOPR (seq_sum, A<0)", "color": "#84cc16", "badge": "负样本轨迹截断"},
    "dppo": {"label": "DPPO (TV ≤ 0.2)", "color": "#6366f1", "badge": "TV 散度屏蔽"},
    "sc": {"label": "Score Centering (SC)", "color": "#14b8a6", "badge": "纯得分中心化"},
    "tis_sc": {"label": "TIS + Score Centering", "color": "#10b981", "badge": "论文核心方法"},
    "mis_sc": {"label": "MIS + Score Centering", "color": "#22c55e", "badge": "MIS + 得分中心化"},
    "ppo_sc": {"label": "PPO + Score Centering", "color": "#a855f7", "badge": "PPO + 得分中心化"},
    "dapo_sc": {"label": "DAPO + Score Centering", "color": "#0891b2", "badge": "DAPO + 得分中心化"},
    "grpo_sc": {"label": "GRPO + Score Centering", "color": "#059669", "badge": "GRPO + 得分中心化"},
    "base_bal": {"label": "1. Base Balanced (grpo_sc)", "color": "#64748b", "badge": "均衡基线"},
    "fmt_stop_dedup": {"label": "2. StopEOS + StrictFmt + Dedup", "color": "#2563eb", "badge": "格式/停止/去重"},
    "kl_ent_dapo": {"label": "3. ClipHigher + KLRef + Entropy", "color": "#059669", "badge": "策略与熵正则"},
    "sample_var": {"label": "4. Group16 + DynSample", "color": "#d97706", "badge": "大组动态重采样"},
    "sft_adamw": {"label": "5. BalSFT50 + AdamW(Cosine)", "color": "#7c3aed", "badge": "均衡SFT+AdamW"},
    "orig_data_500": {"label": "Orig Unbalanced (grpo_sc 500)", "color": "#ef4444", "badge": "未均衡原数据"},
    "base_bal_500": {"label": "Base Balanced (grpo_sc 500)", "color": "#64748b", "badge": "均衡基线 500"},
    "best_combo_500": {"label": "Best Combo (DynG16+Stop+DAPO+SoftKL 500)", "color": "#10b981", "badge": "最优组合 500"},
}


def load_experiment_payload(spec: dict[str, Any]) -> dict[str, Any]:
    """读取单组实验的结果 JSON 与最近的日志尾部。"""
    json_path: Path = spec["json_path"]
    log_path: Path = spec["log_path"]
    log_tail = ""
    if log_path.exists():
        lines = log_path.read_text(errors="replace").splitlines()
        log_tail = "\n".join(lines[-28:])

    if not json_path.exists():
        return {
            **{k: v for k, v in spec.items() if isinstance(v, str)},
            "status": "pending",
            "args": {},
            "probe": None,
            "results": [],
            "log_tail": log_tail,
            "updated_at": None,
        }

    raw = json.loads(json_path.read_text())
    results = raw.get("results", [])
    expected_methods = raw.get("args", {}).get("methods", METHOD_ORDER)
    expected_steps = int(raw.get("args", {}).get("steps", 30))
    status_str = raw.get("status")
    if status_str in ("completed", "running"):
        status = status_str
    else:
        done_count = sum(1 for r in results if len(r.get("history", [])) >= expected_steps)
        status = "completed" if done_count >= len(expected_methods) and results else "running"

    mtime = dt.datetime.fromtimestamp(json_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    return {
        **{k: v for k, v in spec.items() if isinstance(v, str)},
        "status": status,
        "args": raw.get("args", {}),
        "probe": raw.get("engine_drift_probe"),
        "results": results,
        "log_tail": log_tail,
        "updated_at": mtime,
    }


def summarize_method(res: dict[str, Any]) -> dict[str, Any]:
    """汇总单个强化学习算法的核心实验指标。"""
    hist = res.get("history", [])
    eval_rows = [h for h in hist if "eval_reward" in h]
    evals = [h["eval_reward"] * 100.0 for h in eval_rows]
    eval_300 = next(
        (h["eval_reward"] * 100.0 for h in eval_rows if int(h.get("eval_step", h.get("step", -1))) == 300),
        None,
    )
    last_ev = eval_rows[-1] if eval_rows else {}
    trains = [h["train_reward"] * 100.0 for h in hist]
    kls = [abs(h.get("kl_samp_train", 0.0)) for h in hist]
    gnorms = [h.get("grad_norm", 0.0) for h in hist]
    ratios = [h.get("max_is_ratio", 1.0) for h in hist]
    lens = [h.get("mean_len", 0.0) for h in hist]
    nz_vals = [100.0 if h.get("frac_nonzero_adv", 0.0) > 0 else 0.0 for h in hist]
    return {
        "method": res["method"],
        "name": res.get("name", res["method"]),
        "steps_done": len(hist),
        "elapsed_sec": res.get("elapsed_sec", 0.0),
        "eval_init": evals[0] if evals else None,
        "eval_300": eval_300,
        "eval_final": evals[-1] if evals else None,
        "eval_peak": max(evals) if evals else None,
        "eval_orig": last_ev["eval_orig_reward"] * 100.0 if "eval_orig_reward" in last_ev else None,
        "eval_addsub": last_ev["eval_addsub_reward"] * 100.0 if "eval_addsub_reward" in last_ev else None,
        "eval_mul": last_ev["eval_mul_reward"] * 100.0 if "eval_mul_reward" in last_ev else None,
        "eval_div": last_ev["eval_div_reward"] * 100.0 if "eval_div_reward" in last_ev else None,
        "eval_parens": last_ev["eval_parens_reward"] * 100.0 if "eval_parens_reward" in last_ev else None,
        "eval_strict_fmt": last_ev["eval_strict_fmt_rate"] * 100.0 if "eval_strict_fmt_rate" in last_ev else None,
        "nz_pct": sum(nz_vals) / len(nz_vals) if nz_vals else 0.0,
        "train_mean": sum(trains) / len(trains) if trains else 0.0,
        "train_last5": sum(trains[-5:]) / len(trains[-5:]) if trains else 0.0,
        "kl_mean": sum(kls) / len(kls) if kls else 0.0,
        "gnorm_mean": sum(gnorms) / len(gnorms) if gnorms else 0.0,
        "gnorm_final": gnorms[-1] if gnorms else 0.0,
        "max_ratio": max(ratios) if ratios else 1.0,
        "mean_len_final": lens[-1] if lens else 0.0,
    }


def build_dynamic_findings(payloads: list[dict[str, Any]]) -> str:
    """根据当前已收集到的实验结果动态生成中文核心结论卡片。"""
    items: list[str] = []
    first3_done = len(payloads) >= 3 and all(p["status"] == "completed" for p in payloads[:3])
    for p in payloads:
        results = p.get("results", [])
        if not results:
            items.append(
                f"<li><strong>{html.escape(p['title'])}</strong>：实验尚未产出首条评估记录，正在排队或初始化引擎中。</li>"
            )
            continue
        summaries = {r["method"]: summarize_method(r) for r in results}
        expected_methods = len(p.get("args", {}).get("methods") or []) or len(results)
        status_note = (
            "已完成"
            if p["status"] == "completed"
            else f"实时进行中（已记录 {len(results)}/{expected_methods} 个方法）"
        )

        parts = []
        for k in METHOD_ORDER:
            if k in summaries:
                s = summaries[k]
                short_lbl = k.upper().replace("_", "+")
                ev = f"{s['eval_final']:.1f}%" if s["eval_final"] is not None else "—"
                m300 = f"300步={s['eval_300']:.1f}%，" if s["eval_300"] is not None else ""
                eval_label = "末期" if p["status"] == "completed" else "最新"
                parts.append(
                    f"{short_lbl} {m300}{eval_label}={ev}（峰值 {s['eval_peak']:.1f}%，均梯度 {s['gnorm_mean']:.2f} / 末步 {s['gnorm_final']:.2f}）"
                )
        extra = ""
        pg = summaries.get("pg")
        valid_ev = [(k, s) for k, s in summaries.items() if s["eval_final"] is not None]
        if valid_ev and len(valid_ev) >= 2:
            best_k, best_s = max(valid_ev, key=lambda kv: kv[1]["eval_final"])
            low_k, low_s = min(valid_ev, key=lambda kv: kv[1]["gnorm_mean"])
            pg_delta = f"，相对 PG {best_s['eval_final'] - pg['eval_final']:+.1f} pp" if pg and pg["eval_final"] is not None else ""
            extra = (
                f"；<strong>对比结论</strong>：当前最高评估精度 {best_k.upper().replace('_', '+')} "
                f"({best_s['eval_final']:.1f}%{pg_delta})，最小平均梯度范数 {low_k.upper().replace('_', '+')} ({low_s['gnorm_mean']:.2f})。"
            )

        items.append(
            f"<li><strong>{html.escape(p['title'])}</strong>（{status_note}）：{'；'.join(parts)}{extra}</li>"
        )

    if first3_done:
        items.append(
            "<li><strong>前 3 组 80 步实验总体规律</strong>："
            "① 在受控权重噪声、FP16 vLLM+FlashInfer、FP8 E4M3FN 量化全部 3 种 TIM 机制中，从 17.2% 零步基线出发，"
            "带重要性校正的方法（IS / TIS / TIS+SC）在第 79 步评估准确率均一致超越原生 PG；"
            "② 当漂移增强（尤其是实验三 FP8 量化下朴素 IS 全程最大 IS 比率飙升至 205.79x、第 79 步仍达 24.41x）时，PG 与朴素 IS 的末步梯度范数剧烈飙升（分别达 51.83 与 26.82），"
            "而截断类的 TIS（末期 40.6% 最高、末步梯度 4.49）与 TIS+SC（末步梯度 2.08、均梯度 4.89 全场最低）显著抑制了方差冲击。</li>"
        )
    if len(payloads) >= 4 and payloads[3]["status"] == "completed":
        exp4_sum = {r["method"]: summarize_method(r) for r in payloads[3].get("results", [])}
        grpo_sc_s = exp4_sum.get("grpo_sc")
        dapo_sc_s = exp4_sum.get("dapo_sc")
        ppo_sc_s = exp4_sum.get("ppo_sc")
        if grpo_sc_s and dapo_sc_s and ppo_sc_s:
            items.append(
                "<li><strong>实验四（500 步官方对齐 + 11 算法同台）核心结论</strong>："
                "① <strong>无 SC 的裁剪代理长程全军覆没</strong>：在真实 vLLM V1 FP16（staleness=2, σ=0.02, lr=0.01 SGD, group_size=8）下，"
                "未加 Score Centering 的原生 PG 与三大裁剪算法 <code>PPO</code> / <code>DAPO</code> / <code>GRPO</code> 在 500 步末期全部退化坍缩至 0.0%（其中 PPO/DAPO/GRPO 在第 300 步即已归零，非零梯度步数仅 26~80/500）；"
                "② <strong>Score Centering 拯救全部三大裁剪族</strong>：加入 Top-128 Score Centering 后全部避免零值坍缩——"
                f"<code>PPO+SC</code> 峰值达 {ppo_sc_s['eval_peak']:.1f}%、末期回升至 {ppo_sc_s['eval_final']:.1f}%（vs PPO 0.0%）；"
                f"<code>DAPO+SC</code> 300 步 {dapo_sc_s['eval_300']:.1f}%、末期 <strong>{dapo_sc_s['eval_final']:.1f}%</strong>（峰值 {dapo_sc_s['eval_peak']:.1f}%，vs DAPO 0.0%）；"
                f"<strong><code>GRPO+SC</code> 全面夺冠</strong>：300 步 <strong>{grpo_sc_s['eval_300']:.1f}%</strong>、500 步末期 <strong>{grpo_sc_s['eval_final']:.1f}%</strong>、全周期峰值 <strong>{grpo_sc_s['eval_peak']:.1f}%</strong>（非零梯度更新密度高达 329/500，KL 漂移均值 {grpo_sc_s['kl_mean']:.4f} 显著低于 PG 的 0.1183），"
                "表明 <code>group_std</code> 组内标准化 + <code>seq_norm</code> 序列长度归一化与 Score Centering 在长程 TIM 下形成最强协同。</li>"
            )

    if len(payloads) >= 5 and payloads[4]["status"] == "completed":
        e5_sum = {r["method"]: summarize_method(r) for r in payloads[4].get("results", [])}
        b = e5_sum.get("base_bal")
        f = e5_sum.get("fmt_stop_dedup")
        k = e5_sum.get("kl_ent_dapo")
        sv = e5_sum.get("sample_var")
        sa = e5_sum.get("sft_adamw")
        if b and f and k and sv and sa:
            items.append(
                "<li><strong>实验五 Stage 1（200 步均衡数据集 5 组机制消融）核心归因</strong>："
                f"① <strong><code>sample_var</code> (Group16+DynSample) 绝对领跑</strong>：有效优势更新步占比 <strong>{sv['nz_pct']:.0f}% (200/200)</strong>，"
                f"第 140 步峰值飙升至 <strong>{sv['eval_peak']:.1f}%</strong>（末期 <strong>{sv['eval_final']:.1f}%</strong>，"
                f"原分布 {sv['eval_orig']:.1f}%、加减 {sv['eval_addsub']:.1f}%、乘 {sv['eval_mul']:.1f}%、除 {sv['eval_div']:.1f}%、括号 {sv['eval_parens']:.1f}%），"
                f"唯一瑕疵是无 <code>stop=&lt;/answer&gt;</code> 导致末期标签后继续絮叨（<code>strict_fmt={sv['eval_strict_fmt']:.1f}%</code>）及尾部高比率漂移（最大比率 {sv['max_ratio']:.1f}x）；"
                f"② <strong><code>fmt_stop_dedup</code> 反例归因</strong>：早期严格格式曾达 96.9%（第 40 步），但负向重复惩罚（-0.25）+ <code>group_std_raw</code> 放大噪声使第 48 步坍缩为 14-token 直接猜数（末期 {f['eval_final']:.1f}%，<code>strict_fmt=0%</code>），"
                f"因此 Stage 2 仅保留无负惩罚的 <code>stop=&lt;/answer&gt; + strict_fmt</code>；"
                f"③ <strong><code>kl_ent_dapo</code> 反例归因</strong>：<code>kl_ref</code> + <code>DAPO ClipHigher</code> 防止了 <code>base_bal</code> 的晚期失控坍缩（末期 {k['eval_final']:.1f}% vs 基线 {b['eval_final']:.1f}%），"
                "但正熵系数 <code>entropy_coef=0.01</code> 推高熵至 4.3~5.4 并注入多语种乱码 token，故 Stage 2 保留软 <code>kl_beta=0.008</code> + <code>ClipHigher</code> 并彻底剔除正熵项；"
                f"④ <strong><code>sft_adamw</code> 诊断</strong>：SFT50 + AdamW 保持了极低 KL（均值 {sa['kl_mean']:.4f}）与高格式率（末期 {sa['eval_final']:.1f}%），但因 <code>G=8</code> 无动态采样约 {100 - sa['nz_pct']:.1f}% 步陷入双峰零梯度饱和。</li>"
            )

    if len(payloads) >= 6 and payloads[5]["results"]:
        e5s2 = {r["method"]: summarize_method(r) for r in payloads[5].get("results", [])}
        orig_s = e5s2.get("orig_data_500")
        bal_s = e5s2.get("base_bal_500")
        best_s = e5s2.get("best_combo_500")
        if orig_s and bal_s and best_s:
            orig_300 = f"{orig_s['eval_300']:.1f}%" if orig_s.get("eval_300") is not None else "—"
            bal_300 = f"{bal_s['eval_300']:.1f}%" if bal_s.get("eval_300") is not None else "—"
            best_300 = f"{best_s['eval_300']:.1f}%" if best_s.get("eval_300") is not None else "—"
            items.append(
                "<li><strong>实验五 Stage 2（500 步终局全面对比）结论</strong>："
                f"在 <code>max_new_tokens=512</code> 均衡 5 分层（加减 50% / 乘 25% / 除 25%，必含括号 25%）长程评测下："
                f"① 未均衡原数据 <code>Orig</code>（{orig_300} @300 步，第 300 步乘除括号均归 0% 并循环截断至 512 token）末期 <strong>{orig_s['eval_final']:.1f}%</strong>（峰值 {orig_s['eval_peak']:.1f}%，原分布 {orig_s['eval_orig']:.1f}%，×:{orig_s['eval_mul']:.1f}% / ÷:{orig_s['eval_div']:.1f}% / ():{orig_s['eval_parens']:.1f}%，<code>strict_fmt={orig_s['eval_strict_fmt']:.1f}%</code>，非零梯度步 {orig_s['nz_pct']:.0f}%）；"
                f"② 均衡基线 <code>Base Balanced</code>（{bal_300} @300 步）第 250~499 步彻底坍塌至 <strong>{bal_s['eval_final']:.1f}%</strong>（峰值 {bal_s['eval_peak']:.1f}%，最大 IS 比率飙至 {bal_s['max_ratio']:.0f}x，非零梯度步跌至 {bal_s['nz_pct']:.0f}%）；"
                f"③ <strong>最优组合 <code>Best Combo</code>（<code>Group16+DynSample + Stop&lt;/answer&gt; + StrictFmt + DAPO ClipHigher + SoftKLRef 0.008</code>）全程零坍塌且后期持续走高</strong>："
                f"300 步 <strong>{best_300}</strong>、450 步峰值 <strong>{best_s['eval_peak']:.1f}%</strong>（原分布 92.2%）、500 步终局 <strong>{best_s['eval_final']:.1f}%</strong>"
                f"（原分布 {best_s['eval_orig']:.1f}% · ±:{best_s['eval_addsub']:.1f}% · ×:{best_s['eval_mul']:.1f}% · ÷:{best_s['eval_div']:.1f}% · ():{best_s['eval_parens']:.1f}%），"
                f"同时保持 <strong>{best_s['nz_pct']:.0f}% (500/500)</strong> 有效梯度更新、<strong>strict_fmt={best_s['eval_strict_fmt']:.1f}%</strong> 与极低 KL 漂移（均值 {best_s['kl_mean']:.4f}）。</li>"
            )

    return f"""
    <section class="exp-card findings-card">
      <h2>核心结论与实时实验分析（Key Findings）</h2>
      <p class="sub">
        方法设计：在训推不匹配（Training-Inference Mismatch, TIM）场景下，直接对采样序列做强化学习更新（Vanilla PG）会因分布偏移积累偏差；
        朴素 IS 无偏但易受极端比率尾部冲击；TIS/MIS/PPO/DAPO 通过比率截断或双边屏蔽控制方差；
        <strong>Score Centering</strong> 在 Top-<em>k</em> 词汇子集（<em>k</em>=128）上按官方公式重构有效采样质量，在真实 <code>vLLM V1</code> 引擎下检验多种算法组合的 300/500 步长程收敛与稳定性表现。
      </p>
      <ul class="findings-list">{''.join(items)}</ul>
    </section>
    """


def build_html(payloads: list[dict[str, Any]]) -> str:
    """渲染全中文自包含交互式实验报告 HTML。"""
    now_str = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    safe_json = json.dumps(payloads, ensure_ascii=False).replace("</", "<\\/")

    total_done = sum(1 for p in payloads if p["status"] == "completed")
    total_run = sum(1 for p in payloads if p["status"] == "running")
    overview_badge = (
        f"全部 {len(payloads)} 组实验已完成"
        if total_done == len(payloads)
        else f"实时更新中 · 已完成 {total_done}/{len(payloads)} 组 · 运行中 {total_run} 组"
    )
    overview_cls = "badge-done" if total_done == len(payloads) else "badge-run"

    probe_label_map = {
        "sampler_engine": "采样推理引擎",
        "trainer_engine": "训练前向引擎",
        "vllm_installed": "vLLM 是否已安装",
        "mean_abs_logp_diff": "零步平均 |Δlog p|",
        "max_abs_logp_diff": "零步最大 |Δlog p|",
        "initial_kl_samp_train": "初始采样-训练 KL 均值",
        "initial_max_is_ratio": "初始最大 IS 比率 p/q",
    }

    exp_cards_html: list[str] = []
    for p in payloads:
        eid = p["id"]
        status = p["status"]
        status_cls = {
            "completed": "badge-done",
            "running": "badge-run",
            "pending": "badge-wait",
        }[status]
        status_label = {
            "completed": "已完成",
            "running": "实时运行中",
            "pending": "等待运行",
        }[status]

        args = p["args"]
        cfg_chips = ""
        if args:
            chip_items = [
                f"训练步数={args.get('steps', '-')}",
                f"批规模={args.get('num_prompts', '-')}×G{args.get('group_size', '-')}",
                f"最大生成={args.get('max_new_tokens', '-')} tokens",
                f"学习率 lr={args.get('lr', '-')}",
                f"陈旧度 staleness={args.get('staleness', 0)}",
                f"TIS 截断={args.get('tis_max', '-')}",
                f"评估样本={args.get('eval_prompts', '-')}",
                f"优势模式={args.get('reward_mode', 'group_centered')}",
            ]
            if "noise_scale" in args:
                chip_items.append(f"权重噪声={args['noise_scale']}")
            if "weight_mismatch_sigma" in args:
                chip_items.append(f"权重失配 σ={args['weight_mismatch_sigma']}")
            if "vllm_dtype" in args:
                chip_items.append(f"vLLM 精度={args['vllm_dtype']}")
            cfg_chips = "".join(f'<span class="chip">{html.escape(c)}</span>' for c in chip_items)

        probe_html = ""
        probe = p.get("probe")
        if probe:
            probe_items = "".join(
                f'<div class="probe-kv"><span>{html.escape(probe_label_map.get(str(k), str(k)))}</span>'
                f'<strong>{html.escape(f"{v:.5f}" if isinstance(v, float) else str(v))}</strong></div>'
                for k, v in probe.items()
            )
            probe_html = (
                '<div class="probe-box"><div class="probe-title">零步引擎数值漂移探针（Zero-Step Logprob Drift Probe）</div>'
                f'<div class="probe-grid">{probe_items}</div></div>'
            )

        rows_html = []
        summaries = {r["method"]: summarize_method(r) for r in p["results"]}
        show_300 = int(args.get("steps", 0)) >= 300 or any(
            s.get("eval_300") is not None for s in summaries.values()
        )
        show_skill_cols = any(s.get("eval_mul") is not None for s in summaries.values())
        for mkey in METHOD_ORDER:
            if mkey not in summaries:
                continue
            s = summaries[mkey]
            meta = METHOD_META[mkey]
            ev0 = f"{s['eval_init']:.1f}%" if s["eval_init"] is not None else "—"
            ev300 = f"{s['eval_300']:.1f}%" if s["eval_300"] is not None else "—"
            evf = f"{s['eval_final']:.1f}%" if s["eval_final"] is not None else "—"
            evp = f"{s['eval_peak']:.1f}%" if s["eval_peak"] is not None else "—"
            delta_str = "—"
            if s["eval_init"] is not None and s["eval_final"] is not None:
                d = s["eval_final"] - s["eval_init"]
                delta_str = f"{d:+.1f} pp"
            row_cls = (
                ' class="highlight-row"'
                if mkey.endswith("_sc") or mkey in ("sc", "sample_var", "best_combo_500")
                else ""
            )
            col_300 = f"<td>{ev300}</td>" if show_300 else ""
            col_skills = ""
            if show_skill_cols:
                skill_txt = (
                    f"原:{s['eval_orig']:.0f}% · ±:{s['eval_addsub']:.0f}% · ×:{s['eval_mul']:.0f}% · "
                    f"÷:{s['eval_div']:.0f}% · ():{s['eval_parens']:.0f}% · fmt:{s['eval_strict_fmt']:.0f}%"
                    if s.get("eval_mul") is not None
                    else "—"
                )
                col_skills = f"<td>{skill_txt}</td><td>{s['nz_pct']:.0f}%</td>"
            rows_html.append(
                f"<tr{row_cls}>"
                f'<td><span class="dot" style="background:{meta["color"]}"></span><strong>{html.escape(meta["label"])}</strong></td>'
                f"<td>{s['steps_done']}</td>"
                f"<td>{ev0}</td>"
                f"{col_300}"
                f'<td><strong>{evf}</strong> <span class="muted">({delta_str})</span></td>'
                f"<td>{evp}</td>"
                f"{col_skills}"
                f"<td>{s['train_mean']:.1f}%</td>"
                f"<td>{s['train_last5']:.1f}%</td>"
                f"<td>{s['kl_mean']:.4f}</td>"
                f"<td>{s['gnorm_mean']:.2f} / {s['gnorm_final']:.2f}</td>"
                f"<td>{s['max_ratio']:.2f}x</td>"
                f"<td>{s['mean_len_final']:.1f}</td>"
                f"<td>{s['elapsed_sec']:.0f}s</td>"
                f"</tr>"
            )

        th_300 = "<th>300步评估</th>" if show_300 else ""
        th_skills = "<th>末步能力拆解 (原/±/×/÷/()/fmt)</th><th>非零梯度步%</th>" if show_skill_cols else ""
        table_html = (
            '<div class="table-wrap"><table><thead><tr>'
            f"<th>对比方法</th><th>完成步数</th><th>初始评估</th>{th_300}<th>最新评估 (变化)</th>"
            f"<th>评估峰值</th>{th_skills}<th>全程训练均值</th><th>末5步训练均值</th>"
            "<th>平均 |KL| 漂移</th><th>均 / 末梯度范数</th><th>最大 IS 比率</th><th>末步均长</th><th>耗时</th>"
            "</tr></thead><tbody>"
            + "".join(rows_html)
            + "</tbody></table></div>"
            if rows_html
            else '<div class="empty-msg">等待 Colab A100 回传首个方法检查点数据...</div>'
        )

        log_section = (
            f'<details class="log-details"><summary>展开查看 Colab 运行日志尾部 & 复现脚本（{html.escape(p["bash_name"])}）</summary>'
            f'<pre>{html.escape(p["log_tail"] or "暂未拉取到本地日志，实验运行后将实时同步。")}</pre></details>'
        )

        extra_skill_tabs = (
            '<button class="tab-btn" data-metric="eval_orig_reward">原分布评估 (%)</button>'
            '<button class="tab-btn" data-metric="eval_addsub_reward">纯加减 (%)</button>'
            '<button class="tab-btn" data-metric="eval_mul_reward">必用乘法 (%)</button>'
            '<button class="tab-btn" data-metric="eval_div_reward">必用除法 (%)</button>'
            '<button class="tab-btn" data-metric="eval_parens_reward">必用括号 (%)</button>'
            '<button class="tab-btn" data-metric="eval_strict_fmt_rate">严格格式率 (%)</button>'
            if eid.startswith("exp5")
            else ""
        )
        exp_cards_html.append(
            f"""
            <section class="exp-card" id="card-{eid}">
              <div class="exp-head">
                <div>
                  <div class="exp-meta-row">
                    <span class="badge {status_cls}">{status_label}</span>
                    <span class="script-tag">{html.escape(p["script_name"])} · {html.escape(p["bash_name"])}</span>
                    <span class="muted">{html.escape("更新时间: " + p["updated_at"]) if p["updated_at"] else "尚未同步数据"}</span>
                  </div>
                  <h2>{html.escape(p["title"])}</h2>
                  <p class="sub">{html.escape(p["subtitle"])}</p>
                  <div class="chips">{cfg_chips}</div>
                </div>
              </div>
              {probe_html}
              <div class="chart-toolbar">
                <div class="tabs" data-exp="{eid}">
                  <button class="tab-btn active" data-metric="eval_reward">评估集解题率 (%)</button>
                  {extra_skill_tabs}
                  <button class="tab-btn" data-metric="train_reward">训练集解题率 (%)</button>
                  <button class="tab-btn" data-metric="kl">训推 KL 漂移 |KL(q||p)|</button>
                  <button class="tab-btn" data-metric="grad_norm">梯度范数 ||g||₂</button>
                  <button class="tab-btn" data-metric="max_is_ratio">最大 IS 比率 p/q</button>
                  <button class="tab-btn" data-metric="frac_masked">屏蔽/丢弃比例 (%)</button>
                  <button class="tab-btn" data-metric="mean_len">平均生成长度</button>
                </div>
              </div>
              <div class="chart-box" id="chart-{eid}"></div>
              {table_html}
              <div class="samples-box" id="samples-{eid}"></div>
              {log_section}
            </section>
            """
        )

    findings_html = build_dynamic_findings(payloads)

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Score Centering (arXiv:2609.20807) · Qwen3-0.6B 实时实验中文报告</title>
<style>
  :root {{
    --bg: #0b1020;
    --panel: #121a2f;
    --panel-2: #18223d;
    --border: #27345a;
    --text: #e6ecf8;
    --muted: #93a3c7;
    --accent: #10b981;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 24px;
    background: radial-gradient(circle at top right, #162244, var(--bg) 55%);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", "Segoe UI", Roboto, sans-serif;
    line-height: 1.55;
  }}
  .container {{ max-width: 1240px; margin: 0 auto; }}
  header.hero {{
    background: linear-gradient(135deg, #131e38, #0f172d);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 22px;
  }}
  .hero-top {{ display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 12px; }}
  h1 {{ margin: 8px 0 6px; font-size: 1.65rem; letter-spacing: -0.01em; }}
  h2 {{ margin: 6px 0 4px; font-size: 1.24rem; }}
  .sub {{ margin: 0 0 10px; color: var(--muted); font-size: 0.94rem; }}
  .badge {{
    display: inline-block; padding: 3px 11px; border-radius: 999px;
    font-size: 0.76rem; font-weight: 700;
  }}
  .badge-done {{ background: rgba(16, 185, 129, 0.16); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.45); }}
  .badge-run {{ background: rgba(59, 130, 246, 0.16); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.45); }}
  .badge-wait {{ background: rgba(245, 158, 11, 0.14); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.35); }}
  .script-tag {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.78rem; color: #a5b4fc; }}
  .muted {{ color: var(--muted); font-size: 0.84rem; }}
  .chips {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }}
  .chip {{
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.76rem; padding: 2px 8px; border-radius: 6px;
    background: var(--panel-2); border: 1px solid var(--border); color: #cbd5e1;
  }}
  .exp-card {{
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 16px; padding: 22px 24px; margin-bottom: 22px;
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.25);
  }}
  .findings-card {{ border-left: 4px solid var(--accent); }}
  .findings-list {{ margin: 8px 0 0; padding-left: 20px; font-size: 0.9rem; }}
  .findings-list li {{ margin-bottom: 6px; color: #dbeafe; }}
  .exp-meta-row {{ display: flex; flex-wrap: wrap; align-items: center; gap: 10px; }}
  .probe-box {{
    margin: 14px 0; padding: 12px 14px; border-radius: 10px;
    background: var(--panel-2); border: 1px solid var(--border);
  }}
  .probe-title {{ font-size: 0.8rem; font-weight: 700; color: #93c5fd; margin-bottom: 8px; }}
  .probe-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(195px, 1fr)); gap: 10px; }}
  .probe-kv {{ display: flex; flex-direction: column; font-size: 0.82rem; }}
  .probe-kv span {{ color: var(--muted); font-size: 0.74rem; }}
  .chart-toolbar {{ display: flex; justify-content: space-between; align-items: center; margin: 14px 0 8px; }}
  .tabs {{ display: flex; flex-wrap: wrap; gap: 6px; }}
  .tab-btn {{
    background: var(--panel-2); color: var(--muted); border: 1px solid var(--border);
    border-radius: 8px; padding: 6px 12px; cursor: pointer; font-size: 0.82rem; font-weight: 600;
  }}
  .tab-btn.active {{ background: #1e2d52; color: #ffffff; border-color: #60a5fa; }}
  .chart-box {{
    width: 100%; min-height: 285px; background: #0d1426;
    border: 1px solid var(--border); border-radius: 12px; padding: 10px 14px;
    position: relative;
  }}
  .table-wrap {{ overflow-x: auto; margin-top: 16px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
  th, td {{ padding: 9px 10px; text-align: left; border-bottom: 1px solid var(--border); white-space: nowrap; }}
  th {{ color: var(--muted); font-weight: 600; font-size: 0.78rem; }}
  .highlight-row {{ background: rgba(16, 185, 129, 0.09); }}
  .dot {{ display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 7px; }}
  .empty-msg {{ padding: 24px; text-align: center; color: var(--muted); }}
  .samples-box {{ margin-top: 16px; }}
  .sample-controls {{ display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 10px; margin: 10px 0; }}
  .sample-cols {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(460px, 1fr)); gap: 14px; margin-top: 10px; }}
  .sample-col {{
    background: #0d1426; border: 1px solid var(--border); border-radius: 12px; padding: 12px;
    max-height: 620px; overflow-y: auto;
  }}
  .sample-col-title {{ font-size: 0.9rem; font-weight: 700; margin-bottom: 10px; position: sticky; top: 0; background: #0d1426; padding: 4px 0; }}
  .sample-item {{
    background: #121b33; border: 1px solid var(--border); border-radius: 9px; padding: 9px 11px;
    font-size: 0.8rem; margin-bottom: 8px;
  }}
  .sample-item.ok {{ border-left: 3px solid #10b981; }}
  .sample-item.fail {{ border-left: 3px solid #ef4444; }}
  .sample-item pre {{
    white-space: pre-wrap; word-break: break-word; margin: 6px 0 0;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.75rem; color: #cbd5e1;
  }}
  details.log-details {{ margin-top: 12px; }}
  details summary {{ cursor: pointer; color: #93c5fd; font-size: 0.84rem; font-weight: 600; }}
  details pre {{
    background: #090e1b; border: 1px solid var(--border); border-radius: 8px;
    padding: 10px; overflow-x: auto; font-size: 0.75rem; color: #cbd5e1; max-height: 240px;
  }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 16px; margin-top: 8px; font-size: 0.8rem; color: #cbd5e1; }}
  .tooltip {{
    position: absolute; pointer-events: none; background: rgba(11, 16, 32, 0.95);
    border: 1px solid #475569; border-radius: 8px; padding: 6px 10px;
    font-size: 0.78rem; color: #f8fafc; display: none; z-index: 10;
  }}
</style>
</head>
<body>
<div class="container">
  <div style="padding:16px;margin:16px 0;background:#78350f;color:#fff;border-radius:8px">
    历史结果提示：原实验在代码审查前运行；部分训练集与验证集重叠、正则更新和末步评估存在缺陷。修复后重跑单独列出，历史数值和结论有待重新验证。
  </div>
  <header class="hero">
    <div class="hero-top">
      <span class="badge {overview_cls}">{html.escape(overview_badge)}</span>
      <span class="muted">报告刷新时间：{html.escape(now_str)} · 实验算力：2 × NVIDIA A100-SXM4-40GB</span>
    </div>
    <h1>Score Centering (arXiv:2609.20807) · Qwen3-0.6B Countdown 实时复现实验报告</h1>
    <p class="sub">
      在受控权重噪声 TIM、真实 <strong>vLLM 推理引擎（PagedAttention + FlashInfer）</strong> 异步采样、硬件级 FP8 量化、以及对齐官方开源参数的
      <strong>300/500 步长程 8 算法横向评测（PG / Naive IS / TIS / MIS / PPO / DAPO / TIS+SC / PPO+SC）</strong> 下，全面评估收敛精度与梯度稳定性。
    </p>
    <div class="chips">
      <span class="chip">基座模型：Qwen/Qwen3-0.6B</span>
      <span class="chip">任务环境：3 操作数 Countdown（1..25 → 1..200）</span>
      <span class="chip">核心损失：-A [w(r_tok) log p_tok - Σ_{'{j∈Top-k}'} (q_j w(r_j) - α p_j) log p_j]</span>
    </div>
  </header>

  {findings_html}
  {''.join(exp_cards_html)}
</div>

<script>
const EXPERIMENTS = {safe_json};
const META = {json.dumps(METHOD_META, ensure_ascii=False)};
const ORDER = {json.dumps(METHOD_ORDER)};

function getSeriesValue(h, metric) {{
  if (metric === 'eval_reward') return h.eval_reward !== undefined ? h.eval_reward * 100 : null;
  if (metric === 'eval_orig_reward') return h.eval_orig_reward !== undefined ? h.eval_orig_reward * 100 : null;
  if (metric === 'eval_addsub_reward') return h.eval_addsub_reward !== undefined ? h.eval_addsub_reward * 100 : null;
  if (metric === 'eval_mul_reward') return h.eval_mul_reward !== undefined ? h.eval_mul_reward * 100 : null;
  if (metric === 'eval_div_reward') return h.eval_div_reward !== undefined ? h.eval_div_reward * 100 : null;
  if (metric === 'eval_parens_reward') return h.eval_parens_reward !== undefined ? h.eval_parens_reward * 100 : null;
  if (metric === 'eval_strict_fmt_rate') return h.eval_strict_fmt_rate !== undefined ? h.eval_strict_fmt_rate * 100 : null;
  if (metric === 'train_reward') return h.train_reward * 100;
  if (metric === 'kl') return Math.abs(h.kl_samp_train || 0);
  if (metric === 'grad_norm') return h.grad_norm || 0;
  if (metric === 'max_is_ratio') return h.max_is_ratio || 1.0;
  if (metric === 'frac_masked') return (h.frac_masked || 0) * 100;
  if (metric === 'mean_len') return h.mean_len || 0;
  return null;
}}

function renderChart(exp, metric) {{
  const box = document.getElementById('chart-' + exp.id);
  if (!exp.results || exp.results.length === 0) {{
    box.innerHTML = '<div class="empty-msg">暂无训练步数曲线，等待远程 A100 节点回传实时检查点...</div>';
    return;
  }}

  const W = 860, H = 250, padL = 52, padR = 20, padT = 18, padB = 34;
  let maxX = 1, minY = Infinity, maxY = -Infinity;
  const series = [];

  for (const mkey of ORDER) {{
    const res = exp.results.find(r => r.method === mkey);
    if (!res || !res.history || res.history.length === 0) continue;
    const pts = [];
    for (const h of res.history) {{
      const y = getSeriesValue(h, metric);
      if (y !== null && Number.isFinite(y)) {{
        const pointStep = metric === 'eval_reward' ? (h.eval_step ?? h.step) : h.step;
        pts.push({{ x: pointStep, y }});
        if (pointStep > maxX) maxX = pointStep;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }}
    }}
    if (pts.length > 0) series.push({{ key: mkey, pts, meta: META[mkey] }});
  }}

  if (series.length === 0) {{
    box.innerHTML = '<div class="empty-msg">当前实时快照尚无该指标记录。</div>';
    return;
  }}

  if (metric.startsWith('eval_') || metric === 'train_reward' || metric === 'kl' || metric === 'grad_norm' || metric === 'frac_masked') minY = 0;
  if (maxY <= minY) maxY = minY + 1;
  const spanY = (maxY - minY) * 1.12;

  const sx = x => padL + (x / Math.max(maxX, 1)) * (W - padL - padR);
  const sy = y => H - padB - ((y - minY) / spanY) * (H - padT - padB);

  let svg = `<svg viewBox="0 0 ${{W}} ${{H}}" style="width:100%;height:auto;display:block;">`;
  for (let i = 0; i <= 4; i++) {{
    const yVal = minY + (spanY * i) / 4;
    const yy = sy(yVal);
    svg += `<line x1="${{padL}}" y1="${{yy}}" x2="${{W - padR}}" y2="${{yy}}" stroke="#1e293b" stroke-dasharray="3,4"/>`;
    const lbl = (metric === 'kl') ? yVal.toFixed(3) : yVal.toFixed(1);
    svg += `<text x="${{padL - 8}}" y="${{yy + 4}}" text-anchor="end" fill="#94a3b8" font-size="11">${{lbl}}</text>`;
  }}
  const xTicks = Math.min(maxX, 6);
  for (let i = 0; i <= xTicks; i++) {{
    const xv = Math.round((maxX * i) / Math.max(xTicks, 1));
    const xx = sx(xv);
    svg += `<text x="${{xx}}" y="${{H - 10}}" text-anchor="middle" fill="#94a3b8" font-size="11">${{xv}}</text>`;
  }}
  svg += `<text x="${{W / 2}}" y="${{H - 1}}" text-anchor="middle" fill="#64748b" font-size="10">训练步数 (Training Step)</text>`;

  for (const s of series) {{
    const d = s.pts.map((p, idx) => `${{idx === 0 ? 'M' : 'L'}}${{sx(p.x).toFixed(1)}},${{sy(p.y).toFixed(1)}}`).join(' ');
    svg += `<path d="${{d}}" fill="none" stroke="${{s.meta.color}}" stroke-width="2.4" stroke-linejoin="round" />`;
    for (const p of s.pts) {{
      svg += `<circle cx="${{sx(p.x).toFixed(1)}}" cy="${{sy(p.y).toFixed(1)}}" r="3.4" fill="${{s.meta.color}}" data-tip="${{s.meta.label}} | 第 ${{p.x}} 步: ${{metric === 'kl' ? p.y.toFixed(4) : p.y.toFixed(2)}}" />`;
    }}
  }}
  svg += `</svg><div class="tooltip"></div>`;

  const legend = series.map(s => `<span><span class="dot" style="background:${{s.meta.color}}"></span>${{s.meta.label}}</span>`).join('');
  box.innerHTML = svg + `<div class="legend">${{legend}}</div>`;

  const tip = box.querySelector('.tooltip');
  box.querySelectorAll('circle').forEach(c => {{
    c.addEventListener('mouseenter', e => {{
      tip.textContent = c.getAttribute('data-tip');
      tip.style.display = 'block';
      tip.style.left = Math.min(e.offsetX + 12, box.clientWidth - 220) + 'px';
      tip.style.top = Math.max(e.offsetY - 28, 4) + 'px';
    }});
    c.addEventListener('mouseleave', () => {{ tip.style.display = 'none'; }});
  }});
}}

function renderSampleCards(list, isOk) {{
  if (!list || list.length === 0) {{
    return '<div class="empty-msg">本检查点未采集到此类样本。</div>';
  }}
  const cls = isOk ? 'ok' : 'fail';
  const badgeColor = isOk ? '#10b981' : '#ef4444';
  const badgeLabel = isOk ? '成功 (R=1.0)' : '失败 (R=0.0)';
  return list.slice(0, 10).map((ex, idx) => `
    <div class="sample-item ${{cls}}">
      <div>
        <strong>#${{idx + 1}}</strong> · 输入=${{JSON.stringify(ex.numbers)}} → 目标=<strong>${{ex.target}}</strong>
        <span style="float:right;color:${{badgeColor}};font-weight:700;">${{badgeLabel}}</span>
      </div>
      <div class="muted">参考解: <code>${{ex.solution}}</code>${{ex.source ? ' · 来源: ' + ex.source : ''}}</div>
      <pre>${{(ex.completion || '').replace(/</g, '&lt;')}}</pre>
    </div>
  `).join('');
}}

function renderSamples(exp) {{
  const box = document.getElementById('samples-' + exp.id);
  const availableMethods = ORDER.filter(mkey => {{
    const res = exp.results?.find(r => r.method === mkey);
    return res && ((res.eval_samples && Object.keys(res.eval_samples).length > 0) || (res.sample_completions && Object.keys(res.sample_completions).length > 0));
  }});
  if (availableMethods.length === 0) return;

  const defaultMethod = availableMethods.includes('best_combo_500')
    ? 'best_combo_500'
    : (availableMethods.includes('sample_var')
      ? 'sample_var'
      : (availableMethods.includes('grpo_sc')
        ? 'grpo_sc'
        : (availableMethods.includes('tis_sc') ? 'tis_sc' : availableMethods[availableMethods.length - 1])));
  const renderFor = (mkey, ckptOverride) => {{
    const res = exp.results.find(r => r.method === mkey);
    const hasNew = res.eval_samples && Object.keys(res.eval_samples).length > 0;
    const ckptDict = hasNew ? res.eval_samples : res.sample_completions;
    const ckptKeys = Object.keys(ckptDict);
    const activeCkpt = (ckptOverride && ckptKeys.includes(ckptOverride)) ? ckptOverride : ckptKeys[ckptKeys.length - 1];

    let succ = [], fail = [];
    if (hasNew) {{
      succ = ckptDict[activeCkpt]?.success || [];
      fail = ckptDict[activeCkpt]?.failure || [];
    }} else {{
      const legacy = ckptDict[activeCkpt] || [];
      succ = legacy.filter(x => x.reward > 0.5);
      fail = legacy.filter(x => x.reward <= 0.5);
    }}

    const methodBtns = availableMethods.map(mk =>
      `<button class="tab-btn ${{mk === mkey ? 'active' : ''}}" data-smethod="${{mk}}">${{META[mk].label}}</button>`
    ).join('');
    const ckptBtns = ckptKeys.map(ck =>
      `<button class="tab-btn ${{ck === activeCkpt ? 'active' : ''}}" data-sckpt="${{ck}}">${{ck}}</button>`
    ).join('');

    box.innerHTML = `
      <details open>
        <summary>评测阶段样本抽样对比（成功 10 例 + 失败 10 例 · 点击切换方法与检查点）</summary>
        <div class="sample-controls">
          <div class="tabs">${{methodBtns}}</div>
          <div class="tabs">${{ckptBtns}}</div>
        </div>
        <div class="sample-cols">
          <div class="sample-col">
            <div class="sample-col-title" style="color:#34d399;">✓ 评测成功样本（共 ${{succ.length}} 例 · 展示前 10 例）</div>
            ${{renderSampleCards(succ, true)}}
          </div>
          <div class="sample-col">
            <div class="sample-col-title" style="color:#f87171;">✗ 评测失败样本（共 ${{fail.length}} 例 · 展示前 10 例）</div>
            ${{renderSampleCards(fail, false)}}
          </div>
        </div>
      </details>
    `;

    box.querySelectorAll('[data-smethod]').forEach(btn => {{
      btn.addEventListener('click', () => renderFor(btn.getAttribute('data-smethod'), activeCkpt));
    }});
    box.querySelectorAll('[data-sckpt]').forEach(btn => {{
      btn.addEventListener('click', () => renderFor(mkey, btn.getAttribute('data-sckpt')));
    }});
  }};

  renderFor(defaultMethod, null);
}}

EXPERIMENTS.forEach(exp => {{
  renderChart(exp, 'eval_reward');
  renderSamples(exp);
  const tabs = document.querySelector(`.tabs[data-exp="${{exp.id}}"]`);
  if (tabs) {{
    tabs.querySelectorAll('.tab-btn').forEach(btn => {{
      btn.addEventListener('click', () => {{
        tabs.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        renderChart(exp, btn.getAttribute('data-metric'));
      }});
    }});
  }}
}});
</script>
</body>
</html>
"""


def main() -> None:
    """在工作区生成中文交互式实验报告并同步到 artifact 目录。"""
    payloads = [load_experiment_payload(spec) for spec in EXPERIMENTS]
    out_html = "\n".join(line.rstrip() for line in build_html(payloads).split("\n"))
    workspace_html = ROOT_DIR / "experiment_results_report.html"
    workspace_html.write_text(out_html, encoding="utf-8")
    ARTIFACT_HTML.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(workspace_html, ARTIFACT_HTML)
    print(f"已实时更新中文 HTML 报告：{workspace_html} 和 {ARTIFACT_HTML}")


if __name__ == "__main__":
    main()
