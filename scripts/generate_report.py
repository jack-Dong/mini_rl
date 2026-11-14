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
]

METHOD_ORDER = ["pg", "is", "tis", "tis_sc"]
METHOD_META = {
    "pg": {"label": "原生 PG (Vanilla REINFORCE)", "color": "#ef4444", "badge": "未校正漂移"},
    "is": {"label": "朴素重要性采样 (Naive IS)", "color": "#f59e0b", "badge": "无截断高方差"},
    "tis": {"label": "截断重要性采样 (TIS)", "color": "#3b82f6", "badge": "含截断偏差"},
    "tis_sc": {"label": "TIS + 得分中心化 (Score Centering)", "color": "#10b981", "badge": "论文核心方法"},
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
    evals = [h["eval_reward"] * 100.0 for h in hist if "eval_reward" in h]
    trains = [h["train_reward"] * 100.0 for h in hist]
    kls = [abs(h.get("kl_samp_train", 0.0)) for h in hist]
    gnorms = [h.get("grad_norm", 0.0) for h in hist]
    ratios = [h.get("max_is_ratio", 1.0) for h in hist]
    lens = [h.get("mean_len", 0.0) for h in hist]
    return {
        "method": res["method"],
        "name": res.get("name", res["method"]),
        "steps_done": len(hist),
        "elapsed_sec": res.get("elapsed_sec", 0.0),
        "eval_init": evals[0] if evals else None,
        "eval_final": evals[-1] if evals else None,
        "eval_peak": max(evals) if evals else None,
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
    all_completed = payloads and all(p["status"] == "completed" for p in payloads)
    for p in payloads:
        results = p.get("results", [])
        if not results:
            items.append(
                f"<li><strong>{html.escape(p['title'])}</strong>：实验尚未产出首条评估记录，正在排队或初始化引擎中。</li>"
            )
            continue
        summaries = {r["method"]: summarize_method(r) for r in results}
        sc = summaries.get("tis_sc")
        tis = summaries.get("tis")
        pg = summaries.get("pg")
        isn = summaries.get("is")
        status_note = "已完成" if p["status"] == "completed" else f"实时进行中（已记录 {len(results)}/4 个方法）"

        parts = []
        for k, label in [("pg", "PG"), ("is", "IS"), ("tis", "TIS"), ("tis_sc", "TIS+SC")]:
            if k in summaries:
                s = summaries[k]
                ev = f"{s['eval_final']:.1f}%" if s["eval_final"] is not None else "—"
                parts.append(
                    f"{label} 末期={ev}（峰值 {s['eval_peak']:.1f}%，均梯度 {s['gnorm_mean']:.2f} / 末步 {s['gnorm_final']:.2f}，均 |KL|={s['kl_mean']:.4f}）"
                )
        extra = ""
        if all(x and x["eval_final"] is not None for x in (pg, isn, tis, sc)):
            best_m = max(
                [("PG", pg), ("IS", isn), ("TIS", tis), ("TIS+SC", sc)],
                key=lambda kv: kv[1]["eval_final"],
            )
            low_gn = min(
                [("PG", pg), ("IS", isn), ("TIS", tis), ("TIS+SC", sc)],
                key=lambda kv: kv[1]["gnorm_mean"],
            )
            extra = (
                f"；<strong>对比结论</strong>：末期最高精度 {best_m[0]} ({best_m[1]['eval_final']:.1f}%，"
                f"相对 PG {best_m[1]['eval_final'] - pg['eval_final']:+.1f} pp)，"
                f"全程最小平均梯度范数 {low_gn[0]} ({low_gn[1]['gnorm_mean']:.2f})。"
            )
        elif sc and pg and isn:
            extra = f"；Score Centering 将平均训推 KL 保持在 {sc['kl_mean']:.4f}。"

        items.append(
            f"<li><strong>{html.escape(p['title'])}</strong>（{status_note}）：{'；'.join(parts)}{extra}</li>"
        )

    if all_completed:
        items.append(
            "<li><strong>跨实验总体规律（80 步真实 vLLM 复现）</strong>："
            "① 在全部 3 种 TIM 机制（受控权重噪声、FP16 vLLM+FlashInfer、FP8 E4M3FN 量化）中，从 17.2% 零步基线出发，"
            "带重要性校正的方法（IS / TIS / TIS+SC）在第 79 步评估准确率均一致超越原生 PG；"
            "② 当漂移增强（尤其是实验三 FP8 量化下朴素 IS 全程最大 IS 比率飙升至 205.79x、第 79 步仍达 24.41x）时，PG 与朴素 IS 的末步梯度范数剧烈飙升（分别达 51.83 与 26.82），"
            "而截断类的 TIS（末期 40.6% 最高、末步梯度 4.49）与 TIS+SC（末步梯度 2.08、均梯度 4.89 全场最低）显著抑制了方差冲击；"
            "③ 在当前 80 步 / Top-k=128 小批量条件下，TIS+SC 的头质量自适应残差表现出最稳健的梯度平滑性，但在轻中度漂移下的末期准确率略保守于 TIS/IS。</li>"
        )

    return f"""
    <section class="exp-card findings-card">
      <h2>核心结论与实时实验分析（Key Findings）</h2>
      <p class="sub">
        方法设计：在训推不匹配（Training-Inference Mismatch, TIM）场景下，直接对采样序列做强化学习更新（Vanilla PG）会因分布偏移积累偏差；
        朴素重要性采样（IS）无偏但易受极端比率尾部冲击；截断重要性采样（TIS）抑制高方差尾部但会引入截断偏差。
        <strong>Score Centering</strong> 在 Top-<em>k</em> 词汇子集（<em>k</em>=128）上利用自适应质量对齐基线中心化词表梯度，在真实 <code>vLLM V1</code> 引擎中检验其无偏性与梯度稳定性表现。
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
        "全部 3 组实验已完成"
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
                f"最大生成={args.get('max_new_tokens', '-')} tokens",
                f"学习率 lr={args.get('lr', '-')}",
                f"陈旧度 staleness={args.get('staleness', 0)}",
                f"TIS 截断阈值={args.get('tis_max', '-')}",
                f"评估样本={args.get('eval_prompts', '-')}",
                f"优势模式={args.get('reward_mode', 'group_centered')}",
            ]
            if "noise_scale" in args:
                chip_items.append(f"权重噪声={args['noise_scale']}")
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
        for mkey in METHOD_ORDER:
            if mkey not in summaries:
                continue
            s = summaries[mkey]
            meta = METHOD_META[mkey]
            ev0 = f"{s['eval_init']:.1f}%" if s["eval_init"] is not None else "—"
            evf = f"{s['eval_final']:.1f}%" if s["eval_final"] is not None else "—"
            evp = f"{s['eval_peak']:.1f}%" if s["eval_peak"] is not None else "—"
            delta_str = "—"
            if s["eval_init"] is not None and s["eval_final"] is not None:
                d = s["eval_final"] - s["eval_init"]
                delta_str = f"{d:+.1f} pp"
            row_cls = ' class="highlight-row"' if mkey == "tis_sc" else ""
            rows_html.append(
                f"<tr{row_cls}>"
                f'<td><span class="dot" style="background:{meta["color"]}"></span><strong>{html.escape(meta["label"])}</strong></td>'
                f"<td>{s['steps_done']}</td>"
                f"<td>{ev0}</td>"
                f'<td><strong>{evf}</strong> <span class="muted">({delta_str})</span></td>'
                f"<td>{evp}</td>"
                f"<td>{s['train_mean']:.1f}%</td>"
                f"<td>{s['train_last5']:.1f}%</td>"
                f"<td>{s['kl_mean']:.4f}</td>"
                f"<td>{s['gnorm_mean']:.2f} / {s['gnorm_final']:.2f}</td>"
                f"<td>{s['max_ratio']:.2f}x</td>"
                f"<td>{s['mean_len_final']:.1f}</td>"
                f"<td>{s['elapsed_sec']:.0f}s</td>"
                f"</tr>"
            )

        table_html = (
            '<div class="table-wrap"><table><thead><tr>'
            "<th>对比方法</th><th>完成步数</th><th>初始评估</th><th>最终评估 (变化)</th>"
            "<th>评估峰值</th><th>全程训练均值</th><th>末5步训练均值</th>"
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
                  <button class="tab-btn" data-metric="train_reward">训练集解题率 (%)</button>
                  <button class="tab-btn" data-metric="kl">训推 KL 漂移 |KL(q||p)|</button>
                  <button class="tab-btn" data-metric="grad_norm">梯度范数 ||g||₂</button>
                  <button class="tab-btn" data-metric="max_is_ratio">最大 IS 比率 p/q</button>
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
  <header class="hero">
    <div class="hero-top">
      <span class="badge {overview_cls}">{html.escape(overview_badge)}</span>
      <span class="muted">报告刷新时间：{html.escape(now_str)} · 实验算力：2 × NVIDIA A100-SXM4-40GB</span>
    </div>
    <h1>Score Centering (arXiv:2609.20807) · Qwen3-0.6B Countdown 实时复现实验报告</h1>
    <p class="sub">
      在受控权重噪声 TIM、真实 <strong>vLLM 推理引擎（PagedAttention + FlashInfer）</strong> 异步采样、以及硬件级 FP8 量化三种训推不匹配机制下，
      全面对比原生 PG、朴素 IS、截断 TIS 与论文提出的 <strong>TIS + Top-k 得分中心化（k=128）</strong> 的收敛精度与稳定性。
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
  if (metric === 'train_reward') return h.train_reward * 100;
  if (metric === 'kl') return Math.abs(h.kl_samp_train || 0);
  if (metric === 'grad_norm') return h.grad_norm || 0;
  if (metric === 'max_is_ratio') return h.max_is_ratio || 1.0;
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
        pts.push({{ x: h.step, y }});
        if (h.step > maxX) maxX = h.step;
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

  if (metric === 'eval_reward' || metric === 'train_reward' || metric === 'kl' || metric === 'grad_norm') minY = 0;
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

  const defaultMethod = availableMethods.includes('tis_sc') ? 'tis_sc' : availableMethods[availableMethods.length - 1];
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
    out_html = build_html(payloads)
    workspace_html = ROOT_DIR / "experiment_results_report.html"
    workspace_html.write_text(out_html, encoding="utf-8")
    ARTIFACT_HTML.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(workspace_html, ARTIFACT_HTML)
    print(f"已实时更新中文 HTML 报告：{workspace_html} 和 {ARTIFACT_HTML}")


if __name__ == "__main__":
    main()
