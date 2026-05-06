"""
Side-by-side comparison of v4 (baseline) / v5 (C1) / v6 (C1 + scores in prompt)
validation runs on the same eICU v4 cohort (150 patients).

Reads:
  validation/results/EICU_results_v4_latest.json     (baseline)
  validation/results/EICU_results_v5_c1_latest.json  (C1 only)
  validation/results/EICU_results_v6_c1_scores_latest.json (C1 + scores) -- optional

Writes:
  validation/results/v4_v5_v6_comparison.json
  validation/docs/EICU_VALIDATION_COMPARISON.md
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent
RES = ROOT / "results"
DOCS = ROOT / "docs"

RUNS = [
    ("v4_t0 baseline (T=0, no C1, no scores)",  RES / "EICU_results_v4_t0_latest.json"),
    ("v5_t0_c1 (T=0, C1 only)",                 RES / "EICU_results_v5_t0_c1_latest.json"),
    ("v6_t0_c1_scores (T=0, C1 + scores)",      RES / "EICU_results_v6_t0_c1_scores_latest.json"),
    ("v7_t0_c1_scores_c2 (T=0, C1+scores+C2)",  RES / "EICU_results_v7_t0_c1_scores_c2_latest.json"),
]


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = (z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def confusion(rows: list) -> dict:
    succ = [r for r in rows if r.get("status") == "success" and r["predicted_sepsis"] in (True, False)]
    tp = sum(1 for r in succ if r["actual_sepsis"] and r["predicted_sepsis"])
    fn = sum(1 for r in succ if r["actual_sepsis"] and not r["predicted_sepsis"])
    fp = sum(1 for r in succ if not r["actual_sepsis"] and r["predicted_sepsis"])
    tn = sum(1 for r in succ if not r["actual_sepsis"] and not r["predicted_sepsis"])
    sens = wilson(tp, tp+fn)
    spec = wilson(tn, tn+fp)
    ppv  = wilson(tp, tp+fp) if (tp+fp) else (0,0,0)
    npv  = wilson(tn, tn+fn) if (tn+fn) else (0,0,0)
    f1 = 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) else 0.0
    return {
        "n": len(succ), "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "sensitivity": sens, "specificity": spec, "ppv": ppv, "npv": npv, "f1": f1,
        "false_alarm_rate": fp/(fp+tn) if (fp+tn) else 0,
    }


def fmt_ci(t: tuple) -> str:
    return f"{100*t[0]:5.1f}% (CI {100*t[1]:.1f}–{100*t[2]:.1f})"


def index_by_pid(rows: list) -> dict:
    return {r["patient_id"]: r for r in rows}


def main():
    # Load whichever runs are available
    runs = []
    for label, path in RUNS:
        if path.exists():
            payload = json.loads(path.read_text())
            runs.append({
                "label": label,
                "path": str(path),
                "rows": payload["results"],
                "metrics": confusion(payload["results"]),
                "ts": payload.get("timestamp", "?"),
            })
            print(f"  loaded {label}: {len(payload['results'])} patients")
        else:
            print(f"  (skip — not found: {path.name})")

    if len(runs) < 2:
        print("\nNeed at least 2 runs to compare.")
        return

    # Build side-by-side metrics
    md = ["# eICU v4 Cohort — Pipeline-iteration comparison",
          "",
          f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}_",
          "",
          "All three runs target the **same 150-patient v4 cohort** "
          "(34 sepsis / 116 controls). Differences are entirely due to "
          "pipeline changes, not data changes.",
          "",
          "## 1. Headline metrics",
          "",
          "| Metric | " + " | ".join(r["label"] for r in runs) + " |",
          "|---|" + "|".join(":--:" for _ in runs) + "|",
    ]
    for metric_key, metric_label in [
        ("sensitivity",      "Sensitivity"),
        ("specificity",      "Specificity"),
        ("ppv",              "PPV"),
        ("npv",              "NPV"),
        ("f1",               "F1 score"),
        ("false_alarm_rate", "False-alarm rate"),
    ]:
        cells = []
        for r in runs:
            v = r["metrics"][metric_key]
            if isinstance(v, tuple):
                cells.append(fmt_ci(v))
            else:
                cells.append(f"{v:.3f}" if metric_key == "f1" else f"{100*v:.1f}%")
        md.append(f"| {metric_label} | " + " | ".join(cells) + " |")

    md.append("")
    md.append("## 2. Confusion matrix")
    md.append("")
    md.append("| | " + " | ".join(r["label"] for r in runs) + " |")
    md.append("|---|" + "|".join(":--:" for _ in runs) + "|")
    for k, lbl in [("tp", "True positives"), ("fn", "False negatives"),
                   ("fp", "False positives"), ("tn", "True negatives")]:
        md.append(f"| {lbl} | " + " | ".join(str(r["metrics"][k]) for r in runs) + " |")

    # Patient-level deltas vs baseline
    if len(runs) >= 2:
        baseline = runs[0]
        md.append("")
        md.append("## 3. Patient-level changes vs baseline")
        md.append("")
        b_idx = index_by_pid(baseline["rows"])

        for r in runs[1:]:
            r_idx = index_by_pid(r["rows"])
            diffs_fp_to_tn = []   # were FP, now TN  (good)
            diffs_tp_to_fn = []   # were TP, now FN  (bad)
            diffs_fn_to_tp = []   # were FN, now TP  (good)
            diffs_tn_to_fp = []   # were TN, now FP  (bad)
            for pid, b in b_idx.items():
                n = r_idx.get(pid)
                if not n: continue
                b_pred = b.get("predicted_sepsis"); n_pred = n.get("predicted_sepsis")
                b_act  = b.get("actual_sepsis");    n_act  = n.get("actual_sepsis")
                if b_pred is True and n_pred is False and not b_act:
                    diffs_fp_to_tn.append((pid, b, n))
                elif b_pred is False and n_pred is True and not b_act:
                    diffs_tn_to_fp.append((pid, b, n))
                elif b_pred is True and n_pred is False and b_act:
                    diffs_tp_to_fn.append((pid, b, n))
                elif b_pred is False and n_pred is True and b_act:
                    diffs_fn_to_tp.append((pid, b, n))

            md.append(f"### {r['label']} vs {baseline['label']}")
            md.append("")
            md.append(f"- **FP → TN (false alarms removed):** {len(diffs_fp_to_tn)}")
            md.append(f"- **TN → FP (new false alarms):** {len(diffs_tn_to_fp)}")
            md.append(f"- **TP → FN (sepsis newly missed):** {len(diffs_tp_to_fn)}")
            md.append(f"- **FN → TP (sepsis newly caught):** {len(diffs_fn_to_tp)}")
            md.append("")

            if diffs_tp_to_fn:
                md.append("**Sepsis cases newly missed (need SME review):**")
                md.append("")
                md.append("| Patient | Baseline | New run | LLM said |")
                md.append("|---|:--:|:--:|---|")
                for pid, b, n in diffs_tp_to_fn:
                    md.append(f"| {pid} | risk={b['risk_score']} {b['priority']} | "
                              f"risk={n['risk_score']} {n['priority']} | "
                              f"{(n.get('reasoning') or '')[:120]}... |")
                md.append("")

            if diffs_fp_to_tn[:5]:
                md.append("**Sample false-alarms removed (top 5):**")
                md.append("")
                for pid, b, n in diffs_fp_to_tn[:5]:
                    md.append(f"- `{pid}`: was `risk={b['risk_score']}/{b['priority']}` → "
                              f"`risk={n['risk_score']}/{n['priority']}`")
                md.append("")

    # Persist
    DOCS.mkdir(exist_ok=True)
    (DOCS / "EICU_VALIDATION_COMPARISON.md").write_text("\n".join(md))
    (RES / "v4_v5_v6_comparison.json").write_text(json.dumps({
        "runs": [{"label": r["label"], "metrics": r["metrics"]} for r in runs],
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }, indent=2, default=str))

    # Console
    print()
    print("=" * 90)
    print(" PIPELINE-ITERATION COMPARISON")
    print("=" * 90)
    print(f"  {'Metric':<12}" + "".join(f"  {r['label']:<25}" for r in runs))
    for k, lbl in [("sensitivity","Sens"),("specificity","Spec"),("ppv","PPV"),("f1","F1")]:
        cells = []
        for r in runs:
            v = r["metrics"][k]
            if isinstance(v, tuple):
                cells.append(f"  {100*v[0]:5.1f}%  ({100*v[1]:.0f}-{100*v[2]:.0f})")
            else:
                cells.append(f"  {v:.3f}")
        print(f"  {lbl:<12}" + "".join(f"{c:<27}" for c in cells))
    print()
    print(f"  Wrote: {DOCS / 'EICU_VALIDATION_COMPARISON.md'}")
    print(f"  Wrote: {RES / 'v4_v5_v6_comparison.json'}")
    print("=" * 90)


if __name__ == "__main__":
    main()
