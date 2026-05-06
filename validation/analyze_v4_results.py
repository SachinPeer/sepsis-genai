"""
Analyse the v4 validation run.

Reads `validation/results/EICU_results_v4_latest.json`, computes the full
suite of binary-classifier metrics with 95% Wilson confidence intervals,
breaks them down by guardrail behaviour and alert level, and writes:

  - validation/results/EICU_v4_metrics.json    # machine-readable
  - validation/docs/EICU_VALIDATION_EXECUTION.md  # appended human report

Usage:
  python3 validation/analyze_v4_results.py
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
DOCS_DIR = Path(__file__).parent / "docs"
LATEST_JSON = RESULTS_DIR / "EICU_results_v4_latest.json"
OUT_METRICS = RESULTS_DIR / "EICU_v4_metrics.json"
OUT_REPORT = DOCS_DIR / "EICU_VALIDATION_EXECUTION.md"


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval — appropriate for proportions at small n."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (p, max(0.0, centre - half), min(1.0, centre + half))


def fmt_pct(p: float) -> str:
    return f"{100 * p:5.1f}%"


def fmt_ci(p: float, lo: float, hi: float) -> str:
    return f"{100 * p:5.1f}% (95% CI: {100 * lo:.1f} – {100 * hi:.1f})"


def confusion(rows: list[dict]) -> dict:
    tp = sum(1 for r in rows if r["actual_sepsis"] and r["predicted_sepsis"] is True)
    fn = sum(1 for r in rows if r["actual_sepsis"] and r["predicted_sepsis"] is False)
    fp = sum(1 for r in rows if not r["actual_sepsis"] and r["predicted_sepsis"] is True)
    tn = sum(1 for r in rows if not r["actual_sepsis"] and r["predicted_sepsis"] is False)
    n_pos = tp + fn
    n_neg = fp + tn
    n = n_pos + n_neg

    sens = wilson_ci(tp, n_pos)
    spec = wilson_ci(tn, n_neg)
    ppv  = wilson_ci(tp, tp + fp) if (tp + fp) else (0.0, 0.0, 0.0)
    npv  = wilson_ci(tn, tn + fn) if (tn + fn) else (0.0, 0.0, 0.0)
    acc  = wilson_ci(tp + tn, n)

    if (tp + fp + fn) > 0:
        f1 = 2 * tp / (2 * tp + fp + fn)
    else:
        f1 = 0.0

    far = fp / n_neg if n_neg else 0.0   # false-alarm rate (1-spec)

    return {
        "n": n, "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "n_positive": n_pos, "n_negative": n_neg,
        "sensitivity":  {"point": sens[0], "ci_low": sens[1], "ci_high": sens[2]},
        "specificity":  {"point": spec[0], "ci_low": spec[1], "ci_high": spec[2]},
        "ppv":          {"point": ppv[0],  "ci_low": ppv[1],  "ci_high": ppv[2]},
        "npv":          {"point": npv[0],  "ci_low": npv[1],  "ci_high": npv[2]},
        "accuracy":     {"point": acc[0],  "ci_low": acc[1],  "ci_high": acc[2]},
        "f1": f1,
        "false_alarm_rate": far,
    }


def filter_successful(rows: list[dict]) -> list[dict]:
    """Drop rows whose API call errored out (no risk score)."""
    return [r for r in rows if r.get("status") == "success"
            and r["predicted_sepsis"] in (True, False)]


def metric_row_md(label: str, m: dict) -> str:
    return f"| {label} | {fmt_ci(m['point'], m['ci_low'], m['ci_high'])} |"


def main():
    if not LATEST_JSON.exists():
        raise SystemExit(f"Missing results: {LATEST_JSON}")

    payload = json.loads(LATEST_JSON.read_text())
    rows_all = payload["results"]
    rows = filter_successful(rows_all)
    err_rows = [r for r in rows_all if r not in rows]

    overall = confusion(rows)

    # ---- Subgroup: empty notes (soft flag F10)
    empty_notes_pids = set()
    cohort_manifest = json.loads(
        (Path(__file__).parent / "eicu_cohort_v4" / "manifest.json").read_text()
    )
    for p in cohort_manifest["patients"]:
        if p.get("notes_empty"):
            empty_notes_pids.add(p["patient_id"])
    rows_with_notes    = [r for r in rows if r["patient_id"] not in empty_notes_pids]
    rows_no_notes      = [r for r in rows if r["patient_id"] in empty_notes_pids]
    notes_split = {
        "with_notes":  confusion(rows_with_notes) if rows_with_notes else None,
        "empty_notes": confusion(rows_no_notes)   if rows_no_notes   else None,
    }

    # ---- Guardrail behaviour summary
    n_overrides = sum(1 for r in rows if r.get("guardrail_override") is True)
    overrides_by_class = {
        "sepsis":  sum(1 for r in rows if r.get("guardrail_override") is True and r["actual_sepsis"]),
        "control": sum(1 for r in rows if r.get("guardrail_override") is True and not r["actual_sepsis"]),
    }

    # ---- Alert-level distribution
    alerts = {"CRITICAL": 0, "HIGH": 0, "STANDARD": 0, "LOW": 0, "OTHER": 0}
    alerts_in_sepsis = {"CRITICAL": 0, "HIGH": 0, "STANDARD": 0, "LOW": 0, "OTHER": 0}
    for r in rows:
        a = (r.get("alert_level") or "OTHER").upper()
        a = a if a in alerts else "OTHER"
        alerts[a] += 1
        if r["actual_sepsis"]:
            alerts_in_sepsis[a] += 1

    # ---- Latency
    times = [r.get("processing_time_ms", 0) for r in rows]
    times = [t for t in times if isinstance(t, (int, float)) and t > 0]
    latency = {
        "n": len(times),
        "mean_ms":   round(sum(times) / len(times), 1) if times else 0,
        "median_ms": round(sorted(times)[len(times)//2], 1) if times else 0,
        "p95_ms":    round(sorted(times)[int(0.95 * len(times))], 1) if times else 0,
        "max_ms":    round(max(times), 1) if times else 0,
    }

    # ---- Persist machine-readable metrics
    out = {
        "cohort": "eICU-CRD Demo v2.0.1 — cohort_v4 (Option B, ICD-only)",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "n_total_attempted": len(rows_all),
        "n_successful": len(rows),
        "n_errors": len(err_rows),
        "overall": overall,
        "subgroups": {
            "with_notes":  notes_split["with_notes"],
            "empty_notes": notes_split["empty_notes"],
        },
        "guardrail": {
            "overrides_total": n_overrides,
            "overrides_in_sepsis":  overrides_by_class["sepsis"],
            "overrides_in_control": overrides_by_class["control"],
        },
        "alert_level_distribution": alerts,
        "alert_level_in_sepsis":     alerts_in_sepsis,
        "latency": latency,
    }
    OUT_METRICS.write_text(json.dumps(out, indent=2))

    # ---- Build markdown report
    md = []
    md.append("# eICU v4 Validation — Run Report")
    md.append("")
    md.append(f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_")
    md.append("")
    md.append("## 1. Cohort & run summary")
    md.append("")
    md.append(f"- **Cohort:** eICU-CRD Demo v2.0.1, cohort_v4 (Option B, ICD-only)")
    md.append(f"- **Patients attempted:** {len(rows_all)}")
    md.append(f"- **Successful classifications:** {len(rows)}")
    md.append(f"- **API errors:** {len(err_rows)}")
    md.append(f"- **Sepsis (positives):** {overall['n_positive']}")
    md.append(f"- **Controls (negatives):** {overall['n_negative']}")
    md.append("")
    md.append("## 2. Headline metrics (overall)")
    md.append("")
    md.append("| Metric | Value (95 % Wilson CI) |")
    md.append("|---|---|")
    md.append(metric_row_md("Sensitivity (Recall)", overall["sensitivity"]))
    md.append(metric_row_md("Specificity",          overall["specificity"]))
    md.append(metric_row_md("PPV (Precision)",      overall["ppv"]))
    md.append(metric_row_md("NPV",                  overall["npv"]))
    md.append(metric_row_md("Accuracy",             overall["accuracy"]))
    md.append(f"| F1 Score | {overall['f1']:.3f} |")
    md.append(f"| False-alarm rate (1 − Spec) | {fmt_pct(overall['false_alarm_rate'])} |")
    md.append("")
    md.append("### Confusion matrix")
    md.append("")
    md.append("| | Predicted: Sepsis | Predicted: No sepsis |")
    md.append("|---|:--:|:--:|")
    md.append(f"| **Actual: Sepsis** | TP = {overall['tp']} | FN = {overall['fn']} |")
    md.append(f"| **Actual: Control** | FP = {overall['fp']} | TN = {overall['tn']} |")
    md.append("")
    md.append("## 3. Sub-group: with vs without nurse notes")
    md.append("")
    if notes_split["with_notes"] and notes_split["empty_notes"]:
        wn = notes_split["with_notes"]; en = notes_split["empty_notes"]
        md.append("| Sub-group | n | Sensitivity | Specificity | F1 |")
        md.append("|---|:--:|:--:|:--:|:--:|")
        md.append(f"| With notes  | {wn['n']} | {fmt_ci(*[wn['sensitivity'][k] for k in ('point','ci_low','ci_high')])} | "
                  f"{fmt_ci(*[wn['specificity'][k] for k in ('point','ci_low','ci_high')])} | {wn['f1']:.3f} |")
        md.append(f"| Empty notes | {en['n']} | {fmt_ci(*[en['sensitivity'][k] for k in ('point','ci_low','ci_high')])} | "
                  f"{fmt_ci(*[en['specificity'][k] for k in ('point','ci_low','ci_high')])} | {en['f1']:.3f} |")
    md.append("")
    md.append("## 4. Guardrail behaviour")
    md.append("")
    md.append(f"- Total guardrail overrides triggered: **{n_overrides}** of {len(rows)}")
    md.append(f"  - Within sepsis cases: {overrides_by_class['sepsis']} of {overall['n_positive']}")
    md.append(f"  - Within controls (false alarms): {overrides_by_class['control']} of {overall['n_negative']}")
    md.append("")
    md.append("## 5. Alert-level distribution")
    md.append("")
    md.append("| Alert level | All patients | of which sepsis |")
    md.append("|---|:--:|:--:|")
    for k in ("CRITICAL", "HIGH", "STANDARD", "LOW", "OTHER"):
        md.append(f"| {k} | {alerts[k]} | {alerts_in_sepsis[k]} |")
    md.append("")
    md.append("## 6. Latency")
    md.append("")
    md.append(f"- Mean : {latency['mean_ms']:,} ms")
    md.append(f"- Median: {latency['median_ms']:,} ms")
    md.append(f"- p95  : {latency['p95_ms']:,} ms")
    md.append(f"- Max  : {latency['max_ms']:,} ms")
    md.append("")
    md.append("## 7. Errors")
    md.append("")
    if err_rows:
        for r in err_rows[:10]:
            md.append(f"- `{r['patient_id']}`: {r.get('error', 'unknown')}")
        if len(err_rows) > 10:
            md.append(f"- ... and {len(err_rows) - 10} more")
    else:
        md.append("- None — all classifications succeeded.")
    md.append("")
    md.append("---")
    md.append(f"_Source data: `validation/results/EICU_results_v4_latest.json`_")
    md.append("")

    DOCS_DIR.mkdir(exist_ok=True)
    OUT_REPORT.write_text("\n".join(md))

    # ---- Console summary
    print("=" * 70)
    print(" v4 VALIDATION METRICS (Wilson 95% CIs)")
    print("=" * 70)
    print(f"  n successful : {len(rows)}  (errors {len(err_rows)})")
    print(f"  Sepsis       : {overall['n_positive']}    Controls: {overall['n_negative']}")
    print(f"  TP/FN/FP/TN  : {overall['tp']}/{overall['fn']}/{overall['fp']}/{overall['tn']}")
    print()
    s = overall["sensitivity"]; print(f"  Sensitivity  : {fmt_ci(s['point'], s['ci_low'], s['ci_high'])}")
    s = overall["specificity"]; print(f"  Specificity  : {fmt_ci(s['point'], s['ci_low'], s['ci_high'])}")
    s = overall["ppv"];         print(f"  PPV          : {fmt_ci(s['point'], s['ci_low'], s['ci_high'])}")
    s = overall["npv"];         print(f"  NPV          : {fmt_ci(s['point'], s['ci_low'], s['ci_high'])}")
    s = overall["accuracy"];    print(f"  Accuracy     : {fmt_ci(s['point'], s['ci_low'], s['ci_high'])}")
    print(f"  F1           : {overall['f1']:.3f}")
    print(f"  False alarm  : {fmt_pct(overall['false_alarm_rate'])}")
    print()
    print(f"  Wrote: {OUT_METRICS}")
    print(f"  Wrote: {OUT_REPORT}")
    print("=" * 70)


if __name__ == "__main__":
    main()
