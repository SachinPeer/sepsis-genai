"""
Re-classify R4 results at threshold = 70 (drop the LLM moderate-risk noise bucket).
No new API calls — instant analysis on existing R4 data.
"""

import json
from pathlib import Path

R4_FILE = Path(__file__).parent / "results" / "validation_results_20260430_171308.json"


def metrics(tp, fp, fn, tn):
    sens = tp / (tp + fn) * 100 if (tp + fn) else 0
    spec = tn / (tn + fp) * 100 if (tn + fp) else 0
    ppv = tp / (tp + fp) * 100 if (tp + fp) else 0
    npv = tn / (tn + fn) * 100 if (tn + fn) else 0
    acc = (tp + tn) / (tp + fp + fn + tn) * 100
    f1 = (2 * tp) / (2 * tp + fp + fn) * 100 if (2 * tp + fp + fn) else 0
    return sens, spec, ppv, npv, acc, f1


def classify_at_threshold(results, threshold, include_priority=True, include_guardrail=True):
    tp = fp = fn = tn = 0
    for r in results:
        risk = r.get("risk_score")
        if not isinstance(risk, (int, float)):
            continue
        actual = r["actual_sepsis"]
        priority_high = r.get("priority") in ("High", "Critical") if include_priority else False
        guard = r.get("guardrail_override", False) if include_guardrail else False
        predicted = (risk >= threshold) or priority_high or guard

        if actual and predicted: tp += 1
        elif actual and not predicted: fn += 1
        elif not actual and predicted: fp += 1
        else: tn += 1
    return tp, fp, fn, tn


def main():
    with open(R4_FILE) as f:
        d = json.load(f)
    results = d["results"]

    print("=" * 78)
    print("  RECLASSIFICATION SWEEP ON R4 (corrected baseline)")
    print("=" * 78)
    print()
    print(f"  {'Strategy':<55} {'Sens':>7} {'Spec':>7} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}")
    print("  " + "-" * 80)

    strategies = [
        ("Current production: risk>=50 OR priority OR guardrail", 50, True, True),
        ("Threshold raised to 60 + priority + guardrail", 60, True, True),
        ("Threshold raised to 65 + priority + guardrail", 65, True, True),
        ("Threshold raised to 70 + priority + guardrail", 70, True, True),
        ("Threshold 70 + GUARDRAIL only (drop priority)", 70, False, True),
        ("Threshold 70 + PRIORITY only (drop guardrail)", 70, True, False),
        ("Threshold 70 ALONE (no priority, no guardrail)", 70, False, False),
        ("Threshold 75 + priority + guardrail", 75, True, True),
    ]

    rows = []
    for label, t, inc_p, inc_g in strategies:
        tp, fp, fn, tn = classify_at_threshold(results, t, inc_p, inc_g)
        sens, spec, ppv, npv, acc, f1 = metrics(tp, fp, fn, tn)
        print(f"  {label:<55} {sens:>6.1f}% {spec:>6.1f}% {tp:>4} {fp:>4} {fn:>4} {tn:>4}")
        rows.append({"label": label, "threshold": t, "include_priority": inc_p, "include_guardrail": inc_g,
                     "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                     "sens": sens, "spec": spec, "ppv": ppv, "npv": npv, "acc": acc, "f1": f1})

    print()
    print("=" * 78)
    print("  RECOMMENDATION")
    print("=" * 78)

    # find baseline
    baseline = rows[0]
    # the threshold 70 + priority + guardrail
    rec = next((r for r in rows if r["threshold"] == 70 and r["include_priority"] and r["include_guardrail"]), None)
    if rec:
        d_sens = rec["sens"] - baseline["sens"]
        d_spec = rec["spec"] - baseline["spec"]
        print(f"\n  Threshold raise 50 -> 70 (keeping priority + guardrail):")
        print(f"    Sensitivity change: {d_sens:+.2f}% ({baseline['sens']:.1f}% -> {rec['sens']:.1f}%)")
        print(f"    Specificity change: {d_spec:+.2f}% ({baseline['spec']:.1f}% -> {rec['spec']:.1f}%)")
        print(f"    TP change: {rec['tp'] - baseline['tp']} | FP change: {rec['fp'] - baseline['fp']}")

    out = Path(__file__).parent / "results" / "threshold_70_reclassification.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
