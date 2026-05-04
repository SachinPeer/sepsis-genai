"""
Threshold sweep analysis on Round 2 results.
Re-classifies all 340 patients at multiple risk_score thresholds (50, 55, 60, ..., 85)
and reports sensitivity / specificity / PPV / NPV at each threshold.
No new API calls — works on existing data.

Two modes evaluated:
  Mode A: Pure LLM risk_score >= threshold (no guardrail / priority / alert)
  Mode B: risk_score >= threshold OR guardrail_override (current production logic)
"""

import json
from pathlib import Path

R2_FILE = Path(__file__).parent / "results" / "validation_results_20260429_175253.json"
OUT_FILE = Path(__file__).parent / "results" / "threshold_sweep.json"

THRESHOLDS = [50, 55, 60, 65, 70, 75, 80, 85]


def metrics(tp, fp, fn, tn):
    sens = tp / (tp + fn) * 100 if (tp + fn) else 0
    spec = tn / (tn + fp) * 100 if (tn + fp) else 0
    ppv = tp / (tp + fp) * 100 if (tp + fp) else 0
    npv = tn / (tn + fn) * 100 if (tn + fn) else 0
    acc = (tp + tn) / (tp + fp + fn + tn) * 100
    f1 = (2 * tp) / (2 * tp + fp + fn) * 100 if (2 * tp + fp + fn) else 0
    far = fp / (fp + tn) * 100 if (fp + tn) else 0
    return {"sens": sens, "spec": spec, "ppv": ppv, "npv": npv, "acc": acc, "f1": f1, "far": far,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def evaluate(results, threshold, mode="A"):
    """
    mode A: risk_score >= threshold
    mode B: risk_score >= threshold OR guardrail_override
    mode C: risk_score >= threshold OR priority in (High, Critical) OR alert in (HIGH, CRITICAL)
    mode D: full original (risk OR guardrail OR priority OR alert)
    """
    tp = fp = fn = tn = 0
    for r in results:
        risk = r.get("risk_score")
        if not isinstance(risk, (int, float)):
            continue
        actual = r["actual_sepsis"]
        priority_high = r.get("priority") in ("High", "Critical")
        alert_high = r.get("alert_level") in ("HIGH", "CRITICAL")
        guard = r.get("guardrail_override", False)

        if mode == "A":
            predicted = risk >= threshold
        elif mode == "B":
            predicted = (risk >= threshold) or guard
        elif mode == "C":
            predicted = (risk >= threshold) or priority_high or alert_high
        elif mode == "D":
            predicted = (risk >= threshold) or priority_high or alert_high or guard
        else:
            predicted = risk >= threshold

        if actual and predicted:
            tp += 1
        elif actual and not predicted:
            fn += 1
        elif not actual and predicted:
            fp += 1
        else:
            tn += 1
    return metrics(tp, fp, fn, tn)


def main():
    with open(R2_FILE) as f:
        d = json.load(f)
    results = d["results"]

    print("=" * 92)
    print(f"  THRESHOLD SWEEP — Round 2 (n={len(results)}, 140 sepsis / 200 non-sepsis)")
    print("=" * 92)

    def run_mode(label, mode_code):
        print(f"\n  MODE {mode_code}: {label}")
        print("  " + "-" * 90)
        print(f"  {'Threshold':<11} {'Sens':>7} {'Spec':>7} {'PPV':>7} {'NPV':>7} {'Acc':>7} {'F1':>7} {'FAR':>7}   TP/FP/FN/TN")
        print("  " + "-" * 90)
        rows = []
        for t in THRESHOLDS:
            m = evaluate(results, t, mode=mode_code)
            rows.append({"threshold": t, **m})
            print(f"  >= {t:<7}  {m['sens']:>6.1f}% {m['spec']:>6.1f}% {m['ppv']:>6.1f}% {m['npv']:>6.1f}% {m['acc']:>6.1f}% {m['f1']:>6.1f}% {m['far']:>6.1f}%   {m['tp']}/{m['fp']}/{m['fn']}/{m['tn']}")
        return rows

    mode_a = run_mode("Pure LLM risk_score >= threshold (no guardrail / priority / alert)", "A")
    mode_b = run_mode("risk_score >= threshold OR guardrail_override", "B")
    mode_c = run_mode("risk_score >= threshold OR priority(High/Critical) OR alert(HIGH/CRITICAL)", "C")
    mode_d = run_mode("FULL ORIGINAL: risk OR guardrail OR priority OR alert", "D")

    print("\n" + "=" * 92)
    print("  RECOMMENDED OPERATING POINTS")
    print("=" * 92)
    for label, mode in [("MODE A (LLM risk only)", mode_a),
                         ("MODE B (LLM risk + Guardrail)", mode_b),
                         ("MODE C (LLM risk + Priority/Alert)", mode_c),
                         ("MODE D (FULL — current production)", mode_d)]:
        best_f1 = max(mode, key=lambda x: x["f1"])
        viable85 = [x for x in mode if x["sens"] >= 85]
        best_sens85 = max(viable85, key=lambda x: x["spec"]) if viable85 else None
        viable90 = [x for x in mode if x["sens"] >= 90]
        best_sens90 = max(viable90, key=lambda x: x["spec"]) if viable90 else None

        print(f"\n  {label}")
        print(f"    Best F1               : threshold={best_f1['threshold']} | Sens={best_f1['sens']:.1f}% Spec={best_f1['spec']:.1f}% F1={best_f1['f1']:.1f}%")
        if best_sens85:
            print(f"    Best Spec @ Sens>=85% : threshold={best_sens85['threshold']} | Sens={best_sens85['sens']:.1f}% Spec={best_sens85['spec']:.1f}%")
        else:
            print(f"    Best Spec @ Sens>=85% : N/A")
        if best_sens90:
            print(f"    Best Spec @ Sens>=90% : threshold={best_sens90['threshold']} | Sens={best_sens90['sens']:.1f}% Spec={best_sens90['spec']:.1f}%")
        else:
            print(f"    Best Spec @ Sens>=90% : N/A (no threshold reaches 90% sensitivity)")

    with open(OUT_FILE, "w") as f:
        json.dump({
            "mode_a_llm_risk_only": mode_a,
            "mode_b_llm_risk_plus_guardrail": mode_b,
            "mode_c_llm_risk_plus_priority": mode_c,
            "mode_d_full_original": mode_d,
        }, f, indent=2)
    print(f"\n  Saved: {OUT_FILE}")


if __name__ == "__main__":
    main()
