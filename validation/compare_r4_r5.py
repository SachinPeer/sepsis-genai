"""
Patient-level diff between R4 (corrected v4 cohort + old prompt) and
R5 (same cohort + new prompt v3.2).

Outputs:
  - Counts by transition class (TP->FN, FN->TP, FP->TN, TN->FP, etc.)
  - Lists of patient IDs in each class
  - Average risk-score delta and priority shifts
"""

import json
from pathlib import Path
from collections import Counter

RESULTS = Path(__file__).parent / "results"

R4_PATH = RESULTS / "validation_results_20260430_171308.json"  # baseline (post-bugfix)
R5_PATH = RESULTS / "R5_validation_results.json"


def load(path):
    data = json.loads(path.read_text())
    return {r["patient_id"]: r for r in data["results"] if r.get("status") == "success"}


def cell(actual, predicted):
    if actual and predicted:
        return "TP"
    if actual and not predicted:
        return "FN"
    if (not actual) and predicted:
        return "FP"
    return "TN"


def main():
    r4 = load(R4_PATH)
    r5 = load(R5_PATH)

    common = sorted(set(r4) & set(r5))
    print(f"R4 patients: {len(r4)} | R5 patients: {len(r5)} | Common: {len(common)}")

    transitions = Counter()
    transition_lists = {}
    risk_deltas = []
    priority_shifts = Counter()

    for pid in common:
        a = r4[pid]["actual_sepsis"]
        r4_class = cell(a, r4[pid]["predicted_sepsis"])
        r5_class = cell(a, r5[pid]["predicted_sepsis"])
        key = f"{r4_class}->{r5_class}"
        transitions[key] += 1
        transition_lists.setdefault(key, []).append(pid)
        try:
            r4_risk = float(r4[pid]["risk_score"])
            r5_risk = float(r5[pid]["risk_score"])
            risk_deltas.append((pid, r5_risk - r4_risk))
        except Exception:
            pass
        priority_shifts[f"{r4[pid]['priority']}->{r5[pid]['priority']}"] += 1

    print("\nTRANSITIONS (R4_class -> R5_class):")
    for k in sorted(transitions, key=lambda x: -transitions[x]):
        print(f"  {k}: {transitions[k]}")

    print("\nKEY MOVEMENTS (the ones that matter):")
    key_movements = ["TP->FN", "FN->TP", "FP->TN", "TN->FP"]
    for k in key_movements:
        ids = transition_lists.get(k, [])
        print(f"  {k}: {len(ids)} patients")
        if ids and len(ids) <= 25:
            print(f"    {ids}")

    print("\nRISK SCORE DELTAS (R5 - R4):")
    if risk_deltas:
        deltas = [d for _, d in risk_deltas]
        avg = sum(deltas) / len(deltas)
        positive = sum(1 for d in deltas if d > 0)
        negative = sum(1 for d in deltas if d < 0)
        zero = sum(1 for d in deltas if d == 0)
        print(f"  Avg delta: {avg:+.1f}")
        print(f"  Patients with risk down: {negative}, up: {positive}, same: {zero}")

    print("\nPRIORITY SHIFTS (R4 -> R5):")
    for k, v in sorted(priority_shifts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {k}: {v}")

    summary = {
        "transitions": dict(transitions),
        "key_movements": {k: transition_lists.get(k, []) for k in key_movements},
        "priority_shifts": dict(priority_shifts),
        "risk_avg_delta": sum(d for _, d in risk_deltas) / len(risk_deltas) if risk_deltas else 0,
    }
    out = RESULTS / "r4_r5_comparison.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
