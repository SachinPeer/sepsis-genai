"""
Patient-level diff between R5 (v4 cohort + prompt v3.2) and
R6 (v5 cohort with 4h-prior snapshot + prompt v3.2).

The same prompt is used in both. Only the input data window changes:
  R5: snapshot at PhysioNet onset_hour (= 6h before clinical sepsis)
  R6: snapshot at onset_hour + 2 (= 4h before clinical sepsis)

Question: does seeing the patient 2 hours later (closer to actual sepsis)
recover any of the 19 sepsis cases R5 missed?
"""

import json
from pathlib import Path
from collections import Counter

RESULTS = Path(__file__).parent / "results"
R5_PATH = RESULTS / "R5_validation_results.json"
R6_PATH = RESULTS / "R6_validation_results.json"
R4_PATH = RESULTS / "validation_results_20260430_171308.json"


def load(path):
    data = json.loads(path.read_text())
    return {r["patient_id"]: r for r in data["results"] if r.get("status") == "success"}


def cell(actual, predicted):
    if actual and predicted: return "TP"
    if actual and not predicted: return "FN"
    if (not actual) and predicted: return "FP"
    return "TN"


def main():
    r4 = load(R4_PATH)
    r5 = load(R5_PATH)
    r6 = load(R6_PATH)

    common = sorted(set(r5) & set(r6))
    print(f"R5 patients: {len(r5)} | R6 patients: {len(r6)} | Common: {len(common)}")

    transitions = Counter()
    transition_lists = {}
    risk_deltas = []
    priority_shifts = Counter()

    for pid in common:
        a = r5[pid]["actual_sepsis"]
        r5_class = cell(a, r5[pid]["predicted_sepsis"])
        r6_class = cell(a, r6[pid]["predicted_sepsis"])
        key = f"{r5_class}->{r6_class}"
        transitions[key] += 1
        transition_lists.setdefault(key, []).append(pid)
        try:
            r5_risk = float(r5[pid]["risk_score"])
            r6_risk = float(r6[pid]["risk_score"])
            risk_deltas.append((pid, r6_risk - r5_risk))
        except Exception:
            pass
        priority_shifts[f"{r5[pid]['priority']}->{r6[pid]['priority']}"] += 1

    print("\nTRANSITIONS (R5_class -> R6_class):")
    for k in sorted(transitions, key=lambda x: -transitions[x]):
        print(f"  {k}: {transitions[k]}")

    print("\nKEY MOVEMENTS:")
    for k in ["FN->TP", "TP->FN", "FP->TN", "TN->FP"]:
        ids = transition_lists.get(k, [])
        print(f"  {k}: {len(ids)} patients")
        if ids and len(ids) <= 30:
            print(f"    {ids}")

    if risk_deltas:
        deltas = [d for _, d in risk_deltas]
        avg = sum(deltas) / len(deltas)
        positive = sum(1 for d in deltas if d > 0)
        negative = sum(1 for d in deltas if d < 0)
        zero = sum(1 for d in deltas if d == 0)
        print(f"\nRISK SCORE DELTAS (R6 - R5):")
        print(f"  Avg: {avg:+.1f}  | Down: {negative}, Up: {positive}, Same: {zero}")

    print("\nPRIORITY SHIFTS (R5 -> R6):")
    for k, v in sorted(priority_shifts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {k}: {v}")

    # Cross R4-R5-R6 trajectory: who was caught in R4, lost in R5, recovered in R6?
    print("\nR4 -> R5 -> R6 LIFECYCLE:")
    paths = Counter()
    for pid in sorted(set(r4) & set(r5) & set(r6)):
        a = r4[pid]["actual_sepsis"]
        c4 = cell(a, r4[pid]["predicted_sepsis"])
        c5 = cell(a, r5[pid]["predicted_sepsis"])
        c6 = cell(a, r6[pid]["predicted_sepsis"])
        paths[f"{c4}->{c5}->{c6}"] += 1

    interesting = ["TP->FN->TP", "TP->FN->FN", "FP->TN->TN", "FP->TN->FP", "FN->FN->TP"]
    print("  Most common paths:")
    for k, v in sorted(paths.items(), key=lambda x: -x[1])[:8]:
        marker = "  *" if k in interesting else "   "
        print(f"  {marker}{k}: {v}")

    print(f"\n  RECOVERED (TP->FN->TP):     {paths.get('TP->FN->TP', 0)}")
    print(f"  STILL MISSED (TP->FN->FN):  {paths.get('TP->FN->FN', 0)}")
    print(f"  STILL FIXED (FP->TN->TN):   {paths.get('FP->TN->TN', 0)}")
    print(f"  RE-LOST (FP->TN->FP):       {paths.get('FP->TN->FP', 0)}")

    summary = {
        "transitions": dict(transitions),
        "key_movements": {k: transition_lists.get(k, []) for k in ["FN->TP", "TP->FN", "FP->TN", "TN->FP"]},
        "lifecycle_paths": dict(paths),
    }
    out = RESULTS / "r5_r6_comparison.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
