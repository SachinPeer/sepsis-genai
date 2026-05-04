"""
Patient-level diff between R5 (v4 cohort + prompt v3.2 + truncated preprocessor)
and R7 (v4 cohort + prompt v3.2 + FULL TREND preprocessor).

Same prompt, same cohort. Only difference: the LLM now sees the full 6h trend
as numeric arrays instead of just current + 1-hour-prev.

Question: does feeding the LLM the full trend information change which patients
get flagged?
"""

import json
from pathlib import Path
from collections import Counter

RESULTS = Path(__file__).parent / "results"
R4_PATH = RESULTS / "validation_results_20260430_171308.json"
R5_PATH = RESULTS / "R5_validation_results.json"
R7_PATH = RESULTS / "R7_validation_results.json"


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
    r7 = load(R7_PATH)

    print(f"R4: {len(r4)} | R5: {len(r5)} | R7: {len(r7)}")

    # ---- R5 vs R7 ----
    common = sorted(set(r5) & set(r7))
    print(f"\nR5 vs R7 (common: {len(common)} patients)")

    transitions = Counter()
    transition_lists = {}
    risk_deltas = []
    priority_shifts = Counter()

    for pid in common:
        a = r5[pid]["actual_sepsis"]
        c5 = cell(a, r5[pid]["predicted_sepsis"])
        c7 = cell(a, r7[pid]["predicted_sepsis"])
        key = f"{c5}->{c7}"
        transitions[key] += 1
        transition_lists.setdefault(key, []).append(pid)
        try:
            risk_deltas.append((pid, float(r7[pid]["risk_score"]) - float(r5[pid]["risk_score"])))
        except Exception:
            pass
        priority_shifts[f"{r5[pid]['priority']}->{r7[pid]['priority']}"] += 1

    print("\nTRANSITIONS (R5 -> R7):")
    for k in sorted(transitions, key=lambda x: -transitions[x]):
        print(f"  {k}: {transitions[k]}")

    print("\nKEY MOVEMENTS:")
    for k in ["FN->TP", "TP->FN", "FP->TN", "TN->FP"]:
        ids = transition_lists.get(k, [])
        print(f"  {k}: {len(ids)} patients")
        if ids and len(ids) <= 35:
            print(f"    {ids}")

    if risk_deltas:
        deltas = [d for _, d in risk_deltas]
        avg = sum(deltas) / len(deltas)
        positive = sum(1 for d in deltas if d > 0)
        negative = sum(1 for d in deltas if d < 0)
        zero = sum(1 for d in deltas if d == 0)
        print(f"\nRISK SCORE DELTAS (R7 - R5): avg={avg:+.1f} | down={negative}, up={positive}, same={zero}")

    print("\nPRIORITY SHIFTS (R5 -> R7):")
    for k, v in sorted(priority_shifts.items(), key=lambda x: -x[1])[:12]:
        print(f"  {k}: {v}")

    # ---- R4 -> R5 -> R7 lifecycle ----
    print("\n--- R4 -> R5 -> R7 LIFECYCLE ---")
    paths = Counter()
    for pid in sorted(set(r4) & set(r5) & set(r7)):
        a = r5[pid]["actual_sepsis"]
        c4 = cell(a, r4[pid]["predicted_sepsis"])
        c5 = cell(a, r5[pid]["predicted_sepsis"])
        c7 = cell(a, r7[pid]["predicted_sepsis"])
        paths[f"{c4}->{c5}->{c7}"] += 1

    print("Top paths:")
    for k, v in sorted(paths.items(), key=lambda x: -x[1])[:12]:
        print(f"  {k}: {v}")

    # The interesting ones:
    print("\nR7-RELATED PATHS:")
    print(f"  FP->FP->TN (R7 alone fixed):       {paths.get('FP->FP->TN', 0)}")
    print(f"  FP->TN->FP (R7 broke it):          {paths.get('FP->TN->FP', 0)}")
    print(f"  TP->TP->FN (R7 alone lost):        {paths.get('TP->TP->FN', 0)}")
    print(f"  TP->FN->TP (R7 recovered):         {paths.get('TP->FN->TP', 0)}")
    print(f"  FN->FN->TP (R7 caught new):        {paths.get('FN->FN->TP', 0)}")
    print(f"  FP->FP->FP (still always FP):      {paths.get('FP->FP->FP', 0)}")

    summary = {
        "transitions_r5_r7": dict(transitions),
        "key_movements": {k: transition_lists.get(k, []) for k in ["FN->TP", "TP->FN", "FP->TN", "TN->FP"]},
        "lifecycle_paths": dict(paths),
    }
    out = RESULTS / "r5_r7_comparison.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
