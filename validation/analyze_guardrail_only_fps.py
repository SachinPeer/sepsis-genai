"""
Deep clinical justifiability analysis of the 12 guardrail-only FPs.
For each case, checks:
  1. Which guardrail rule(s) fired
  2. Vital values at the snapshot (was the rule's number actually critical?)
  3. PhysioNet trajectory: did patient deteriorate or develop sepsis later?
  4. Clinical justifiability of the override
"""

import json
import csv
from pathlib import Path

ROOT = Path(__file__).parent.parent
RECHECK = Path(__file__).parent / "results" / "guardrail_recheck.json"
COHORT_V2 = Path(__file__).parent / "selected_cohort_v2"
RAW_DIR = Path(__file__).parent / "raw_data"
OUT_FILE = Path(__file__).parent / "results" / "guardrail_only_fp_clinical_review.json"


def parse_psv(filepath):
    rows = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            rows.append(row)
    return rows


def safe_float(v):
    if v is None or str(v).strip() == "" or str(v).strip() == "NaN":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def analyze_trajectory(patient_id, snapshot_idx_from_cohort):
    """Look at the full PhysioNet trajectory for this patient AFTER the snapshot point."""
    psv_path = RAW_DIR / f"{patient_id}.psv"
    if not psv_path.exists():
        return None

    rows = parse_psv(psv_path)
    total_hours = len(rows)

    sepsis_label_seen = False
    sepsis_onset = None
    for i, r in enumerate(rows):
        if r.get("SepsisLabel", "0").strip() == "1":
            sepsis_label_seen = True
            sepsis_onset = i
            break

    # Look at vitals after the snapshot point
    after_snapshot = rows[snapshot_idx_from_cohort:] if snapshot_idx_from_cohort < total_hours else []

    deterioration_signals = []
    if len(after_snapshot) >= 6:
        # check next 24 hours
        future_window = after_snapshot[:24]

        sbp_min = min((safe_float(r.get("SBP")) for r in future_window if safe_float(r.get("SBP")) is not None), default=None)
        map_min = min((safe_float(r.get("MAP")) for r in future_window if safe_float(r.get("MAP")) is not None), default=None)
        lac_max = max((safe_float(r.get("Lactate")) for r in future_window if safe_float(r.get("Lactate")) is not None), default=None)
        cr_max = max((safe_float(r.get("Creatinine")) for r in future_window if safe_float(r.get("Creatinine")) is not None), default=None)
        wbc_max = max((safe_float(r.get("WBC")) for r in future_window if safe_float(r.get("WBC")) is not None), default=None)
        temp_max = max((safe_float(r.get("Temp")) for r in future_window if safe_float(r.get("Temp")) is not None), default=None)
        hr_max = max((safe_float(r.get("HR")) for r in future_window if safe_float(r.get("HR")) is not None), default=None)

        if sbp_min and sbp_min < 90:
            deterioration_signals.append(f"SBP fell to {sbp_min:.0f} (from <=90 threshold)")
        if map_min and map_min < 65:
            deterioration_signals.append(f"MAP fell to {map_min:.0f} (<65)")
        if lac_max and lac_max > 4:
            deterioration_signals.append(f"Lactate rose to {lac_max:.1f} (>4)")
        elif lac_max and lac_max > 2:
            deterioration_signals.append(f"Lactate rose to {lac_max:.1f} (>2)")
        if cr_max and cr_max > 2:
            deterioration_signals.append(f"Creatinine rose to {cr_max:.1f}")
        if wbc_max and wbc_max > 15:
            deterioration_signals.append(f"WBC rose to {wbc_max:.1f}")
        if temp_max and temp_max > 38.5:
            deterioration_signals.append(f"Fever to {temp_max:.1f}C")
        if hr_max and hr_max > 130:
            deterioration_signals.append(f"HR rose to {hr_max:.0f}")

    return {
        "total_hours_in_icu": total_hours,
        "physionet_sepsis_label": sepsis_label_seen,
        "physionet_sepsis_onset_hour": sepsis_onset,
        "snapshot_hour": snapshot_idx_from_cohort,
        "hours_after_snapshot": total_hours - snapshot_idx_from_cohort,
        "deterioration_in_next_24h": deterioration_signals,
    }


def assess_justifiability(case):
    """Clinical safety assessment of the override."""
    reasons = case["override_reasons"]
    trajectory = case.get("trajectory", {})
    deterioration = trajectory.get("deterioration_in_next_24h", []) if trajectory else []

    # Categorize the rule(s) that fired
    rule_categories = []
    for r in reasons:
        r_lower = r.lower()
        if "hypotension" in r_lower or "sbp" in r_lower or "map" in r_lower or "dbp" in r_lower:
            rule_categories.append("hemodynamic")
        if "lactate" in r_lower:
            rule_categories.append("perfusion")
        if "tachypnea" in r_lower or "rr" in r_lower:
            rule_categories.append("respiratory")
        if "aki" in r_lower or "creatinine" in r_lower or "bun" in r_lower:
            rule_categories.append("renal")
        if "hypoglycemia" in r_lower or "glucose" in r_lower:
            rule_categories.append("metabolic")
        if "anemia" in r_lower or "hgb" in r_lower:
            rule_categories.append("hematologic")
        if "hypercapnia" in r_lower or "paco2" in r_lower:
            rule_categories.append("respiratory")
        if "bicarbonate" in r_lower or "hco3" in r_lower:
            rule_categories.append("metabolic")
        if "septic shock" in r_lower:
            rule_categories.append("septic-shock-criteria")

    rule_categories = list(set(rule_categories))

    # Rule single-trigger analysis
    single_rule = len(reasons) == 1
    if "septic-shock-criteria" in rule_categories:
        category = "STRONG: Septic shock criteria explicitly met"
    elif len(rule_categories) >= 2:
        category = f"MULTI-SYSTEM: {len(rule_categories)} organ systems flagged"
    elif single_rule:
        category = f"SINGLE-TRIGGER ({rule_categories[0] if rule_categories else 'unknown'}): may overfire"
    else:
        category = "MIXED"

    # Did the patient actually deteriorate?
    if deterioration:
        outcome = f"DETERIORATED: {len(deterioration)} concerning signals in next 24h"
    else:
        outcome = "STABLE: No deterioration in next 24h"

    # Final justifiability
    if "septic-shock-criteria" in rule_categories:
        justifiable = "JUSTIFIED — septic shock criteria are textbook sepsis indicators"
    elif len(rule_categories) >= 2 and len(deterioration) >= 2:
        justifiable = "JUSTIFIED — multi-system + actual deterioration"
    elif len(rule_categories) >= 2:
        justifiable = "DEFENSIBLE — multi-system flag, even if patient stable"
    elif single_rule and len(deterioration) >= 1:
        justifiable = "DEFENSIBLE — single trigger but patient did deteriorate"
    elif single_rule and not deterioration:
        justifiable = "QUESTIONABLE — single borderline trigger, no deterioration"
    else:
        justifiable = "MIXED"

    return {
        "rule_categories": rule_categories,
        "rule_strength": category,
        "patient_outcome": outcome,
        "justifiability": justifiable,
    }


def main():
    with open(RECHECK) as f:
        recheck = json.load(f)

    guardrail_only = [r for r in recheck if not r["would_still_be_fp_without_guardrail"]]
    print(f"Analyzing {len(guardrail_only)} guardrail-only FPs (LLM said <50 risk, guardrail forced 95)\n")

    cases = []
    for r in guardrail_only:
        pid = r["patient_id"]
        # Get snapshot hour from cohort manifest
        with open(COHORT_V2 / "cohort_manifest.json") as f:
            manifest = json.load(f)
        patient_meta = next((p for p in manifest["patients"] if p["patient_id"] == pid), None)
        snapshot_hour = patient_meta.get("snapshot_hour", 0) if patient_meta else 0

        trajectory = analyze_trajectory(pid, snapshot_hour)
        case = {**r, "trajectory": trajectory}
        case["assessment"] = assess_justifiability(case)
        cases.append(case)

    # Print report
    print("=" * 80)
    print("  CLINICAL JUSTIFIABILITY REVIEW — 12 Guardrail-Only False Positives")
    print("=" * 80)

    for i, c in enumerate(cases, 1):
        a = c["assessment"]
        t = c.get("trajectory") or {}
        print(f"\n  Case {i}/12: {c['patient_id']} | LLM risk: {c['original_llm_risk']} → forced 95")
        print(f"    Override reason(s): {c['override_reasons']}")
        print(f"    Rule strength:      {a['rule_strength']}")
        print(f"    PhysioNet label:    sepsis={t.get('physionet_sepsis_label')} (onset hr {t.get('physionet_sepsis_onset_hour')})")
        print(f"    Snapshot at hr:     {t.get('snapshot_hour')} of {t.get('total_hours_in_icu')} total")
        if t.get("deterioration_in_next_24h"):
            print(f"    Next 24h:           {'; '.join(t['deterioration_in_next_24h'])}")
        else:
            print(f"    Next 24h:           No major deterioration")
        print(f"    Patient outcome:    {a['patient_outcome']}")
        print(f"    JUSTIFIABILITY:     {a['justifiability']}")

    # Summary
    print(f"\n{'=' * 80}")
    print("  SUMMARY")
    print("=" * 80)
    from collections import Counter
    j_counts = Counter(c["assessment"]["justifiability"].split(" — ")[0] for c in cases)
    for k, v in j_counts.most_common():
        print(f"  {k:<20} {v} cases")

    # Cases that actually deteriorated despite PhysioNet "no sepsis" label
    deteriorated = [c for c in cases if c["assessment"]["patient_outcome"].startswith("DETERIORATED")]
    print(f"\n  → {len(deteriorated)}/{len(cases)} 'false alarms' actually showed concerning deterioration in next 24h")
    print(f"  → These may be TRUE early warnings that PhysioNet labeling missed")

    with open(OUT_FILE, "w") as f:
        json.dump(cases, f, indent=2)
    print(f"\n  Full review saved: {OUT_FILE}")


if __name__ == "__main__":
    main()
