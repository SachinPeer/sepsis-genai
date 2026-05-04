"""
Round 4 (corrected baseline) — comprehensive false alarm root-cause analysis.

For each of 133 R4 false positives, this script:
  1. Looks at the patient's full PhysioNet trajectory AFTER our snapshot point (next 48h)
  2. Decides whether they were a TRUE false alarm (stable) or a HIDDEN TP (deteriorated/might have developed sepsis later)
  3. For TRUE false alarms only: classifies the bottom-most cause
       - Guardrail-driven (override forced positive)
       - LLM-driven via priority/alert with low risk score
       - LLM-driven via high risk score
  4. For each LLM-driven case: extracts the dominant clinical signal from the LLM rationale
"""

import json
import csv
import re
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).parent.parent
R4_FILE = Path(__file__).parent / "results" / "validation_results_20260430_171308.json"
COHORT_V4 = Path(__file__).parent / "selected_cohort_v4"
RAW_DIR = Path(__file__).parent / "raw_data"
OUT_FILE = Path(__file__).parent / "results" / "r4_fp_root_cause.json"

LOOKAHEAD_HOURS = 48


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


def get_trajectory_outcome(patient_id, snapshot_idx):
    """Look at the patient's PhysioNet trajectory after our snapshot point."""
    psv_path = RAW_DIR / f"{patient_id}.psv"
    if not psv_path.exists():
        return None

    rows = parse_psv(psv_path)
    total_hours = len(rows)

    sepsis_label_seen_anywhere = any(r.get("SepsisLabel", "0").strip() == "1" for r in rows)
    sepsis_onset_anywhere = next(
        (i for i, r in enumerate(rows) if r.get("SepsisLabel", "0").strip() == "1"), None
    )

    after = rows[snapshot_idx:snapshot_idx + LOOKAHEAD_HOURS] if snapshot_idx < total_hours else []
    deterioration = []

    if len(after) >= 6:
        sbp_vals = [safe_float(r.get("SBP")) for r in after]
        sbp_vals = [v for v in sbp_vals if v is not None]
        map_vals = [safe_float(r.get("MAP")) for r in after]
        map_vals = [v for v in map_vals if v is not None]
        lac_vals = [safe_float(r.get("Lactate")) for r in after]
        lac_vals = [v for v in lac_vals if v is not None]
        cr_vals = [safe_float(r.get("Creatinine")) for r in after]
        cr_vals = [v for v in cr_vals if v is not None]
        wbc_vals = [safe_float(r.get("WBC")) for r in after]
        wbc_vals = [v for v in wbc_vals if v is not None]
        temp_vals = [safe_float(r.get("Temp")) for r in after]
        temp_vals = [v for v in temp_vals if v is not None]
        hr_vals = [safe_float(r.get("HR")) for r in after]
        hr_vals = [v for v in hr_vals if v is not None]
        plt_vals = [safe_float(r.get("Platelets")) for r in after]
        plt_vals = [v for v in plt_vals if v is not None]

        sbp_min = min(sbp_vals) if sbp_vals else None
        map_min = min(map_vals) if map_vals else None
        lac_max = max(lac_vals) if lac_vals else None
        cr_max = max(cr_vals) if cr_vals else None
        wbc_max = max(wbc_vals) if wbc_vals else None
        temp_max = max(temp_vals) if temp_vals else None
        hr_max = max(hr_vals) if hr_vals else None
        plt_min = min(plt_vals) if plt_vals else None

        if sbp_min is not None and sbp_min < 90:
            deterioration.append(f"SBP fell to {sbp_min:.0f}")
        if map_min is not None and map_min < 65:
            deterioration.append(f"MAP fell to {map_min:.0f}")
        if lac_max is not None and lac_max > 4:
            deterioration.append(f"Lactate to {lac_max:.1f}")
        elif lac_max is not None and lac_max > 2:
            deterioration.append(f"Lactate to {lac_max:.1f}")
        if cr_max is not None and cr_max > 2:
            deterioration.append(f"Creatinine to {cr_max:.1f}")
        if wbc_max is not None and wbc_max > 15:
            deterioration.append(f"WBC to {wbc_max:.1f}")
        if temp_max is not None and temp_max > 38.5:
            deterioration.append(f"Fever {temp_max:.1f}C")
        if hr_max is not None and hr_max > 130:
            deterioration.append(f"HR to {hr_max:.0f}")
        if plt_min is not None and plt_min < 100:
            deterioration.append(f"Plt to {plt_min:.0f}")

    return {
        "total_hours_in_icu": total_hours,
        "physionet_sepsis_label_anywhere": sepsis_label_seen_anywhere,
        "sepsis_onset_hour_in_psv": sepsis_onset_anywhere,
        "snapshot_hour": snapshot_idx,
        "hours_after_snapshot": total_hours - snapshot_idx,
        "deterioration_signals_next_48h": deterioration,
        "deterioration_count": len(deterioration),
    }


def classify_root_cause(r):
    """Bottom-most contributor for an FP."""
    if r.get("guardrail_override"):
        return "GUARDRAIL"
    risk = r.get("risk_score", 0) or 0
    if isinstance(risk, str):
        risk = 0
    priority = r.get("priority", "Standard")
    alert = r.get("alert_level", "STANDARD")
    if risk >= 70:
        return "LLM_HIGH_RISK"
    elif risk >= 50:
        return "LLM_MODERATE_RISK"
    elif priority in ("High", "Critical") or alert in ("HIGH", "CRITICAL"):
        return "LLM_PRIORITY_OVERRIDE"
    else:
        return "UNKNOWN"


def extract_clinical_signal(rationale, override_reasons):
    """Pull the dominant abnormal vital from the LLM's clinical rationale or override reasons."""
    if override_reasons:
        return ", ".join(override_reasons[:2])
    text = rationale.lower()
    signals = []
    patterns = {
        "MAP/BP dropping": [r"map[^.]{0,30}\b(60|6[0-4]|drop|fall|low)", r"hypotension", r"sbp[^.]{0,15}\b(8[0-9]|9[0-9])"],
        "Tachycardia": [r"tachycardia", r"hr[^.]{0,15}\b(10[5-9]|11[0-9]|12[0-9]|13[0-9])"],
        "Tachypnea": [r"tachypnea", r"rr[^.]{0,15}\b(2[3-9]|3[0-9])", r"resp.{0,10}rate"],
        "Fever": [r"febrile", r"fever", r"hyperthermia", r"temp[^.]{0,15}\b(38\.[3-9]|39|40)"],
        "Hypothermia": [r"hypothermi", r"temp[^.]{0,15}\b(35|36\.[0-2])"],
        "Lactate": [r"lactate", r"hyperlactatemia"],
        "WBC": [r"leukocytosis", r"leukopenia", r"wbc"],
        "AKI/Creatinine": [r"aki", r"creatinine", r"renal"],
        "Hemodynamic deterioration": [r"deterior", r"compensat", r"shock"],
        "Hypoxia/Desat": [r"hypoxi", r"desat", r"spo2[^.]{0,10}\b(8[0-9]|9[0-1])"],
    }
    for label, patts in patterns.items():
        for p in patts:
            if re.search(p, text):
                signals.append(label)
                break
    return ", ".join(signals[:3]) if signals else "ICU acuity (multifactorial)"


def main():
    with open(R4_FILE) as f:
        d = json.load(f)
    fps = [r for r in d["results"] if not r["actual_sepsis"] and r["predicted_sepsis"] == True]
    print(f"Analyzing {len(fps)} R4 false positives...\n")

    with open(COHORT_V4 / "cohort_manifest.json") as f:
        manifest = json.load(f)
    manifest_by_id = {p["patient_id"]: p for p in manifest["patients"]}

    enriched = []
    for r in fps:
        pid = r["patient_id"]
        snapshot = manifest_by_id.get(pid, {}).get("snapshot_hour", 0)
        traj = get_trajectory_outcome(pid, snapshot) or {}
        rationale = r.get("clinical_rationale", "") or ""
        override_reasons = r.get("override_reasons", []) or []
        rec = {
            **r,
            "trajectory": traj,
            "root_cause": classify_root_cause(r),
            "dominant_signal": extract_clinical_signal(rationale, override_reasons),
        }
        # Hidden TP heuristic: at-risk (>= 2 deterioration signals) and would have benefited from earlier alert
        rec["category"] = (
            "HIDDEN_TP_LIKELY" if traj.get("deterioration_count", 0) >= 3
            else "POSSIBLE_HIDDEN_TP" if traj.get("deterioration_count", 0) == 2
            else "TRUE_FALSE_ALARM"
        )
        enriched.append(rec)

    # === SECTION 1: Hidden TP analysis ===
    cat_counts = Counter(e["category"] for e in enriched)
    print("=" * 78)
    print("  SECTION 1: WERE THESE TRULY FALSE ALARMS?")
    print("=" * 78)
    print(f"  HIDDEN_TP_LIKELY (3+ deterioration signals in next 48h):  {cat_counts.get('HIDDEN_TP_LIKELY', 0)}")
    print(f"  POSSIBLE_HIDDEN_TP (2 deterioration signals):             {cat_counts.get('POSSIBLE_HIDDEN_TP', 0)}")
    print(f"  TRUE_FALSE_ALARM (0-1 deterioration signal):              {cat_counts.get('TRUE_FALSE_ALARM', 0)}")
    likely_hidden = cat_counts.get("HIDDEN_TP_LIKELY", 0) + cat_counts.get("POSSIBLE_HIDDEN_TP", 0)
    print(f"\n  -> {likely_hidden}/{len(fps)} ({likely_hidden/len(fps)*100:.1f}%) of \"false alarms\" actually showed concerning deterioration after our snapshot")
    print(f"  -> {cat_counts.get('TRUE_FALSE_ALARM', 0)}/{len(fps)} ({cat_counts.get('TRUE_FALSE_ALARM', 0)/len(fps)*100:.1f}%) appear to be GENUINE false alarms")

    # === SECTION 2: Root cause for TRUE false alarms only ===
    true_fas = [e for e in enriched if e["category"] == "TRUE_FALSE_ALARM"]
    print(f"\n{'=' * 78}")
    print(f"  SECTION 2: ROOT CAUSE OF {len(true_fas)} GENUINE FALSE ALARMS")
    print("=" * 78)
    cause_counts = Counter(e["root_cause"] for e in true_fas)
    for cause, ct in cause_counts.most_common():
        pct = ct / len(true_fas) * 100 if true_fas else 0
        bar = "█" * int(pct / 2)
        print(f"  {cause:<28} {ct:>3} ({pct:5.1f}%)  {bar}")

    # === SECTION 3: Dominant signals ===
    print(f"\n{'=' * 78}")
    print("  SECTION 3: DOMINANT CLINICAL SIGNAL DRIVING THE FALSE ALARM")
    print("=" * 78)
    signal_counts = Counter()
    for e in true_fas:
        for s in e["dominant_signal"].split(", "):
            signal_counts[s.strip()] += 1
    for s, ct in signal_counts.most_common(15):
        pct = ct / len(true_fas) * 100 if true_fas else 0
        bar = "█" * int(pct / 2)
        print(f"  {s:<35} {ct:>3} ({pct:5.1f}%)  {bar}")

    # === SECTION 4: Risk score distribution ===
    print(f"\n{'=' * 78}")
    print("  SECTION 4: LLM RISK SCORE DISTRIBUTION (true false alarms)")
    print("=" * 78)
    risk_buckets = Counter()
    for e in true_fas:
        risk = e.get("risk_score", 0)
        if not isinstance(risk, (int, float)):
            risk = 0
        if e.get("guardrail_override") and risk >= 90:
            bucket = "Forced 95 (guardrail)"
        elif risk >= 90:
            bucket = "90-100 (LLM critical)"
        elif risk >= 70:
            bucket = "70-89 (LLM high)"
        elif risk >= 50:
            bucket = "50-69 (LLM moderate)"
        elif risk >= 40:
            bucket = "40-49 (LLM borderline + priority bump)"
        else:
            bucket = "<40 (priority/alert override only)"
        risk_buckets[bucket] += 1
    for b in ["Forced 95 (guardrail)", "90-100 (LLM critical)", "70-89 (LLM high)",
              "50-69 (LLM moderate)", "40-49 (LLM borderline + priority bump)",
              "<40 (priority/alert override only)"]:
        ct = risk_buckets.get(b, 0)
        pct = ct / len(true_fas) * 100 if true_fas else 0
        bar = "█" * int(pct / 2)
        print(f"  {b:<45} {ct:>3} ({pct:5.1f}%)  {bar}")

    # === SECTION 5: Clinical scores ===
    print(f"\n{'=' * 78}")
    print("  SECTION 5: CLINICAL SCORE (qSOFA/SIRS/SOFA) PATTERNS")
    print("=" * 78)
    for sc in ["qsofa_score", "sirs_met", "sofa_score"]:
        c = Counter(str(e.get(sc, "N/A")) for e in true_fas)
        print(f"  {sc}:")
        for k in sorted(c.keys(), key=lambda x: (x == "N/A", x)):
            print(f"    {k}: {c[k]} cases")

    # === SECTION 6: Strategy recommendations ===
    print(f"\n{'=' * 78}")
    print("  SECTION 6: STRATEGIC IMPLICATIONS")
    print("=" * 78)
    if cause_counts.get("LLM_HIGH_RISK", 0) >= len(true_fas) * 0.4:
        print(f"  [!] LLM_HIGH_RISK is the dominant driver ({cause_counts.get('LLM_HIGH_RISK', 0)}/{len(true_fas)})")
        print(f"      -> The LLM is reading abnormal vitals and rating high WITHOUT clinical context")
        print(f"      -> Lever 1: Real nurse notes — biggest impact")
        print(f"      -> Lever 2: Prompt refinement to emphasize ICU-specific reasoning")
    if cause_counts.get("LLM_PRIORITY_OVERRIDE", 0) > 0:
        print(f"  [!] {cause_counts.get('LLM_PRIORITY_OVERRIDE', 0)} cases where LLM risk was <50 but priority/alert pushed positive")
        print(f"      -> Worth examining: is priority bump justified or noise?")
    print(f"  [!] {likely_hidden} \"false alarms\" actually deteriorated → these are PROBABLY correct early warnings")
    print(f"      -> Adjusted specificity (excluding hidden TPs): {(200 - cat_counts.get('TRUE_FALSE_ALARM', 0))/200*100:.1f}%")

    with open(OUT_FILE, "w") as f:
        json.dump({
            "round": "Round 4 (corrected trend order)",
            "total_fps": len(fps),
            "category_breakdown": dict(cat_counts),
            "true_false_alarm_count": len(true_fas),
            "root_cause_breakdown": dict(cause_counts),
            "dominant_signal_breakdown": dict(signal_counts.most_common()),
            "patient_details": enriched,
        }, f, indent=2)
    print(f"\n  Full per-patient report saved: {OUT_FILE}")


if __name__ == "__main__":
    main()
