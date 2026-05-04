"""
Deep-dive analysis of Round 2 false positives (152 patients).
Categorizes each FP by root cause: guardrail override, LLM risk score,
clinical scores (qSOFA/SIRS/SOFA), and abnormal vital trends.
"""

import json
from pathlib import Path
from collections import Counter, defaultdict

R2_FILE = Path(__file__).parent / "results" / "validation_results_20260429_175253.json"
COHORT_DIR = Path(__file__).parent / "selected_cohort_v2"
OUTPUT_DIR = Path(__file__).parent / "results"


def load_r2_results():
    with open(R2_FILE) as f:
        return json.load(f)["results"]


def load_patient_vitals(patient_id):
    p_file = COHORT_DIR / f"{patient_id}.json"
    if p_file.exists():
        with open(p_file) as f:
            return json.load(f)
    return None


def get_latest_vital(vitals, key):
    v = vitals.get(key)
    if v is None:
        return None
    if isinstance(v, list):
        if len(v) == 0:
            return None
        last = v[-1]
        if isinstance(last, dict):
            return last.get("val")
        return last
    return v


def analyze_vital_triggers(vitals):
    """Identify which vital signs are abnormal and could trigger false alarm."""
    triggers = []

    hr = get_latest_vital(vitals, "HR")
    if hr and hr > 100:
        triggers.append(f"Tachycardia (HR={hr:.0f})")
    if hr and hr > 120:
        triggers[-1] = f"Severe Tachycardia (HR={hr:.0f})"

    sbp = get_latest_vital(vitals, "SBP")
    if sbp and sbp < 100:
        triggers.append(f"Hypotension (SBP={sbp:.0f})")
    if sbp and sbp < 90:
        triggers[-1] = f"Severe Hypotension (SBP={sbp:.0f})"

    map_v = get_latest_vital(vitals, "MAP")
    if map_v and map_v < 65:
        triggers.append(f"Low MAP ({map_v:.0f})")

    temp = get_latest_vital(vitals, "Temp")
    if temp and temp > 38.3:
        triggers.append(f"Fever (Temp={temp:.1f})")
    elif temp and temp < 36.0:
        triggers.append(f"Hypothermia (Temp={temp:.1f})")

    resp = get_latest_vital(vitals, "Resp")
    if resp and resp > 22:
        triggers.append(f"Tachypnea (RR={resp:.0f})")

    o2 = get_latest_vital(vitals, "O2Sat")
    if o2 and o2 < 92:
        triggers.append(f"Desaturation (SpO2={o2:.0f}%)")

    wbc = get_latest_vital(vitals, "WBC")
    if wbc:
        if isinstance(wbc, list):
            wbc = wbc[0] if wbc else None
    if wbc and wbc > 12:
        triggers.append(f"Leukocytosis (WBC={wbc:.1f})")
    elif wbc and wbc < 4:
        triggers.append(f"Leukopenia (WBC={wbc:.1f})")

    lac = get_latest_vital(vitals, "Lactate")
    if lac:
        if isinstance(lac, list):
            lac = lac[0] if lac else None
    if lac and lac > 2:
        triggers.append(f"Elevated Lactate ({lac:.1f})")
    if lac and lac > 4:
        triggers[-1] = f"Critical Lactate ({lac:.1f})"

    cr = get_latest_vital(vitals, "Creatinine")
    if cr:
        if isinstance(cr, list):
            cr = cr[0] if cr else None
    if cr and cr > 1.5:
        triggers.append(f"Elevated Creatinine ({cr:.1f})")

    plt = get_latest_vital(vitals, "Platelets")
    if plt:
        if isinstance(plt, list):
            plt = plt[0] if plt else None
    if plt and plt < 100:
        triggers.append(f"Thrombocytopenia (Plt={plt:.0f})")

    # Trend analysis
    hr_vals = vitals.get("HR", [])
    if isinstance(hr_vals, list) and len(hr_vals) >= 2:
        if all(isinstance(x, dict) for x in hr_vals):
            first_hr = hr_vals[0]["val"]
            last_hr = hr_vals[-1]["val"]
            if last_hr - first_hr > 15:
                triggers.append(f"HR Rising Trend ({first_hr:.0f}→{last_hr:.0f})")

    sbp_vals = vitals.get("SBP", [])
    if isinstance(sbp_vals, list) and len(sbp_vals) >= 2:
        if all(isinstance(x, dict) for x in sbp_vals):
            first_sbp = sbp_vals[0]["val"]
            last_sbp = sbp_vals[-1]["val"]
            if first_sbp - last_sbp > 15:
                triggers.append(f"BP Falling Trend ({first_sbp:.0f}→{last_sbp:.0f})")

    return triggers


def classify_fp_cause(result, vitals_data):
    """Determine the primary and contributing causes of a false positive."""
    causes = []

    if result.get("guardrail_override"):
        causes.append("GUARDRAIL_OVERRIDE")

    risk = result.get("risk_score", 0)
    if isinstance(risk, str):
        risk = int(risk) if risk.isdigit() else 0

    qsofa = result.get("qsofa_score", 0)
    if isinstance(qsofa, str):
        qsofa = int(qsofa) if qsofa.isdigit() else 0

    sirs = result.get("sirs_met", 0)
    if isinstance(sirs, str):
        sirs = int(sirs) if sirs.isdigit() else 0

    sofa = result.get("sofa_score", 0)
    if isinstance(sofa, str):
        sofa = int(sofa) if sofa.isdigit() else 0

    if qsofa >= 2:
        causes.append("HIGH_QSOFA")
    if sirs >= 2:
        causes.append("HIGH_SIRS")
    if sofa >= 2:
        causes.append("HIGH_SOFA")

    if risk >= 80:
        causes.append("LLM_VERY_HIGH_RISK")
    elif risk >= 60:
        causes.append("LLM_HIGH_RISK")
    elif risk >= 50:
        causes.append("LLM_MODERATE_RISK")

    vital_triggers = []
    if vitals_data:
        vital_triggers = analyze_vital_triggers(vitals_data.get("api_input", {}).get("vitals", {}))
        if len(vital_triggers) >= 3:
            causes.append("MULTIPLE_ABNORMAL_VITALS")

    primary = "GUARDRAIL_OVERRIDE" if "GUARDRAIL_OVERRIDE" in causes else \
              "LLM_VERY_HIGH_RISK" if "LLM_VERY_HIGH_RISK" in causes else \
              "HIGH_QSOFA" if "HIGH_QSOFA" in causes else \
              "LLM_HIGH_RISK" if "LLM_HIGH_RISK" in causes else \
              "LLM_MODERATE_RISK" if "LLM_MODERATE_RISK" in causes else \
              "CLINICAL_SCORES" if ("HIGH_SIRS" in causes or "HIGH_SOFA" in causes) else \
              "UNKNOWN"

    return primary, causes, vital_triggers


def main():
    results = load_r2_results()

    fps = [r for r in results if not r["actual_sepsis"] and r["predicted_sepsis"] == True]
    print(f"{'='*70}")
    print(f"  FALSE ALARM DEEP-DIVE ANALYSIS — Round 2 ({len(fps)} False Positives)")
    print(f"{'='*70}\n")

    primary_counts = Counter()
    all_causes = Counter()
    all_vital_triggers = Counter()
    risk_distribution = Counter()
    priority_distribution = Counter()
    guardrail_count = 0
    fp_details = []

    for r in fps:
        patient_id = r["patient_id"]
        vitals_data = load_patient_vitals(patient_id)
        primary, causes, vital_triggers = classify_fp_cause(r, vitals_data)

        primary_counts[primary] += 1
        for c in causes:
            all_causes[c] += 1
        for vt in vital_triggers:
            cat = vt.split("(")[0].strip()
            all_vital_triggers[cat] += 1

        risk = r.get("risk_score", 0)
        if risk >= 90:
            risk_distribution["90-100 (Critical)"] += 1
        elif risk >= 70:
            risk_distribution["70-89 (High)"] += 1
        elif risk >= 50:
            risk_distribution["50-69 (Moderate)"] += 1
        else:
            risk_distribution["<50 (Should not be FP)"] += 1

        priority_distribution[r.get("priority", "Unknown")] += 1

        if r.get("guardrail_override"):
            guardrail_count += 1

        fp_details.append({
            "patient_id": patient_id,
            "risk_score": r.get("risk_score"),
            "priority": r.get("priority"),
            "alert_level": r.get("alert_level"),
            "guardrail_override": r.get("guardrail_override"),
            "qsofa": r.get("qsofa_score"),
            "sirs_met": r.get("sirs_met"),
            "sofa": r.get("sofa_score"),
            "primary_cause": primary,
            "all_causes": causes,
            "vital_triggers": vital_triggers,
            "clinical_rationale": r.get("clinical_rationale", "")[:300],
            "age": r.get("age"),
            "gender": r.get("gender"),
        })

    # === PRINT REPORT ===
    print("=" * 70)
    print("  SECTION 1: PRIMARY CAUSE BREAKDOWN")
    print("=" * 70)
    for cause, count in primary_counts.most_common():
        pct = count / len(fps) * 100
        bar = "█" * int(pct / 2)
        print(f"  {cause:<30} {count:>4} ({pct:5.1f}%)  {bar}")

    print(f"\n{'='*70}")
    print("  SECTION 2: GUARDRAIL vs LLM vs CLINICAL SCORES")
    print("=" * 70)
    guardrail_fps = [d for d in fp_details if d["guardrail_override"]]
    llm_only_fps = [d for d in fp_details if not d["guardrail_override"]]
    print(f"  Guardrail Override caused FP:  {len(guardrail_fps):>4} ({len(guardrail_fps)/len(fps)*100:.1f}%)")
    print(f"  LLM alone caused FP:           {len(llm_only_fps):>4} ({len(llm_only_fps)/len(fps)*100:.1f}%)")

    # Among guardrail FPs, what was the LLM's original assessment?
    guardrail_llm_low = sum(1 for d in guardrail_fps if d["risk_score"] and int(str(d["risk_score"])) < 50)
    guardrail_llm_high = len(guardrail_fps) - guardrail_llm_low
    print(f"\n  Among {len(guardrail_fps)} guardrail overrides:")
    print(f"    LLM said low risk but guardrail escalated:  {guardrail_llm_low}")
    print(f"    LLM already said high risk + guardrail:     {guardrail_llm_high}")

    print(f"\n{'='*70}")
    print("  SECTION 3: RISK SCORE DISTRIBUTION (False Alarms)")
    print("=" * 70)
    for bucket in ["90-100 (Critical)", "70-89 (High)", "50-69 (Moderate)", "<50 (Should not be FP)"]:
        count = risk_distribution.get(bucket, 0)
        pct = count / len(fps) * 100
        bar = "█" * int(pct / 2)
        print(f"  {bucket:<30} {count:>4} ({pct:5.1f}%)  {bar}")

    print(f"\n{'='*70}")
    print("  SECTION 4: PRIORITY DISTRIBUTION (False Alarms)")
    print("=" * 70)
    for priority in ["Critical", "High", "Standard", "Low"]:
        count = priority_distribution.get(priority, 0)
        pct = count / len(fps) * 100
        bar = "█" * int(pct / 2)
        print(f"  {priority:<30} {count:>4} ({pct:5.1f}%)  {bar}")

    print(f"\n{'='*70}")
    print("  SECTION 5: ABNORMAL VITALS TRIGGERING FALSE ALARMS")
    print("=" * 70)
    for trigger, count in all_vital_triggers.most_common(15):
        pct = count / len(fps) * 100
        bar = "█" * int(pct / 2)
        print(f"  {trigger:<30} {count:>4} ({pct:5.1f}%)  {bar}")

    # Combination patterns
    print(f"\n{'='*70}")
    print("  SECTION 6: COMMON VITAL ABNORMALITY COMBINATIONS")
    print("=" * 70)
    combo_counter = Counter()
    for d in fp_details:
        if d["vital_triggers"]:
            combo_key = " + ".join(sorted([vt.split("(")[0].strip() for vt in d["vital_triggers"]]))
            combo_counter[combo_key] += 1
    for combo, count in combo_counter.most_common(10):
        pct = count / len(fps) * 100
        print(f"  {combo[:60]:<62} {count:>3} ({pct:4.1f}%)")

    # qSOFA/SIRS/SOFA analysis
    print(f"\n{'='*70}")
    print("  SECTION 7: CLINICAL SCORE PATTERNS IN FALSE ALARMS")
    print("=" * 70)
    qsofa_dist = Counter()
    sirs_dist = Counter()
    sofa_dist = Counter()
    for d in fp_details:
        q = d.get("qsofa", "N/A")
        s = d.get("sirs_met", "N/A")
        so = d.get("sofa", "N/A")
        qsofa_dist[str(q)] += 1
        sirs_dist[str(s)] += 1
        sofa_dist[str(so)] += 1

    print("  qSOFA Score Distribution:")
    for k in sorted(qsofa_dist.keys()):
        print(f"    qSOFA={k}: {qsofa_dist[k]} patients ({qsofa_dist[k]/len(fps)*100:.1f}%)")
    print("  SIRS Criteria Met:")
    for k in sorted(sirs_dist.keys()):
        print(f"    SIRS={k}: {sirs_dist[k]} patients ({sirs_dist[k]/len(fps)*100:.1f}%)")
    print("  SOFA Score Distribution:")
    for k in sorted(sofa_dist.keys()):
        print(f"    SOFA={k}: {sofa_dist[k]} patients ({sofa_dist[k]/len(fps)*100:.1f}%)")

    # Actionable recommendations
    print(f"\n{'='*70}")
    print("  SECTION 8: ACTIONABLE RECOMMENDATIONS")
    print("=" * 70)

    if len(guardrail_fps) > len(fps) * 0.3:
        print(f"  [!] GUARDRAIL OVERRIDES are a major cause ({len(guardrail_fps)}/{len(fps)} = {len(guardrail_fps)/len(fps)*100:.0f}%)")
        print(f"      → Review guardrail thresholds. Consider raising the bar for override.")
        if guardrail_llm_low > 0:
            print(f"      → {guardrail_llm_low} cases where LLM said LOW risk but guardrail escalated — strongest signal for tuning")

    high_risk_fps = risk_distribution.get("90-100 (Critical)", 0)
    if high_risk_fps > len(fps) * 0.2:
        print(f"  [!] {high_risk_fps} FPs with risk 90-100 — LLM is very confident but wrong")
        print(f"      → These patients likely have severe ICU acuity (not sepsis).")
        print(f"      → Without clinical context notes, LLM cannot distinguish.")

    tachy_count = all_vital_triggers.get("Tachycardia", 0) + all_vital_triggers.get("Severe Tachycardia", 0)
    if tachy_count > len(fps) * 0.4:
        print(f"  [!] Tachycardia present in {tachy_count}/{len(fps)} ({tachy_count/len(fps)*100:.0f}%) of FPs")
        print(f"      → ICU patients are commonly tachycardic; consider adjusting HR threshold or weighting")

    tachy_pnea = all_vital_triggers.get("Tachypnea", 0)
    if tachy_pnea > len(fps) * 0.3:
        print(f"  [!] Tachypnea present in {tachy_pnea}/{len(fps)} ({tachy_pnea/len(fps)*100:.0f}%) of FPs")
        print(f"      → High RR common in ventilated ICU patients; consider context-aware thresholds")

    print()

    # Save full report
    report = {
        "round": "Round 2 (Trend-enriched data)",
        "total_false_positives": len(fps),
        "primary_cause_breakdown": dict(primary_counts),
        "guardrail_vs_llm": {
            "guardrail_override_fps": len(guardrail_fps),
            "llm_only_fps": len(llm_only_fps),
            "guardrail_llm_low_risk": guardrail_llm_low,
            "guardrail_llm_high_risk": guardrail_llm_high,
        },
        "risk_distribution": dict(risk_distribution),
        "priority_distribution": dict(priority_distribution),
        "vital_trigger_frequency": dict(all_vital_triggers.most_common()),
        "common_combinations": dict(combo_counter.most_common(10)),
        "clinical_scores": {
            "qsofa": dict(qsofa_dist),
            "sirs": dict(sirs_dist),
            "sofa": dict(sofa_dist),
        },
        "patient_details": fp_details,
    }

    report_path = OUTPUT_DIR / "false_alarm_analysis.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Full report saved: {report_path}")


if __name__ == "__main__":
    main()
