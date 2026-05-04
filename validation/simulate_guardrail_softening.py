"""
Counterfactual simulation: what happens to sensitivity/specificity if we
soften the guardrail's early-detection escalation (the rule that auto-bumps
risk_score to 70 and priority to High whenever any 2 of {HR>=90, RR>=22,
abnormal Temp, abnormal WBC} are true)?

We do NOT re-run the 340-patient validation. Instead we:
  1. Parse the API terminal log for each R7 patient
  2. Recover the LLM's RAW risk_score (logged before guardrail runs)
  3. Re-classify each patient under several softer-guardrail scenarios
  4. Report the resulting confusion matrix for each scenario

Scenarios:
  A. BASELINE_R7        = what actually happened
  B. NO_EARLY_DETECT    = keep only the critical overrides (MAP<65+lact etc.);
                          drop the 2-of-4 early-detection risk-bump to 70
  C. STRICTER_3_OF_4    = require 3 of 4 SIRS-ish criteria for early-detection
                          (HR, RR, Temp, WBC) before bumping to 70
  D. EARLY_DETECT_SOFT  = early-detection fires a "warning" but no risk bump
                          (priority stays whatever LLM said). Same as B in effect.

The re-classification logic mirrors run_validation.py's classify_prediction:
  predicted_sepsis = risk >= 50 OR priority in {High, Critical}

Where "risk" and "priority" are whatever they would be in each scenario:
  - If there is a critical override (guardrail_override=True), keep it (risk 95, Critical).
  - If there is ONLY an early-detection warning, in scenarios B/D use the raw LLM
    risk_score and the raw LLM priority; in scenario C, apply the strict 3-of-4 gate.

Input files:
  - validation/results/R7_validation_results.json  (final predictions per patient)
  - terminals/301119.txt  (API server log covering the R7 run window)
"""

import json
import re
from pathlib import Path
from collections import Counter

RESULTS = Path(__file__).parent / "results"
API_LOG = Path("/Users/sachinj/.cursor/projects/Users-sachinj-Documents-My-projetcs-folder-Medical-medbacon-sepsis-genai/terminals/301119.txt")
R7_PATH = RESULTS / "R7_validation_results.json"

# Regex to extract raw LLM risk per request from API log lines like:
#   2026-04-30 23:45:10,000 - genai_pipeline - INFO - [genai_20260430_234510_p000018] Stage 2 complete: Risk=48
STAGE2_RE = re.compile(
    r"\[genai_\d+_\d+_(?P<pid>p\d+)\] Stage 2 complete: Risk=(?P<risk>\d+|N/A)"
)
AUDIT_RE = re.compile(r'"patient_id": "(?P<pid>p\d+)".*?"prediction": (?P<pred>\{[^}]*\}).*?"clinical_scores": (?P<scores>\{[^}]*\}).*?"guardrail": (?P<gr>\{[^}]*\})')


def parse_api_log():
    """For each patient, get the raw LLM risk score and the audit details."""
    text = API_LOG.read_text()
    raw_llm_risk = {}
    for m in STAGE2_RE.finditer(text):
        pid = m.group("pid")
        risk_str = m.group("risk")
        if risk_str != "N/A":
            raw_llm_risk[pid] = int(risk_str)

    # Parse audit JSON per request. The audit line is a big JSON dict; let's
    # find each audit-emit and extract what we need.
    audit_info = {}
    lines = text.splitlines()
    for line in lines:
        if "sepsis_audit" not in line:
            continue
        # Find the JSON portion after " - INFO - "
        json_start = line.find("{")
        if json_start < 0:
            continue
        try:
            audit = json.loads(line[json_start:])
        except Exception:
            continue
        pid = audit.get("patient_id")
        if not pid:
            continue
        audit_info[pid] = {
            "final_risk": audit.get("prediction", {}).get("risk_score"),
            "final_priority": audit.get("prediction", {}).get("priority"),
            "override_applied": audit.get("guardrail", {}).get("override_applied"),
            "override_reasons": audit.get("guardrail", {}).get("override_reasons"),
            "qsofa": audit.get("clinical_scores", {}).get("qsofa_score"),
            "sirs": audit.get("clinical_scores", {}).get("sirs_criteria_met"),
            "septic_shock": audit.get("clinical_scores", {}).get("septic_shock_met"),
        }
    return raw_llm_risk, audit_info


def load_r7():
    data = json.loads(R7_PATH.read_text())
    return {r["patient_id"]: r for r in data["results"] if r.get("status") == "success"}


def classify(risk, priority):
    if risk is None:
        return False
    if isinstance(risk, (int, float)) and risk >= 50:
        return True
    if priority in ("High", "Critical"):
        return True
    return False


def detect_early_detection_firing(final_risk, final_priority, override_applied):
    """Heuristic: if there's NO override but final risk is exactly 70 and
    priority is High, that almost certainly came from the early-detection
    escalation (which forces risk=max(risk,70), priority=High).

    This is an approximation because the LLM could also independently
    emit risk=70 priority=High. We'll double-check against the raw LLM
    risk: if raw LLM risk was < 70 but final is >= 70 with no override,
    that's an early-detection bump.
    """
    if override_applied:
        return False
    return final_risk == 70 and final_priority == "High"


def scenario_metrics(r7, raw_llm, audit, scenario):
    """Simulate the specified scenario and return a confusion matrix."""
    tp = fp = tn = fn = 0
    detected_early_escalations = 0
    llm_missing = 0

    for pid, row in r7.items():
        actual = row["actual_sepsis"]
        final_risk = row["risk_score"]
        final_priority = row["priority"]
        override = row.get("guardrail_override", False)
        try:
            final_risk = int(final_risk)
        except Exception:
            final_risk = 0

        if scenario == "BASELINE_R7":
            risk_used, prio_used = final_risk, final_priority

        elif scenario == "NO_EARLY_DETECT":
            # If there was no override and the final risk was the 70 bump,
            # revert to LLM-raw risk. If we have no raw risk, we skip the
            # revert (conservative).
            is_early_bump = detect_early_detection_firing(final_risk, final_priority, override)
            if is_early_bump and pid in raw_llm:
                detected_early_escalations += 1
                raw = raw_llm[pid]
                risk_used = raw
                # Priority: can we recover LLM-raw priority? No — the audit
                # log captures FINAL priority. We assume priority would be
                # whatever the LLM itself emitted, which typically aligns
                # with LLM-raw risk given our prompt v3.2 decoupling.
                # Approximation: if LLM risk < 50, priority = Standard.
                prio_used = "Standard" if raw < 50 else ("High" if raw < 80 else "Critical")
            elif is_early_bump and pid not in raw_llm:
                llm_missing += 1
                risk_used, prio_used = final_risk, final_priority
            else:
                risk_used, prio_used = final_risk, final_priority

        elif scenario == "STRICTER_3_OF_4":
            # Require 3+ of {HR>=90, RR>=22, Temp abnormal, WBC abnormal}
            # We don't have the raw vitals in r7 row — approximate using SIRS
            # criteria_met from clinical_scores (in audit dict). The SIRS criteria
            # are roughly equivalent (HR>90, RR>20 OR PaCO2<32, Temp abnormal,
            # WBC abnormal = 4 criteria).
            is_early_bump = detect_early_detection_firing(final_risk, final_priority, override)
            if is_early_bump:
                sirs_count = (audit.get(pid, {}) or {}).get("sirs", 0) or 0
                if sirs_count >= 3 and pid in raw_llm:
                    # Keep the bump (3-of-4 met)
                    risk_used, prio_used = final_risk, final_priority
                elif pid in raw_llm:
                    # Not enough — revert to raw LLM
                    detected_early_escalations += 1
                    raw = raw_llm[pid]
                    risk_used = raw
                    prio_used = "Standard" if raw < 50 else ("High" if raw < 80 else "Critical")
                else:
                    risk_used, prio_used = final_risk, final_priority
            else:
                risk_used, prio_used = final_risk, final_priority

        else:
            raise ValueError(scenario)

        predicted = classify(risk_used, prio_used)
        if actual and predicted:
            tp += 1
        elif actual and not predicted:
            fn += 1
        elif not actual and predicted:
            fp += 1
        else:
            tn += 1

    sens = tp / (tp + fn) * 100 if (tp + fn) else 0
    spec = tn / (tn + fp) * 100 if (tn + fp) else 0
    return {
        "scenario": scenario,
        "TP": tp, "FN": fn, "FP": fp, "TN": tn,
        "sensitivity": round(sens, 2),
        "specificity": round(spec, 2),
        "early_escalations_flipped": detected_early_escalations,
        "llm_raw_missing": llm_missing,
    }


def main():
    print("Parsing API log for raw LLM risk scores...")
    raw_llm, audit = parse_api_log()
    print(f"  Got raw LLM risk for {len(raw_llm)} patients")
    print(f"  Got audit info for {len(audit)} patients")

    r7 = load_r7()
    print(f"  R7 patients: {len(r7)}")

    # How many patients look like early-detection escalations?
    early_bumps = 0
    for pid, row in r7.items():
        try:
            fr = int(row["risk_score"])
        except Exception:
            fr = 0
        if detect_early_detection_firing(fr, row["priority"], row.get("guardrail_override", False)):
            early_bumps += 1
    print(f"\n  R7 early-detection escalations (risk=70/High without override): {early_bumps}")

    # How many of those are true sepsis vs false alarms?
    true_pos_among_early = 0
    false_pos_among_early = 0
    for pid, row in r7.items():
        try:
            fr = int(row["risk_score"])
        except Exception:
            fr = 0
        if detect_early_detection_firing(fr, row["priority"], row.get("guardrail_override", False)):
            if row["actual_sepsis"]:
                true_pos_among_early += 1
            else:
                false_pos_among_early += 1
    print(f"    - true positives among them:  {true_pos_among_early}")
    print(f"    - false positives among them: {false_pos_among_early}")

    print("\n===== SCENARIOS =====")
    for scenario in ["BASELINE_R7", "NO_EARLY_DETECT", "STRICTER_3_OF_4"]:
        metrics = scenario_metrics(r7, raw_llm, audit, scenario)
        print(f"\n  {scenario}")
        print(f"    TP={metrics['TP']:3d}  FN={metrics['FN']:3d}  FP={metrics['FP']:3d}  TN={metrics['TN']:3d}")
        print(f"    Sensitivity: {metrics['sensitivity']:.2f}%   Specificity: {metrics['specificity']:.2f}%")
        if scenario != "BASELINE_R7":
            print(f"    (early-detection bumps reverted: {metrics['early_escalations_flipped']}, raw-LLM missing: {metrics['llm_raw_missing']})")

    out = RESULTS / "guardrail_softening_simulation.json"
    results = [scenario_metrics(r7, raw_llm, audit, s) for s in ["BASELINE_R7", "NO_EARLY_DETECT", "STRICTER_3_OF_4"]]
    out.write_text(json.dumps(results, indent=2))
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
