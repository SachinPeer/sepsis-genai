"""
Simulate the impact of tightening the guardrail's early-detection escalation.

Now enhanced with LLM RAW risk scores recovered from the R7 API server log
(terminal 301119.txt), so we can tell whether a patient would still cross the
alert threshold AFTER the escalation is removed.

Method:
  - For each R7 patient, parse the "Stage 2 complete: Risk=X" log line to get
    llm_raw_risk (pre-guardrail).
  - Detect if "Early detection escalation" fired for that patient (log line
    WARNING within the same request_id block).
  - Replay the pipeline under alternative rules:
        prob of alert := (final_risk >= 50) OR (final_priority in {High, Critical})
    where:
        If escalation no longer fires under the new rule:
          final_risk = llm_raw_risk
          final_priority = LLM raw priority  (unknown — we assume Standard when
                          llm_raw_risk < 50, else leave as in the original result)
        Else: result unchanged.

The Standard-priority assumption when llm_raw_risk < 50 is CONSERVATIVE for
sensitivity (over-reports loss). v3.2 prompt has decoupled risk from priority,
so some of those LLM raw outputs might in fact already say "High" and the
patient would remain alerted even without the guardrail bump. Our numbers are
therefore an UPPER BOUND on sensitivity loss.
"""

import json
import re
from pathlib import Path
from collections import Counter

BASE = Path(__file__).parent
RESULTS = BASE / "results"
COHORT_DIR = BASE / "selected_cohort_v4"
R7_PATH = RESULTS / "R7_validation_results.json"
API_LOG = Path(
    "/Users/sachinj/.cursor/projects/"
    "Users-sachinj-Documents-My-projetcs-folder-Medical-medbacon-sepsis-genai/"
    "terminals/301119.txt"
)


STAGE2_RE = re.compile(
    r"\[genai_[\d_]+_(?P<pid>p\d+)\] Stage 2 complete: Risk=(?P<risk>\d+)"
)
ESCAL_RE = re.compile(r"Early detection escalation:")


def get_current_vital(vitals_obj, *keys):
    for k in keys:
        v = vitals_obj.get(k)
        if v is None: continue
        if isinstance(v, list) and len(v) > 0:
            first = v[0]
            if isinstance(first, dict): return first.get("val")
            return first
        return v
    return None


def count_early_detection_criteria(vitals_obj):
    hr = get_current_vital(vitals_obj, "HR")
    rr = get_current_vital(vitals_obj, "Resp", "RR")
    temp = get_current_vital(vitals_obj, "Temp")
    wbc = get_current_vital(vitals_obj, "WBC")
    n = 0; flags = []
    if hr is not None and hr >= 90: n += 1; flags.append(f"HR={hr:.0f}")
    if rr is not None and rr >= 22: n += 1; flags.append(f"RR={rr:.0f}")
    if temp is not None and (temp >= 38.0 or temp < 36.0): n += 1; flags.append(f"T={temp:.1f}")
    if wbc is not None and (wbc >= 12 or wbc <= 4): n += 1; flags.append(f"WBC={wbc:.1f}")
    return n, flags


def parse_api_log(log_path):
    """Return: {patient_id: {"raw_risk": int, "escalated": bool}}
    We keep the LAST occurrence per patient (R7 run only — the run started at 23:12)."""
    out = {}
    current_pid = None
    current_raw_risk = None
    current_escalated = False

    text = log_path.read_text()
    lines = text.splitlines()

    for line in lines:
        m = STAGE2_RE.search(line)
        if m:
            if current_pid is not None:
                out[current_pid] = {
                    "raw_risk": current_raw_risk,
                    "escalated": current_escalated,
                }
            current_pid = m.group("pid")
            current_raw_risk = int(m.group("risk"))
            current_escalated = False
            continue
        if ESCAL_RE.search(line) and current_pid is not None:
            current_escalated = True
    if current_pid is not None:
        out[current_pid] = {
            "raw_risk": current_raw_risk,
            "escalated": current_escalated,
        }
    return out


def classify(actual, predicted):
    if actual and predicted: return "TP"
    if actual and not predicted: return "FN"
    if (not actual) and predicted: return "FP"
    return "TN"


def predicted_from(risk, priority):
    return risk >= 50 or priority in ("High", "Critical")


def main():
    results = json.loads(R7_PATH.read_text())["results"]
    log_info = parse_api_log(API_LOG)
    print(f"Parsed {len(log_info)} patients from API log")

    per_patient = []
    for r in results:
        if r.get("status") != "success": continue
        pid = r["patient_id"]
        cohort_file = COHORT_DIR / f"{pid}.json"
        if not cohort_file.exists(): continue
        vitals = json.loads(cohort_file.read_text())["api_input"]["vitals"]

        actual = r["actual_sepsis"]
        pred = r["predicted_sepsis"]
        risk = float(r["risk_score"])
        priority = r["priority"]
        gr_override = r.get("guardrail_override") in (True, "True", "true")
        qsofa = r.get("qsofa_score", 0)
        sirs = r.get("sirs_met", 0)

        n_crit, crit_flags = count_early_detection_criteria(vitals)

        info = log_info.get(pid)
        if info is None:
            raw_risk = int(risk)
            escalated = False
        else:
            raw_risk = info["raw_risk"]
            escalated = info["escalated"]

        per_patient.append({
            "pid": pid,
            "actual": actual,
            "pred": pred,
            "cell": classify(actual, pred),
            "risk": risk,
            "priority": priority,
            "override": gr_override,
            "n_crit": n_crit,
            "crit_flags": crit_flags,
            "qsofa": qsofa,
            "sirs_met": sirs,
            "raw_risk": raw_risk,
            "escalated": escalated,
        })

    cells = Counter(p["cell"] for p in per_patient)
    base_tp, base_fn, base_fp, base_tn = cells['TP'], cells['FN'], cells['FP'], cells['TN']
    base_sens = base_tp/(base_tp+base_fn)*100
    base_spec = base_tn/(base_tn+base_fp)*100
    print(f"\nR7 baseline: TP={base_tp} FP={base_fp} TN={base_tn} FN={base_fn}")
    print(f"             Sens={base_sens:.1f}%   Spec={base_spec:.1f}%")

    esc = [p for p in per_patient if p["escalated"] and not p["override"]]
    print(f"\nPatients where early-detection escalation FIRED (no other override): {len(esc)}")
    print(f"  By class: {dict(Counter(p['cell'] for p in esc))}")
    print(f"  Raw-risk distribution among escalated:")
    bins = {"<30": 0, "30-49": 0, "50-69": 0, ">=70": 0}
    for p in esc:
        rr = p["raw_risk"]
        if rr < 30: bins["<30"] += 1
        elif rr < 50: bins["30-49"] += 1
        elif rr < 70: bins["50-69"] += 1
        else: bins[">=70"] += 1
    print(f"  {bins}")

    print()
    print("=" * 72)
    print("SCENARIOS — What happens if the escalation rule is changed?")
    print("=" * 72)
    print("Assumption: when escalation no longer fires, priority falls to the LLM's")
    print("raw priority. Because we don't log LLM raw priority, we conservatively")
    print("assume 'Standard' when raw_risk < 50 (upper bound on sensitivity loss).")
    print("If raw_risk >= 50, the patient still triggers on the risk gate alone.")

    scenarios = [
        {"id": "A", "name": "Tighten to 3-of-4 SIRS-on-vitals criteria",
         "fires": lambda p: p["n_crit"] >= 3},
        {"id": "B", "name": "Require 2-of-4 criteria AND qSOFA >= 1",
         "fires": lambda p: p["n_crit"] >= 2 and p["qsofa"] >= 1},
        {"id": "C", "name": "Require 2-of-4 criteria AND qSOFA >= 2  (strict)",
         "fires": lambda p: p["n_crit"] >= 2 and p["qsofa"] >= 2},
        {"id": "D", "name": "Disable escalation entirely (warn only, no bump)",
         "fires": lambda p: False},
        {"id": "E", "name": "Current rule (2-of-4) but cap escalation priority at 'Standard' (still bump risk to 70)",
         "fires": lambda p: p["n_crit"] >= 2, "priority_only_standard": True},
    ]

    rows = []
    for s in scenarios:
        fires = s["fires"]
        priority_only_std = s.get("priority_only_standard", False)
        tp = fp = tn = fn = 0
        flipped_fp_tn = []
        flipped_tp_fn = []
        for p in per_patient:
            new_risk = p["risk"]
            new_priority = p["priority"]

            if p["escalated"] and not p["override"]:
                if priority_only_std:
                    # S keeps bumping risk to 70 but priority stays LLM-raw-ish
                    new_risk = max(p["raw_risk"], 70)
                    new_priority = "Standard" if p["raw_risk"] < 50 else p["priority"]
                elif not fires(p):
                    new_risk = p["raw_risk"]
                    new_priority = "Standard" if p["raw_risk"] < 50 else p["priority"]
            new_pred = predicted_from(new_risk, new_priority)

            if p["actual"] and new_pred: tp += 1
            elif p["actual"] and not new_pred:
                fn += 1
                if p["cell"] == "TP": flipped_tp_fn.append(p["pid"])
            elif (not p["actual"]) and new_pred: fp += 1
            else:
                tn += 1
                if p["cell"] == "FP": flipped_fp_tn.append(p["pid"])

        sens = tp/(tp+fn)*100 if (tp+fn) else 0
        spec = tn/(tn+fp)*100 if (tn+fp) else 0
        row = {
            "id": s["id"], "name": s["name"],
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "sens": sens, "spec": spec,
            "flipped_fp_tn": flipped_fp_tn,
            "flipped_tp_fn": flipped_tp_fn,
        }
        rows.append(row)
        print(f"\n[{s['id']}] {s['name']}")
        print(f"    FP → TN: {len(flipped_fp_tn):>2}     TP → FN: {len(flipped_tp_fn):>2}")
        print(f"    New:  TP={tp:>3} FP={fp:>3} TN={tn:>3} FN={fn:>3}")
        print(f"    Sens: {sens:5.1f}% ({sens-base_sens:+.1f}%)   Spec: {spec:5.1f}% ({spec-base_spec:+.1f}%)")

    # Print side-by-side table
    print()
    print("=" * 72)
    print("SIDE-BY-SIDE SUMMARY")
    print("=" * 72)
    print(f"{'Scenario':<55} {'Sens':>7} {'Spec':>7} {'FP':>4} {'TP':>4}")
    print(f"{'R7 baseline (current guardrail)':<55} {base_sens:>6.1f}% {base_spec:>6.1f}% {base_fp:>4} {base_tp:>4}")
    for r in rows:
        label = f"[{r['id']}] {r['name'][:50]}"
        print(f"{label:<55} {r['sens']:>6.1f}% {r['spec']:>6.1f}% {r['FP']:>4} {r['TP']:>4}")

    # Save
    out = {
        "baseline": {"TP": base_tp, "FP": base_fp, "TN": base_tn, "FN": base_fn,
                     "sens": base_sens, "spec": base_spec},
        "escalations_fired": [p["pid"] for p in esc],
        "escalations_raw_risk_bins": bins,
        "scenarios": rows,
    }
    (RESULTS / "guardrail_tune_simulation.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {RESULTS / 'guardrail_tune_simulation.json'}")


if __name__ == "__main__":
    main()
