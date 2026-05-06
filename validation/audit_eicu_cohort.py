"""
Audit the current eICU validation cohort and flag patients that should be
removed before scaling. Produces:

  - validation/eicu_cohort/audit_report.json  (full per-patient details)
  - validation/eicu_cohort/audit_summary.csv  (one row per patient)

A patient is flagged for removal if any of the following are true:

  AGE_NEONATE       : Age == 0 or numeric age missing
  AGE_PEDIATRIC     : Age < 18
  AGE_OUTLIER       : Age impossible (>120) — likely encoding error
  VITAL_HR_THIN     : HR readings in trend window < 4
  VITAL_RESP_THIN   : Resp readings in trend window < 4
  VITAL_TYPES_THIN  : Fewer than 3 vital types with >= 4 readings each
  LABS_MISSING      : Zero labs in the 24h window
  LABS_THIN         : Fewer than 2 labs across the entire panel
  NOTES_EMPTY       : Notes field is "No free-text notes or structured observations..."

Usage:
    python3 validation/audit_eicu_cohort.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import os
COHORT = Path(__file__).parent / os.environ.get("COHORT_DIR", "eicu_cohort")
OUT_JSON = COHORT / "audit_report.json"
OUT_CSV = COHORT / "audit_summary.csv"

VITAL_KEYS = ["HR", "SBP", "DBP", "MAP", "Temp", "Resp", "O2Sat"]
LAB_KEYS = ["WBC", "Lactate", "Creatinine", "Platelets", "Glucose",
            "BUN", "pH", "HCO3", "PaCO2", "Hgb", "FiO2"]

NO_NOTES_MARKER = "No free-text notes or structured observations"

MIN_VITAL_READINGS = 4   # per type, in 6h window
MIN_VITAL_TYPES = 3      # number of vital types meeting MIN_VITAL_READINGS
MIN_LAB_COUNT = 2        # any 2 labs anywhere in panel


def audit_one(path: Path) -> dict:
    p = json.loads(path.read_text())
    pid = p.get("patient_id", path.stem)
    demo = p.get("patient_demographics", {}) or {}
    vit = p.get("patient_vitals", {}) or {}
    notes = p.get("patient_notes", "") or ""
    gt = p.get("ground_truth", {}) or {}

    flags = []

    # --- Age checks
    age = demo.get("Age")
    try:
        age_num = float(age) if age is not None else None
    except Exception:
        age_num = None
    if age_num is None or age_num == 0:
        flags.append("AGE_NEONATE")
    elif age_num < 18:
        flags.append("AGE_PEDIATRIC")
    elif age_num > 120:
        flags.append("AGE_OUTLIER")

    # --- Vital coverage
    vital_counts = {k: len(vit.get(k, []) or []) for k in VITAL_KEYS}
    types_meeting = sum(1 for k in VITAL_KEYS if vital_counts[k] >= MIN_VITAL_READINGS)
    if vital_counts["HR"] < MIN_VITAL_READINGS:
        flags.append("VITAL_HR_THIN")
    if vital_counts["Resp"] < MIN_VITAL_READINGS:
        flags.append("VITAL_RESP_THIN")
    if types_meeting < MIN_VITAL_TYPES:
        flags.append("VITAL_TYPES_THIN")

    # --- Lab coverage
    lab_counts = {k: len(vit.get(k, []) or []) for k in LAB_KEYS}
    n_labs = sum(1 for c in lab_counts.values() if c > 0)
    if n_labs == 0:
        flags.append("LABS_MISSING")
    elif n_labs < MIN_LAB_COUNT:
        flags.append("LABS_THIN")

    # --- Notes
    if NO_NOTES_MARKER in notes:
        flags.append("NOTES_EMPTY")

    return {
        "patient_id": pid,
        "patientunitstayid": p.get("source", {}).get("patientunitstayid"),
        "age": age_num,
        "gender": demo.get("Gender"),
        "unit_type": demo.get("UnitType"),
        "hospital_id": demo.get("HospitalId"),
        "is_sepsis": bool(gt.get("actual_sepsis")),
        "vital_counts": vital_counts,
        "lab_counts": lab_counts,
        "vital_types_with_min_readings": types_meeting,
        "labs_present": n_labs,
        "notes_empty": NO_NOTES_MARKER in notes,
        "notes_chars": len(notes),
        "flags": flags,
        "remove": bool(flags),
    }


def main():
    files = sorted(COHORT.glob("eicu_p*.json"))
    if not files:
        raise SystemExit(f"No cohort files in {COHORT}")
    rows = [audit_one(f) for f in files]

    n_total = len(rows)
    n_remove = sum(1 for r in rows if r["remove"])
    n_keep = n_total - n_remove
    n_sepsis = sum(1 for r in rows if r["is_sepsis"])
    n_sepsis_keep = sum(1 for r in rows if r["is_sepsis"] and not r["remove"])
    n_ctrl_keep = n_keep - n_sepsis_keep

    flag_counts: dict[str, int] = {}
    for r in rows:
        for f in r["flags"]:
            flag_counts[f] = flag_counts.get(f, 0) + 1

    summary = {
        "n_total": n_total,
        "n_keep": n_keep,
        "n_remove": n_remove,
        "n_sepsis_total": n_sepsis,
        "n_sepsis_keep": n_sepsis_keep,
        "n_control_keep": n_ctrl_keep,
        "flag_counts": flag_counts,
        "patients": rows,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=str))

    with OUT_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "patient_id", "patientunitstayid", "age", "gender", "unit_type",
            "hospital_id", "is_sepsis",
            "HR", "SBP", "DBP", "MAP", "Temp", "Resp", "O2Sat",
            "labs_present", "vital_types_ok", "notes_chars",
            "remove", "flags",
        ])
        for r in rows:
            vc = r["vital_counts"]
            w.writerow([
                r["patient_id"], r["patientunitstayid"], r["age"], r["gender"],
                r["unit_type"], r["hospital_id"], r["is_sepsis"],
                vc["HR"], vc["SBP"], vc["DBP"], vc["MAP"],
                vc["Temp"], vc["Resp"], vc["O2Sat"],
                r["labs_present"], r["vital_types_with_min_readings"],
                r["notes_chars"], r["remove"], "|".join(r["flags"]),
            ])

    # --- print human report
    print("=" * 70)
    print(" eICU COHORT AUDIT")
    print("=" * 70)
    print(f" Files inspected   : {n_total}")
    print(f" Will remove       : {n_remove}")
    print(f" Will keep         : {n_keep}")
    print(f"   - Sepsis kept   : {n_sepsis_keep} (of {n_sepsis} total sepsis)")
    print(f"   - Controls kept : {n_ctrl_keep}")
    print()
    print(" Flag counts:")
    for k in sorted(flag_counts, key=lambda x: -flag_counts[x]):
        print(f"   {k:18s}: {flag_counts[k]}")
    print()
    print(f" Outputs:\n   {OUT_JSON.relative_to(COHORT.parent.parent)}\n   {OUT_CSV.relative_to(COHORT.parent.parent)}")


if __name__ == "__main__":
    main()
