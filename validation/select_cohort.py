"""
Phase 1: Select validation cohort from already-downloaded PhysioNet data.
Scans local .psv files and selects 140 sepsis + 200 non-sepsis patients.
"""

import os
import csv
import json
import random
from pathlib import Path

RAW_DIR = Path(__file__).parent / "raw_data"
COHORT_DIR = Path(__file__).parent / "selected_cohort"
COHORT_DIR.mkdir(exist_ok=True)

TARGET_SEPSIS = 140
TARGET_NON_SEPSIS = 200


def parse_psv(filepath):
    rows = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            rows.append(row)
    return rows


def safe_float(val):
    if val is None or val.strip() == "" or val.strip() == "NaN":
        return None
    try:
        return float(val)
    except ValueError:
        return None


def get_sepsis_onset_index(rows):
    for i, row in enumerate(rows):
        if row.get("SepsisLabel", "0").strip() == "1":
            return i
    return None


def extract_snapshot(rows, target_idx):
    vital_map = {
        "HR": "heart_rate", "O2Sat": "o2_saturation", "Temp": "temperature",
        "SBP": "sbp", "MAP": "map", "DBP": "dbp", "Resp": "respiratory_rate",
    }
    lab_map = {
        "BaseExcess": "base_excess", "HCO3": "bicarbonate", "FiO2": "fio2",
        "pH": "ph", "PaCO2": "paco2", "AST": "ast", "BUN": "bun",
        "Creatinine": "creatinine", "Glucose": "glucose", "Lactate": "lactate",
        "Potassium": "potassium", "Bilirubin_total": "bilirubin",
        "TroponinI": "troponin", "Hgb": "hemoglobin", "PTT": "ptt",
        "WBC": "wbc", "Fibrinogen": "fibrinogen", "Platelets": "platelets",
    }

    lookback = max(0, target_idx - 6)
    recent_rows = rows[lookback:target_idx + 1]

    combined = {}
    for psv_col, api_field in {**vital_map, **lab_map}.items():
        val = None
        for r in reversed(recent_rows):
            val = safe_float(r.get(psv_col))
            if val is not None:
                break
        if val is not None:
            combined[api_field] = val

    age = safe_float(rows[target_idx].get("Age"))
    gender_code = safe_float(rows[target_idx].get("Gender"))
    gender = "Male" if gender_code == 1 else "Female" if gender_code == 0 else None
    if age is not None:
        combined["age"] = age
    if gender is not None:
        combined["gender"] = gender

    return combined


def build_trend_note(rows, target_idx):
    lookback = max(0, target_idx - 5)
    recent = rows[lookback:target_idx + 1]
    notes = []

    hr_vals = [safe_float(r.get("HR")) for r in recent if safe_float(r.get("HR")) is not None]
    sbp_vals = [safe_float(r.get("SBP")) for r in recent if safe_float(r.get("SBP")) is not None]
    temp_vals = [safe_float(r.get("Temp")) for r in recent if safe_float(r.get("Temp")) is not None]
    resp_vals = [safe_float(r.get("Resp")) for r in recent if safe_float(r.get("Resp")) is not None]
    o2_vals = [safe_float(r.get("O2Sat")) for r in recent if safe_float(r.get("O2Sat")) is not None]

    if len(hr_vals) >= 2:
        delta = hr_vals[-1] - hr_vals[0]
        direction = "rising" if delta > 5 else "falling" if delta < -5 else "stable"
        notes.append(f"HR trend {direction} ({hr_vals[0]:.0f} -> {hr_vals[-1]:.0f})")

    if len(sbp_vals) >= 2:
        delta = sbp_vals[-1] - sbp_vals[0]
        direction = "dropping" if delta < -5 else "rising" if delta > 5 else "stable"
        notes.append(f"SBP trend {direction} ({sbp_vals[0]:.0f} -> {sbp_vals[-1]:.0f})")

    if len(temp_vals) >= 1:
        t = temp_vals[-1]
        if t > 38.3:
            notes.append(f"Febrile at {t:.1f}C")
        elif t < 36.0:
            notes.append(f"Hypothermic at {t:.1f}C")
        else:
            notes.append(f"Temp {t:.1f}C within normal range")

    if len(resp_vals) >= 2:
        delta = resp_vals[-1] - resp_vals[0]
        if delta > 3:
            notes.append(f"Respiratory rate increasing ({resp_vals[0]:.0f} -> {resp_vals[-1]:.0f})")
        elif resp_vals[-1] > 22:
            notes.append(f"Tachypneic at RR {resp_vals[-1]:.0f}")

    if len(o2_vals) >= 1 and o2_vals[-1] < 94:
        notes.append(f"O2 sat concerning at {o2_vals[-1]:.0f}%")

    return ". ".join(notes) if notes else "Routine monitoring, vitals within expected range."


def main():
    print("=" * 60)
    print("Phase 1: Select Validation Cohort from Local Data")
    print("=" * 60)

    psv_files = sorted([f for f in os.listdir(RAW_DIR) if f.endswith(".psv")])
    print(f"Found {len(psv_files)} local .psv files\n")

    sepsis_candidates = []
    non_sepsis_candidates = []

    for i, filename in enumerate(psv_files):
        filepath = RAW_DIR / filename
        rows = parse_psv(filepath)

        onset = get_sepsis_onset_index(rows)
        if onset is not None and onset >= 3:
            sepsis_candidates.append({"filename": filename, "rows": rows, "onset_idx": onset})
        elif onset is None and len(rows) >= 6:
            non_sepsis_candidates.append({"filename": filename, "rows": rows})

        if (i + 1) % 200 == 0:
            print(f"  Scanned {i+1}/{len(psv_files)} | "
                  f"Sepsis candidates: {len(sepsis_candidates)} | "
                  f"Non-sepsis candidates: {len(non_sepsis_candidates)}")

    print(f"\nScan complete:")
    print(f"  Sepsis candidates: {len(sepsis_candidates)}")
    print(f"  Non-sepsis candidates: {len(non_sepsis_candidates)}")

    random.seed(42)
    selected_sepsis = sepsis_candidates[:TARGET_SEPSIS]
    random.shuffle(non_sepsis_candidates)
    selected_non_sepsis = non_sepsis_candidates[:TARGET_NON_SEPSIS]

    print(f"\nSelected cohort:")
    print(f"  Sepsis: {len(selected_sepsis)}")
    print(f"  Non-sepsis: {len(selected_non_sepsis)}")
    print(f"  Total: {len(selected_sepsis) + len(selected_non_sepsis)}")

    cohort_manifest = []

    for p in selected_sepsis:
        patient_id = p["filename"].replace(".psv", "")
        snapshot = extract_snapshot(p["rows"], p["onset_idx"])
        notes = build_trend_note(p["rows"], p["onset_idx"])

        api_input = {"patient_id": patient_id, "vitals": snapshot, "notes": notes}
        ground_truth = {
            "patient_id": patient_id,
            "actual_sepsis": True,
            "onset_hour": p["onset_idx"],
            "total_hours": len(p["rows"]),
            "age": snapshot.get("age"),
            "gender": snapshot.get("gender")
        }

        out_path = COHORT_DIR / f"{patient_id}.json"
        with open(out_path, "w") as f:
            json.dump({"api_input": api_input, "ground_truth": ground_truth}, f, indent=2)
        cohort_manifest.append(ground_truth)

    for p in selected_non_sepsis:
        patient_id = p["filename"].replace(".psv", "")
        mid_idx = len(p["rows"]) // 2
        snapshot = extract_snapshot(p["rows"], mid_idx)
        notes = build_trend_note(p["rows"], mid_idx)

        api_input = {"patient_id": patient_id, "vitals": snapshot, "notes": notes}
        ground_truth = {
            "patient_id": patient_id,
            "actual_sepsis": False,
            "snapshot_hour": mid_idx,
            "total_hours": len(p["rows"]),
            "age": snapshot.get("age"),
            "gender": snapshot.get("gender")
        }

        out_path = COHORT_DIR / f"{patient_id}.json"
        with open(out_path, "w") as f:
            json.dump({"api_input": api_input, "ground_truth": ground_truth}, f, indent=2)
        cohort_manifest.append(ground_truth)

    manifest_path = COHORT_DIR / "cohort_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "total_patients": len(cohort_manifest),
            "sepsis_positive": len(selected_sepsis),
            "sepsis_negative": len(selected_non_sepsis),
            "source": "PhysioNet Challenge 2019 - Training Set A",
            "url": "https://physionet.org/content/challenge-2019/",
            "selection_seed": 42,
            "patients": cohort_manifest
        }, f, indent=2)

    print(f"\nCohort saved to: {COHORT_DIR}")
    print(f"Manifest: {manifest_path}")

    ages = [p.get("age") for p in cohort_manifest if p.get("age") is not None]
    print(f"\nCohort demographics:")
    print(f"  Age range: {min(ages):.0f} - {max(ages):.0f}")
    print(f"  Mean age: {sum(ages)/len(ages):.1f}")
    genders = [p.get("gender") for p in cohort_manifest if p.get("gender")]
    males = sum(1 for g in genders if g == "Male")
    print(f"  Male: {males}, Female: {len(genders) - males}")

    print("\nPhase 1 Complete!")


if __name__ == "__main__":
    main()
