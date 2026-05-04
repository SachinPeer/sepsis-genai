"""
Phase 0 + 1: Download PhysioNet Challenge 2019 data and select validation cohort.
Downloads .psv files, scans for sepsis/non-sepsis, selects 340 patients.
"""

import os
import csv
import json
import random
import time
import urllib.request
import re
from pathlib import Path

BASE_URL = "https://physionet.org/files/challenge-2019/1.0.0/training/training_setA"
RAW_DIR = Path(__file__).parent / "raw_data"
COHORT_DIR = Path(__file__).parent / "selected_cohort"
RAW_DIR.mkdir(exist_ok=True)
COHORT_DIR.mkdir(exist_ok=True)

TARGET_SEPSIS = 140
TARGET_NON_SEPSIS = 200
SCAN_LIMIT = 5000


def list_patient_files():
    """Scrape the directory listing for .psv filenames."""
    url = f"{BASE_URL}/"
    print(f"Fetching file list from {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        html = resp.read().decode()
    files = re.findall(r'href="(p\d+\.psv)"', html)
    print(f"Found {len(files)} patient files")
    return sorted(files)


def download_file(filename):
    """Download a single .psv file if not already cached."""
    local_path = RAW_DIR / filename
    if local_path.exists() and local_path.stat().st_size > 100:
        return local_path
    url = f"{BASE_URL}/{filename}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
    local_path.write_bytes(data)
    return local_path


def parse_psv(filepath):
    """Parse a pipe-separated values file into list of dicts."""
    rows = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            rows.append(row)
    return rows


def is_sepsis_patient(rows):
    """Check if any row has SepsisLabel == 1."""
    return any(row.get("SepsisLabel", "0").strip() == "1" for row in rows)


def get_sepsis_onset_index(rows):
    """Return the index of the first row where SepsisLabel == 1."""
    for i, row in enumerate(rows):
        if row.get("SepsisLabel", "0").strip() == "1":
            return i
    return None


def safe_float(val):
    """Convert a string to float, return None for NaN or empty."""
    if val is None or val.strip() == "" or val.strip() == "NaN":
        return None
    try:
        return float(val)
    except ValueError:
        return None


def extract_snapshot(rows, target_idx):
    """
    Extract a clinical snapshot for API input.
    Uses data at target_idx, forward-fills from recent rows for missing values.
    """
    target = rows[target_idx]

    vitals = {}
    labs = {}

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

    for psv_col, api_field in vital_map.items():
        val = None
        for r in reversed(recent_rows):
            val = safe_float(r.get(psv_col))
            if val is not None:
                break
        if val is not None:
            vitals[api_field] = val

    for psv_col, api_field in lab_map.items():
        val = None
        for r in reversed(recent_rows):
            val = safe_float(r.get(psv_col))
            if val is not None:
                break
        if val is not None:
            labs[api_field] = val

    age = safe_float(target.get("Age"))
    gender_code = safe_float(target.get("Gender"))
    gender = "Male" if gender_code == 1 else "Female" if gender_code == 0 else None

    combined = {**vitals, **labs}
    if age is not None:
        combined["age"] = age
    if gender is not None:
        combined["gender"] = gender

    return combined


def build_trend_note(rows, target_idx):
    """Build a synthetic clinician note from vital sign trends."""
    lookback = max(0, target_idx - 5)
    recent = rows[lookback:target_idx + 1]

    notes = []
    hr_vals = [safe_float(r.get("HR")) for r in recent if safe_float(r.get("HR")) is not None]
    sbp_vals = [safe_float(r.get("SBP")) for r in recent if safe_float(r.get("SBP")) is not None]
    temp_vals = [safe_float(r.get("Temp")) for r in recent if safe_float(r.get("Temp")) is not None]
    resp_vals = [safe_float(r.get("Resp")) for r in recent if safe_float(r.get("Resp")) is not None]

    if len(hr_vals) >= 2:
        delta = hr_vals[-1] - hr_vals[0]
        direction = "rising" if delta > 5 else "falling" if delta < -5 else "stable"
        notes.append(f"HR trend {direction} ({hr_vals[0]:.0f} -> {hr_vals[-1]:.0f})")

    if len(sbp_vals) >= 2:
        delta = sbp_vals[-1] - sbp_vals[0]
        direction = "rising" if delta > 5 else "falling" if delta < -5 else "stable"
        notes.append(f"SBP trend {direction} ({sbp_vals[0]:.0f} -> {sbp_vals[-1]:.0f})")

    if len(temp_vals) >= 1:
        t = temp_vals[-1]
        if t > 38.3:
            notes.append(f"Febrile at {t:.1f}C")
        elif t < 36.0:
            notes.append(f"Hypothermic at {t:.1f}C")

    if len(resp_vals) >= 2:
        delta = resp_vals[-1] - resp_vals[0]
        if delta > 3:
            notes.append(f"RR increasing ({resp_vals[0]:.0f} -> {resp_vals[-1]:.0f})")

    return ". ".join(notes) if notes else "Routine monitoring, no acute changes noted."


def main():
    print("=" * 60)
    print("PhysioNet Sepsis Validation - Download & Select Cohort")
    print("=" * 60)

    all_files = list_patient_files()

    random.seed(42)
    files_to_scan = all_files[:SCAN_LIMIT]

    sepsis_patients = []
    non_sepsis_patients = []
    downloaded = 0
    errors = 0

    print(f"\nScanning up to {SCAN_LIMIT} patients to find cohort...")
    print(f"Target: {TARGET_SEPSIS} sepsis + {TARGET_NON_SEPSIS} non-sepsis = {TARGET_SEPSIS + TARGET_NON_SEPSIS} total\n")

    for i, filename in enumerate(files_to_scan):
        if len(sepsis_patients) >= TARGET_SEPSIS and len(non_sepsis_patients) >= TARGET_NON_SEPSIS:
            break

        try:
            filepath = download_file(filename)
            downloaded += 1
            rows = parse_psv(filepath)

            if is_sepsis_patient(rows):
                if len(sepsis_patients) < TARGET_SEPSIS:
                    onset_idx = get_sepsis_onset_index(rows)
                    if onset_idx and onset_idx >= 3:
                        sepsis_patients.append({
                            "filename": filename,
                            "rows": rows,
                            "onset_idx": onset_idx,
                            "is_sepsis": True
                        })
            else:
                if len(non_sepsis_patients) < TARGET_NON_SEPSIS:
                    if len(rows) >= 6:
                        mid_idx = len(rows) // 2
                        non_sepsis_patients.append({
                            "filename": filename,
                            "rows": rows,
                            "target_idx": mid_idx,
                            "is_sepsis": False
                        })

            if (i + 1) % 100 == 0:
                print(f"  Scanned {i+1}/{SCAN_LIMIT} | "
                      f"Sepsis: {len(sepsis_patients)}/{TARGET_SEPSIS} | "
                      f"Non-sepsis: {len(non_sepsis_patients)}/{TARGET_NON_SEPSIS}")

        except Exception as e:
            errors += 1
            if errors < 5:
                print(f"  Error on {filename}: {e}")
            continue

    print(f"\n--- Download Complete ---")
    print(f"Files downloaded: {downloaded}")
    print(f"Sepsis patients found: {len(sepsis_patients)}")
    print(f"Non-sepsis patients found: {len(non_sepsis_patients)}")
    print(f"Errors: {errors}")

    print(f"\n--- Building Validation Cohort ---")
    cohort_manifest = []

    for p in sepsis_patients:
        patient_id = p["filename"].replace(".psv", "")
        snapshot = extract_snapshot(p["rows"], p["onset_idx"])
        notes = build_trend_note(p["rows"], p["onset_idx"])

        api_input = {
            "patient_id": patient_id,
            "vitals": snapshot,
            "notes": notes
        }

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

    for p in non_sepsis_patients:
        patient_id = p["filename"].replace(".psv", "")
        snapshot = extract_snapshot(p["rows"], p["target_idx"])
        notes = build_trend_note(p["rows"], p["target_idx"])

        api_input = {
            "patient_id": patient_id,
            "vitals": snapshot,
            "notes": notes
        }

        ground_truth = {
            "patient_id": patient_id,
            "actual_sepsis": False,
            "snapshot_hour": p["target_idx"],
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
            "sepsis_positive": len(sepsis_patients),
            "sepsis_negative": len(non_sepsis_patients),
            "source": "PhysioNet Challenge 2019 - Training Set A",
            "selection_seed": 42,
            "patients": cohort_manifest
        }, f, indent=2)

    print(f"\nCohort saved to: {COHORT_DIR}")
    print(f"Manifest: {manifest_path}")
    print(f"Total patients: {len(cohort_manifest)}")
    print(f"  Sepsis-positive: {len(sepsis_patients)}")
    print(f"  Non-sepsis: {len(non_sepsis_patients)}")
    print("\nPhase 0 + 1 Complete!")


if __name__ == "__main__":
    main()
