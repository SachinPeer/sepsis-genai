"""
Phase 1 v3: Select validation cohort with TREND DATA + ICU CONTEXT.
Adds clinical context note that these are ICU patients admitted for various reasons,
not necessarily sepsis — testing whether context improves specificity.
"""

import os
import csv
import json
import random
from pathlib import Path
from datetime import datetime, timedelta

RAW_DIR = Path(__file__).parent / "raw_data"
COHORT_DIR = Path(__file__).parent / "selected_cohort_v3"
COHORT_DIR.mkdir(exist_ok=True)

TARGET_SEPSIS = 140
TARGET_NON_SEPSIS = 200
TREND_HOURS = 6


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


def extract_trend_vitals(rows, target_idx):
    """Extract 6 hours of trend data in our API's expected format."""
    start_idx = max(0, target_idx - TREND_HOURS + 1)
    trend_rows = rows[start_idx:target_idx + 1]

    base_time = datetime(2026, 4, 29, 8, 0, 0)

    vital_cols = {
        "HR": "HR", "SBP": "SBP", "DBP": "DBP",
        "Temp": "Temp", "Resp": "Resp", "O2Sat": "O2Sat", "MAP": "MAP"
    }
    lab_cols = {
        "WBC": "WBC", "Lactate": "Lactate", "Creatinine": "Creatinine",
        "Platelets": "Platelets", "Bilirubin_total": "Bilirubin_total",
        "BUN": "BUN", "Glucose": "Glucose", "pH": "pH",
        "BaseExcess": "BaseExcess", "HCO3": "HCO3", "PaCO2": "PaCO2",
        "AST": "AST", "Potassium": "Potassium", "Hgb": "Hgb",
        "PTT": "PTT", "Fibrinogen": "Fibrinogen", "TroponinI": "TroponinI",
        "FiO2": "FiO2",
    }

    vitals = {}

    for col_name, api_name in vital_cols.items():
        trend_points = []
        for j, r in enumerate(trend_rows):
            val = safe_float(r.get(col_name))
            if val is not None:
                hours_offset = -(len(trend_rows) - 1 - j)
                ts = (base_time + timedelta(hours=hours_offset)).strftime("%Y-%m-%dT%H:%M")
                trend_points.append({"val": val, "ts": ts})
        if trend_points:
            vitals[api_name] = trend_points

    for col_name, api_name in lab_cols.items():
        latest_val = None
        for r in reversed(trend_rows):
            latest_val = safe_float(r.get(col_name))
            if latest_val is not None:
                break
        if latest_val is not None:
            vitals[api_name] = [latest_val]

    age = safe_float(rows[target_idx].get("Age"))
    gender_code = safe_float(rows[target_idx].get("Gender"))
    if age is not None:
        vitals["Age"] = age
    if gender_code is not None:
        vitals["Gender"] = "Male" if gender_code == 1 else "Female"

    return vitals


def build_rich_notes(rows, target_idx):
    """Build detailed nurse notes from 6 hours of trend data."""
    start_idx = max(0, target_idx - TREND_HOURS + 1)
    trend_rows = rows[start_idx:target_idx + 1]
    notes_parts = []

    def get_vals(col):
        return [(i, safe_float(r.get(col))) for i, r in enumerate(trend_rows)
                if safe_float(r.get(col)) is not None]

    hr_data = get_vals("HR")
    sbp_data = get_vals("SBP")
    temp_data = get_vals("Temp")
    resp_data = get_vals("Resp")
    o2_data = get_vals("O2Sat")
    map_data = get_vals("MAP")
    lac_data = get_vals("Lactate")
    wbc_data = get_vals("WBC")
    cr_data = get_vals("Creatinine")
    plt_data = get_vals("Platelets")

    if len(hr_data) >= 2:
        first, last = hr_data[0][1], hr_data[-1][1]
        delta = last - first
        if delta > 10:
            notes_parts.append(f"Heart rate rising significantly over past {len(trend_rows)} hours ({first:.0f} -> {last:.0f} bpm)")
        elif delta < -10:
            notes_parts.append(f"Heart rate falling over past hours ({first:.0f} -> {last:.0f} bpm)")
        elif last > 100:
            notes_parts.append(f"Persistent tachycardia, HR {last:.0f} bpm")
        elif last < 50:
            notes_parts.append(f"Bradycardic, HR {last:.0f} bpm")
    elif len(hr_data) == 1:
        v = hr_data[0][1]
        if v > 100:
            notes_parts.append(f"Tachycardic at HR {v:.0f}")

    if len(sbp_data) >= 2:
        first, last = sbp_data[0][1], sbp_data[-1][1]
        delta = last - first
        if delta < -15:
            notes_parts.append(f"Blood pressure dropping ({first:.0f} -> {last:.0f} mmHg systolic), concerning for hemodynamic instability")
        elif last < 90:
            notes_parts.append(f"Hypotensive, SBP {last:.0f} mmHg")
        elif delta > 15:
            notes_parts.append(f"BP rising ({first:.0f} -> {last:.0f} systolic)")

    if len(map_data) >= 1:
        last_map = map_data[-1][1]
        if last_map < 65:
            notes_parts.append(f"MAP {last_map:.0f} below target of 65, vasopressor may be needed")

    if len(temp_data) >= 1:
        last_temp = temp_data[-1][1]
        if last_temp > 38.3:
            notes_parts.append(f"Febrile at {last_temp:.1f}C, consider infectious workup")
        elif last_temp < 36.0:
            notes_parts.append(f"Hypothermic at {last_temp:.1f}C, concerning in setting of possible infection")
        if len(temp_data) >= 2:
            first_t = temp_data[0][1]
            if last_temp - first_t > 1.0:
                notes_parts.append(f"Temperature trending up ({first_t:.1f} -> {last_temp:.1f}C)")

    if len(resp_data) >= 2:
        first, last = resp_data[0][1], resp_data[-1][1]
        if last > 22:
            notes_parts.append(f"Tachypneic, RR {last:.0f} (was {first:.0f})")
        if last - first > 5:
            notes_parts.append(f"Respiratory rate increasing, work of breathing noted")
    elif len(resp_data) == 1 and resp_data[0][1] > 22:
        notes_parts.append(f"Elevated respiratory rate at {resp_data[0][1]:.0f}")

    if len(o2_data) >= 1:
        last_o2 = o2_data[-1][1]
        if last_o2 < 92:
            notes_parts.append(f"Desaturation noted, SpO2 {last_o2:.0f}%, supplemental O2 adjusted")
        elif last_o2 < 95:
            notes_parts.append(f"SpO2 borderline at {last_o2:.0f}%")

    if len(lac_data) >= 1:
        lac = lac_data[-1][1]
        if lac > 4:
            notes_parts.append(f"Lactate critically elevated at {lac:.1f} mmol/L, tissue hypoperfusion suspected")
        elif lac > 2:
            notes_parts.append(f"Lactate mildly elevated at {lac:.1f} mmol/L")

    if len(wbc_data) >= 1:
        wbc = wbc_data[-1][1]
        if wbc > 12:
            notes_parts.append(f"Leukocytosis, WBC {wbc:.1f}")
        elif wbc < 4:
            notes_parts.append(f"Leukopenia, WBC {wbc:.1f}, immunocompromised concern")

    if len(cr_data) >= 1:
        cr = cr_data[-1][1]
        if cr > 1.5:
            notes_parts.append(f"Creatinine elevated at {cr:.1f}, monitoring renal function")

    if len(plt_data) >= 1:
        plt = plt_data[-1][1]
        if plt < 100:
            notes_parts.append(f"Thrombocytopenia, platelets {plt:.0f}")

    combined_flags = []
    last_hr = hr_data[-1][1] if hr_data else None
    last_sbp = sbp_data[-1][1] if sbp_data else None
    last_lac = lac_data[-1][1] if lac_data else None
    last_temp_v = temp_data[-1][1] if temp_data else None

    if last_hr and last_sbp and last_hr > 100 and last_sbp < 100:
        combined_flags.append("Tachycardia with hypotension — possible compensated shock pattern")
    if last_lac and last_lac > 2 and last_temp_v and last_temp_v > 38.3:
        combined_flags.append("Elevated lactate with fever — high suspicion for sepsis")
    if last_hr and last_sbp and len(hr_data) >= 2 and len(sbp_data) >= 2:
        hr_delta = hr_data[-1][1] - hr_data[0][1]
        sbp_delta = sbp_data[-1][1] - sbp_data[0][1]
        if hr_delta > 10 and sbp_delta < -10:
            combined_flags.append("Diverging HR/BP trends — hemodynamic deterioration pattern")

    if combined_flags:
        notes_parts.extend(combined_flags)

    icu_context = ("Patient is in ICU for close monitoring. Admission may be for surgical recovery, "
                   "trauma, cardiac event, respiratory failure, or other non-infectious cause — "
                   "not necessarily sepsis. Abnormal vitals may reflect underlying ICU condition rather than new infection. ")

    if not notes_parts:
        return icu_context + "Routine monitoring. Vitals within expected range for ICU baseline. No acute changes noted."

    return icu_context + ". ".join(notes_parts) + "."


def main():
    print("=" * 60)
    print("Phase 1 v3: Select Cohort with TREND DATA + ICU CONTEXT")
    print("=" * 60)

    psv_files = sorted([f for f in os.listdir(RAW_DIR) if f.endswith(".psv")])
    print(f"Found {len(psv_files)} local .psv files\n")

    sepsis_candidates = []
    non_sepsis_candidates = []

    for i, filename in enumerate(psv_files):
        filepath = RAW_DIR / filename
        rows = parse_psv(filepath)

        onset = get_sepsis_onset_index(rows)
        if onset is not None and onset >= TREND_HOURS:
            sepsis_candidates.append({"filename": filename, "rows": rows, "onset_idx": onset})
        elif onset is None and len(rows) >= TREND_HOURS + 3:
            non_sepsis_candidates.append({"filename": filename, "rows": rows})

        if (i + 1) % 400 == 0:
            print(f"  Scanned {i+1}/{len(psv_files)} | "
                  f"Sepsis: {len(sepsis_candidates)} | Non-sepsis: {len(non_sepsis_candidates)}")

    print(f"\nScan complete: {len(sepsis_candidates)} sepsis, {len(non_sepsis_candidates)} non-sepsis")

    random.seed(42)
    selected_sepsis = sepsis_candidates[:TARGET_SEPSIS]
    random.shuffle(non_sepsis_candidates)
    selected_non_sepsis = non_sepsis_candidates[:TARGET_NON_SEPSIS]

    print(f"Selected: {len(selected_sepsis)} sepsis + {len(selected_non_sepsis)} non-sepsis = {len(selected_sepsis) + len(selected_non_sepsis)} total")

    cohort_manifest = []

    for p in selected_sepsis:
        patient_id = p["filename"].replace(".psv", "")
        vitals = extract_trend_vitals(p["rows"], p["onset_idx"])
        notes = build_rich_notes(p["rows"], p["onset_idx"])

        api_input = {"patient_id": patient_id, "vitals": vitals, "notes": notes}
        ground_truth = {
            "patient_id": patient_id, "actual_sepsis": True,
            "onset_hour": p["onset_idx"], "total_hours": len(p["rows"]),
            "age": vitals.get("Age"), "gender": vitals.get("Gender")
        }

        out_path = COHORT_DIR / f"{patient_id}.json"
        with open(out_path, "w") as f:
            json.dump({"api_input": api_input, "ground_truth": ground_truth}, f, indent=2)
        cohort_manifest.append(ground_truth)

    for p in selected_non_sepsis:
        patient_id = p["filename"].replace(".psv", "")
        mid_idx = len(p["rows"]) // 2
        vitals = extract_trend_vitals(p["rows"], mid_idx)
        notes = build_rich_notes(p["rows"], mid_idx)

        api_input = {"patient_id": patient_id, "vitals": vitals, "notes": notes}
        ground_truth = {
            "patient_id": patient_id, "actual_sepsis": False,
            "snapshot_hour": mid_idx, "total_hours": len(p["rows"]),
            "age": vitals.get("Age"), "gender": vitals.get("Gender")
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
            "version": "v3 - 6-hour trend data + ICU context notes",
            "trend_hours": TREND_HOURS,
            "selection_seed": 42,
            "patients": cohort_manifest
        }, f, indent=2)

    print(f"\nCohort saved to: {COHORT_DIR}")
    print(f"Total: {len(cohort_manifest)} patients with {TREND_HOURS}-hour trend data")

    sample = COHORT_DIR / f"{selected_sepsis[0]['filename'].replace('.psv', '')}.json"
    with open(sample, "r") as f:
        s = json.load(f)
    print(f"\nSample patient notes ({s['ground_truth']['patient_id']}):")
    print(f"  {s['api_input']['notes'][:200]}...")


if __name__ == "__main__":
    main()
