"""
eICU Validation Runner — sends each patient in validation/eicu_cohort through
the local /genai-classify API and collects results.

Converts our cohort's {patient_demographics, patient_vitals, patient_notes,
ground_truth} structure into the API's flat {vitals: {...}, notes, patient_id}
structure on the fly.

Usage:
  python validation/run_eicu_validation.py                # all patients
  python validation/run_eicu_validation.py --limit 5      # smoke test
"""

from __future__ import annotations
import argparse
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path
import requests
from dotenv import load_dotenv

COHORT_DIR = Path(__file__).parent / "eicu_cohort"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path, override=True)
API_KEY = os.getenv("API_KEY", "sepsis_api_key_2024")
API_URL = "http://localhost:8000/classify"

DELAY_BETWEEN_CALLS = 2


def build_api_input(patient_json: dict) -> dict:
    """Convert our cohort JSON to the /classify input shape. Drops empty-list
    vitals because the downstream clinical scorer doesn't handle them cleanly."""
    demo = patient_json.get("patient_demographics", {})
    raw_vitals = patient_json.get("patient_vitals", {})
    vitals = {k: v for k, v in raw_vitals.items() if v not in (None, [], "")}
    vitals["Age"] = demo.get("Age")
    vitals["Gender"] = demo.get("Gender") or "Unknown"
    return {
        "vitals": vitals,
        "notes": patient_json.get("patient_notes", ""),
        "patient_id": patient_json.get("patient_id", "unknown"),
    }


def call_api(api_input):
    headers = {"x-api-key": API_KEY, "Content-Type": "application/json"}
    try:
        resp = requests.post(API_URL, json=api_input, headers=headers, timeout=120)
        if resp.status_code == 200:
            return resp.json(), None
        else:
            return None, f"HTTP {resp.status_code}: {resp.text[:300]}"
    except requests.exceptions.Timeout:
        return None, "Request timeout (120s)"
    except Exception as e:
        return None, str(e)


def classify_prediction(result):
    risk = result.get("risk_score", 0)
    prio = result.get("priority", "Standard")
    alert = result.get("alert_level", "LOW")
    try:
        risk = float(risk)
    except Exception:
        risk = 0.0
    if risk >= 50 or prio in ("High", "Critical") or alert in ("HIGH", "CRITICAL"):
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="Smoke-test mode: only first N patients")
    args = ap.parse_args()

    manifest_path = COHORT_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    patients = manifest["patients"]
    if args.limit:
        patients = patients[:args.limit]
    total = len(patients)

    print("=" * 70)
    print("eICU-CRD Demo Validation Run")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Cohort size: {total} patients (sepsis: {sum(1 for p in patients if p['actual_sepsis'])}, "
          f"controls: {sum(1 for p in patients if not p['actual_sepsis'])})")
    print(f"API: {API_URL}")
    print("=" * 70, flush=True)

    results = []
    success = errors = 0
    start_all = time.time()

    for i, pinfo in enumerate(patients):
        pid = pinfo["patient_id"]
        patient_file = COHORT_DIR / pinfo["file"]
        patient_json = json.loads(patient_file.read_text())
        actual_sepsis = pinfo["actual_sepsis"]

        api_input = build_api_input(patient_json)

        t0 = time.time()
        api_result, error = call_api(api_input)
        elapsed = (time.time() - t0) * 1000

        if api_result:
            predicted = classify_prediction(api_result)
            row = {
                "patient_id": pid,
                "actual_sepsis": actual_sepsis,
                "predicted_sepsis": predicted,
                "risk_score": api_result.get("risk_score", "N/A"),
                "priority": api_result.get("priority", "N/A"),
                "confidence_level": api_result.get("confidence_level", "N/A"),
                "alert_level": api_result.get("alert_level", "N/A"),
                "guardrail_override": api_result.get("guardrail_override", False),
                "original_risk_score": api_result.get("original_risk_score"),
                "early_warnings": ", ".join(api_result.get("early_warnings", []) or []),
                "sepsis_probability_6h": api_result.get("sepsis_probability_6h"),
                "qsofa": api_result.get("clinical_scores", {}).get("qSOFA", {}).get("score"),
                "sirs_met": api_result.get("clinical_scores", {}).get("SIRS", {}).get("criteria_met"),
                "sofa": api_result.get("clinical_scores", {}).get("SOFA", {}).get("score"),
                "patientunitstayid": pinfo.get("patientunitstayid"),
                "icd9_codes": ", ".join(pinfo.get("icd9_codes", [])) if actual_sepsis else "",
                "processing_time_ms": round(elapsed, 1),
                "status": "success",
                "error": "",
                "reasoning": (api_result.get("clinical_rationale") or api_result.get("reasoning") or "")[:400],
            }
            success += 1
        else:
            row = {
                "patient_id": pid,
                "actual_sepsis": actual_sepsis,
                "predicted_sepsis": "ERROR",
                "risk_score": "N/A",
                "priority": "N/A",
                "confidence_level": "N/A",
                "alert_level": "N/A",
                "guardrail_override": "N/A",
                "original_risk_score": "N/A",
                "early_warnings": "",
                "sepsis_probability_6h": "N/A",
                "qsofa": "N/A",
                "sirs_met": "N/A",
                "sofa": "N/A",
                "patientunitstayid": pinfo.get("patientunitstayid"),
                "icd9_codes": "",
                "processing_time_ms": round(elapsed, 1),
                "status": "error",
                "error": error,
                "reasoning": "",
            }
            errors += 1

        results.append(row)

        actual_char = "S" if actual_sepsis else "N"
        pred_char = "+" if row["predicted_sepsis"] is True else ("-" if row["predicted_sepsis"] is False else "!")
        note_len = len(patient_json.get("patient_notes", ""))
        print(f"  [{i+1:2d}/{total}] {pid} | actual:{actual_char} pred:{pred_char} | "
              f"risk:{row['risk_score']:>4} pri:{row['priority']:<8} | "
              f"notes:{note_len:>4}ch | {elapsed:.0f}ms", flush=True)

        if i < total - 1:
            time.sleep(DELAY_BETWEEN_CALLS)

    total_elapsed = time.time() - start_all
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"EICU_results_{ts}.csv"
    json_path = RESULTS_DIR / f"EICU_results_{ts}.json"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    with open(json_path, "w") as f:
        json.dump({
            "timestamp": ts,
            "cohort": "eICU-CRD Demo v2.0.1",
            "total": total,
            "success": success,
            "errors": errors,
            "total_elapsed_sec": round(total_elapsed, 1),
            "results": results,
        }, f, indent=2, default=str)

    # Also copy to stable names for downstream analysis
    import shutil
    shutil.copy(csv_path, RESULTS_DIR / "EICU_results_latest.csv")
    shutil.copy(json_path, RESULTS_DIR / "EICU_results_latest.json")

    # Quick summary metrics
    tp = sum(1 for r in results if r["actual_sepsis"] and r["predicted_sepsis"] is True)
    fn = sum(1 for r in results if r["actual_sepsis"] and r["predicted_sepsis"] is False)
    fp = sum(1 for r in results if not r["actual_sepsis"] and r["predicted_sepsis"] is True)
    tn = sum(1 for r in results if not r["actual_sepsis"] and r["predicted_sepsis"] is False)

    print()
    print("=" * 70)
    print("RUN COMPLETE")
    print(f"  Total: {total}  Success: {success}  Errors: {errors}")
    print(f"  Elapsed: {total_elapsed:.1f}s  (avg {total_elapsed/max(1,total):.1f}s/pt)")
    print(f"  TP={tp}  FN={fn}  FP={fp}  TN={tn}")
    if (tp + fn) > 0:
        print(f"  Sensitivity: {100*tp/(tp+fn):.2f}%")
    if (tn + fp) > 0:
        print(f"  Specificity: {100*tn/(tn+fp):.2f}%")
    print(f"  Results: {csv_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
