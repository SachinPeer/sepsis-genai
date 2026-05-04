"""
Phase 3: Run validation cohort through the local API.
Calls /classify for each of 340 patients and records results.
"""

import os
import json
import time
import csv
import requests
from pathlib import Path
from datetime import datetime

COHORT_DIR = Path(__file__).parent / "selected_cohort_v4"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

API_URL = "http://localhost:8000/classify"
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path, override=True)
API_KEY = os.getenv("API_KEY", "sepsis_api_key_2024")

BATCH_SIZE = 10
DELAY_BETWEEN_CALLS = 2
DELAY_BETWEEN_BATCHES = 10


def load_cohort():
    manifest_path = COHORT_DIR / "cohort_manifest.json"
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    return manifest


def call_api(api_input):
    headers = {"x-api-key": API_KEY, "Content-Type": "application/json"}
    try:
        resp = requests.post(API_URL, json=api_input, headers=headers, timeout=120)
        if resp.status_code == 200:
            return resp.json(), None
        else:
            return None, f"HTTP {resp.status_code}: {resp.text[:200]}"
    except requests.exceptions.Timeout:
        return None, "Request timeout (120s)"
    except Exception as e:
        return None, str(e)


def classify_prediction(result):
    """Determine if our system predicted sepsis based on risk_score and priority."""
    risk_score = result.get("risk_score", 0)
    priority = result.get("priority", "Standard")
    alert_level = result.get("alert_level", "LOW")

    if risk_score >= 50 or priority in ("High", "Critical") or alert_level in ("HIGH", "CRITICAL"):
        return True
    return False


def main():
    print("=" * 60)
    print("Phase 3: Validation Run — 340 Patients Through API")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    manifest = load_cohort()
    patients = manifest["patients"]
    total = len(patients)

    print(f"Cohort: {total} patients ({manifest['sepsis_positive']} sepsis, {manifest['sepsis_negative']} non-sepsis)")
    print(f"API: {API_URL}")
    print(f"Delay: {DELAY_BETWEEN_CALLS}s between calls, {DELAY_BETWEEN_BATCHES}s between batches of {BATCH_SIZE}\n")

    results = []
    success = 0
    errors = 0

    for i, patient_info in enumerate(patients):
        patient_id = patient_info["patient_id"]
        patient_file = COHORT_DIR / f"{patient_id}.json"

        with open(patient_file, "r") as f:
            data = json.load(f)

        api_input = data["api_input"]
        ground_truth = data["ground_truth"]

        start_time = time.time()
        api_result, error = call_api(api_input)
        elapsed = (time.time() - start_time) * 1000

        if api_result:
            predicted_sepsis = classify_prediction(api_result)

            result_row = {
                "patient_id": patient_id,
                "actual_sepsis": ground_truth["actual_sepsis"],
                "predicted_sepsis": predicted_sepsis,
                "risk_score": api_result.get("risk_score", "N/A"),
                "priority": api_result.get("priority", "N/A"),
                "confidence_level": api_result.get("confidence_level", "N/A"),
                "alert_level": api_result.get("alert_level", "N/A"),
                "guardrail_override": api_result.get("guardrail_override", False),
                "qsofa_score": api_result.get("clinical_scores", {}).get("qSOFA", {}).get("score", "N/A"),
                "sirs_met": api_result.get("clinical_scores", {}).get("SIRS", {}).get("criteria_met", "N/A"),
                "sofa_score": api_result.get("clinical_scores", {}).get("SOFA", {}).get("score", "N/A"),
                "clinical_rationale": api_result.get("clinical_rationale", "")[:200],
                "processing_time_ms": round(elapsed, 1),
                "age": ground_truth.get("age", "N/A"),
                "gender": ground_truth.get("gender", "N/A"),
                "status": "success",
                "error": ""
            }
            success += 1
        else:
            result_row = {
                "patient_id": patient_id,
                "actual_sepsis": ground_truth["actual_sepsis"],
                "predicted_sepsis": "ERROR",
                "risk_score": "N/A",
                "priority": "N/A",
                "confidence_level": "N/A",
                "alert_level": "N/A",
                "guardrail_override": "N/A",
                "qsofa_score": "N/A",
                "sirs_met": "N/A",
                "sofa_score": "N/A",
                "clinical_rationale": "",
                "processing_time_ms": round(elapsed, 1),
                "age": ground_truth.get("age", "N/A"),
                "gender": ground_truth.get("gender", "N/A"),
                "status": "error",
                "error": error
            }
            errors += 1

        results.append(result_row)

        status_char = "+" if result_row.get("predicted_sepsis") == True else "-" if result_row.get("predicted_sepsis") == False else "!"
        actual_char = "S" if ground_truth["actual_sepsis"] else "N"
        print(f"  [{i+1:3d}/{total}] {patient_id} | Actual:{actual_char} Pred:{status_char} | "
              f"Risk:{result_row['risk_score']:>3} | {result_row['priority']:>8} | "
              f"{elapsed:.0f}ms", flush=True)

        if (i + 1) % BATCH_SIZE == 0 and (i + 1) < total:
            print(f"  --- Batch pause ({DELAY_BETWEEN_BATCHES}s) | Success: {success}, Errors: {errors} ---", flush=True)
            time.sleep(DELAY_BETWEEN_BATCHES)
        else:
            time.sleep(DELAY_BETWEEN_CALLS)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"validation_results_{timestamp}.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    json_path = RESULTS_DIR / f"validation_results_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump({
            "run_timestamp": timestamp,
            "total_patients": total,
            "successful": success,
            "errors": errors,
            "results": results
        }, f, indent=2)

    latest_csv = RESULTS_DIR / "validation_results_latest.csv"
    latest_json = RESULTS_DIR / "validation_results_latest.json"
    import shutil
    shutil.copy(csv_path, latest_csv)
    shutil.copy(json_path, latest_json)

    print(f"\n{'=' * 60}")
    print(f"Validation Run Complete")
    print(f"  Total: {total} | Success: {success} | Errors: {errors}")
    print(f"  Results: {csv_path}")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
