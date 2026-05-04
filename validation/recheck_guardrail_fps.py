"""
Re-runs only the 56 guardrail-override false positives from Round 2,
capturing the LLM's ORIGINAL risk score (before guardrail bumped it).
This answers: "Would these patients still be FPs without guardrails?"
"""

import os
import json
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env", override=True)

R2_FILE = Path(__file__).parent / "results" / "validation_results_20260429_175253.json"
COHORT_DIR = Path(__file__).parent / "selected_cohort_v2"
OUT_FILE = Path(__file__).parent / "results" / "guardrail_recheck.json"

API_URL = "http://localhost:8000/classify"
API_KEY = os.getenv("API_KEY", "sepsis_api_key_2024")


def main():
    with open(R2_FILE) as f:
        d = json.load(f)
    fps = [r for r in d["results"] if not r["actual_sepsis"] and r["predicted_sepsis"] == True]
    guardrail_fps = [r for r in fps if r.get("guardrail_override")]
    print(f"Re-running {len(guardrail_fps)} guardrail-override FPs to capture original LLM risk score\n")

    headers = {"x-api-key": API_KEY, "Content-Type": "application/json"}
    recheck = []

    for i, r in enumerate(guardrail_fps):
        pid = r["patient_id"]
        with open(COHORT_DIR / f"{pid}.json") as pf:
            api_input = json.load(pf)["api_input"]

        try:
            resp = requests.post(API_URL, json=api_input, headers=headers, timeout=120)
            if resp.status_code != 200:
                print(f"  [{i+1}/{len(guardrail_fps)}] {pid}: HTTP {resp.status_code}")
                continue
            data = resp.json()
            original = data.get("original_risk_score")
            final = data.get("risk_score")
            guard = data.get("guardrail_override")
            reasons = data.get("override_reasons", [])

            row = {
                "patient_id": pid,
                "original_llm_risk": original,
                "final_risk_after_guardrail": final,
                "guardrail_override": guard,
                "override_reasons": reasons,
                "would_still_be_fp_without_guardrail": (original is not None and original >= 50),
            }
            recheck.append(row)
            mark = "STILL FP" if row["would_still_be_fp_without_guardrail"] else "FIXED!"
            print(f"  [{i+1:2d}/{len(guardrail_fps)}] {pid}: orig_LLM={original} → final={final} | {mark}")
        except Exception as e:
            print(f"  [{i+1}/{len(guardrail_fps)}] {pid}: ERROR {e}")

        time.sleep(2)
        if (i + 1) % 10 == 0:
            print(f"  --- pause 10s ---")
            time.sleep(10)

    with open(OUT_FILE, "w") as f:
        json.dump(recheck, f, indent=2)

    still_fp = sum(1 for r in recheck if r["would_still_be_fp_without_guardrail"])
    fixed = len(recheck) - still_fp

    print()
    print("=" * 60)
    print("  GUARDRAIL ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"  Total guardrail-override FPs re-checked: {len(recheck)}")
    print(f"  Would STILL be FP without guardrails:    {still_fp} ({still_fp/len(recheck)*100:.1f}%)")
    print(f"  Would be FIXED by removing guardrails:   {fixed} ({fixed/len(recheck)*100:.1f}%)")
    print()
    print("  → Removing/loosening guardrails would only fix:", fixed, "FPs")
    print("  → The remaining", still_fp, "FPs are LLM-driven (would still flag positive)")
    print(f"  Saved: {OUT_FILE}")


if __name__ == "__main__":
    main()
