"""
Smoke test for prompt v3.1: run 10 patients (5 sepsis, 5 non-sepsis) through the
running API and inspect results for evidence of decoupled priority assignment.

The signal we want to see:
  - More patients with risk_score >= 50 but priority="Standard" (decoupling working)
  - Critical/High priority backed by clear rationale (qSOFA/SIRS/discordance)
  - Specificity-leaning behavior on non-sepsis ICU patients
"""

import os
import json
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env", override=True)

COHORT_DIR = Path(__file__).parent / "selected_cohort_v4"
API_URL = "http://localhost:8000/classify"
API_KEY = os.getenv("API_KEY", "sepsis_api_key_2024")

N_PER_CLASS = 8


def call_api(api_input):
    headers = {"x-api-key": API_KEY, "Content-Type": "application/json"}
    resp = requests.post(API_URL, json=api_input, headers=headers, timeout=120)
    return resp.json() if resp.status_code == 200 else {"error": resp.text[:200], "status": resp.status_code}


def main():
    manifest = json.loads((COHORT_DIR / "cohort_manifest.json").read_text())
    patients = manifest["patients"]
    sepsis_pos = [p for p in patients if p["actual_sepsis"]][:N_PER_CLASS]
    sepsis_neg = [p for p in patients if not p["actual_sepsis"]][:N_PER_CLASS]
    sample = sepsis_pos + sepsis_neg

    print(f"Smoke test: {len(sample)} patients ({N_PER_CLASS} sepsis, {N_PER_CLASS} non-sepsis)")
    print(f"API: {API_URL}")
    print(f"Cohort dir: {COHORT_DIR}\n")
    print(f"{'Patient':<14} {'Actual':<8} {'Risk':<6} {'P6h':<10} {'Priority':<10} {'GR':<4}  Rationale")
    print("-" * 140)

    results = []
    for p in sample:
        pid = p["patient_id"]
        data = json.loads((COHORT_DIR / f"{pid}.json").read_text())
        r = call_api(data["api_input"])
        if "error" in r:
            print(f"{pid:<14} ERROR: {r['error']}")
            continue

        actual = "SEPSIS" if p["actual_sepsis"] else "NON"
        risk = r.get("risk_score", "N/A")
        prio = r.get("priority", "N/A")
        p6h = r.get("sepsis_probability_6h", "N/A")
        gr = "Y" if r.get("guardrail_override") else "N"
        rationale = (r.get("clinical_rationale", "") or "")[:90]

        print(f"{pid:<14} {actual:<8} {str(risk):<6} {p6h:<10} {prio:<10} {gr:<4}  {rationale}")
        results.append({
            "patient_id": pid,
            "actual": actual,
            "risk": risk,
            "sepsis_probability_6h": p6h,
            "priority": prio,
            "guardrail_override": gr,
            "rationale": r.get("clinical_rationale", ""),
        })
        time.sleep(1)

    print("\n" + "=" * 60)
    print("Decoupling check:")
    decoupled_priority = [r for r in results if isinstance(r["risk"], (int, float)) and r["risk"] >= 50 and r["priority"] == "Standard"]
    decoupled_p6h = [r for r in results if r["sepsis_probability_6h"] == "Low" and r["priority"] in ("High", "Critical")]
    print(f"  Risk>=50 but priority=Standard:         {len(decoupled_priority)}")
    print(f"  P6h=Low but priority=High/Critical (CONTRADICTION): {len(decoupled_p6h)}")
    if decoupled_p6h:
        for d in decoupled_p6h:
            print(f"    [BAD] {d['patient_id']}: P6h=Low, priority={d['priority']}")

    print("\nNon-sepsis triage (priority-driven alerts):")
    neg = [r for r in results if r["actual"] == "NON"]
    flagged_priority = [r for r in neg if r["priority"] in ("High", "Critical")]
    flagged_gr = [r for r in flagged_priority if r["guardrail_override"] == "Y"]
    flagged_llm = [r for r in flagged_priority if r["guardrail_override"] == "N"]
    print(f"  Non-sepsis flagged High/Critical: {len(flagged_priority)}/{len(neg)}  (LLM={len(flagged_llm)}, Guardrail={len(flagged_gr)})")

    print("\nSepsis catch (priority-driven):")
    pos = [r for r in results if r["actual"] == "SEPSIS"]
    caught = [r for r in pos if r["priority"] in ("High", "Critical")]
    print(f"  Sepsis caught (priority>=High): {len(caught)}/{len(pos)}")

    out = Path(__file__).parent / "results" / "smoke_test_v3_1.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
