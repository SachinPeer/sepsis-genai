"""
v7.3 pilot — run only the 7 sepsis patients v7.2 lost as TPs.

Acceptance for proceeding to full 150-patient run: at least 6/7 caught.

Reuses validation/run_eicu_validation.py helpers so the API-input shape
is exactly what the full validation will use.
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "validation"))

from run_eicu_validation import build_api_input, call_api, classify_prediction  # noqa: E402

PILOT_IDS = [
    "eicu_p00005",
    "eicu_p00006",
    "eicu_p00007",
    "eicu_p00009",
    "eicu_p00010",
    "eicu_p00014",
    "eicu_p00017",
]
COHORT_DIR = ROOT / "validation" / "eicu_cohort_v4"

# Pull API key from env / .env
api_key = os.getenv("API_KEY")
if not api_key:
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("API_KEY="):
            api_key = line.split("=", 1)[1].strip()
            break
os.environ["API_KEY"] = api_key

print("=" * 78)
print("v7.3 PILOT — 7 sepsis patients v7.2 lost")
print("Tripwire: catch >= 6/7 to proceed to full run")
print("=" * 78)

caught = 0
results = []
for pid in PILOT_IDS:
    pjson = json.loads((COHORT_DIR / f"{pid}.json").read_text())
    api_input = build_api_input(pjson)
    api_result, err = call_api(api_input)
    if err:
        print(f"  ERROR {pid}: {err[:120]}")
        continue
    predicted = classify_prediction(api_result)
    risk = api_result.get("risk_score")
    pri = api_result.get("priority")
    alert = api_result.get("alert_level")
    llm_risk = api_result.get("llm_initial_risk_score")
    llm_pri = api_result.get("llm_initial_priority")
    c1 = api_result.get("c1_suppression_applied")
    c2 = api_result.get("c2_suppression_applied")
    rationale = (api_result.get("clinical_rationale") or api_result.get("reasoning") or "")[:130]

    sym = "PASS" if predicted else "FAIL"
    if predicted:
        caught += 1
    print(f"  [{sym}] {pid}  final risk={risk}/{pri}  "
          f"LLM={llm_risk}/{llm_pri}  C1={c1} C2={c2}")
    print(f"          rationale: {rationale}...")
    results.append({"pid": pid, "predicted": predicted, "risk": risk, "pri": pri,
                    "llm_risk": llm_risk, "llm_pri": llm_pri,
                    "c1": c1, "c2": c2, "rationale": rationale})

print()
print("=" * 78)
print(f"Pilot result: {caught} / 7 sepsis caught")
print("v7 baseline : 7 / 7 (these patients were all TP under v7)")
print("v7.2 result : 0 / 7 (these are the patients we lost)")
print("v7.3 target : >= 6 / 7 to proceed")
if caught >= 6:
    print(">> PILOT PASSED — proceed to full 150-patient run")
    sys.exit(0)
else:
    print(">> PILOT FAILED — abort full run; revert to v7 inline default")
    sys.exit(1)
