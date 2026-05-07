"""
Final smoke test for the GenAI Sepsis Prediction pipeline against the
canonical demo trial cohort in samples/genai_test_patients.json.

Verifies that the full v7 stack (LLM + deterministic scores + C1 + C2)
produces the *expected clinical behaviour* across the 6-case severity
ladder plus 3 batch sanity cases:

    1. Healthy post-op           ->  Standard / no sepsis flag
    2. Silent-sepsis (notes)     ->  Catch via notes, elevated risk
    3. SIRS                      ->  High risk, sepsis flag
    4. Sepsis (organ dysfn)      ->  Critical
    5. Severe sepsis (MOF)       ->  Critical, max severity
    6. Septic shock              ->  Critical, max severity

Each case has explicit expectations; fail-loud if any deviates.

    python scripts/demo_smoke_test.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
TEST_FILE = ROOT / "samples" / "genai_test_patients.json"
API = "http://localhost:8000/classify"
TIMEOUT = 60

# Pull API key from .env so we don't hard-code it.
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.strip().startswith("API_KEY="):
                API_KEY = line.split("=", 1)[1].strip()
                break


# ---------- expected behaviour per case ---------------------------------
EXPECT = {
    # patient_id: dict(min/max risk, expected sepsis flag, allowed priorities)
    "GENAI_TEST_001": {  # healthy post-op
        "label": "Healthy post-op",
        "predicted_sepsis": False,
        "risk_max": 49,
        "priorities": {"Standard"},
        "notes": "Should NOT flag - all vitals normal, post-op recovery",
    },
    "GENAI_TEST_002": {  # silent sepsis - vitals stable but notes concerning
        "label": "Silent sepsis (notes-driven)",
        "predicted_sepsis": True,
        "risk_min": 50,
        "priorities": {"High", "Critical"},
        "notes": "MUST catch from notes ('confusion', 'cool skin', "
                 "'decreased UO') even though vitals look mild",
    },
    "GENAI_TEST_003": {  # SIRS
        "label": "SIRS",
        "predicted_sepsis": True,
        "risk_min": 50,
        "priorities": {"High", "Critical"},
        "notes": "Fever + tachycardia + antibiotics started",
    },
    "GENAI_TEST_004": {  # sepsis with organ dysfn
        "label": "Sepsis with organ dysfunction",
        "predicted_sepsis": True,
        "risk_min": 70,
        "priorities": {"High", "Critical"},
        "notes": "AMS, mottled skin, lactate 2.8 - clear sepsis",
    },
    "GENAI_TEST_005": {  # severe sepsis MOF
        "label": "Severe sepsis (multi-organ failure)",
        "predicted_sepsis": True,
        "risk_min": 80,
        "priorities": {"Critical"},
        "notes": "Hypothermia, lactate 4.2, vasopressors started, "
                 "intubation - critical",
    },
    "GENAI_TEST_006": {  # septic shock
        "label": "Septic shock",
        "predicted_sepsis": True,
        "risk_min": 90,
        "priorities": {"Critical"},
        "notes": "Max vasopressors, lactate 7.8, DIC - extreme severity",
    },
    "BATCH_001": {
        "label": "Batch: routine stable",
        "predicted_sepsis": False,
        "risk_max": 49,
        "priorities": {"Standard"},
        "notes": "Routine check, all normal",
    },
    "BATCH_002": {
        "label": "Batch: mild SIRS",
        "predicted_sepsis": True,
        "risk_min": 50,
        "priorities": {"High", "Critical"},
        "notes": "Fever + chills + leukocytosis",
    },
    "BATCH_003": {
        "label": "Batch: septic shock-like",
        "predicted_sepsis": True,
        "risk_min": 70,
        "priorities": {"High", "Critical"},
        "notes": "AMS + mottled skin + lactate 4.5 - sepsis",
    },
}


# ----------------------- runner ----------------------------------------
def call_classify(case: dict) -> dict:
    payload = {
        "patient_id": case["patient_id"],
        "vitals": case["vitals"],
    }
    if "notes" in case:
        payload["notes"] = case["notes"]
    headers = {"x-api-key": API_KEY, "Content-Type": "application/json"} \
        if API_KEY else {"Content-Type": "application/json"}
    r = requests.post(API, json=payload, headers=headers, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def derive_predicted_sepsis(result: dict) -> bool:
    """Same binary-classification rule used by validation/run_eicu_validation.py:
    risk >= 50 OR priority in {High, Critical} OR alert_level in {HIGH, CRITICAL}."""
    risk = float(result.get("risk_score", 0) or 0)
    pri = result.get("priority", "")
    alert = (result.get("alert_level") or "").upper()
    return (risk >= 50
            or pri in ("High", "Critical")
            or alert in ("HIGH", "CRITICAL"))


def check(case: dict, result: dict, expected: dict) -> tuple[bool, list]:
    """Return (pass, list of mismatches)."""
    misses = []
    risk = float(result.get("risk_score", 0) or 0)
    pri = result.get("priority", "")
    pred = derive_predicted_sepsis(result)

    if pred != expected["predicted_sepsis"]:
        misses.append(
            f"predicted_sepsis={pred} but expected "
            f"{expected['predicted_sepsis']}"
        )
    if "risk_min" in expected and risk < expected["risk_min"]:
        misses.append(
            f"risk_score={risk:.0f} < expected min {expected['risk_min']}"
        )
    if "risk_max" in expected and risk > expected["risk_max"]:
        misses.append(
            f"risk_score={risk:.0f} > expected max {expected['risk_max']}"
        )
    if expected["priorities"] and pri not in expected["priorities"]:
        misses.append(
            f"priority={pri!r} not in expected {expected['priorities']}"
        )
    return len(misses) == 0, misses


def fmt_guardrail_audit(result: dict) -> str:
    """Concise summary of guardrail action fields (LLM_init, C1, C2, override, ED)."""
    parts = []
    init_risk = result.get("llm_initial_risk_score")
    if init_risk is not None:
        parts.append(f"LLM_init={init_risk:g}")
    if result.get("c1_suppression_applied"):
        parts.append(f"C1\u2713({result.get('c1_suppression_path') or '?'})")
    if result.get("c2_suppression_applied"):
        parts.append(f"C2\u2713({result.get('c2_branch') or '?'})")
    if result.get("guardrail_override"):
        reasons = result.get("override_reasons") or []
        first = reasons[0] if reasons else "yes"
        parts.append(f"OVR({first[:30]})")
    early = str(result.get("early_warnings") or "")
    if "Early Detection" in early:
        parts.append("ED-bump")
    return " ".join(parts) if parts else "LLM-only"


def main() -> int:
    if not TEST_FILE.exists():
        print(f"FAIL: test file missing - {TEST_FILE}")
        return 2

    data = json.loads(TEST_FILE.read_text())
    cases = list(data.get("test_cases", []))
    for p in data.get("batch_test", {}).get("patients", []):
        cases.append(p)

    print(f"\n{'='*88}")
    print(f"GenAI Sepsis Prediction — final smoke test ({len(cases)} cases)")
    print(f"API: {API}")
    print(f"{'='*88}\n")

    pass_n, fail_n = 0, 0
    failures = []
    rows = []

    for i, case in enumerate(cases, 1):
        pid = case["patient_id"]
        expected = EXPECT.get(pid)
        if not expected:
            print(f"[{i}/{len(cases)}] {pid}: SKIP (no expectation set)")
            continue

        label = expected["label"]
        print(f"[{i}/{len(cases)}] {pid}  -  {label}")
        t0 = time.time()
        try:
            result = call_classify(case)
        except Exception as e:
            print(f"    ERROR: {type(e).__name__}: {e}")
            failures.append((pid, label, [f"API error: {e}"]))
            fail_n += 1
            continue
        latency = time.time() - t0

        ok, misses = check(case, result, expected)
        risk = float(result.get("risk_score", 0))
        pri = result.get("priority", "")
        pred = derive_predicted_sepsis(result)
        audit = fmt_guardrail_audit(result)
        status = "PASS" if ok else "FAIL"

        print(f"    risk={risk:5.1f}  priority={pri:<8s}  "
              f"sepsis={str(pred):<5s}  {audit}  ({latency:.1f}s)  "
              f"-> {status}")
        if not ok:
            for m in misses:
                print(f"        - {m}")
            failures.append((pid, label, misses))
            fail_n += 1
        else:
            pass_n += 1
        rows.append((pid, label, risk, pri, pred, audit, ok))

    print(f"\n{'='*88}")
    print(f"Result: {pass_n} pass / {fail_n} fail "
          f"out of {pass_n + fail_n} executed")
    print(f"{'='*88}\n")

    if failures:
        print("FAILURES:")
        for pid, label, misses in failures:
            print(f"  {pid} ({label}):")
            for m in misses:
                print(f"      - {m}")
        return 1

    print("\nClinical-severity ladder verified end-to-end:")
    print("    healthy -> silent-sepsis -> SIRS -> sepsis -> severe -> shock")
    print("All 6 demo cases + 3 batch cases match expected behaviour.")
    print("Ready to push & containerise.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
