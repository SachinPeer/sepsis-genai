"""
Full LLM Model Comparison — All test patients x All models.
Outputs CSV to docs/ for leadership evidence.
8 patients x 4 models = 32 API calls.
"""

import boto3
import json
import time
import os
import sys
import csv
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing.genai_preprocess import SepsisPreprocessor

REGION = "us-east-1"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS = [
    {"id": "us.anthropic.claude-sonnet-4-20250514-v1:0", "name": "Sonnet 4", "input_cost": 3.0, "output_cost": 15.0},
    {"id": "us.anthropic.claude-sonnet-4-5-20250929-v1:0", "name": "Sonnet 4.5", "input_cost": 3.0, "output_cost": 15.0},
    {"id": "us.anthropic.claude-sonnet-4-6", "name": "Sonnet 4.6", "input_cost": 3.0, "output_cost": 15.0},
    {"id": "us.anthropic.claude-opus-4-6-v1", "name": "Opus 4.6", "input_cost": 5.0, "output_cost": 25.0},
]


def load_prompt():
    with open(os.path.join(BASE_DIR, "docs", "prompt.md"), "r") as f:
        return f.read()


def load_all_patients():
    patients = []
    preprocessor = SepsisPreprocessor()

    # test_trend.json
    with open(os.path.join(BASE_DIR, "test_trend.json"), "r") as f:
        data = json.load(f)
    patients.append({
        "patient_id": data["patient_id"],
        "case_desc": "Post-op Day 3, trending vitals",
        "expected_severity": "High",
        "narrative": preprocessor.process(data.get("vitals", data), data.get("notes", "")),
    })

    # test_notes.json
    with open(os.path.join(BASE_DIR, "test_notes.json"), "r") as f:
        data = json.load(f)
    patients.append({
        "patient_id": data["patient_id"],
        "case_desc": "UTI + confusion + mottling (Silent Sepsis)",
        "expected_severity": "High",
        "narrative": preprocessor.process(data.get("vitals", data), data.get("notes", "")),
    })

    # genai_test_patients.json
    with open(os.path.join(BASE_DIR, "genai_test_patients.json"), "r") as f:
        data = json.load(f)

    case_meta = [
        ("Normal healthy post-op", "Standard"),
        ("Silent Sepsis — notes concerning", "High"),
        ("SIRS — inflammatory response", "High"),
        ("Sepsis — organ dysfunction", "Critical"),
        ("Severe Sepsis — multi-organ", "Critical"),
        ("Septic Shock — refractory", "Critical"),
    ]

    for i, tc in enumerate(data["test_cases"]):
        desc, expected = case_meta[i]
        patients.append({
            "patient_id": tc["patient_id"],
            "case_desc": desc,
            "expected_severity": expected,
            "narrative": preprocessor.process(tc.get("vitals", tc), tc.get("notes", "")),
        })

    return patients


def invoke_model(client, model_id, system_prompt, narrative):
    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2048,
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": f"Analyze this patient for sepsis risk:\n\n{narrative}"}
        ]
    }

    start = time.time()
    response = client.invoke_model(
        modelId=model_id,
        body=json.dumps(request_body),
        contentType="application/json",
        accept="application/json"
    )
    elapsed_ms = (time.time() - start) * 1000

    result = json.loads(response["body"].read())
    output_text = result["content"][0]["text"]
    input_tokens = result.get("usage", {}).get("input_tokens", 0)
    output_tokens = result.get("usage", {}).get("output_tokens", 0)

    return output_text, elapsed_ms, input_tokens, output_tokens


def parse_output(raw_text):
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
    if text.endswith("```"):
        text = text[:text.rfind("```")]
    text = text.strip()

    try:
        parsed = json.loads(text)
        pred = parsed.get("prediction", {})
        metrics = parsed.get("clinical_metrics", {})
        logic = parsed.get("logic_gate", {})
        return {
            "json_valid": True,
            "risk_score": pred.get("risk_score_0_100", ""),
            "priority": pred.get("priority", ""),
            "probability_6h": pred.get("sepsis_probability_6h", ""),
            "confidence": pred.get("confidence_level", ""),
            "confidence_reasoning": pred.get("confidence_reasoning", ""),
            "rationale": pred.get("clinical_rationale", ""),
            "qsofa": metrics.get("qSOFA_score", ""),
            "sirs_met": metrics.get("SIRS_met", ""),
            "trend_velocity": metrics.get("trend_velocity", ""),
            "organ_stress": "; ".join(metrics.get("organ_stress_indicators", [])),
            "discordance": logic.get("discordance_detected", ""),
            "primary_driver": logic.get("primary_driver", ""),
            "missing_params": "; ".join(logic.get("missing_parameters", [])),
        }
    except json.JSONDecodeError:
        return {
            "json_valid": False,
            "risk_score": "PARSE_ERROR",
            "priority": "", "probability_6h": "", "confidence": "",
            "confidence_reasoning": "", "rationale": raw_text[:200],
            "qsofa": "", "sirs_met": "", "trend_velocity": "",
            "organ_stress": "", "discordance": "", "primary_driver": "",
            "missing_params": "",
        }


def run_full_comparison():
    print("=" * 70)
    print("SEPSIS GenAI — FULL MODEL COMPARISON")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    system_prompt = load_prompt()
    patients = load_all_patients()
    print(f"\nPatients loaded: {len(patients)}")
    print(f"Models to test: {len(MODELS)}")
    print(f"Total API calls: {len(patients) * len(MODELS)}")
    print()

    client = boto3.client(
        "bedrock-runtime",
        region_name=REGION,
        config=boto3.session.Config(
            retries={"max_attempts": 3, "mode": "adaptive"},
            read_timeout=120,
            connect_timeout=10
        )
    )

    all_results = []
    call_num = 0
    total_calls = len(patients) * len(MODELS)

    for patient in patients:
        for model in MODELS:
            call_num += 1
            print(f"[{call_num}/{total_calls}] {patient['patient_id']} → {model['name']}...", end=" ", flush=True)

            try:
                raw_text, elapsed_ms, in_tokens, out_tokens = invoke_model(
                    client, model["id"], system_prompt, patient["narrative"]
                )

                input_cost = (in_tokens / 1_000_000) * model["input_cost"]
                output_cost = (out_tokens / 1_000_000) * model["output_cost"]
                total_cost = input_cost + output_cost

                parsed = parse_output(raw_text)

                row = {
                    "test_date": datetime.now().strftime("%Y-%m-%d"),
                    "patient_id": patient["patient_id"],
                    "case_description": patient["case_desc"],
                    "expected_severity": patient["expected_severity"],
                    "model_name": model["name"],
                    "model_id": model["id"],
                    "response_time_ms": round(elapsed_ms, 1),
                    "input_tokens": in_tokens,
                    "output_tokens": out_tokens,
                    "cost_per_call_usd": round(total_cost, 6),
                    "cost_per_1k_patients_usd": round(total_cost * 1000, 2),
                    **parsed,
                }

                print(f"✓ {elapsed_ms:.0f}ms | Risk:{parsed['risk_score']} | {parsed['priority']}")

            except Exception as e:
                print(f"✗ ERROR: {e}")
                row = {
                    "test_date": datetime.now().strftime("%Y-%m-%d"),
                    "patient_id": patient["patient_id"],
                    "case_description": patient["case_desc"],
                    "expected_severity": patient["expected_severity"],
                    "model_name": model["name"],
                    "model_id": model["id"],
                    "response_time_ms": 0,
                    "input_tokens": 0, "output_tokens": 0,
                    "cost_per_call_usd": 0, "cost_per_1k_patients_usd": 0,
                    "json_valid": False, "risk_score": "ERROR",
                    "priority": "", "probability_6h": "", "confidence": "",
                    "confidence_reasoning": "", "rationale": str(e)[:200],
                    "qsofa": "", "sirs_met": "", "trend_velocity": "",
                    "organ_stress": "", "discordance": "", "primary_driver": "",
                    "missing_params": "",
                }

            all_results.append(row)
            time.sleep(1)

    # Write CSV
    csv_path = os.path.join(BASE_DIR, "docs", "LLM_Model_Comparison_Results.csv")
    fieldnames = [
        "test_date", "patient_id", "case_description", "expected_severity",
        "model_name", "model_id", "response_time_ms",
        "input_tokens", "output_tokens", "cost_per_call_usd", "cost_per_1k_patients_usd",
        "json_valid", "risk_score", "priority", "probability_6h", "confidence",
        "confidence_reasoning", "rationale",
        "qsofa", "sirs_met", "trend_velocity", "organ_stress",
        "discordance", "primary_driver", "missing_params",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n{'=' * 70}")
    print(f"CSV saved: {csv_path}")
    print(f"Total calls: {len(all_results)}")
    print(f"{'=' * 70}")

    # Print summary table
    print(f"\n{'MODEL SUMMARY':=^70}")
    for model in MODELS:
        model_rows = [r for r in all_results if r["model_name"] == model["name"]]
        valid = [r for r in model_rows if r["json_valid"]]
        avg_time = sum(r["response_time_ms"] for r in valid) / len(valid) if valid else 0
        avg_cost = sum(r["cost_per_call_usd"] for r in valid) / len(valid) if valid else 0
        scores = [r["risk_score"] for r in valid if isinstance(r["risk_score"], int)]
        print(f"\n  {model['name']}:")
        print(f"    Avg response time: {avg_time:.0f}ms")
        print(f"    Avg cost/call: ${avg_cost:.6f} (${avg_cost * 1000:.2f}/1K patients)")
        print(f"    JSON valid: {len(valid)}/{len(model_rows)}")
        print(f"    Risk scores: {scores}")

    # Write raw JSON too
    json_path = os.path.join(BASE_DIR, "model_comparison_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results


if __name__ == "__main__":
    run_full_comparison()
