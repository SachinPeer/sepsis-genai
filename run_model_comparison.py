"""
Compare Claude models on Bedrock for sepsis prediction.
Runs the same patient through each model, captures output and timing.
"""

import boto3
import json
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from knowledge.genai_proprocess import SepsisPreprocessor

REGION = "us-east-1"

MODELS_TO_TEST = [
    {"id": "us.anthropic.claude-sonnet-4-20250514-v1:0", "name": "Claude Sonnet 4", "input_cost": 3.0, "output_cost": 15.0},
    {"id": "us.anthropic.claude-sonnet-4-5-20250929-v1:0", "name": "Claude Sonnet 4.5", "input_cost": 3.0, "output_cost": 15.0},
    {"id": "us.anthropic.claude-sonnet-4-6", "name": "Claude Sonnet 4.6", "input_cost": 3.0, "output_cost": 15.0},
    {"id": "us.anthropic.claude-opus-4-6-v1", "name": "Claude Opus 4.6", "input_cost": 5.0, "output_cost": 25.0},
]

def load_prompt():
    prompt_path = os.path.join(os.path.dirname(__file__), "docs", "prompt.md")
    with open(prompt_path, "r") as f:
        return f.read()

def preprocess_patient(patient_file):
    with open(patient_file, "r") as f:
        data = json.load(f)
    preprocessor = SepsisPreprocessor()
    vitals = data.get("vitals", data)
    notes = data.get("notes", "")
    narrative = preprocessor.process(vitals, notes)
    return narrative, data.get("patient_id", "unknown")

def invoke_model(client, model_id, system_prompt, patient_narrative):
    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2048,
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": f"Analyze this patient for sepsis risk:\n\n{patient_narrative}"}
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

    return {
        "output": output_text,
        "elapsed_ms": round(elapsed_ms, 1),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def run_comparison():
    print("=" * 70)
    print("SEPSIS GenAI — LLM MODEL COMPARISON")
    print("=" * 70)

    system_prompt = load_prompt()
    test_file = os.path.join(os.path.dirname(__file__), "test_trend.json")
    narrative, patient_id = preprocess_patient(test_file)
    print(f"\nPatient: {patient_id}")
    print(f"Prompt length: {len(system_prompt)} chars")
    print(f"Narrative length: {len(narrative)} chars\n")

    client = boto3.client(
        "bedrock-runtime",
        region_name=REGION,
        config=boto3.session.Config(
            retries={"max_attempts": 3, "mode": "adaptive"},
            read_timeout=120,
            connect_timeout=10
        )
    )

    results = []

    for model in MODELS_TO_TEST:
        print(f"\n{'─' * 50}")
        print(f"Testing: {model['name']} ({model['id']})")
        print(f"{'─' * 50}")

        try:
            result = invoke_model(client, model["id"], system_prompt, narrative)

            input_cost = (result["input_tokens"] / 1_000_000) * model["input_cost"]
            output_cost = (result["output_tokens"] / 1_000_000) * model["output_cost"]
            total_cost = input_cost + output_cost
            cost_per_1k = total_cost * 1000

            try:
                raw_text = result["output"].strip()
                if raw_text.startswith("```"):
                    raw_text = raw_text.split("\n", 1)[1] if "\n" in raw_text else raw_text
                if raw_text.endswith("```"):
                    raw_text = raw_text[:raw_text.rfind("```")]
                raw_text = raw_text.strip()
                parsed = json.loads(raw_text)
                prediction = parsed.get("prediction", {})
                risk_score = prediction.get("risk_score_0_100", "N/A")
                priority = prediction.get("priority", "N/A")
                probability = prediction.get("sepsis_probability_6h", "N/A")
                confidence = prediction.get("confidence_level", "N/A")
                confidence_reasoning = prediction.get("confidence_reasoning", "N/A")
                rationale = prediction.get("clinical_rationale", "N/A")
                metrics = parsed.get("clinical_metrics", {})
                logic_gate = parsed.get("logic_gate", {})
                json_valid = True
            except json.JSONDecodeError:
                risk_score = "PARSE ERROR"
                priority = "N/A"
                probability = "N/A"
                confidence = "N/A"
                confidence_reasoning = "N/A"
                rationale = result["output"][:200]
                metrics = {}
                logic_gate = {}
                json_valid = False

            model_result = {
                "model_name": model["name"],
                "model_id": model["id"],
                "response_time_ms": result["elapsed_ms"],
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "input_cost_per_call": round(input_cost, 6),
                "output_cost_per_call": round(output_cost, 6),
                "total_cost_per_call": round(total_cost, 6),
                "cost_per_1k_patients": round(cost_per_1k, 2),
                "risk_score": risk_score,
                "priority": priority,
                "probability_6h": probability,
                "confidence": confidence,
                "confidence_reasoning": confidence_reasoning,
                "rationale": rationale,
                "qsofa": metrics.get("qSOFA_score", "N/A"),
                "sirs_met": metrics.get("SIRS_met", "N/A"),
                "trend_velocity": metrics.get("trend_velocity", "N/A"),
                "organ_stress": metrics.get("organ_stress_indicators", []),
                "discordance": logic_gate.get("discordance_detected", "N/A"),
                "primary_driver": logic_gate.get("primary_driver", "N/A"),
                "missing_params": logic_gate.get("missing_parameters", []),
                "json_valid": json_valid,
                "raw_output": result["output"],
                "pricing": {"input": model["input_cost"], "output": model["output_cost"]},
                "status": "SUCCESS"
            }

            print(f"  Response time: {result['elapsed_ms']}ms")
            print(f"  Tokens: {result['input_tokens']} in / {result['output_tokens']} out")
            print(f"  Cost per call: ${total_cost:.6f} | Per 1K patients: ${cost_per_1k:.2f}")
            print(f"  Risk Score: {risk_score} | Priority: {priority} | Confidence: {confidence}")
            print(f"  JSON Valid: {json_valid}")

        except Exception as e:
            print(f"  ERROR: {e}")
            model_result = {
                "model_name": model["name"],
                "model_id": model["id"],
                "status": "FAILED",
                "error": str(e),
            }

        results.append(model_result)
        time.sleep(2)

    output_file = os.path.join(os.path.dirname(__file__), "model_comparison_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    run_comparison()
