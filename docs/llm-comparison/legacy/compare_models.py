#!/usr/bin/env python3
"""
Model Comparison Script - Azure OpenAI GPT-4o vs AWS Bedrock Claude 3.5

This script runs the same test cases through both LLM providers and 
compares their predictions, latency, and costs.

Usage:
    # Compare both models on default test cases
    python compare_models.py

    # Compare using a specific test file
    python compare_models.py --test-file test_trend.json

    # Run only Azure
    python compare_models.py --provider azure

    # Run only Bedrock
    python compare_models.py --provider bedrock

    # Output results to JSON file
    python compare_models.py --output comparison_results.json
"""

import os
import sys
import json
import time
import argparse
import logging
from datetime import datetime
from typing import Dict, Any, List

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

from genai_inference_service import GenAIInferenceService
from genai_pipeline import GenAISepsisPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# DEFAULT TEST CASES
# =============================================================================

DEFAULT_TEST_CASES = [
    {
        "name": "Normal Post-Op Patient",
        "expected_risk": "Low",
        "vitals": {
            "HR": [{"val": 72, "ts": "2026-02-10T08:00"}],
            "SBP": [{"val": 118, "ts": "2026-02-10T08:00"}],
            "Temp": [36.8],
            "WBC": [7.2],
            "Lactate": [0.9],
            "Age": 45
        },
        "notes": "Post-op day 2, patient ambulatory, tolerating diet well. Wound clean and dry."
    },
    {
        "name": "Trending Toward Sepsis",
        "expected_risk": "Moderate-High",
        "vitals": {
            "HR": [
                {"val": 105, "ts": "2026-02-11T08:00"},
                {"val": 92, "ts": "2026-02-11T06:00"},
                {"val": 78, "ts": "2026-02-11T04:00"},
                {"val": 68, "ts": "2026-02-11T02:00"}
            ],
            "SBP": [
                {"val": 98, "ts": "2026-02-11T08:00"},
                {"val": 108, "ts": "2026-02-11T06:00"},
                {"val": 118, "ts": "2026-02-11T04:00"},
                {"val": 128, "ts": "2026-02-11T02:00"}
            ],
            "Temp": [{"val": 38.4, "ts": "2026-02-11T08:00"}],
            "WBC": [12.5],
            "Lactate": [1.6],
            "Age": 58
        },
        "notes": "Day 3 post cholecystectomy. Patient reports feeling not quite right but no specific complaints."
    },
    {
        "name": "Silent Sepsis - Notes Critical",
        "expected_risk": "High",
        "vitals": {
            "HR": [{"val": 88, "ts": "2026-02-11T08:00"}],
            "SBP": [{"val": 112, "ts": "2026-02-11T08:00"}],
            "Temp": [37.6],
            "WBC": [10.8],
            "Lactate": [1.3],
            "Age": 71
        },
        "notes": "Patient was oriented x3 at shift start. RN noted patient asked same question 3 times, seems not herself per family. Mottling on bilateral knees - new finding. Foley output only 80mL over past 4 hours. Family states she just does not look right. UTI diagnosed yesterday."
    },
    {
        "name": "Septic Shock",
        "expected_risk": "Critical",
        "vitals": {
            "HR": [{"val": 128, "ts": "2026-02-10T08:00"}],
            "SBP": [{"val": 88, "ts": "2026-02-10T08:00"}],
            "MAP": [{"val": 62, "ts": "2026-02-10T08:00"}],
            "Temp": [35.8],
            "WBC": [22.0],
            "Lactate": [4.2],
            "Creatinine": [3.5],
            "Platelets": [85],
            "Age": 68
        },
        "notes": "CRITICAL: Patient unresponsive to verbal stimuli. Now hypothermic. 3rd liter of fluids running. Norepinephrine drip started."
    }
]


# =============================================================================
# COMPARISON FUNCTIONS
# =============================================================================

def run_single_prediction(provider: str, vitals: Dict, notes: str, patient_id: str) -> Dict[str, Any]:
    """Run a single prediction with timing."""
    try:
        # Use the full pipeline for consistency
        from genai_inference_service import get_llm_provider
        from genai_pipeline import GenAISepsisPipeline
        
        # Create a new pipeline with specific provider
        service = GenAIInferenceService(provider=provider)
        
        # Process through preprocessor manually for direct comparison
        from knowledge.genai_proprocess import SepsisPreprocessor
        preprocessor = SepsisPreprocessor()
        narrative = preprocessor.process(vitals, notes)
        
        start_time = time.time()
        result = service.predict(narrative)
        elapsed = time.time() - start_time
        
        return {
            "status": "success",
            "provider": provider,
            "model": service.provider.model_name,
            "latency_seconds": round(elapsed, 2),
            "risk_score": result.get("prediction", {}).get("risk_score_0_100"),
            "priority": result.get("prediction", {}).get("priority"),
            "probability_6h": result.get("prediction", {}).get("sepsis_probability_6h"),
            "rationale": result.get("prediction", {}).get("clinical_rationale", "")[:200],
            "tokens_used": result.get("_metadata", {}).get("tokens_used"),
            "full_result": result
        }
    except Exception as e:
        return {
            "status": "error",
            "provider": provider,
            "error": str(e),
            "latency_seconds": 0,
            "risk_score": None
        }


def compare_models(test_cases: List[Dict], providers: List[str] = ["azure", "bedrock"]) -> Dict[str, Any]:
    """Run comparison across all test cases and providers."""
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "providers_tested": providers,
        "test_cases": [],
        "summary": {}
    }
    
    for i, test_case in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"Test Case {i+1}: {test_case['name']}")
        print(f"Expected Risk: {test_case['expected_risk']}")
        print('='*60)
        
        case_result = {
            "name": test_case["name"],
            "expected_risk": test_case["expected_risk"],
            "predictions": {}
        }
        
        for provider in providers:
            print(f"\n  Running {provider.upper()}...", end=" ", flush=True)
            
            prediction = run_single_prediction(
                provider=provider,
                vitals=test_case["vitals"],
                notes=test_case.get("notes", ""),
                patient_id=f"compare_{i}_{provider}"
            )
            
            case_result["predictions"][provider] = prediction
            
            if prediction["status"] == "success":
                print(f"Done in {prediction['latency_seconds']}s")
                print(f"    Risk Score: {prediction['risk_score']}")
                print(f"    Priority: {prediction['priority']}")
                print(f"    Tokens: {prediction['tokens_used']}")
            else:
                print(f"ERROR: {prediction['error']}")
        
        results["test_cases"].append(case_result)
    
    # Generate summary
    results["summary"] = generate_summary(results)
    
    return results


def generate_summary(results: Dict) -> Dict[str, Any]:
    """Generate comparison summary statistics."""
    
    summary = {provider: {
        "total_cases": 0,
        "successful": 0,
        "errors": 0,
        "avg_latency": 0,
        "avg_risk_score": 0,
        "total_tokens": 0
    } for provider in results["providers_tested"]}
    
    for case in results["test_cases"]:
        for provider, pred in case["predictions"].items():
            summary[provider]["total_cases"] += 1
            
            if pred["status"] == "success":
                summary[provider]["successful"] += 1
                summary[provider]["avg_latency"] += pred["latency_seconds"]
                summary[provider]["avg_risk_score"] += pred["risk_score"] or 0
                summary[provider]["total_tokens"] += pred["tokens_used"] or 0
            else:
                summary[provider]["errors"] += 1
    
    # Calculate averages
    for provider in summary:
        n = summary[provider]["successful"]
        if n > 0:
            summary[provider]["avg_latency"] = round(summary[provider]["avg_latency"] / n, 2)
            summary[provider]["avg_risk_score"] = round(summary[provider]["avg_risk_score"] / n, 1)
    
    return summary


def print_summary(summary: Dict):
    """Print a formatted summary table."""
    
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    headers = ["Metric", "Azure OpenAI", "AWS Bedrock"]
    row_format = "{:<25} {:>20} {:>20}"
    
    print(row_format.format(*headers))
    print("-"*70)
    
    metrics = [
        ("Model", "gpt-4o", "claude-3.5-sonnet"),
        ("Successful Runs", summary.get("azure", {}).get("successful", 0), summary.get("bedrock", {}).get("successful", 0)),
        ("Errors", summary.get("azure", {}).get("errors", 0), summary.get("bedrock", {}).get("errors", 0)),
        ("Avg Latency (sec)", summary.get("azure", {}).get("avg_latency", 0), summary.get("bedrock", {}).get("avg_latency", 0)),
        ("Avg Risk Score", summary.get("azure", {}).get("avg_risk_score", 0), summary.get("bedrock", {}).get("avg_risk_score", 0)),
        ("Total Tokens", summary.get("azure", {}).get("total_tokens", 0), summary.get("bedrock", {}).get("total_tokens", 0)),
    ]
    
    for metric in metrics:
        print(row_format.format(str(metric[0]), str(metric[1]), str(metric[2])))
    
    print("="*70)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Compare Azure OpenAI vs AWS Bedrock Claude for Sepsis Prediction")
    parser.add_argument("--provider", choices=["azure", "bedrock", "both"], default="both",
                        help="Which provider to test (default: both)")
    parser.add_argument("--test-file", type=str, help="JSON file with test cases")
    parser.add_argument("--output", type=str, help="Output file for results (JSON)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Determine providers to test
    if args.provider == "both":
        providers = ["azure", "bedrock"]
    else:
        providers = [args.provider]
    
    # Load test cases
    if args.test_file:
        with open(args.test_file, 'r') as f:
            data = json.load(f)
            # Support both single case and array formats
            if isinstance(data, list):
                test_cases = data
            elif "vitals" in data:
                test_cases = [{"name": data.get("patient_id", "Custom"), "expected_risk": "Unknown", **data}]
            else:
                test_cases = data.get("test_cases", [data])
    else:
        test_cases = DEFAULT_TEST_CASES
    
    print("\n" + "="*70)
    print("SEPSIS GENAI - MODEL COMPARISON")
    print(f"Providers: {', '.join(providers)}")
    print(f"Test Cases: {len(test_cases)}")
    print("="*70)
    
    # Run comparison
    results = compare_models(test_cases, providers)
    
    # Print summary
    print_summary(results["summary"])
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output}")
    
    return results


if __name__ == "__main__":
    main()
