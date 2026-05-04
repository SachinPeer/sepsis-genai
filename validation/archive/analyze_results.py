"""
Phase 4: Analyze validation results — calculate sensitivity, specificity, and other metrics.
"""

import json
import math
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path(__file__).parent / "results"


def load_results():
    path = RESULTS_DIR / "validation_results_latest.json"
    with open(path, "r") as f:
        return json.load(f)


def calculate_metrics(results):
    tp = fp = tn = fn = 0
    errors = 0
    override_count = 0
    total_time = 0
    times = []

    for r in results:
        if r["status"] != "success":
            errors += 1
            continue

        actual = r["actual_sepsis"]
        predicted = r["predicted_sepsis"]
        total_time += r.get("processing_time_ms", 0)
        times.append(r.get("processing_time_ms", 0))

        if r.get("guardrail_override"):
            override_count += 1

        if actual and predicted:
            tp += 1
        elif actual and not predicted:
            fn += 1
        elif not actual and predicted:
            fp += 1
        elif not actual and not predicted:
            tn += 1

    total_valid = tp + fp + tn + fn
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    accuracy = (tp + tn) / total_valid if total_valid > 0 else 0
    f1 = 2 * (ppv * sensitivity) / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

    def confidence_interval(p, n, z=1.96):
        if n == 0:
            return (0, 0)
        se = math.sqrt(p * (1 - p) / n)
        return (max(0, p - z * se), min(1, p + z * se))

    sens_ci = confidence_interval(sensitivity, tp + fn)
    spec_ci = confidence_interval(specificity, tn + fp)

    avg_time = total_time / len(times) if times else 0
    median_time = sorted(times)[len(times) // 2] if times else 0

    return {
        "confusion_matrix": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
        "total_valid": total_valid,
        "errors": errors,
        "sensitivity": round(sensitivity * 100, 2),
        "sensitivity_ci": (round(sens_ci[0] * 100, 2), round(sens_ci[1] * 100, 2)),
        "specificity": round(specificity * 100, 2),
        "specificity_ci": (round(spec_ci[0] * 100, 2), round(spec_ci[1] * 100, 2)),
        "ppv": round(ppv * 100, 2),
        "npv": round(npv * 100, 2),
        "accuracy": round(accuracy * 100, 2),
        "f1_score": round(f1 * 100, 2),
        "false_alarm_rate": round(false_alarm_rate * 100, 2),
        "guardrail_overrides": override_count,
        "avg_processing_ms": round(avg_time, 1),
        "median_processing_ms": round(median_time, 1),
    }


def print_report(metrics, results_data):
    print("\n" + "=" * 70)
    print("   SEPSIS GenAI VALIDATION STUDY — RESULTS REPORT")
    print("=" * 70)

    print(f"\n  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Dataset: PhysioNet Challenge 2019 (Training Set A)")
    print(f"  Patients: {results_data['total_patients']} "
          f"(Success: {results_data['successful']}, Errors: {results_data['errors']})")

    cm = metrics["confusion_matrix"]
    print(f"\n  CONFUSION MATRIX")
    print(f"  {'':>20} | {'Predicted Sepsis':>16} | {'Predicted No Sepsis':>20}")
    print(f"  {'-'*60}")
    print(f"  {'Actual Sepsis':>20} | {'TP: ' + str(cm['TP']):>16} | {'FN: ' + str(cm['FN']):>20}")
    print(f"  {'Actual No Sepsis':>20} | {'FP: ' + str(cm['FP']):>16} | {'TN: ' + str(cm['TN']):>20}")

    print(f"\n  PRIMARY METRICS")
    print(f"  {'Sensitivity (Recall)':>25}: {metrics['sensitivity']:>6.2f}%  "
          f"(95% CI: {metrics['sensitivity_ci'][0]:.2f}% - {metrics['sensitivity_ci'][1]:.2f}%)")
    print(f"  {'Specificity':>25}: {metrics['specificity']:>6.2f}%  "
          f"(95% CI: {metrics['specificity_ci'][0]:.2f}% - {metrics['specificity_ci'][1]:.2f}%)")
    print(f"  {'PPV (Precision)':>25}: {metrics['ppv']:>6.2f}%")
    print(f"  {'NPV':>25}: {metrics['npv']:>6.2f}%")
    print(f"  {'Accuracy':>25}: {metrics['accuracy']:>6.2f}%")
    print(f"  {'F1 Score':>25}: {metrics['f1_score']:>6.2f}%")
    print(f"  {'False Alarm Rate':>25}: {metrics['false_alarm_rate']:>6.2f}%")

    print(f"\n  GUARDRAIL PERFORMANCE")
    print(f"  {'Guardrail Overrides':>25}: {metrics['guardrail_overrides']}")

    print(f"\n  PERFORMANCE")
    print(f"  {'Avg Processing Time':>25}: {metrics['avg_processing_ms']:>8.1f} ms")
    print(f"  {'Median Processing Time':>25}: {metrics['median_processing_ms']:>8.1f} ms")

    sens_target = 90.0
    spec_target = 85.0
    print(f"\n  TARGET ASSESSMENT")
    sens_status = "PASS" if metrics['sensitivity'] >= sens_target else "BELOW TARGET"
    spec_status = "PASS" if metrics['specificity'] >= spec_target else "BELOW TARGET"
    print(f"  {'Sensitivity target (≥90%)':>30}: {sens_status} ({metrics['sensitivity']:.2f}%)")
    print(f"  {'Specificity target (≥85%)':>30}: {spec_status} ({metrics['specificity']:.2f}%)")

    print("\n" + "=" * 70)


def save_report(metrics, results_data):
    report = {
        "report_date": datetime.now().isoformat(),
        "dataset": "PhysioNet Challenge 2019 - Training Set A",
        "total_patients": results_data["total_patients"],
        "successful_runs": results_data["successful"],
        "failed_runs": results_data["errors"],
        "metrics": metrics,
        "targets": {
            "sensitivity_target": 90.0,
            "sensitivity_met": metrics["sensitivity"] >= 90.0,
            "specificity_target": 85.0,
            "specificity_met": metrics["specificity"] >= 85.0
        }
    }

    path = RESULTS_DIR / "validation_analysis.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved: {path}")
    return report


def main():
    data = load_results()
    results = data["results"]
    metrics = calculate_metrics(results)
    print_report(metrics, data)
    save_report(metrics, data)


if __name__ == "__main__":
    main()
