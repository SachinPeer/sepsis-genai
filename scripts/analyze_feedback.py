#!/usr/bin/env python3
"""
Feedback Analysis Script for Sepsis GenAI

Analyzes doctor feedback from MongoDB to identify patterns
and generate prompt improvement recommendations.

Usage:
    python analyze_feedback.py --mongo-uri "mongodb://..." --output prompt_updates.md
    
Example:
    python analyze_feedback.py \
        --mongo-uri "mongodb://localhost:27017" \
        --database "medbacon" \
        --days 90 \
        --output ../docs/feedback_report.md
"""

import os
import json
import argparse
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Any, Optional

# MongoDB driver
try:
    from pymongo import MongoClient
except ImportError:
    print("Error: pymongo not installed.")
    print("Install with: pip install pymongo")
    exit(1)


class FeedbackAnalyzer:
    """Analyzes doctor feedback to improve AI predictions."""
    
    def __init__(self, mongo_uri: str, database: str = "medbacon"):
        """
        Initialize the analyzer with MongoDB connection.
        
        Args:
            mongo_uri: MongoDB connection string
            database: Database name (default: medbacon)
        """
        self.client = MongoClient(mongo_uri)
        self.db = self.client[database]
        self.feedback_collection = self.db["sepsis_feedback"]
        
    def get_feedback(self, days: int = 90) -> List[Dict]:
        """
        Retrieve recent feedback from MongoDB.
        
        Args:
            days: Number of days to look back
            
        Returns:
            List of feedback documents
        """
        cutoff = datetime.utcnow() - timedelta(days=days)
        return list(self.feedback_collection.find({
            "timestamp": {"$gte": cutoff}
        }))
    
    def analyze_accuracy(self, feedback: List[Dict]) -> Dict[str, Any]:
        """
        Calculate overall accuracy metrics.
        
        Args:
            feedback: List of feedback documents
            
        Returns:
            Dictionary with accuracy metrics
        """
        total = len(feedback)
        if total == 0:
            return {"error": "No feedback data", "total_feedback": 0}
        
        agreement_counts = defaultdict(int)
        for f in feedback:
            agreement = f.get("doctor_feedback", {}).get("agreement", "unknown")
            agreement_counts[agreement] += 1
        
        # Calculate accuracy rate (agree + strongly_agree)
        positive = agreement_counts.get("strongly_agree", 0) + agreement_counts.get("agree", 0)
        accuracy_rate = (positive / total) * 100 if total > 0 else 0
        
        return {
            "total_feedback": total,
            "agreement_distribution": dict(agreement_counts),
            "accuracy_rate": round(accuracy_rate, 1),
            "needs_improvement": accuracy_rate < 80
        }
    
    def analyze_score_calibration(self, feedback: List[Dict]) -> Dict[str, Any]:
        """
        Analyze if AI over/under estimates risk scores.
        
        Args:
            feedback: List of feedback documents
            
        Returns:
            Dictionary with calibration analysis
        """
        overestimates = []
        underestimates = []
        accurate = []
        
        for f in feedback:
            ai_score = f.get("ai_prediction", {}).get("risk_score")
            corrected = f.get("doctor_feedback", {}).get("corrected_risk_score")
            
            if ai_score is None or corrected is None:
                continue
                
            diff = ai_score - corrected
            
            if diff > 10:  # AI overestimated by more than 10 points
                overestimates.append({
                    "ai_score": ai_score,
                    "corrected": corrected,
                    "diff": diff,
                    "context": f.get("context", {}),
                    "notes": f.get("doctor_feedback", {}).get("notes", "")
                })
            elif diff < -10:  # AI underestimated by more than 10 points
                underestimates.append({
                    "ai_score": ai_score,
                    "corrected": corrected,
                    "diff": diff,
                    "context": f.get("context", {}),
                    "notes": f.get("doctor_feedback", {}).get("notes", "")
                })
            else:
                accurate.append(diff)
        
        return {
            "overestimates": {
                "count": len(overestimates),
                "avg_overestimate": round(sum(e["diff"] for e in overestimates) / len(overestimates), 1) if overestimates else 0,
                "common_contexts": self._find_common_patterns(overestimates),
                "examples": overestimates[:3]
            },
            "underestimates": {
                "count": len(underestimates),
                "avg_underestimate": round(sum(e["diff"] for e in underestimates) / len(underestimates), 1) if underestimates else 0,
                "common_contexts": self._find_common_patterns(underestimates),
                "examples": underestimates[:3]
            },
            "accurate_count": len(accurate),
            "calibration_bias": "overestimates" if len(overestimates) > len(underestimates) * 1.5 else "underestimates" if len(underestimates) > len(overestimates) * 1.5 else "balanced"
        }
    
    def analyze_by_context(self, feedback: List[Dict]) -> Dict[str, Any]:
        """
        Analyze accuracy by clinical context.
        
        Args:
            feedback: List of feedback documents
            
        Returns:
            Dictionary with context-specific accuracy
        """
        context_accuracy = defaultdict(lambda: {"total": 0, "accurate": 0, "issues": []})
        
        for f in feedback:
            context = f.get("context", {}).get("clinical_context", "unknown")
            agreement = f.get("doctor_feedback", {}).get("agreement", "")
            
            context_accuracy[context]["total"] += 1
            
            if agreement in ["strongly_agree", "agree"]:
                context_accuracy[context]["accurate"] += 1
            else:
                context_accuracy[context]["issues"].append({
                    "ai_score": f.get("ai_prediction", {}).get("risk_score"),
                    "corrected": f.get("doctor_feedback", {}).get("corrected_risk_score"),
                    "notes": f.get("doctor_feedback", {}).get("notes", "")
                })
        
        results = {}
        for context, data in context_accuracy.items():
            accuracy = (data["accurate"] / data["total"]) * 100 if data["total"] > 0 else 0
            results[context] = {
                "total": data["total"],
                "accuracy": round(accuracy, 1),
                "needs_attention": accuracy < 75 and data["total"] >= 5,
                "sample_issues": data["issues"][:3]
            }
        
        return results
    
    def analyze_outcomes(self, feedback: List[Dict]) -> Dict[str, Any]:
        """
        Analyze prediction vs actual patient outcomes.
        
        Args:
            feedback: List of feedback documents
            
        Returns:
            Dictionary with outcome analysis
        """
        outcome_matrix = defaultdict(lambda: defaultdict(int))
        
        for f in feedback:
            ai_priority = f.get("ai_prediction", {}).get("priority", "unknown")
            outcome = f.get("doctor_feedback", {}).get("actual_outcome", "unknown")
            outcome_matrix[ai_priority][outcome] += 1
        
        # Calculate sensitivity/specificity for Critical predictions
        critical_data = outcome_matrix.get("Critical", {})
        high_data = outcome_matrix.get("High", {})
        standard_data = outcome_matrix.get("Standard", {})
        
        # True positives: Critical prediction + sepsis confirmed
        true_pos = critical_data.get("sepsis_confirmed", 0) + critical_data.get("septic_shock", 0)
        
        # False positives: Critical prediction + no sepsis
        false_pos = critical_data.get("no_infection", 0) + critical_data.get("infection_no_sepsis", 0)
        
        # False negatives: Standard/High prediction but patient developed sepsis
        false_neg = (
            standard_data.get("sepsis_confirmed", 0) + 
            standard_data.get("septic_shock", 0) +
            high_data.get("septic_shock", 0)
        )
        
        # True negatives
        true_neg = (
            standard_data.get("no_infection", 0) + 
            standard_data.get("infection_no_sepsis", 0) +
            high_data.get("no_infection", 0)
        )
        
        sensitivity = (true_pos / (true_pos + false_neg) * 100) if (true_pos + false_neg) > 0 else 0
        specificity = (true_neg / (true_neg + false_pos) * 100) if (true_neg + false_pos) > 0 else 0
        ppv = (true_pos / (true_pos + false_pos) * 100) if (true_pos + false_pos) > 0 else 0
        
        return {
            "outcome_matrix": {k: dict(v) for k, v in outcome_matrix.items()},
            "sensitivity": round(sensitivity, 1),
            "specificity": round(specificity, 1),
            "positive_predictive_value": round(ppv, 1),
            "false_positive_rate": round(100 - specificity, 1),
            "missed_sepsis_cases": false_neg,
            "false_alarms": false_pos
        }
    
    def analyze_confidence_calibration(self, feedback: List[Dict]) -> Dict[str, Any]:
        """
        Analyze if confidence levels match actual accuracy.
        
        Args:
            feedback: List of feedback documents
            
        Returns:
            Dictionary with confidence calibration
        """
        confidence_accuracy = defaultdict(lambda: {"total": 0, "accurate": 0})
        
        for f in feedback:
            confidence = f.get("ai_prediction", {}).get("confidence_level", "unknown")
            agreement = f.get("doctor_feedback", {}).get("agreement", "")
            
            confidence_accuracy[confidence]["total"] += 1
            if agreement in ["strongly_agree", "agree"]:
                confidence_accuracy[confidence]["accurate"] += 1
        
        results = {}
        for conf, data in confidence_accuracy.items():
            accuracy = (data["accurate"] / data["total"]) * 100 if data["total"] > 0 else 0
            
            # Expected accuracy based on confidence level
            expected = {"High": 90, "Medium": 70, "Low": 50}.get(conf, 70)
            
            results[conf] = {
                "total": data["total"],
                "actual_accuracy": round(accuracy, 1),
                "expected_accuracy": expected,
                "calibrated": abs(accuracy - expected) <= 15
            }
        
        return results
    
    def _find_common_patterns(self, cases: List[Dict]) -> List[str]:
        """
        Find common patterns in a set of error cases.
        
        Args:
            cases: List of error case dictionaries
            
        Returns:
            List of common pattern strings
        """
        patterns = defaultdict(int)
        
        for case in cases:
            context = case.get("context", {})
            
            # Clinical context
            clinical = context.get("clinical_context", "")
            if clinical:
                patterns[f"clinical_context: {clinical}"] += 1
            
            # Demographics
            age_range = context.get("patient_demographics", {}).get("age_range", "")
            if age_range:
                patterns[f"age_range: {age_range}"] += 1
            
            # Comorbidities
            comorbidities = context.get("patient_demographics", {}).get("comorbidities", [])
            for c in comorbidities:
                patterns[f"comorbidity: {c}"] += 1
            
            # Keywords from notes
            notes = case.get("notes", "").lower()
            keywords = [
                "lactate", "trend", "post-op", "elderly", "renal", "cardiac",
                "immunocompromised", "diabetes", "fever", "hypothermia",
                "tachycardia", "hypotension", "mental status", "urine output"
            ]
            for kw in keywords:
                if kw in notes:
                    patterns[f"keyword: {kw}"] += 1
        
        # Return top patterns (sorted by frequency)
        sorted_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)
        return [p[0] for p in sorted_patterns[:5]]
    
    def generate_prompt_recommendations(self, feedback: List[Dict]) -> str:
        """
        Generate specific recommendations for prompt improvements.
        
        Args:
            feedback: List of feedback documents
            
        Returns:
            Markdown string with recommendations
        """
        calibration = self.analyze_score_calibration(feedback)
        context_analysis = self.analyze_by_context(feedback)
        outcomes = self.analyze_outcomes(feedback)
        confidence = self.analyze_confidence_calibration(feedback)
        
        recommendations = []
        
        # Calibration recommendations
        if calibration["calibration_bias"] == "overestimates":
            patterns = calibration["overestimates"]["common_contexts"]
            avg = calibration["overestimates"]["avg_overestimate"]
            recommendations.append(f"""
### Overestimation Pattern Detected

The AI tends to **OVERESTIMATE** risk by an average of {avg} points.

**Common contexts where this occurs:**
{chr(10).join(f'- {p}' for p in patterns)}

**Recommended prompt addition:**
```
CALIBRATION - OVERESTIMATION TENDENCY:
You have been overestimating risk scores. Be more conservative when:
{chr(10).join(f'- {p.replace("clinical_context: ", "Patient is ").replace("keyword: ", "Notes mention ").replace("comorbidity: ", "Patient has ")}' for p in patterns[:3])}
Consider reducing your initial risk estimate by 10-15 points in these scenarios.
```
""")
        
        elif calibration["calibration_bias"] == "underestimates":
            patterns = calibration["underestimates"]["common_contexts"]
            avg = abs(calibration["underestimates"]["avg_underestimate"])
            recommendations.append(f"""
### Underestimation Pattern Detected

The AI tends to **UNDERESTIMATE** risk by an average of {avg} points.

**Common contexts where this occurs:**
{chr(10).join(f'- {p}' for p in patterns)}

**Recommended prompt addition:**
```
CALIBRATION - UNDERESTIMATION TENDENCY:
You have been underestimating risk scores. Be more aggressive when:
{chr(10).join(f'- {p.replace("clinical_context: ", "Patient is ").replace("keyword: ", "Notes mention ").replace("comorbidity: ", "Patient has ")}' for p in patterns[:3])}
Consider increasing your initial risk estimate by 10-15 points in these scenarios.
```
""")
        
        # Context-specific recommendations
        problem_contexts = [(ctx, data) for ctx, data in context_analysis.items() 
                          if data.get("needs_attention") and data["total"] >= 5]
        
        if problem_contexts:
            for context, data in problem_contexts:
                sample_notes = [i.get("notes", "") for i in data["sample_issues"] if i.get("notes")]
                recommendations.append(f"""
### Low Accuracy in {context.upper()} Context

**Accuracy:** {data['accuracy']}% (below 75% threshold)  
**Sample size:** {data['total']} cases

**Doctor feedback notes:**
{chr(10).join(f'- "{note}"' for note in sample_notes[:3] if note)}

**Recommended prompt addition:**
```
CONTEXT-SPECIFIC GUIDANCE - {context.upper()}:
Based on clinical feedback, pay special attention when patient context is {context}.
[Review the doctor notes above and add specific guidance]
```
""")
        
        # Outcome-based recommendations
        if outcomes["missed_sepsis_cases"] > 0:
            recommendations.append(f"""
### Missed Sepsis Cases

**{outcomes['missed_sepsis_cases']}** cases where AI predicted Standard/High but patient developed sepsis/septic shock.

**Current metrics:**
- Sensitivity: {outcomes['sensitivity']}%
- Missed cases: {outcomes['missed_sepsis_cases']}

**Recommended prompt addition:**
```
SAFETY PRIORITY:
Clinical feedback shows {outcomes['missed_sepsis_cases']} missed sepsis cases.
When in doubt, escalate priority. A false alarm is preferable to a missed sepsis case.
If ANY of these are present, consider High/Critical priority:
- Lactate trending upward (even if still normal)
- Subtle mental status changes
- Decreasing urine output
- Mottled skin
```
""")
        
        # Confidence calibration recommendations
        miscalibrated = [conf for conf, data in confidence.items() 
                        if not data.get("calibrated") and data["total"] >= 10]
        
        if miscalibrated:
            conf_details = []
            for conf in miscalibrated:
                data = confidence[conf]
                if data["actual_accuracy"] < data["expected_accuracy"]:
                    conf_details.append(f"- {conf} confidence: Actual accuracy {data['actual_accuracy']}% vs expected {data['expected_accuracy']}% (overconfident)")
                else:
                    conf_details.append(f"- {conf} confidence: Actual accuracy {data['actual_accuracy']}% vs expected {data['expected_accuracy']}% (underconfident)")
            
            recommendations.append(f"""
### Confidence Calibration Issues

**Miscalibrated confidence levels:**
{chr(10).join(conf_details)}

**Recommended prompt addition:**
```
CONFIDENCE CALIBRATION:
{chr(10).join(f'- Use "{conf}" confidence only when...' for conf in miscalibrated)}
```
""")
        
        if not recommendations:
            return "No significant patterns requiring prompt changes detected. The model appears well-calibrated."
        
        return "\n---\n".join(recommendations)
    
    def generate_full_report(self, days: int = 90) -> str:
        """
        Generate a comprehensive analysis report.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Markdown report string
        """
        feedback = self.get_feedback(days)
        
        if not feedback:
            return f"""# Feedback Analysis Report

**Generated:** {datetime.now().isoformat()}  
**Analysis Period:** Last {days} days  

## No Data Available

No feedback records found in the database for the specified period.

### Next Steps
1. Ensure the frontend is storing feedback to MongoDB collection `sepsis_feedback`
2. Wait for doctors to provide feedback on AI predictions
3. Re-run this analysis once you have at least 20-30 feedback records
"""
        
        accuracy = self.analyze_accuracy(feedback)
        calibration = self.analyze_score_calibration(feedback)
        context = self.analyze_by_context(feedback)
        outcomes = self.analyze_outcomes(feedback)
        confidence = self.analyze_confidence_calibration(feedback)
        recommendations = self.generate_prompt_recommendations(feedback)
        
        report = f"""# Feedback Analysis Report

**Generated:** {datetime.now().isoformat()}  
**Analysis Period:** Last {days} days  
**Total Feedback Records:** {accuracy['total_feedback']}

---

## 1. Executive Summary

| Metric | Value | Status |
|--------|-------|--------|
| Overall Accuracy | {accuracy['accuracy_rate']}% | {'✅ Good' if accuracy['accuracy_rate'] >= 80 else '⚠️ Needs Improvement'} |
| Calibration Bias | {calibration['calibration_bias']} | {'✅ Balanced' if calibration['calibration_bias'] == 'balanced' else '⚠️ Biased'} |
| Sensitivity | {outcomes['sensitivity']}% | {'✅ Good' if outcomes['sensitivity'] >= 85 else '⚠️ Low'} |
| Missed Sepsis Cases | {outcomes['missed_sepsis_cases']} | {'✅ None' if outcomes['missed_sepsis_cases'] == 0 else '🚨 Attention Needed'} |

---

## 2. Agreement Distribution

| Response | Count | Percentage |
|----------|-------|------------|
{chr(10).join(f"| {k} | {v} | {round(v/accuracy['total_feedback']*100, 1)}% |" for k, v in accuracy['agreement_distribution'].items())}

---

## 3. Score Calibration Analysis

### Overestimates (AI score > Corrected by >10 pts)
- **Count:** {calibration['overestimates']['count']}
- **Average overestimate:** {calibration['overestimates']['avg_overestimate']} points
- **Common patterns:** {', '.join(calibration['overestimates']['common_contexts'][:3]) or 'None identified'}

### Underestimates (AI score < Corrected by >10 pts)
- **Count:** {calibration['underestimates']['count']}
- **Average underestimate:** {calibration['underestimates']['avg_underestimate']} points
- **Common patterns:** {', '.join(calibration['underestimates']['common_contexts'][:3]) or 'None identified'}

### Accurate Predictions (within ±10 pts)
- **Count:** {calibration['accurate_count']}

---

## 4. Accuracy by Clinical Context

| Context | Total | Accuracy | Status |
|---------|-------|----------|--------|
{chr(10).join(f"| {ctx} | {data['total']} | {data['accuracy']}% | {'✅' if not data.get('needs_attention') else '⚠️'} |" for ctx, data in sorted(context.items(), key=lambda x: x[1]['total'], reverse=True))}

---

## 5. Outcome Analysis

### Confusion Matrix Summary

| AI Priority | Sepsis Confirmed | Septic Shock | No Infection | Other |
|-------------|------------------|--------------|--------------|-------|
| Critical | {outcomes['outcome_matrix'].get('Critical', {}).get('sepsis_confirmed', 0)} | {outcomes['outcome_matrix'].get('Critical', {}).get('septic_shock', 0)} | {outcomes['outcome_matrix'].get('Critical', {}).get('no_infection', 0)} | - |
| High | {outcomes['outcome_matrix'].get('High', {}).get('sepsis_confirmed', 0)} | {outcomes['outcome_matrix'].get('High', {}).get('septic_shock', 0)} | {outcomes['outcome_matrix'].get('High', {}).get('no_infection', 0)} | - |
| Standard | {outcomes['outcome_matrix'].get('Standard', {}).get('sepsis_confirmed', 0)} | {outcomes['outcome_matrix'].get('Standard', {}).get('septic_shock', 0)} | {outcomes['outcome_matrix'].get('Standard', {}).get('no_infection', 0)} | - |

### Key Metrics
- **Sensitivity:** {outcomes['sensitivity']}% (ability to catch true sepsis)
- **Specificity:** {outcomes['specificity']}% (ability to avoid false alarms)
- **Positive Predictive Value:** {outcomes['positive_predictive_value']}%
- **Missed Sepsis Cases:** {outcomes['missed_sepsis_cases']}
- **False Alarms:** {outcomes['false_alarms']}

---

## 6. Confidence Level Calibration

| Confidence | Total | Actual Accuracy | Expected | Calibrated? |
|------------|-------|-----------------|----------|-------------|
{chr(10).join(f"| {conf} | {data['total']} | {data['actual_accuracy']}% | {data['expected_accuracy']}% | {'✅' if data.get('calibrated') else '❌'} |" for conf, data in confidence.items() if conf != 'unknown')}

---

## 7. Recommendations for Prompt Update

{recommendations}

---

## 8. Suggested Prompt Addition

Copy this block and add it to your system prompt (`knowledge/Guide bank/prompt.md`):

```
## LEARNED CALIBRATIONS
Updated: {datetime.now().strftime('%Y-%m-%d')}
Based on: {accuracy['total_feedback']} clinical feedback records

### Overall Performance
- Agreement rate: {accuracy['accuracy_rate']}%
- Calibration bias: {calibration['calibration_bias']}
- Sensitivity: {outcomes['sensitivity']}%

### Adjustments
{'- Tendency to overestimate risk. Be more conservative in borderline cases.' if calibration['calibration_bias'] == 'overestimates' else '- Tendency to underestimate risk. Be more aggressive in escalating priority.' if calibration['calibration_bias'] == 'underestimates' else '- Calibration is balanced. Maintain current approach.'}

### Context-Specific Rules
{chr(10).join(f'- {ctx}: Accuracy {data["accuracy"]}% - {"needs attention" if data.get("needs_attention") else "performing well"}' for ctx, data in context.items() if data['total'] >= 5)}
```

---

## 9. Action Items

1. {"✅ No immediate action needed" if accuracy['accuracy_rate'] >= 80 and outcomes['missed_sepsis_cases'] == 0 else "⚠️ Review recommendations above and update prompt"}
2. Re-run this analysis in 30 days
3. Target: {max(accuracy['accuracy_rate'] + 5, 85)}% accuracy for next period

---

*Report generated by analyze_feedback.py*
"""
        
        return report


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Analyze doctor feedback for Sepsis GenAI improvement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze last 90 days of feedback
    python analyze_feedback.py --mongo-uri "mongodb://localhost:27017"
    
    # Analyze last 30 days and save to specific file
    python analyze_feedback.py --mongo-uri "mongodb://user:pass@host:27017" --days 30 --output report.md
    
    # Use specific database
    python analyze_feedback.py --mongo-uri "mongodb://localhost:27017" --database "medbacon_prod"
        """
    )
    
    parser.add_argument(
        "--mongo-uri", 
        required=True, 
        help="MongoDB connection URI (e.g., mongodb://localhost:27017)"
    )
    parser.add_argument(
        "--database", 
        default="medbacon", 
        help="Database name (default: medbacon)"
    )
    parser.add_argument(
        "--days", 
        type=int, 
        default=90, 
        help="Number of days of feedback to analyze (default: 90)"
    )
    parser.add_argument(
        "--output", 
        default="feedback_report.md", 
        help="Output file path (default: feedback_report.md)"
    )
    
    args = parser.parse_args()
    
    print(f"=" * 60)
    print(f"Sepsis GenAI Feedback Analyzer")
    print(f"=" * 60)
    
    print(f"\nConnecting to MongoDB: {args.mongo_uri.split('@')[-1] if '@' in args.mongo_uri else args.mongo_uri}")
    print(f"Database: {args.database}")
    
    try:
        analyzer = FeedbackAnalyzer(args.mongo_uri, args.database)
        
        print(f"\nAnalyzing feedback from last {args.days} days...")
        report = analyzer.generate_full_report(args.days)
        
        print(f"Writing report to: {args.output}")
        with open(args.output, 'w') as f:
            f.write(report)
        
        print(f"\n✅ Done! Report saved to {args.output}")
        print(f"\nNext steps:")
        print(f"  1. Review the report: cat {args.output}")
        print(f"  2. Update prompt with recommendations")
        print(f"  3. Test and deploy updated container")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print(f"\nTroubleshooting:")
        print(f"  - Check MongoDB URI is correct")
        print(f"  - Ensure MongoDB is running")
        print(f"  - Verify database and collection exist")
        exit(1)


if __name__ == "__main__":
    main()
