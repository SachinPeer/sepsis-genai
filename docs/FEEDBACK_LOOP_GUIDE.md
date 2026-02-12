# Doctor Feedback Loop: Complete Guide

This guide explains how to collect, store, analyze doctor feedback, and use it to improve the AI system's predictions.

---

## Overview

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  1. COLLECT │───▶│  2. STORE   │───▶│ 3. ANALYZE  │───▶│ 4. IMPROVE  │
│  Feedback   │    │  MongoDB    │    │  Patterns   │    │  Prompt     │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

---

## 1. Collecting Feedback

### What to Capture (Frontend Implementation)

The frontend team should capture this data when a doctor reviews an AI prediction:

```json
{
  "feedback_id": "fb_20260212_001",
  "request_id": "genai_20260212_050653_TREND_DEMO_001",
  "patient_id": "TREND_DEMO_001",
  "timestamp": "2026-02-12T10:30:00Z",
  
  "doctor_info": {
    "doctor_id": "DR_12345",
    "specialty": "ICU",
    "years_experience": 15
  },
  
  "ai_prediction": {
    "risk_score": 65,
    "confidence_level": "Medium",
    "priority": "High",
    "clinical_rationale": "Post-surgical patient showing early compensatory signs..."
  },
  
  "doctor_feedback": {
    "agreement": "partially_agree",
    "corrected_risk_score": 75,
    "corrected_priority": "High",
    "feedback_reason": "AI underestimated - lactate trend was concerning",
    "actual_outcome": "sepsis_confirmed",
    "outcome_timeframe_hours": 4,
    "notes": "Patient developed septic shock 4 hours later. AI should weigh lactate trends more heavily."
  },
  
  "context": {
    "patient_demographics": {
      "age_range": "65-75",
      "gender": "M",
      "comorbidities": ["diabetes", "CKD"]
    },
    "clinical_context": "post_surgical",
    "time_of_day": "night_shift"
  }
}
```

### Agreement Options

```javascript
const agreementOptions = [
  "strongly_agree",    // AI was spot-on
  "agree",             // AI was close enough
  "partially_agree",   // Right direction, wrong magnitude
  "disagree",          // AI was significantly off
  "strongly_disagree"  // AI was dangerously wrong
];
```

### Outcome Options

```javascript
const outcomeOptions = [
  "sepsis_confirmed",      // Blood culture positive / clinical sepsis
  "septic_shock",          // Required vasopressors
  "infection_no_sepsis",   // Infection but no organ dysfunction
  "no_infection",          // False alarm
  "unknown",               // Patient transferred / lost to follow-up
  "pending"                // Still monitoring
];
```

---

## 2. Storing Feedback (MongoDB)

### Collection Schema

```javascript
// MongoDB Collection: sepsis_feedback
db.createCollection("sepsis_feedback", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["feedback_id", "request_id", "timestamp", "doctor_feedback"],
      properties: {
        feedback_id: { bsonType: "string" },
        request_id: { bsonType: "string" },
        patient_id: { bsonType: "string" },
        timestamp: { bsonType: "date" },
        doctor_info: {
          bsonType: "object",
          properties: {
            doctor_id: { bsonType: "string" },
            specialty: { bsonType: "string" },
            years_experience: { bsonType: "int" }
          }
        },
        ai_prediction: {
          bsonType: "object",
          properties: {
            risk_score: { bsonType: "int" },
            confidence_level: { bsonType: "string" },
            priority: { bsonType: "string" }
          }
        },
        doctor_feedback: {
          bsonType: "object",
          required: ["agreement"],
          properties: {
            agreement: { bsonType: "string" },
            corrected_risk_score: { bsonType: "int" },
            corrected_priority: { bsonType: "string" },
            actual_outcome: { bsonType: "string" },
            notes: { bsonType: "string" }
          }
        }
      }
    }
  }
});

// Indexes for analysis queries
db.sepsis_feedback.createIndex({ "timestamp": -1 });
db.sepsis_feedback.createIndex({ "doctor_feedback.agreement": 1 });
db.sepsis_feedback.createIndex({ "doctor_feedback.actual_outcome": 1 });
db.sepsis_feedback.createIndex({ "ai_prediction.risk_score": 1 });
db.sepsis_feedback.createIndex({ "context.clinical_context": 1 });
```

---

## 3. Analyzing Feedback

### Analysis Script

Save this as `scripts/analyze_feedback.py`:

```python
#!/usr/bin/env python3
"""
Feedback Analysis Script for Sepsis GenAI

Analyzes doctor feedback from MongoDB to identify patterns
and generate prompt improvement recommendations.

Usage:
    python analyze_feedback.py --mongo-uri "mongodb://..." --output prompt_updates.md
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
    print("Install pymongo: pip install pymongo")
    exit(1)


class FeedbackAnalyzer:
    """Analyzes doctor feedback to improve AI predictions."""
    
    def __init__(self, mongo_uri: str, database: str = "medbacon"):
        self.client = MongoClient(mongo_uri)
        self.db = self.client[database]
        self.feedback_collection = self.db["sepsis_feedback"]
        
    def get_feedback(self, days: int = 90) -> List[Dict]:
        """Retrieve recent feedback."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        return list(self.feedback_collection.find({
            "timestamp": {"$gte": cutoff}
        }))
    
    def analyze_accuracy(self, feedback: List[Dict]) -> Dict[str, Any]:
        """Calculate overall accuracy metrics."""
        total = len(feedback)
        if total == 0:
            return {"error": "No feedback data"}
        
        agreement_counts = defaultdict(int)
        for f in feedback:
            agreement = f.get("doctor_feedback", {}).get("agreement", "unknown")
            agreement_counts[agreement] += 1
        
        # Calculate accuracy rate
        positive = agreement_counts.get("strongly_agree", 0) + agreement_counts.get("agree", 0)
        accuracy_rate = (positive / total) * 100 if total > 0 else 0
        
        return {
            "total_feedback": total,
            "agreement_distribution": dict(agreement_counts),
            "accuracy_rate": round(accuracy_rate, 1),
            "needs_improvement": accuracy_rate < 80
        }
    
    def analyze_score_calibration(self, feedback: List[Dict]) -> Dict[str, Any]:
        """Analyze if AI over/under estimates risk."""
        overestimates = []
        underestimates = []
        accurate = []
        
        for f in feedback:
            ai_score = f.get("ai_prediction", {}).get("risk_score")
            corrected = f.get("doctor_feedback", {}).get("corrected_risk_score")
            
            if ai_score is None or corrected is None:
                continue
                
            diff = ai_score - corrected
            
            if diff > 10:
                overestimates.append({
                    "ai_score": ai_score,
                    "corrected": corrected,
                    "diff": diff,
                    "context": f.get("context", {}),
                    "notes": f.get("doctor_feedback", {}).get("notes", "")
                })
            elif diff < -10:
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
                "avg_overestimate": sum(e["diff"] for e in overestimates) / len(overestimates) if overestimates else 0,
                "common_contexts": self._find_common_patterns(overestimates)
            },
            "underestimates": {
                "count": len(underestimates),
                "avg_underestimate": sum(e["diff"] for e in underestimates) / len(underestimates) if underestimates else 0,
                "common_contexts": self._find_common_patterns(underestimates)
            },
            "accurate_count": len(accurate),
            "calibration_bias": "overestimates" if len(overestimates) > len(underestimates) else "underestimates" if len(underestimates) > len(overestimates) else "balanced"
        }
    
    def analyze_by_context(self, feedback: List[Dict]) -> Dict[str, Any]:
        """Analyze accuracy by clinical context."""
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
                "needs_attention": accuracy < 75,
                "sample_issues": data["issues"][:3]  # Top 3 issues
            }
        
        return results
    
    def analyze_outcomes(self, feedback: List[Dict]) -> Dict[str, Any]:
        """Analyze prediction vs actual outcomes."""
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
        
        # False negatives: Standard/High prediction + sepsis
        false_neg = (standard_data.get("sepsis_confirmed", 0) + standard_data.get("septic_shock", 0) +
                     high_data.get("septic_shock", 0))
        
        sensitivity = true_pos / (true_pos + false_neg) * 100 if (true_pos + false_neg) > 0 else 0
        
        return {
            "outcome_matrix": {k: dict(v) for k, v in outcome_matrix.items()},
            "sensitivity_critical": round(sensitivity, 1),
            "false_positive_rate": round(false_pos / (false_pos + true_pos) * 100, 1) if (false_pos + true_pos) > 0 else 0,
            "missed_sepsis_cases": false_neg
        }
    
    def _find_common_patterns(self, cases: List[Dict]) -> List[str]:
        """Find common patterns in a set of cases."""
        patterns = defaultdict(int)
        
        for case in cases:
            context = case.get("context", {})
            clinical = context.get("clinical_context", "")
            if clinical:
                patterns[f"clinical_context: {clinical}"] += 1
            
            # Extract keywords from notes
            notes = case.get("notes", "").lower()
            keywords = ["lactate", "trend", "post-op", "elderly", "renal", "cardiac", 
                       "immunocompromised", "diabetes", "fever", "hypothermia"]
            for kw in keywords:
                if kw in notes:
                    patterns[f"keyword: {kw}"] += 1
        
        # Return top patterns
        sorted_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)
        return [p[0] for p in sorted_patterns[:5]]
    
    def generate_prompt_recommendations(self, feedback: List[Dict]) -> str:
        """Generate recommendations for prompt improvements."""
        calibration = self.analyze_score_calibration(feedback)
        context_analysis = self.analyze_by_context(feedback)
        outcomes = self.analyze_outcomes(feedback)
        
        recommendations = []
        
        # Calibration recommendations
        if calibration["overestimates"]["count"] > calibration["underestimates"]["count"]:
            patterns = calibration["overestimates"]["common_contexts"]
            recommendations.append(f"""
## Overestimation Pattern Detected

The AI tends to OVERESTIMATE risk (avg: +{calibration['overestimates']['avg_overestimate']:.0f} points).

Common contexts where this occurs:
{chr(10).join(f'- {p}' for p in patterns)}

**Recommended prompt addition:**
```
CALIBRATION NOTE: You have a tendency to overestimate risk. Be more conservative when:
{chr(10).join(f'- {p.replace("clinical_context: ", "Patient is ").replace("keyword: ", "Notes mention ")}' for p in patterns[:3])}
```
""")
        
        elif calibration["underestimates"]["count"] > calibration["overestimates"]["count"]:
            patterns = calibration["underestimates"]["common_contexts"]
            recommendations.append(f"""
## Underestimation Pattern Detected

The AI tends to UNDERESTIMATE risk (avg: {calibration['underestimates']['avg_underestimate']:.0f} points).

Common contexts where this occurs:
{chr(10).join(f'- {p}' for p in patterns)}

**Recommended prompt addition:**
```
CALIBRATION NOTE: Be more aggressive with risk scoring when:
{chr(10).join(f'- {p.replace("clinical_context: ", "Patient is ").replace("keyword: ", "Notes mention ")}' for p in patterns[:3])}
```
""")
        
        # Context-specific recommendations
        for context, data in context_analysis.items():
            if data["needs_attention"] and data["total"] >= 5:
                sample_notes = [i.get("notes", "") for i in data["sample_issues"] if i.get("notes")]
                recommendations.append(f"""
## Low Accuracy in {context.upper()} Context

Accuracy: {data['accuracy']}% (below 75% threshold)
Sample size: {data['total']}

Doctor feedback notes:
{chr(10).join(f'- "{note}"' for note in sample_notes[:3])}

**Recommended prompt addition:**
```
CONTEXT-SPECIFIC GUIDANCE for {context}:
[Add specific guidance based on the feedback patterns above]
```
""")
        
        # Outcome-based recommendations
        if outcomes["missed_sepsis_cases"] > 0:
            recommendations.append(f"""
## Missed Sepsis Cases Detected

{outcomes['missed_sepsis_cases']} cases where AI predicted Standard/High but patient developed sepsis/septic shock.

Current sensitivity for Critical predictions: {outcomes['sensitivity_critical']}%

**Recommended prompt addition:**
```
SAFETY NOTE: Err on the side of caution. When in doubt, escalate priority.
A missed sepsis case is worse than a false alarm.
```
""")
        
        return "\n\n---\n\n".join(recommendations) if recommendations else "No significant patterns requiring prompt changes detected."
    
    def generate_full_report(self, days: int = 90) -> str:
        """Generate a full analysis report."""
        feedback = self.get_feedback(days)
        
        if not feedback:
            return "# Feedback Analysis Report\n\nNo feedback data available for analysis."
        
        accuracy = self.analyze_accuracy(feedback)
        calibration = self.analyze_score_calibration(feedback)
        context = self.analyze_by_context(feedback)
        outcomes = self.analyze_outcomes(feedback)
        recommendations = self.generate_prompt_recommendations(feedback)
        
        report = f"""# Feedback Analysis Report

Generated: {datetime.now().isoformat()}
Analysis Period: Last {days} days
Total Feedback Records: {accuracy['total_feedback']}

---

## 1. Overall Accuracy

| Metric | Value |
|--------|-------|
| Total Feedback | {accuracy['total_feedback']} |
| Accuracy Rate | {accuracy['accuracy_rate']}% |
| Status | {'✅ Good' if not accuracy['needs_improvement'] else '⚠️ Needs Improvement'} |

### Agreement Distribution
{json.dumps(accuracy['agreement_distribution'], indent=2)}

---

## 2. Score Calibration

| Metric | Value |
|--------|-------|
| Overestimates (>10 pts) | {calibration['overestimates']['count']} |
| Underestimates (>10 pts) | {calibration['underestimates']['count']} |
| Accurate (±10 pts) | {calibration['accurate_count']} |
| Overall Bias | {calibration['calibration_bias']} |

---

## 3. Accuracy by Clinical Context

| Context | Total | Accuracy | Status |
|---------|-------|----------|--------|
{chr(10).join(f"| {ctx} | {data['total']} | {data['accuracy']}% | {'✅' if not data['needs_attention'] else '⚠️'} |" for ctx, data in context.items())}

---

## 4. Outcome Analysis

Sensitivity (Critical predictions): {outcomes['sensitivity_critical']}%
False Positive Rate: {outcomes['false_positive_rate']}%
Missed Sepsis Cases: {outcomes['missed_sepsis_cases']}

---

## 5. Prompt Improvement Recommendations

{recommendations}

---

## 6. Suggested Prompt Update

Based on the analysis above, here is the recommended addition to your system prompt:

```
LEARNED CALIBRATIONS (Updated {datetime.now().strftime('%Y-%m-%d')}):

Based on clinical feedback analysis ({accuracy['total_feedback']} cases):

1. Overall calibration bias: {calibration['calibration_bias']}
2. Sensitivity for critical cases: {outcomes['sensitivity_critical']}%
3. Contexts needing extra attention: {', '.join(ctx for ctx, data in context.items() if data['needs_attention'])}

Apply these adjustments to your risk assessment.
```

---

## Next Steps

1. Review the recommendations above
2. Update the system prompt in `knowledge/Guide bank/prompt.md`
3. Test with sample cases
4. Monitor next batch of feedback for improvement
"""
        
        return report


def main():
    parser = argparse.ArgumentParser(description="Analyze doctor feedback for AI improvement")
    parser.add_argument("--mongo-uri", required=True, help="MongoDB connection URI")
    parser.add_argument("--database", default="medbacon", help="Database name")
    parser.add_argument("--days", type=int, default=90, help="Days of feedback to analyze")
    parser.add_argument("--output", default="feedback_report.md", help="Output file path")
    
    args = parser.parse_args()
    
    print(f"Connecting to MongoDB...")
    analyzer = FeedbackAnalyzer(args.mongo_uri, args.database)
    
    print(f"Analyzing feedback from last {args.days} days...")
    report = analyzer.generate_full_report(args.days)
    
    print(f"Writing report to {args.output}...")
    with open(args.output, 'w') as f:
        f.write(report)
    
    print(f"Done! Report saved to {args.output}")


if __name__ == "__main__":
    main()
```

---

## 4. Embedding Insights in Prompt

### Where to Update

The system prompt is located at:
```
knowledge/Guide bank/prompt.md
```

### How to Update

After running the analysis script, add the recommended calibrations to your prompt:

```markdown
## SYSTEM ROLE

You are a Board-Certified ICU Intensivist...

## LEARNED CALIBRATIONS (Updated 2026-02-15)

Based on analysis of 250 doctor feedback records:

### Risk Score Adjustments
- Post-surgical patients: Tendency to overestimate by ~15 points. Be more conservative.
- Elderly (>75) with borderline lactate (1.8-2.2): Tendency to underestimate. Add 10 points.
- Immunocompromised patients: Current scoring is well-calibrated. Maintain approach.

### Confidence Level Guidance
- Missing temperature: Always use "Low" confidence
- Missing lactate: Use "Medium" confidence maximum
- Complete vital set with clear trends: Can use "High" confidence

### Context-Specific Rules
- Night shift admissions: Higher threshold for "Standard" priority (nurses may miss subtle signs)
- Post-cardiac surgery: Tachycardia alone is common; require additional signs for High priority

## OUTPUT FORMAT
...
```

### Update Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    MONTHLY PROMPT UPDATE WORKFLOW                │
└─────────────────────────────────────────────────────────────────┘

Week 1-3: Collect feedback
    ↓
Week 4: Run analysis
    python scripts/analyze_feedback.py --mongo-uri "..." --output report.md
    ↓
Review report.md
    ↓
Update knowledge/Guide bank/prompt.md with calibrations
    ↓
Test with 10 sample cases
    ↓
Deploy updated container
    ↓
Monitor next month's feedback for improvement
```

---

## 5. Measuring Improvement

Track these metrics month-over-month:

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Agreement Rate | >80% | % of "agree" + "strongly_agree" |
| Calibration Error | <10 pts | Avg difference between AI and corrected score |
| Sensitivity | >90% | % of actual sepsis cases predicted as High/Critical |
| False Positive Rate | <30% | % of Critical predictions that were no-sepsis |

### Create a Tracking Dashboard

```javascript
// MongoDB aggregation for monthly metrics
db.sepsis_feedback.aggregate([
  {
    $group: {
      _id: { $month: "$timestamp" },
      total: { $sum: 1 },
      agrees: {
        $sum: {
          $cond: [
            { $in: ["$doctor_feedback.agreement", ["agree", "strongly_agree"]] },
            1, 0
          ]
        }
      },
      avg_score_diff: {
        $avg: {
          $subtract: [
            "$ai_prediction.risk_score",
            "$doctor_feedback.corrected_risk_score"
          ]
        }
      }
    }
  },
  { $sort: { "_id": 1 } }
]);
```

---

## Quick Reference

### Commands

```bash
# Run feedback analysis
python scripts/analyze_feedback.py \
  --mongo-uri "mongodb://localhost:27017" \
  --database "medbacon" \
  --days 90 \
  --output docs/feedback_report.md

# View report
cat docs/feedback_report.md
```

### File Locations

| File | Purpose |
|------|---------|
| `scripts/analyze_feedback.py` | Analysis script |
| `knowledge/Guide bank/prompt.md` | System prompt (update here) |
| `docs/feedback_report.md` | Generated analysis report |
| `docs/FEEDBACK_LOOP_GUIDE.md` | This guide |

---

## Summary

1. **Collect**: Frontend captures doctor feedback → MongoDB
2. **Store**: MongoDB with proper schema and indexes
3. **Analyze**: Run `analyze_feedback.py` monthly
4. **Improve**: Update system prompt with calibrations
5. **Measure**: Track accuracy metrics month-over-month

This creates a continuous improvement loop without needing traditional model fine-tuning.
