# GenAI Sepsis Classification - Output Explained

## Overview

The GenAI pathway uses a **3-Stage Architecture** to analyze patient data and predict sepsis risk 6 hours ahead. It provides a **continuous risk score (0-100)** with clinical reasoning.

---

## How It Works: 3-Stage Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        GenAI 3-STAGE PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT                                                                      │
│  ┌─────────────────┐  ┌─────────────────┐                                  │
│  │ Patient Vitals  │  │ Clinician Notes │                                  │
│  │ (Structured)    │  │ (Unstructured)  │                                  │
│  └────────┬────────┘  └────────┬────────┘                                  │
│           │                    │                                            │
│           └──────────┬─────────┘                                            │
│                      ▼                                                      │
│  ╔═══════════════════════════════════════════════════════════════════════╗ │
│  ║  STAGE 1: PREPROCESSOR (Narrative Serialization)                      ║ │
│  ║  • Converts vitals + notes into clinical prose                        ║ │
│  ║  • Calculates trends (rising/falling)                                 ║ │
│  ║  • Identifies missing parameters                                      ║ │
│  ╚═══════════════════════════════════════════════════════════════════════╝ │
│                      │                                                      │
│                      ▼                                                      │
│  ╔═══════════════════════════════════════════════════════════════════════╗ │
│  ║  STAGE 2: AI INFERENCE (Clinical Intelligence Engine)                 ║ │
│  ║  • Analyzes clinical narrative as ICU specialist                      ║ │
│  ║  • Detects "Silent Sepsis" (discordance between vitals & notes)       ║ │
│  ║  • Calculates qSOFA, checks SIRS criteria                             ║ │
│  ║  • Returns structured JSON prediction                                 ║ │
│  ╚═══════════════════════════════════════════════════════════════════════╝ │
│                      │                                                      │
│                      ▼                                                      │
│  ╔═══════════════════════════════════════════════════════════════════════╗ │
│  ║  STAGE 3: GUARDRAIL (Deterministic Safety Validation)                 ║ │
│  ║  • Checks 45+ critical thresholds                                     ║ │
│  ║  • Detects septic shock criteria                                      ║ │
│  ║  • Detects DIC criteria                                               ║ │
│  ║  • OVERRIDES AI if critical thresholds breached                       ║ │
│  ╚═══════════════════════════════════════════════════════════════════════╝ │
│                      │                                                      │
│                      ▼                                                      │
│  OUTPUT                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Risk Score (0-100) + Priority + Clinical Rationale + Alerts        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Input Format

### Required: Patient Vitals

The system accepts vitals in multiple formats:

**Simple Format:**
```json
{
  "HR": 118,
  "SBP": 85,
  "Temp": 38.9,
  "Resp": 24,
  "O2Sat": 92,
  "Lactate": 3.2,
  "WBC": 15.8,
  "Creatinine": 2.1
}
```

**Red Rover Time-Series Format:**
```json
{
  "HR": [{"val": 118, "ts": "2026-02-07T18:30"}, {"val": 105, "ts": "2026-02-07T18:25"}],
  "SBP": [{"val": 85, "ts": "2026-02-07T18:30"}, {"val": 95, "ts": "2026-02-07T18:25"}]
}
```

### Optional: Clinician Notes

Unstructured nursing or physician notes provide critical context:

```
"Patient complains of chills. Altered mental status noted. 
Skin appears mottled on lower extremities. Required 2L fluid bolus 
with minimal BP improvement."
```

---

## Output Format

### Complete API Response Structure

```json
{
  "request_id": "genai_20260210_143022_P001",
  "patient_id": "P001",
  "status": "success",
  
  "risk_score": 85,
  "priority": "Critical",
  "sepsis_probability_6h": "High",
  "clinical_rationale": "Rising HR trend (105→118) with falling SBP (95→85) indicates hemodynamic deterioration. Nursing notes document mottled skin and minimal fluid response consistent with early septic shock.",
  
  "alert_level": "CRITICAL",
  "alert_color": "red",
  "action_required": "Immediate clinical review",
  
  "guardrail_override": false,
  "override_reasons": [],
  
  "total_processing_time_ms": 2847.5
}
```

---

## Output Fields Explained

### Primary Classification

| Field | Type | Description |
|-------|------|-------------|
| `risk_score` | Integer (0-100) | Overall sepsis risk percentage |
| `priority` | String | `Standard`, `High`, or `Critical` |
| `sepsis_probability_6h` | String | `Low`, `Moderate`, or `High` |
| `clinical_rationale` | String | AI model's reasoning for the assessment |

### Alert Information

| Field | Type | Description |
|-------|------|-------------|
| `alert_level` | String | `STANDARD`, `HIGH`, or `CRITICAL` |
| `alert_color` | String | `green`, `orange`, or `red` |
| `action_required` | String | Recommended clinical action |

### Guardrail Information

| Field | Type | Description |
|-------|------|-------------|
| `guardrail_override` | Boolean | `true` if guardrail overrode AI prediction |
| `override_reasons` | Array | List of critical thresholds breached |

---

## Priority Levels

### How Priority is Determined

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      PRIORITY DETERMINATION                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  CRITICAL (Risk 80-100, Red Alert)                                      │
│  ├── Septic shock criteria met (hypotension + elevated lactate)         │
│  ├── DIC criteria met (3+ of: low platelets, high INR, etc.)            │
│  ├── Any critical threshold breached (guardrail override)               │
│  └── AI assessment of imminent deterioration                            │
│                                                                         │
│  HIGH (Risk 50-79, Orange Alert)                                        │
│  ├── Multiple concerning findings                                       │
│  ├── Discordance detected (stable vitals but concerning notes)          │
│  ├── Deteriorating trend velocity                                       │
│  └── qSOFA score 2-3                                                    │
│                                                                         │
│  STANDARD (Risk 0-49, Green Alert)                                      │
│  ├── Stable or improving vitals                                         │
│  ├── No concerning patterns in notes                                    │
│  └── qSOFA score 0-1                                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Priority Mapping Table

| Risk Score | Priority | Alert Level | Alert Color | Action |
|------------|----------|-------------|-------------|--------|
| 0-29 | Standard | STANDARD | Green | Routine monitoring |
| 30-49 | Standard | STANDARD | Green | Routine monitoring |
| 50-69 | High | HIGH | Orange | Monitor closely |
| 70-79 | High | HIGH | Orange | Monitor closely |
| 80-89 | Critical | CRITICAL | Red | Immediate review |
| 90-100 | Critical | CRITICAL | Red | Immediate review |

---

## AI Reasoning Framework

The GenAI model applies three layers of analysis:

### 1. Discordance Analysis (Silent Sepsis Detection)

Compares structured vitals vs. unstructured notes:

| Scenario | Vitals | Notes | Interpretation |
|----------|--------|-------|----------------|
| **Discordant** | BP 115/70 (stable) | "Required 2L bolus, skin mottled" | **Compensated Shock** - Escalate! |
| **Concordant** | BP 85/50 (low) | "Hypotensive, on pressors" | Overt shock - matches |
| **Concordant** | BP 120/80 (normal) | "Patient stable, alert" | Stable - no escalation |

### 2. Velocity of Change (Trends)

The model analyzes direction, not just current values:

| Parameter | Current | Previous | Trend | Impact on Risk |
|-----------|---------|----------|-------|----------------|
| Heart Rate | 118 | 105 | ↑ Rising | Increases risk |
| SBP | 85 | 95 | ↓ Falling | Increases risk |
| Lactate | 2.1 | 2.1 | → Stable | Neutral |
| Creatinine | 1.8 | 1.2 | ↑ Rising | Increases risk |

### 3. Clinical Scoring

| Score | Criteria | Interpretation |
|-------|----------|----------------|
| **qSOFA** | RR ≥22, SBP ≤100, Altered mentation | Score 2-3 = High risk |
| **SIRS** | 2+ of: HR>90, RR>20, Temp abnormal, WBC abnormal | Indicates systemic inflammation |
| **Organ Stress** | Rising creatinine, falling platelets, acidosis | Organ dysfunction markers |

---

## Guardrail Override Logic

The guardrail can **override** the AI's assessment if critical thresholds are breached:

### When Override Occurs

```
IF (Any Critical Threshold Breached) AND (AI Risk Score < 80)
THEN → Force Risk Score to 95, Priority to "Critical"
```

### Critical Thresholds That Trigger Override

| Category | Parameter | Threshold |
|----------|-----------|-----------|
| **Hemodynamic** | SBP | ≤ 90 mmHg |
| **Hemodynamic** | MAP | < 65 mmHg |
| **Perfusion** | Lactate | ≥ 4.0 mmol/L |
| **Respiratory** | SpO2 | < 88% |
| **Respiratory** | RR | ≥ 30/min |
| **Renal** | Creatinine | ≥ 3.5 mg/dL |
| **Hematologic** | Platelets | < 50 K/µL |
| **Metabolic** | pH | < 7.25 |
| **Neurologic** | GCS | ≤ 8 |

### Septic Shock Auto-Detection

```
IF (SBP ≤ 90 OR MAP < 65) AND (Lactate ≥ 2.0)
THEN → Immediate escalation to Critical with "Septic Shock" flag
```

### DIC Auto-Detection

```
IF 3 or more of:
  - Platelets < 50 K/µL
  - INR ≥ 2.5
  - Fibrinogen < 100 mg/dL
  - D-Dimer ≥ 10 µg/mL
THEN → Flag as DIC, recommend hematology consult
```

---

## Example Outputs

### Example 1: Standard Risk Patient

**Input:**
```json
{
  "vitals": {"HR": 72, "SBP": 120, "Temp": 36.8, "Resp": 16, "O2Sat": 99},
  "notes": "Patient resting comfortably. No complaints."
}
```

**Output:**
```json
{
  "risk_score": 15,
  "priority": "Standard",
  "sepsis_probability_6h": "Low",
  "clinical_rationale": "Vitals within normal limits with stable trends. No concerning findings in nursing documentation.",
  "alert_level": "STANDARD",
  "alert_color": "green",
  "action_required": "Routine monitoring",
  "guardrail_override": false
}
```

---

### Example 2: High Risk Patient (Discordance Detected)

**Input:**
```json
{
  "vitals": {"HR": 95, "SBP": 108, "Temp": 37.2, "Resp": 20, "Lactate": 1.8},
  "notes": "Patient appears lethargic. Skin cool to touch with delayed cap refill. Required fluid bolus x2."
}
```

**Output:**
```json
{
  "risk_score": 72,
  "priority": "High",
  "sepsis_probability_6h": "Moderate",
  "clinical_rationale": "DISCORDANCE DETECTED: Vitals appear borderline stable but nursing notes indicate signs of hypoperfusion (cool skin, delayed cap refill, fluid requirements). This pattern suggests compensated shock - risk escalated.",
  "alert_level": "HIGH",
  "alert_color": "orange",
  "action_required": "Monitor closely",
  "guardrail_override": false
}
```

---

### Example 3: Critical Patient (Guardrail Override)

**Input:**
```json
{
  "vitals": {"HR": 125, "SBP": 78, "MAP": 52, "Temp": 39.2, "Lactate": 5.8, "Creatinine": 3.2},
  "notes": "Patient on norepinephrine. Urine output minimal."
}
```

**Output:**
```json
{
  "risk_score": 95,
  "priority": "Critical",
  "sepsis_probability_6h": "High",
  "clinical_rationale": "Multiple critical findings: Severe hypotension (SBP 78, MAP 52), elevated lactate (5.8), renal dysfunction (Cr 3.2). Meets septic shock criteria. Currently on vasopressor support.",
  "alert_level": "CRITICAL",
  "alert_color": "red",
  "action_required": "Immediate clinical review",
  "guardrail_override": true,
  "override_reasons": [
    "SBP 78 ≤ 90 (Critical hypotension)",
    "MAP 52 < 65 (Inadequate perfusion)",
    "Lactate 5.8 ≥ 4.0 (Severe hyperlactatemia)",
    "SEPTIC SHOCK CRITERIA MET: Hypotension + Lactate ≥ 2.0"
  ]
}
```

---

## API Endpoint

```
POST /genai-classify
```

**Headers:**
```
Content-Type: application/json
X-API-Key: your_api_key
```

**Request Body:**
```json
{
  "patient_id": "P001",
  "vitals": {
    "HR": 118,
    "SBP": 85,
    "Temp": 38.9,
    "Lactate": 3.2
  },
  "notes": "Patient appears lethargic, skin mottled."
}
```

---

## Summary

The GenAI pathway provides:

1. **Risk Score (0-100)** - Continuous measure of sepsis likelihood
2. **Priority Level** - Standard / High / Critical
3. **6-Hour Probability** - Low / Moderate / High
4. **Clinical Rationale** - AI reasoning in natural language
5. **Guardrail Protection** - Deterministic safety override when critical thresholds breached
6. **Discordance Detection** - Identifies "Silent Sepsis" when vitals and notes don't match

This approach focuses on **predicting sepsis 6 hours ahead** rather than just classifying current state.

---

*Document Version: 1.1*
*Last Updated: February 2026*
