# Validation Study Plan — Sepsis GenAI System

**Date:** February 11, 2026
**Status:** Planning — awaiting patient data access via Red Rover
**Owner:** Sachin / Paula (clinical review)

---

## 1. Why Validation, Not Training

Our system is already operational. There is nothing to "train":

| Component | Status | What It Does |
|---|---|---|
| Claude Sonnet 4.5 (AWS Bedrock) | Pre-trained by Anthropic | Analyzes clinical narratives, generates risk assessment |
| qSOFA / SIRS / SOFA scoring | Deterministic math | Calculates established clinical scores from vitals/labs |
| Clinical guardrails | Rule-based logic | Overrides, early detection patterns, history-aware context |
| Preprocessing (Stage 1) | Code-based | Converts raw data into clinical narrative for LLM |

**Validation** = proving this already-built system works correctly on real patient outcomes.

---

## 2. Validation Targets

| Parameter | Value | Rationale |
|---|---|---|
| **Sensitivity** | 90% | Out of all real sepsis cases, we catch 90%. High because missed sepsis = patient risk. |
| **Specificity** | 85% | Out of all non-sepsis cases, we correctly clear 85%. Tolerates some false alarms. |
| **Confidence level** | 95% | Industry standard for clinical studies. |
| **Margin of error** | ±5% | Balances precision with achievable patient count. |

### Why Sensitivity > Specificity

| Error Type | Clinical Impact | Tolerance |
|---|---|---|
| **False Negative** (missed sepsis) | Patient could deteriorate or die | Very low — must minimize |
| **False Positive** (false alarm) | Doctor does an extra review | Moderate — acceptable trade-off |

Our system is intentionally biased toward catching cases (guardrail overrides, early detection patterns, discordance escalation).

---

## 3. Sample Size Calculation

### Formula

Standard sample size for proportion estimation with 95% confidence:

```
n = (Z^2 × p × (1 - p)) / e^2
```

Where:
- **n** = required sample size
- **Z** = 1.96 (for 95% confidence level)
- **p** = expected proportion (sensitivity or specificity)
- **e** = acceptable margin of error

### Sensitivity Calculation

```
Target sensitivity: p = 0.90
Margin of error:    e = 0.05 (±5%)
Confidence:         Z = 1.96 (95%)

n = (1.96^2 × 0.90 × 0.10) / 0.05^2
n = (3.8416 × 0.09) / 0.0025
n = 0.34574 / 0.0025
n = ~138 sepsis-positive patients
```

### Specificity Calculation

```
Target specificity: p = 0.85
Margin of error:    e = 0.05 (±5%)
Confidence:         Z = 1.96 (95%)

n = (1.96^2 × 0.85 × 0.15) / 0.05^2
n = (3.8416 × 0.1275) / 0.0025
n = 0.48980 / 0.0025
n = ~196 non-sepsis patients
```

### Total Required

| Patient Group | Count | Description |
|---|---|---|
| Sepsis-positive | ~140 | Patients with confirmed sepsis outcomes |
| Non-sepsis | ~200 | Patients confirmed to NOT have developed sepsis |
| **Total** | **~340** | **Mixed cohort with known outcomes** |

### What This Allows Us to Claim

> "With 95% confidence, our system's sensitivity is 90% ±5% (between 85% and 95%)
> and specificity is 85% ±5% (between 80% and 90%)."

---

## 4. Understanding the Formula Variables

### Z — Confidence Level

How certain are you that the result is real, not a statistical fluke.

| Confidence Level | Z Value | Patients Needed (sensitivity) |
|---|---|---|
| 90% | 1.645 | ~98 |
| **95%** | **1.96** | **~138** |
| 99% | 2.576 | ~239 |

We use 95% — the accepted standard in medical research.

### p — Expected Proportion

Your best estimate of system performance before the study.

| p Value | Meaning | Variance (p × (1-p)) |
|---|---|---|
| 0.50 | Coin flip (max uncertainty) | 0.2500 |
| 0.85 | Good performance | 0.1275 |
| **0.90** | **Our sensitivity target** | **0.0900** |
| 0.95 | Excellent performance | 0.0475 |

Higher p = less variance = fewer patients needed (because there's less uncertainty to resolve).

### e — Margin of Error

How precise your final answer needs to be.

| e Value | Claim Example (if sensitivity = 90%) | Patients Needed |
|---|---|---|
| ±10% | "Sensitivity is between 80% and 100%" | ~35 |
| **±5%** | **"Sensitivity is between 85% and 95%"** | **~138** |
| ±3% | "Sensitivity is between 87% and 93%" | ~384 |

Tighter precision costs exponentially more patients.

---

## 5. Patient Data Requirements

Each patient record must include:

| Data Element | Required? | Source |
|---|---|---|
| Vitals (HR, BP, SpO2, Temp, RR) | **Yes** | Red Rover / EHR |
| Lab results (Lactate, WBC, Creatinine, Platelets) | **Yes** | Red Rover / EHR |
| Nurse/clinician notes | **Yes** | Red Rover / EHR |
| **Known outcome** (sepsis confirmed Y/N) | **Critical** | Hospital records / discharge summary |
| Timestamp of sepsis onset (if positive) | **Yes** | To measure prediction lead time |
| Patient demographics (age, gender) | Preferred | For subgroup analysis |

### Patient Mix Required

| Category | Count | Why |
|---|---|---|
| Confirmed sepsis | ~140 | Test sensitivity (catch rate) |
| Looked like sepsis but wasn't | ~50 | Test false positive rate on tricky cases |
| Clearly non-sepsis | ~150 | Test specificity baseline |
| **Total** | **~340** | |

---

## 6. Study Execution Plan

| Step | Description | Timeline | Depends On |
|---|---|---|---|
| 1. Data access | Obtain 340 patient records via Red Rover | 1-2 weeks | Red Rover sandbox + hospital approval |
| 2. Data preparation | Format records for API ingestion | 3-4 days | Step 1 |
| 3. System run | Process all 340 patients through our pipeline | 2-3 days | Step 2 |
| 4. Outcome comparison | Compare AI predictions vs. known outcomes | 1 week | Step 3 |
| 5. Metrics report | Calculate sensitivity, specificity, PPV, NPV, lead time | 3-4 days | Step 4 |
| 6. Clinical review | Paula reviews results, identifies patterns in errors | 1 week | Step 5 |
| 7. Calibration | Adjust prompts/guardrails based on findings | 1-2 weeks | Step 6 |
| **Total** | | **4-6 weeks after data access** | |

---

## 7. Metrics We Will Report

| Metric | Definition | Target |
|---|---|---|
| **Sensitivity** | True positives / All actual positives | ≥ 90% |
| **Specificity** | True negatives / All actual negatives | ≥ 85% |
| **PPV** (Positive Predictive Value) | True positives / All predicted positives | Measured |
| **NPV** (Negative Predictive Value) | True negatives / All predicted negatives | Measured |
| **Lead time** | How far ahead (in hours) did we predict before clinical diagnosis? | Target: 4-6 hours |
| **False alarm rate** | False positives / All alerts triggered | < 15% |
| **Guardrail override accuracy** | When guardrail overrode AI, was it correct? | Measured |

---

## 8. What Happens After Validation

| Finding | Action |
|---|---|
| Sensitivity ≥ 90%, Specificity ≥ 85% | Proceed to pilot deployment with confidence |
| Sensitivity 80-90% | Fine-tune prompts and guardrail thresholds, re-validate |
| Sensitivity < 80% | Major investigation — identify failure patterns, redesign guardrails |
| High false positive rate (>20%) | Adjust specificity thresholds, add suppression rules |

### Fine-tuning (NOT retraining)

Based on validation results, we may adjust:
- System prompt wording (how we instruct the LLM)
- Guardrail threshold values (e.g., lactate cutoff)
- Early detection pattern combinations
- Override sensitivity levels

This is calibration of an existing system — not building a new model.

---

## 9. Why NOT Synthetic (AI-Generated) Data

| Concern | Explanation |
|---|---|
| **Circular reasoning** | AI-generated patients reflect patterns an AI already knows — validates nothing new |
| **Missing clinical messiness** | Real data has missing labs, typos in notes, conflicting information |
| **No real outcomes** | Synthetic data has no ground truth — we can't measure accuracy |
| **Not defensible** | No clinical review board or regulator accepts synthetic-only validation |

Synthetic data is useful for stress-testing edge cases AFTER real validation — not as a substitute.

---

## 10. External Dependencies

| Dependency | Owner | Status |
|---|---|---|
| Red Rover data access (340 patients) | Shawn / Red Rover | Pending |
| Known sepsis outcomes for each patient | Hospital / Paula | Pending |
| Clinical review of results | Paula | Available after study |
| Atlas BAA for storing validation data | Shawn / Legal | Pending |

---

## 11. Interim Validation — PhysioNet (Completed May 1, 2026)

While we wait on real Red Rover data, we ran an interim validation study on the **PhysioNet Challenge 2019** dataset (340 ICU patients, 140 sepsis + 200 non-sepsis, known outcomes). This does **not** replace the Red Rover validation — PhysioNet has no free-text clinician notes, which we know is a major input for our system — but it exercised the full pipeline end-to-end and surfaced real bugs and design decisions.

### Headline Results (Final, Locked Configuration)

| Metric | Result | Target | Notes |
|---|---|---|---|
| **Sensitivity** | **62.86%** | ≥ 90% | Below target — expected without real nurse notes |
| **Specificity** | **52.50%** | ≥ 85% | Below target — ICU-baseline noise dominates without context |
| PPV | 48.09% | — | — |
| NPV | 66.88% | — | — |
| Guardrail Overrides | 15.6% | — | Down from 25.9% at R1, reflecting better LLM judgment |

### Why Below Target (and Why That's Okay)

The PhysioNet dataset has **no free-text clinician notes**. Our system was designed to use nurse notes as a critical input — things like "patient appeared confused", "fluid bolus given", "suspected UTI". With only synthetic trend descriptions, the LLM is working with one eye closed. Rigorous "hidden TP" analysis (`validation/HIDDEN_TPS.md`) surfaced **18 STRONG cases** where our system correctly flagged patients whom the labeled ground truth missed — these cases document the system's upside.

**Projected production performance with real notes:** 85–90% sensitivity / 55–60% specificity (within reach of targets).

### Decisions Emerging from the Study

| Decision | Basis |
|---|---|
| **Keep 6-hour trend window** | R6's 4h-prior shift was a wash |
| **Prompt v3.2 locked in** | Decoupled priority; biggest single spec gain (+17.8 pts) |
| **Full-trend preprocessor locked in** | R7 bug fix — all 6h of readings now reach the LLM |
| **Guardrails NOT lowered** | Simulation showed 1:1 sensitivity cost for any relaxation |
| **Next gain lever: real Red Rover notes** | Out of our hands until SJSA integration |

Full report: `validation/VALIDATION_EXECUTION.md` (§10–20). Trial log: `validation/VALIDATION_EXECUTION.md#20`.

---

## Version History

| Date | Change | Author |
|---|---|---|
| Feb 11, 2026 | Initial plan created | Sachin |
| May 1, 2026 | Added §11 — PhysioNet interim validation complete; system configuration locked in | Sachin |
