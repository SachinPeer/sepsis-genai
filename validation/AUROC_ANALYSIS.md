# AUROC Analysis — Phase 1B eICU Demo (n=90)

**Date:** May 1, 2026
**Input:** `validation/results/EICU_results_20260501_205255.csv` (Phase 1B locked run)
**Purpose:** Provide a threshold-independent discrimination metric (AUROC) for head-to-head comparison with published sepsis-prediction models, and identify the Youden-optimal operating point.

---

## Headline AUROC

| Cohort | n | P (sepsis) | N (controls) | **AUROC** |
|---|---|---|---|---|
| Full Phase 1B | 90 | 30 | 60 | **0.745** |
| Excluding 5 neonates | 85 | 29 | 56 | **0.765** |

ROC curve: `validation/results/phase1b_roc_curve.png`

## Operating points table

| Threshold | Sens | Spec | PPV | Youden J | TP | FN | FP | TN |
|---|---|---|---|---|---|---|---|---|
| ≥15 | 100.0% | 0.0% | 33.3% | 0.000 | 30 | 0 | 60 | 0 |
| ≥25 | 90.0% | 26.7% | 38.0% | 0.167 | 27 | 3 | 44 | 16 |
| ≥35 | 86.7% | 51.7% | 47.3% | 0.383 | 26 | 4 | 29 | 31 |
| ≥42 | 76.7% | 58.3% | 47.9% | 0.350 | 23 | 7 | 25 | 35 |
| ≥45 | 76.7% | 61.7% | 50.0% | 0.383 | 23 | 7 | 23 | 37 |
| ≥58 | 73.3% | 63.3% | 50.0% | 0.367 | 22 | 8 | 22 | 38 |
| **≥70 (current prod)** | **73.3%** | **65.0%** | **51.2%** | **0.383** | 22 | 8 | 21 | 39 |
| **≥72 (Youden-optimal)** | **66.7%** | **81.7%** | **64.5%** | **0.483** | **20** | **10** | **11** | **49** |
| ≥75 | 60.0% | 85.0% | 66.7% | 0.450 | 18 | 12 | 9 | 51 |
| ≥82 | 56.7% | 85.0% | 65.4% | 0.417 | 17 | 13 | 9 | 51 |
| ≥85 | 46.7% | 86.7% | 63.6% | 0.333 | 14 | 16 | 8 | 52 |
| ≥92 | 40.0% | 86.7% | 60.0% | 0.267 | 12 | 18 | 8 | 52 |
| ≥95 | 30.0% | 86.7% | 52.9% | 0.167 | 9 | 21 | 8 | 52 |

Note: the risk-score distribution is quantized (13 unique values) because the LLM + guardrail system produces canonical scores (15, 25, 35, 42, 45, 58, 70, 72, 82, 85, 92, 95). The curve is therefore step-wise rather than smooth.

## Benchmark context

| Model / Study | Evaluation regime | AUROC |
|---|---|---|
| **PhysioNet Challenge 2019 winner** (Morrill et al., Oxford — "Can I get your signature?") | Hourly, across entire ICU stay, 40k train patients, structured-only data | **0.868** |
| **Our Phase 1B** | Single snapshot 6h pre-onset, no training, with nurse-note context | **0.745** (0.765 excl. neonates) |

**Why the comparison is not apples-to-apples:**
- Morrill scored hourly predictions across the full stay; we score once per patient at a fixed pre-onset snapshot.
- Morrill had 40,000 training patients; our system uses zero training data (pure LLM + clinical rules).
- Morrill's data had no nurse notes (not provided in the Challenge); our system integrates structured + free-text observations.
- Morrill's data used the Challenge's Sepsis-3 derivation; eICU labels us via ICD-9 (a weaker proxy).

## Observations

1. **Current production threshold (≥65-70) is close to the Sens/Spec balance point**, but Youden's J is maximized one step higher at **threshold 72** (66.7% Sens / 81.7% Spec).

2. **Specificity gain of +17 pts is available by moving threshold from 70 → 72**, at a sensitivity cost of 6.6 pts (4 missed sepsis of our 30 go uncaught). This is exactly the kind of trade-off we discussed with the "guardrail lowering" simulation and rejected — but here it's achievable via threshold-only tuning with no guardrail change.

3. **The curve is "two-plateau" shaped**: big jumps in sensitivity at low thresholds (adding controls doesn't help much), then a second cluster at high thresholds (severity confirms sepsis). Flat regions in between suggest our quantized scoring leaves some discrimination on the table — a continuous-score version of the model would likely AUROC-improve without any logic changes.

4. **Excluding neonates improves AUROC by 0.02**, roughly consistent with the specificity gain from the earlier adjusted-view analysis.

## What this enables

- **Single-number benchmark** for future PPTs and management updates: "AUROC 0.745 on 90-patient open-data validation" — a metric that's stable across thresholds and directly comparable to published sepsis papers.
- **Threshold-tuning roadmap** already mapped (see table above) — Red Rover can pick their operating point based on local ICU false-alarm tolerance, without any code change.
- **Continuous-score improvement** is a low-risk future optimization if we ever need a cleaner ROC.

---

## Files

| Artifact | Path |
|---|---|
| ROC curve image | `validation/results/phase1b_roc_curve.png` |
| Source results | `validation/results/EICU_results_20260501_205255.csv` |
| This analysis | `validation/AUROC_ANALYSIS.md` |
