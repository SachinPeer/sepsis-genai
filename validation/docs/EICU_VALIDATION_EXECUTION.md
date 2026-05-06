# eICU v4 Validation — Run Report

_Generated: 2026-05-06 14:24_

> **TL;DR.** On the v4 cohort the pipeline detects **82 % of true sepsis
> patients 6 hours before clinical recognition** (sensitivity 82.4 %,
> 95 % CI 66.5–91.7 %), but it also flags more than half of non-sepsis
> patients (specificity 44.8 %, 95 % CI 36.1–53.9 %). The miss rate is
> small and the safety-net behaviour we want; the false-alarm rate is too
> high for unattended deployment and is dominated by the guardrail's
> *Early Detection Escalation* rule. See §8 for tuning options.

## 1. Cohort & run summary

- **Cohort:** eICU-CRD Demo v2.0.1, cohort_v4 (Option B, ICD-only)
- **Patients attempted:** 150
- **Successful classifications:** 150
- **API errors:** 0
- **Sepsis (positives):** 34
- **Controls (negatives):** 116
- **Pipeline:** Preprocess → Claude Sonnet 4.5 (Bedrock) → Clinical Guardrail
- **Run duration:** 2,325 s (≈ 39 min); median per-patient latency 10.7 s

## 2. Headline metrics (overall)

| Metric | Value (95 % Wilson CI) |
|---|---|
| Sensitivity (Recall) |  82.4% (95% CI: 66.5 – 91.7) |
| Specificity |  44.8% (95% CI: 36.1 – 53.9) |
| PPV (Precision) |  30.4% (95% CI: 22.0 – 40.5) |
| NPV |  89.7% (95% CI: 79.2 – 95.2) |
| Accuracy |  53.3% (95% CI: 45.4 – 61.1) |
| F1 Score | 0.444 |
| False-alarm rate (1 − Spec) |  55.2% |

### Confusion matrix

| | Predicted: Sepsis | Predicted: No sepsis |
|---|:--:|:--:|
| **Actual: Sepsis** | TP = 28 | FN = 6 |
| **Actual: Control** | FP = 64 | TN = 52 |

## 3. Sub-group: with vs without nurse notes

| Sub-group | n | Sensitivity | Specificity | F1 |
|---|:--:|:--:|:--:|:--:|
| With notes  | 144 |  83.9% (95% CI: 67.4 – 92.9) |  46.0% (95% CI: 37.1 – 55.2) | 0.441 |
| Empty notes | 6 |  66.7% (95% CI: 20.8 – 93.9) |   0.0% (95% CI: 0.0 – 56.2) | 0.500 |

## 4. Guardrail behaviour

- Total guardrail overrides triggered: **43** of 150
  - Within sepsis cases: 8 of 34
  - Within controls (false alarms): 35 of 116

## 5. Alert-level distribution

| Alert level | All patients | of which sepsis |
|---|:--:|:--:|
| CRITICAL | 60 | 19 |
| HIGH | 32 | 9 |
| STANDARD | 58 | 6 |
| LOW | 0 | 0 |
| OTHER | 0 | 0 |

## 6. Latency

- Mean : 14,503.0 ms
- Median: 10,704.1 ms
- p95  : 13,402.2 ms
- Max  : 587,683.6 ms

## 7. Errors

- None — all classifications succeeded.

## 8. Root-cause analysis of the 64 false alarms

| Source of FP | Count | Share |
|---|:--:|:--:|
| Guardrail's *Early Detection Escalation* bumped a "low" LLM verdict to High | 27 | 42 % |
| LLM itself returned risk ≥ 50 (without guardrail bump) | 37 | 58 % |

The Early Detection rule fires when **2 or more** of the following are
present in the most recent vitals window: HR ≥ 90, RR ≥ 22, abnormal Temp,
abnormal WBC. In the ICU, these thresholds are met by the majority of
non-sepsis patients (post-op tachycardia, ventilator weaning, pain, fever
from non-infectious cause, etc.).

The LLM's own 37 false-positives reflect a different failure mode: the LLM
correctly identifies that the patient is *acutely ill* (the dominant signal
in any ICU patient), but cannot tell from vitals + notes alone that the
illness is non-infectious (e.g., post-arrest, DKA, GI bleed). It currently
has no negative-label corpus to anchor on.

### Top early-warning triggers seen in FPs (raw token counts)

| Trigger | Count |
|---|:--:|
| HR ≥ 90 (Early Detection) | 16 |
| RR ≥ 22 (Early Detection) | 12 |
| WBC ≥ 12 (Early Detection) | 2 |

### The 6 false negatives

| Patient | Risk | Alert | Override | Note |
|---|:--:|:--:|:--:|---|
| eicu_p00011 | 25 | Standard | No | LLM said low, no guardrail trigger |
| eicu_p00016 | 35 | Standard | No | Same |
| eicu_p00019 | 15 | Standard | No | Same |
| eicu_p00022 | 35 | Standard | No | Same |
| eicu_p00025 | 35 | Standard | No | Same |
| eicu_p00026 | 35 | Standard | No | Same |

> All six FNs sit at risk-score 15–35 with no guardrail bump.
> Recommendation: queue these for SME review (Paula) before publication —
> they may be ICD-9 coding artefacts (sepsis listed as differential, never
> confirmed) rather than true model misses.

## 9. Threshold sweep

The default cutoff for "predicted sepsis" is risk ≥ 50. Sweeping the cutoff
shows the distribution is bimodal — most controls are at 70+ (because the
guardrail bumps them) and most "clean" controls are at 25–35:

| Cutoff | TP / FN | FP / TN | Sens | Spec | PPV | F1 | False alarm |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 50 *(current)* | 28 / 6 | 64 / 52 | 82.4 % | 44.8 % | 30.4 % | 0.444 | 55.2 % |
| 60 | 28 / 6 | 64 / 52 | 82.4 % | 44.8 % | 30.4 % | 0.444 | 55.2 % |
| 70 | 27 / 7 | 63 / 53 | 79.4 % | 45.7 % | 30.0 % | 0.435 | 54.3 % |
| **75** | **20 / 14** | **41 / 75** | **58.8 %** | **64.7 %** | **32.8 %** | **0.421** | **35.3 %** |
| 80 | 19 / 15 | 41 / 75 | 55.9 % | 64.7 % | 31.7 % | 0.404 | 35.3 % |
| 85 | 15 / 19 | 39 / 77 | 44.1 % | 66.4 % | 27.8 % | 0.341 | 33.6 % |

> Pure threshold tuning will not get us out of this trade-off. Anything
> that lifts specificity above ~65 % drops sensitivity below 60 %. The
> bottleneck is **not** where we set the cut; it is the guardrail rule that
> bumps everything above 70 in the first place.

## 10. Remediation paths (for SME review with Paula)

| # | Path | Expected effect | Effort |
|:-:|---|---|:-:|
| A | **Tighten Early Detection rule:** require 3-of-4 instead of 2-of-4, OR raise HR threshold to 100, RR to 24 | Drops FPs by an estimated 15–20; sens hit ~ 5 % | Low — single guardrail.json edit |
| B | Add **negative anchors** to the LLM prompt (e.g. "Recent surgery, DKA, GI bleed and post-arrest can mimic SIRS but are not sepsis") | Targets the 37 LLM-only FPs; no effect on guardrail-bumps | Low — prompt edit |
| C | Add an **infection-context check** before guardrail escalation — only bump if labs/notes show infectious signals | Cleanest fix; estimated FP drop 30–40 with sens loss < 3 % | Medium — needs guardrail extension |
| D | Manual SME review of 6 FNs to confirm they are real misses | Doesn't change metrics; protects the headline | Low — Paula's queue |

Recommend A + B + D as the immediate next iteration; C as the second iteration
once A/B numbers are in.

## 11. How to interpret the headline

At n = 34 sepsis the 95 % Wilson CI on sensitivity is wide (66.5–91.7 %).
The same point estimate at n = 100 would be reported as roughly
82 % (74–88 %). See `EICU_DATASET_AND_COHORT.md` §4.2 for the full power
analysis. The headline statement we can defend today is:

> *"On the 150-patient eICU pilot cohort, the system flagged 82 % of true
> sepsis cases six hours before clinical recognition (95 % CI 67–92 %).
> Specificity in this run was 45 % — a 55 % false-alarm rate driven
> primarily by the guardrail's early-detection rule firing on common
> non-infectious ICU presentations. We are tuning that rule before the
> next run."*

---
_Source data: `validation/results/EICU_results_v4_latest.json`_
_Metrics JSON: `validation/results/EICU_v4_metrics.json`_
