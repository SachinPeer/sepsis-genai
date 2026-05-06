# eICU v4 Cohort — Pipeline-iteration comparison

_Generated 2026-05-06 22:53_

All three runs target the **same 150-patient v4 cohort** (34 sepsis / 116 controls). Differences are entirely due to pipeline changes, not data changes.

## 1. Headline metrics

| Metric | v4_t0 baseline (T=0, no C1, no scores) | v5_t0_c1 (T=0, C1 only) | v6_t0_c1_scores (T=0, C1 + scores) | v7_t0_c1_scores_c2 (T=0, C1+scores+C2) |
|---|:--:|:--:|:--:|:--:|
| Sensitivity |  82.4% (CI 66.5–91.7) |  82.4% (CI 66.5–91.7) |  82.4% (CI 66.5–91.7) |  82.4% (CI 66.5–91.7) |
| Specificity |  30.2% (CI 22.6–39.1) |  31.9% (CI 24.1–40.8) |  39.7% (CI 31.2–48.8) |  62.9% (CI 53.9–71.2) |
| PPV |  25.7% (CI 18.4–34.6) |  26.2% (CI 18.8–35.2) |  28.6% (CI 20.6–38.2) |  39.4% (CI 28.9–51.1) |
| NPV |  85.4% (CI 71.6–93.1) |  86.0% (CI 72.7–93.4) |  88.5% (CI 77.0–94.6) |  92.4% (CI 84.4–96.5) |
| F1 score | 0.392 | 0.397 | 0.424 | 0.533 |
| False-alarm rate | 69.8% | 68.1% | 60.3% | 37.1% |

## 2. Confusion matrix

| | v4_t0 baseline (T=0, no C1, no scores) | v5_t0_c1 (T=0, C1 only) | v6_t0_c1_scores (T=0, C1 + scores) | v7_t0_c1_scores_c2 (T=0, C1+scores+C2) |
|---|:--:|:--:|:--:|:--:|
| True positives | 28 | 28 | 28 | 28 |
| False negatives | 6 | 6 | 6 | 6 |
| False positives | 81 | 79 | 70 | 43 |
| True negatives | 35 | 37 | 46 | 73 |

## 3. Patient-level changes vs baseline

### v5_t0_c1 (T=0, C1 only) vs v4_t0 baseline (T=0, no C1, no scores)

- **FP → TN (false alarms removed):** 2
- **TN → FP (new false alarms):** 0
- **TP → FN (sepsis newly missed):** 0
- **FN → TP (sepsis newly caught):** 0

**Sample false-alarms removed (top 5):**

- `eicu_p00083`: was `risk=95/Critical` → `risk=35/Standard`
- `eicu_p00139`: was `risk=35/High` → `risk=35/Standard`

### v6_t0_c1_scores (T=0, C1 + scores) vs v4_t0 baseline (T=0, no C1, no scores)

- **FP → TN (false alarms removed):** 12
- **TN → FP (new false alarms):** 1
- **TP → FN (sepsis newly missed):** 0
- **FN → TP (sepsis newly caught):** 0

**Sample false-alarms removed (top 5):**

- `eicu_p00052`: was `risk=95/Critical` → `risk=28/Standard`
- `eicu_p00053`: was `risk=42/High` → `risk=35/Standard`
- `eicu_p00069`: was `risk=72/High` → `risk=25/Standard`
- `eicu_p00070`: was `risk=95/Critical` → `risk=25/Standard`
- `eicu_p00084`: was `risk=35/High` → `risk=25/Standard`

### v7_t0_c1_scores_c2 (T=0, C1+scores+C2) vs v4_t0 baseline (T=0, no C1, no scores)

- **FP → TN (false alarms removed):** 38
- **TN → FP (new false alarms):** 0
- **TP → FN (sepsis newly missed):** 0
- **FN → TP (sepsis newly caught):** 0

**Sample false-alarms removed (top 5):**

- `eicu_p00035`: was `risk=95/Critical` → `risk=35/Standard`
- `eicu_p00040`: was `risk=95/Critical` → `risk=45.0/Standard`
- `eicu_p00041`: was `risk=72/High` → `risk=35.0/Standard`
- `eicu_p00043`: was `risk=70/High` → `risk=25.0/Standard`
- `eicu_p00044`: was `risk=70/High` → `risk=35.0/Standard`
