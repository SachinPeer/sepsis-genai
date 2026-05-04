# Validation

All validation work — past, current, and planned — for the Sepsis GenAI system.

## `docs/` — reports and plans (start here)

| File | What it covers |
|---|---|
| `EICU_VALIDATION_EXECUTION.md` | **Current master record.** All eICU-CRD Demo runs (Phase 1A n=29, Phase 1B n=90), bug fixes, cohort-versioning history. |
| `PHASE1B_MANAGEMENT_SUMMARY.md` | 1-page executive summary of the Phase 1B result. |
| `AUROC_ANALYSIS.md` | AUROC = 0.745 on Phase 1B; operating-point table; ROC curve in `results/phase1b_roc_curve.png`. |
| `EICU_DEMO_PLAN.md` | The plan we executed for the eICU pivot. |
| `VALIDATION_EXECUTION.md` | Historical record of the pre-eICU PhysioNet 2019 runs (R1–R7). |
| `HIDDEN_TPS.md` | Deep-dive: evidence that some "false positives" were actually early catches. |
| `MIMIC_IV_PLAN.md` | Plan for the next (MIMIC-IV) validation phase. |

## Current working set (validation/ root)

**eICU-CRD Demo pipeline (active):**

| File | Purpose |
|---|---|
| `eicu_cohort_builder.py` | Builds the eICU cohort JSONs. Reads raw data from `eicu_demo/`, writes patient files to `eicu_cohort/`. Deterministic (seeded). |
| `run_eicu_validation.py` | Runs each patient in `eicu_cohort/` through the API, writes results to `results/`. |
| `eicu_cohort/` | Generated patient JSONs (n=90 for Phase 1B). |
| `eicu_demo/` | Raw eICU-CRD Demo CSVs. *Gitignored* — download separately. |

**PhysioNet 2019 pipeline (locked baseline):**

| File | Purpose |
|---|---|
| `select_cohort_v5.py` | Final PhysioNet cohort builder (R7). |
| `selected_cohort_v5/` | 340-patient locked cohort. |
| `download_and_select.py` | Utility to fetch PhysioNet raw data into `raw_data/`. |
| `raw_data/` | PhysioNet raw `.psv` files. *Gitignored.* |

**Results (both datasets):** `results/` — CSVs, JSON dumps, ROC plots.

## `experiments/` — sensitivity / what-if analyses

One-off simulations on locked run outputs (no runtime dependency on the API):

| File | What it does |
|---|---|
| `simulate_guardrail_softening.py` | Measures sensitivity impact of lowering guardrail thresholds. |
| `simulate_guardrail_tune.py` | Per-rule guardrail sensitivity sweep. |
| `threshold_sweep.py` | Classification-threshold sweep on R2 data. |
| `threshold_70_reclassify.py` | Re-classifies R4 at threshold = 70. |

## `archive/` — superseded / historical

Previous cohort versions (v1–v4), earlier PhysioNet runners, R4-era analyzers, and one-off comparers. Kept for traceability; not intended to be re-run. Relative paths inside these scripts point to their sibling cohort folders in this same `archive/` directory.

## Running the current pipeline (eICU)

```bash
# Build the cohort (deterministic; n=90 by default)
cd validation
python eicu_cohort_builder.py

# Run validation against a local API
python run_eicu_validation.py

# Latest results land in results/EICU_results_latest.csv
```
