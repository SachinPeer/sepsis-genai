# validation/archive — Historical / Superseded

Scripts and cohort snapshots from earlier PhysioNet validation rounds (R1–R4) that are no longer part of the active pipeline. Kept for:

1. **Traceability** — reproducing or auditing historical results.
2. **Diff-against-current** — understanding how the methodology evolved.

## Contents

**Cohort builders + their output cohorts (kept together so relative paths resolve):**

| Builder | Cohort dir | Round |
|---|---|---|
| `select_cohort.py` | `selected_cohort/` | R1 baseline |
| `select_cohort_v2.py` | `selected_cohort_v2/` | R2 — adds ICU context note |
| `select_cohort_v3.py` | `selected_cohort_v3/` | R3 — trend-enriched |
| `select_cohort_v4.py` | `selected_cohort_v4/` | R4 — 6h trend + full notes |

`select_cohort_v5.py` (R7 / locked baseline) remains in `validation/` root.

**Runners:** `run_validation.py` — R4-era runner (v4 cohort). Superseded by `run_eicu_validation.py`.

**Analyzers / deep-dives** (one-offs for specific rounds):
- `analyze_false_alarms.py`, `analyze_guardrail_only_fps.py`, `analyze_results.py`
- `recheck_guardrail_fps.py`
- `r4_false_alarm_deepdive.py`
- `hidden_tp_evidence.py`
- `smoke_test_prompt.py`
- `compare_r4_r5.py`, `compare_r5_r6.py`, `compare_r5_r7.py`

## Running these (if needed)

Most scripts use `Path(__file__).parent / "selected_cohort_vN"`, so re-running from this folder works because cohort dirs were moved here together. Results in `results/` referenced by some scripts resolve to `archive/results/` which does **not** exist — adjust paths or copy the needed result file in if you really need to re-run an old analyzer.
