# Phase 1 Validation Plan — MIMIC-IV Recent-Era Cohort

**Created:** May 1, 2026
**Owner:** Sachin
**Goal:** Validate Medbeacon Sepsis GenAI on 50-100 recent (post-Sepsis-3) ICU patients from MIMIC-IV, using MIMIC-IV-Note radiology reports for note-context. Directly comparable to our R7 PhysioNet baseline (62.86% sens / 52.50% spec).
**Why:** PhysioNet Challenge 2019 data predates the 2016 Sepsis-3 consensus. Clinical practice, documentation norms, and the definition of sepsis itself have evolved. MIMIC-IV (2008-2019) captures modern practice and is Sepsis-3 compatible.

---

## 1. Dataset Choice (Locked)

**Combination:** MIMIC-IV v2.2 (structured) + MIMIC-IV-Note v2.2 (free-text notes)

| What we get | What we don't get |
|---|---|
| Recent (2008-2019) Beth Israel Deaconess ICU data | Hourly nursing progress notes (these are in MIMIC-III, not IV) |
| Structured: vitals, labs, meds, procedures, diagnoses (ICD-10) | Shift-level "patient appears septic" style commentary |
| Radiology reports with timestamps (CT, CXR, US) | Physician hourly notes |
| Discharge summaries (use for outcome confirmation, NOT as input) | Triage notes (those are in MIMIC-IV-ED) |
| Sepsis-3 derivable via MIT's `mimic-code` repo | |

### What "nurse notes" looks like in MIMIC-IV

MIMIC-IV does NOT give us free-text nursing narratives. Instead we reconstruct an equivalent "nurse observation trail" from `chartevents`, which contains what nurses actually recorded:
- Vital signs (same as our PhysioNet set)
- GCS / mental status scores
- Pain scores
- Skin assessment entries
- Oxygen delivery changes
- Urine output (from `outputevents`)
- Fluid boluses, pressors (from `inputevents`)

This is functionally the same data a bedside nurse would write as "patient confused, pressors started, fluid bolus given" — just structured instead of free-text. We feed this to the LLM as a generated narrative.

Radiology reports are the one free-text input we'll have. If a patient has a CT abdomen or CXR timestamped in our 6-hour window, we inject that report into the LLM narrative.

---

## 2. Critical Path — PhysioNet Credentialed Access

**This is the gating item.** Nothing else can happen without it. Typical timeline: **3-5 business days**.

### Onboarding Checklist (for Sachin)

| # | Step | Estimated time | Dependencies | Status |
|---|------|----------------|--------------|--------|
| P1 | Create free account at https://physionet.org/register/ | 5 min | — | ❌ |
| P2 | Complete CITI Program course: "Data or Specimens Only Research" | 2-3 hrs | P1 | ❌ |
|    | - Register at https://www.citiprogram.org/ | | | |
|    | - Affiliate with "Massachusetts Institute of Technology Affiliates" | | | |
|    | - Complete all modules + quizzes (~90-120 min of content) | | | |
|    | - Download the completion certificate (PDF) | | | |
| P3 | Upload CITI certificate to PhysioNet profile | 5 min | P2 | ❌ |
| P4 | Apply for credentialing: PhysioNet profile -> "Credentialing" | 10 min | P3 | ❌ |
|    | - Upload CITI certificate | | | |
|    | - Provide reference (supervisor, colleague, or peer — easy) | | | |
|    | - Describe research purpose in 2-3 sentences | | | |
| P5 | Wait for credentialing approval | 1-3 business days | P4 | ⏳ |
| P6 | Sign DUA for MIMIC-IV v2.2 (open project page, click "Request access") | 5 min | P5 | ❌ |
|    | - Review terms, sign electronically | | | |
| P7 | Sign DUA for MIMIC-IV-Note v2.2 (separate project, separate DUA) | 5 min | P5 | ❌ |
|    | - Requires the same credentialing, plus second DUA | | | |
| P8 | Verify download access | 5 min | P6, P7 | ❌ |
|    | - From PhysioNet: "Files" tab visible on both project pages | | | |

### Research-purpose language to use in P4

Copy-paste this into the PhysioNet credentialing application (adapt as needed):

> I am validating a generative-AI-based early sepsis detection system (Medbeacon Sepsis GenAI) that runs in a hospital ICU environment. My goal is to measure the system's sensitivity and specificity on a recent-era (post-Sepsis-3) ICU cohort before deploying to partner hospitals. MIMIC-IV provides the structured physiological data and MIMIC-IV-Note provides the radiology context I need to measure realistic performance. No data will leave secure local environments; no re-identification attempts will be made; results will only be published as aggregate validation metrics.

### Reference for P4

A peer or colleague (Narendra, Shawn, or any co-worker with a research email) can serve as the reference. They receive a 1-click confirmation email from PhysioNet.

---

## 3. Parallel Development Plan — What I Build While Access Clears

While you do P1-P8, I build the entire harness using only **public schemas and open code** (MIT's `mimic-code` repo is Apache 2.0, no credentialing needed). When access arrives, we point the harness at the downloaded data and run in under an hour.

### Build Plan (no credentialing needed)

| # | Task | Output | Depends on |
|---|------|--------|------------|
| B1 | Clone MIT `mimic-code` repo; understand Sepsis-3 derivation SQL | Local reference for `sepsis3.sql` | — |
| B2 | Write `validation/mimic4_sepsis3_derive.py` — pure-Python port of the Sepsis-3 query | Produces `(subject_id, hadm_id, stay_id, onset_time)` | B1 |
| B3 | Write `validation/mimic4_cohort_builder.py` — mirrors our `select_cohort_v4.py` logic | For each patient: 6h trend of HR/SBP/DBP/MAP/Temp/Resp/O2Sat/WBC, labs, demographics, nurse-observation synthetic note | B2 |
| B4 | Write `validation/mimic4_note_injector.py` — pulls any radiology report timestamped in the 6h window into the note | Augmented patient JSON | B3 |
| B5 | Adapt `run_validation.py` to accept MIMIC-IV cohort format | Reusable validation runner | B3, B4 |
| B6 | Adapt `analyze_results.py` — same metrics, separate output folder | MIMIC-IV validation report | B5 |
| B7 | Smoke-test the whole pipeline on synthetic MIMIC-IV-shaped fixtures | End-to-end dry run succeeds | B5 |

**Total effort:** ~6-8 hours of build time, done in parallel with your access process.

### Data sources for build phase

- `https://github.com/MIT-LCP/mimic-code` — SQL and Python for all standard cohort derivations, including Sepsis-3 (BigQuery SQL, straightforwardly portable)
- `https://mimic.mit.edu/docs/iv/` — public schema documentation (tables, columns, units)
- `https://physionet.org/content/mimiciv/2.2/` — public project landing page (no data, but lists file layout)

---

## 4. Execution Plan (after access clears)

Once all P-steps and B-steps are done:

| # | Task | Time | Notes |
|---|------|------|-------|
| X1 | Download MIMIC-IV v2.2 structured data | 1-2 hrs | ~7 GB gzipped CSVs; use `wget --user --password` with PhysioNet token |
| X2 | Download MIMIC-IV-Note v2.2 | 30-60 min | ~1.5 GB gzipped |
| X3 | Load into local SQLite or DuckDB | 30-60 min | We skip full Postgres for a 100-patient pilot |
| X4 | Run `mimic4_sepsis3_derive.py` to get all Sepsis-3 patients | 5-10 min | Typically ~8-10k patients eligible |
| X5 | Sample 50-100 patients (stratified: ~40-50 sepsis, ~50-60 non-sepsis controls) | 5 min | Matched for age/ICU stay length |
| X6 | Run `mimic4_cohort_builder.py` to produce patient JSONs | 10-15 min | Output format identical to our PhysioNet v4 cohort |
| X7 | Run `mimic4_note_injector.py` to add radiology reports | 5 min | |
| X8 | Smoke-test 5 patients end-to-end through the API | 10 min | Catch any format drift |
| X9 | Run full validation (50-100 patients) | 20-45 min | Sonnet 4.5 latency + guardrail |
| X10 | Generate `MIMIC_IV_VALIDATION_EXECUTION.md` with side-by-side vs R7 | 30 min | Follows our VALIDATION_EXECUTION.md template |

**Total execution phase:** ~1/2 day once data is in hand.

---

## 5. Decision — Cohort Size (Locked)

**50-100 patients pilot.**

### Sampling plan

- 40-50 Sepsis-3 positive patients (modern sepsis definition, onset timestamped)
- 50-60 ICU controls (non-sepsis, matched for ICU stay length ≥24h, any organ dysfunction flag)
- For each patient: snapshot at **6 hours before Sepsis-3 onset** (or a matched random hour for controls)
- Trend window: 6 hours (locked per R7 decision)

### Stratification (fairness guardrails)

We ensure the cohort is not trivially skewed:
- Age: balanced across ≥65 / <65
- Sex: roughly 50/50
- First-ICU-day vs late-ICU-day: 50/50 split (sepsis later in ICU is harder; we should see both)

---

## 6. Success Criteria for Phase 1

Phase 1 is primarily a **pipeline shake-out and sanity check**, not a definitive validation. Success means:

| Criterion | Target | If not met |
|---|---|---|
| Pipeline works end-to-end on MIMIC-IV-shaped data | No runtime errors on 50-100 patients | Fix format / adapter bugs |
| Sensitivity within ±10 pts of PhysioNet R7 (62.86%) | 53-73% | Investigate data-drift (e.g., Sepsis-3 labels differ from PhysioNet SepsisLabel) |
| Specificity within ±10 pts of PhysioNet R7 (52.50%) | 43-63% | Investigate note-context impact |
| Radiology reports are hitting the LLM when available | >60% of sepsis patients have ≥1 radiology report in window | Adjust window or exclude radiology from pipeline |
| No guardrail behavioral regressions | Override rate 10-25% of cohort | Review thresholds vs MIMIC-IV population norms |

### What Phase 1 will NOT tell us (manage expectations)

- Whether real nurse narratives help — MIMIC-IV-Note doesn't have them, so we're still note-deprived vs what Red Rover will deliver
- True production-like performance on a 1000-patient cohort — that's Phase 2

### What Phase 1 WILL tell us

- Does our system generalize beyond PhysioNet to a real hospital's ICU population?
- How does modern sepsis (post-Sepsis-3) differ from the 2019 PhysioNet dataset?
- Does radiology-report context move the specificity needle?
- Do our guardrails generalize or do MIMIC-IV patients trip them differently?

---

## 7. Follow-on Phases (Post-MIMIC-IV)

Once Phase 1 clears, we have a choice depending on results:

| Phase | Trigger | Focus |
|---|---|---|
| **Phase 2a** | Phase 1 sensitivity < 60% | MIMIC-III with full hourly nursing notes (older data but validates nurse-note value hypothesis) |
| **Phase 2b** | Phase 1 sensitivity acceptable | Scale to 500-1000 MIMIC-IV patients for statistical power |
| **Phase 3** | Red Rover sandbox ready (depends on A3/A4/A5 in `docs/planning/PROJECT_TRACKER.md`) | Real SJSA patient data — the production validation |

---

## 8. File Plan

Files we'll create in Phase 1:

```
validation/
├── MIMIC_IV_PLAN.md                        (this file)
├── mimic4_sepsis3_derive.py                (MIT sepsis3.sql -> Python)
├── mimic4_cohort_builder.py                (like select_cohort_v4.py, for MIMIC-IV)
├── mimic4_note_injector.py                 (radiology report timing + injection)
├── MIMIC_IV_EXECUTION.md                   (results doc, like VALIDATION_EXECUTION.md)
├── mimic4_cohort/                          (generated patient JSONs)
│   ├── manifest.json
│   └── p*.json
└── results/
    ├── MIMIC_IV_results.json
    ├── MIMIC_IV_results.csv
    └── MIMIC_IV_analysis.json
```

---

## 9. Status & Version History

| Date | Status | Notes |
|------|--------|-------|
| May 1, 2026 | PLAN created | Awaiting user to start P1-P8 (PhysioNet onboarding) |
| | | |

---

## Appendix A — MIT mimic-code Sepsis-3 derivation (summary)

The standard Sepsis-3 derivation in MIMIC-IV uses three ingredients:

1. **Suspicion of infection** — an antibiotic prescription + a blood culture order within a narrow window (24h antibiotic-before-culture, 72h culture-before-antibiotic)
2. **SOFA >= 2 point increase** from 48h-before to 24h-after suspicion
3. **Onset time** = earlier of antibiotic order or culture order, if both SOFA and SOI are satisfied

Output: `(subject_id, hadm_id, stay_id, sepsis_onset_time)`. Our cohort builder reads this and sets the snapshot to `sepsis_onset_time - 6 hours`.

Reference: `https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iv/concepts/sepsis`
