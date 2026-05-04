# Phase 1 Validation Plan — eICU-CRD Demo (Open Access)

**Created:** May 1, 2026
**Owner:** Sachin
**Supersedes:** `MIMIC_IV_PLAN.md` for Phase 1 (MIMIC-IV now deferred to Phase 2)
**Goal:** Validate Medbeacon Sepsis GenAI on 30-40 recent (2014-2015) ICU patients from the eICU Collaborative Research Database Demo, using **real hourly nursing observations** (the signal PhysioNet never had).

## Why We Pivoted From MIMIC-IV

| Factor | MIMIC-IV (original plan) | eICU Demo (new plan) |
|---|---|---|
| Credentialing | Required (3-5 business days) | **None** — click-through open license |
| Nursing notes | Structured only (v2.2 note = discharge + radiology) | **Structured + free-text nurse notes** |
| Multi-center | 1 hospital | **20 hospitals** (generalizability) |
| Era | 2008-2019 | **2014-2015** (post-Sepsis-3) |
| Time to results | ~1 week | **~1-2 days** |

MIMIC-IV is still the right Phase 2 dataset for scale, but eICU Demo lets us answer the "do nurse notes help?" question immediately.

---

## 1. Dataset — Locked

**Source:** eICU Collaborative Research Database Demo v2.0.1
**Publisher:** MIT Laboratory for Computational Physiology + Philips Healthcare
**License:** Open Data Commons ODbL v1.0 (PhysioNet open access, no credentialing)
**URL:** https://physionet.org/content/eicu-crd-demo/2.0.1/
**DOI:** 10.13026/4mxk-na84
**Size:** 130 MB compressed, ~2,500 unit stays across 20 US hospitals
**Era:** 2014-2015

### Tables We Use

| Table | Purpose in our pipeline |
|---|---|
| `patient.csv.gz` | Demographics + ICU admit/discharge timestamps + outcome |
| `diagnosis.csv.gz` | ICD-9 codes with time offsets — **the doctor's verdict** |
| `vitalPeriodic.csv.gz` | Continuous HR, SBP, DBP, MAP, Temp, Resp, O2Sat → our 6h trend window |
| `vitalAperiodic.csv.gz` | Intermittent BP, periodic vitals |
| `lab.csv.gz` | WBC, lactate, creatinine, platelets, etc. |
| `nurseCharting.csv.gz` | **Hourly nurse-recorded observations** (GCS, pain, mental status) |
| `nurseAssessment.csv.gz` | **Structured nursing assessments** at regular intervals |
| `nurseCare.csv.gz` | Care plan activities executed |
| `note.csv.gz` | **Free-text clinical notes** (sparse but present) |
| `physicalExam.csv.gz` | Documented physical exam findings |
| `carePlanInfectiousDisease.csv.gz` | Infection flags (sepsis suspicion) |
| `microLab.csv.gz` | Microbiology / culture results |
| `treatment.csv.gz` | Antibiotics, pressors, fluid boluses (with timestamps) |
| `medication.csv.gz` | Drug orders |
| `infusiondrug.csv.gz` | Infusion administrations |

### Sepsis-3 Derivation for eICU

eICU has structured diagnosis records with ICD-9 codes (038.x, 995.91, 995.92, 785.52) with time-from-admission offsets. For Sepsis-3 onset time we use the canonical approach:

1. **Suspicion of Infection (SOI):** antibiotic order within a window of microLab culture order
2. **Organ dysfunction:** ≥2-point SOFA rise around SOI window (derived from vitalPeriodic + lab)
3. **Onset:** earlier of antibiotic or culture, if both conditions met
4. **Cross-check:** ICD-9 sepsis diagnosis present (038.x, 995.9x, 785.52)

Reference SQL: MIT's `eicu-code` repo (Apache 2.0, public) — `concepts/sepsis/sepsis3.sql`.

---

## 2. Cohort Selection — 30-40 Patients Target

### Sampling

- **15-20 sepsis positive**: Sepsis-3 derivable, onset ≥6h after ICU admission (so we have snapshot room)
- **15-20 non-sepsis controls**: ICU stay ≥24h, no sepsis-related ICD codes, no antibiotics within 48h of admission
- **Total target:** 30-40 patients

### Stratification

- Age: mix of <65 / ≥65
- Hospital: draw from ≥8 different hospital_ids for generalizability
- Sex: roughly 50/50
- ICU unit type: mix of Med-Surg, MICU, SICU, CCU

### Snapshot Placement

- **Sepsis patients:** snapshot at `sepsis_onset_time - 6 hours` (matches our R7 locked convention)
- **Controls:** snapshot at a random hour ≥12h into ICU stay (so there's adequate trend window)

---

## 3. Patient JSON Format (same as Red Rover contract)

Each patient in `validation/eicu_cohort/p*.json` will have the exact same schema as our PhysioNet v4 cohort, so `run_validation.py` works unchanged:

```json
{
  "patient_id": "eicu_p00001",
  "patient_demographics": {
    "age": 67,
    "gender": "Male",
    "admission_type": "Emergency",
    "hospital_id": "hospital_7",
    "unit_type": "MICU"
  },
  "patient_vitals": {
    "HR": [{"val": 112, "ts": "2014-03-15T18:00:00"}, ...newest-first 6h],
    "SBP": [...], "DBP": [...], "MAP": [...],
    "Temp": [...], "Resp": [...], "O2Sat": [...],
    "WBC": [{"val": 14.2, "ts": "..."}], "Lactate": [...], ...
  },
  "patient_notes": "Nursing assessment (T-5h): Patient alert but confused, skin warm and flushed. Pain 4/10 unchanged. Central line site clean. Nursing assessment (T-3h): Patient became diaphoretic, GCS drop from 15 to 13. MD notified. Nursing note (T-1h): Cultures drawn, IV fluids running. Vitals trending down per protocol.",
  "ground_truth": {
    "sepsis_onset_offset_min": 360,
    "icd9_codes": ["038.9", "995.92"],
    "derived_sepsis3": true,
    "soi_met": true,
    "sofa_delta": 3
  }
}
```

### How we build `patient_notes` (the key new piece)

For each patient's 6h snapshot window, we stitch together (in chronological order):

1. All `note.csv.gz` free-text entries timestamped in [T-6h, T] → raw text
2. All `nurseAssessment.csv.gz` entries in window → formatted as "Nursing assessment (T-Xh): {field: value; ...}"
3. All `nurseCharting.csv.gz` entries for mental status, pain, skin, GCS → formatted similarly
4. Any `physicalExam.csv.gz` finding in window → formatted
5. Any `treatment.csv.gz` entries (new antibiotic, pressor, fluid bolus) → formatted as "MD order (T-Xh): {treatment}"

The final `patient_notes` field is a clean narrative that looks like a nurse's shift report + any MD orders — exactly what the Red Rover integration will produce.

---

## 4. Build Plan (starts now)

| # | Task | Output | Est. time |
|---|------|--------|-----------|
| B1 | Unpack eICU Demo into DuckDB (single-file SQL engine, no setup) | `validation/eicu_demo/eicu.duckdb` | 10 min |
| B2 | Write `eicu_sepsis3_derive.py` | Returns DataFrame of (patientunitstayid, onset_offset_min) | 1.5 hr |
| B3 | Write `eicu_cohort_builder.py` | 30-40 patient JSONs in `validation/eicu_cohort/` | 2 hr |
| B4 | Write `eicu_note_stitcher.py` (part of B3) | Narrative notes field | 1 hr |
| B5 | Adapt `run_validation.py` to accept cohort-dir parameter | Reusable runner | 15 min |
| B6 | Smoke-test on 3-5 patients end-to-end via API | Dry run confirmation | 30 min |
| B7 | Full run (30-40 patients) | Results JSON + CSV | 30-45 min |
| B8 | Adapt `analyze_results.py` + write `EICU_VALIDATION_EXECUTION.md` | Comparison report vs R7 | 1 hr |

**Total: ~7 hours of active work, spread over 1-2 days depending on iteration.**

---

## 5. Success Criteria

| Criterion | Target | Meaning if hit |
|---|---|---|
| Pipeline works E2E on eICU-shaped data | 100% success rate on 30-40 patients | Architecture is portable beyond PhysioNet |
| Sensitivity | **>70%** (vs 62.86% PhysioNet R7 floor) | **Nurse notes help** — validates the core Red Rover hypothesis |
| Specificity | **>55%** (vs 52.50% PhysioNet R7 floor) | Notes also reduce false alarms |
| Notes actually being used | ≥80% of patients have non-trivial `patient_notes` (>200 chars) | Data pipeline is doing its job |
| No guardrail regressions | Override rate 10-25% | Guardrails generalize |

### What Phase 1 WILL tell us
- Does real nurse documentation move the sensitivity needle?
- How portable is our pipeline to a different EHR (Philips eICU vs PhysioNet snapshot)?
- Are there documentation patterns in multi-center data that we should exploit (e.g., infectious-disease care plans)?

### What Phase 1 WILL NOT tell us
- Scale performance (only 30-40 patients — narrow CIs)
- Production performance in a single hospital (eICU is remote-telehealth, not bedside — different documentation style than Red Rover/SJSA)
- Direct comparability with R7 (different cohort, different era, different case mix)

---

## 6. Follow-on Phases

| Phase | Trigger | Focus |
|---|---|---|
| **Phase 2 — MIMIC-IV** (deferred from current plan) | Phase 1 shows nurse notes help | Larger cohort (500-1000 patients), single-center, modern era. Credentialing in parallel with Phase 1. |
| **Phase 2b — MIMIC-III** | Phase 1 shows nurse notes help AND we want deeper free-text validation | Older but has hourly nursing progress notes |
| **Phase 3 — Red Rover SJSA** | Real sandbox ready (A3/A4/A5 in PROJECT_TRACKER) | The production validation |

---

## 7. File Plan

```
validation/
├── EICU_DEMO_PLAN.md                   (this file)
├── MIMIC_IV_PLAN.md                    (Phase 2 - deferred)
├── eicu_demo/                          (raw downloads, gitignored)
│   ├── *.csv.gz
│   └── eicu.duckdb
├── eicu_sepsis3_derive.py              (Sepsis-3 derivation)
├── eicu_cohort_builder.py              (cohort selector + JSON writer)
├── eicu_note_stitcher.py               (narrative builder, called by cohort builder)
├── eicu_cohort/                        (generated patient JSONs)
│   ├── manifest.json
│   └── p*.json
├── EICU_VALIDATION_EXECUTION.md        (results doc)
└── results/
    ├── EICU_results.json
    ├── EICU_results.csv
    └── EICU_analysis.json
```

---

## 8. Status & Version History

| Date | Status | Notes |
|------|--------|-------|
| May 1, 2026 | PLAN created; download kicked off | Pivoted from MIMIC-IV after finding eICU Demo has nurse notes + no credentialing |
| | | |

---

## Appendix — Why "telehealth" eICU documentation is still valuable

eICU is a remote-monitoring telehealth platform: the documentation style differs from bedside but not in the ways that matter for us:
- **Bedside nurses still chart in the same EHR** — their entries flow into `nurseCharting` and `nurseAssessment`
- **Remote critical care physicians add formal notes** — these are in `note.csv.gz`
- **Structured assessments are standardized** — actually *better* for our pipeline than free-text only
- **Multi-center coverage** reduces single-hospital bias

The one genuine difference: eICU may under-document subjective nurse observations ("patient looks septic") that don't fit structured fields. Red Rover's real SJSA data will have more of this. We flag this as a known limitation in our Phase 1 report.
