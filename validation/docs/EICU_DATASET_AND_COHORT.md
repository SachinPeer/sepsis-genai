# eICU-CRD Validation — Dataset & Cohort Reference

> Living document. Updated whenever we modify the cohort, the filters,
> or the validation methodology. Each major change adds a row to the
> Cohort History table at the bottom.

| | |
|---|---|
| **Dataset** | eICU Collaborative Research Database — Demo v2.0.1 |
| **Provenance** | Real ICU patient data (Philips eICU telehealth program, 2014–2015) |
| **De-identification** | HIPAA Safe Harbor — no names, no MRN, no DOB (age in years, 90+ binned), dates randomly shifted per patient |
| **Access** | Public, open access — no credentialing required |
| **License** | Open Database License (ODbL) |
| **Owner of doc** | Sachin Jadhav |
| **Last updated** | 2026-02-11 |

---

## 1. Dataset Q & A

### 1.1 Real patients or synthetic?

**Real patients. Real ICUs. Real care.** All identifiers stripped per HIPAA Safe
Harbor methodology.

| | |
|---|---|
| Hospitals in our demo | **186** across 4 US census regions |
| ICU stays in demo | 2,520 |
| Unique patients | 1,841 |
| Time period | 2014–2015 |
| Care model | Philips eICU telehealth — clinicians actually monitoring patients |

This is **not** synthetic, **not** generated, **not** "looks-real" data. It's the
gold standard of open ICU research data, alongside MIMIC-IV. Same provenance
class, used in 200+ peer-reviewed papers including FDA decision-support
submissions.

### 1.2 Is sepsis labelled?

**Yes — via real physicians' ICD-9 / ICD-10 codes**, not a special label
column. The treating doctor coded the diagnosis at a real point in time
(`diagnosisoffset`) which we use as ground truth.

ICD codes counted as sepsis in our cohort:

| ICD-9 / ICD-10 | Meaning | Demo records |
|---|---|---|
| 038.x / A41.9 | Septicaemia / sepsis | 326 |
| 995.91 / R65.20 | Sepsis | 150 |
| 995.92 / R65.20 | Severe sepsis | 224 + 78 |
| 785.52 / R65.21 | Septic shock | 204 |
| 995.90 | SIRS / signs of sepsis | 107 + 61 |

> **322 unique ICU stays in the demo carry a sepsis ICD code.**

To weed out coding noise we apply a Sepsis-3-style cross-check: the patient
must additionally have either (a) a broad-spectrum antibiotic ordered around
the diagnosis time, or (b) a microbiology culture drawn. This brings the
cohort from 322 → 27 confirmed cases (see §3 for breakdown).

> Difference vs PhysioNet 2019: PhysioNet had a synthetic per-hour
> `SepsisLabel` column shifted 6h ahead by the organisers. eICU has *real
> physician diagnoses with real timestamps* — closer to clinical reality,
> harder to game.

### 1.3 Do we get vitals trends?

**Yes — eICU is the densest open ICU vitals dataset there is.**

| | Value |
|---|---|
| Total `vitalPeriodic` rows in demo | 1,634,960 |
| Stays covered | 2,375 |
| Average rows per stay | ~688 |
| Cadence | ~ every 5 minutes (Philips bedside monitor) |
| Typical 6-hour window | ~73 readings |
| Variables | HR, SBP, DBP, MAP, SpO2, Resp, Temperature, ETCO2, CVP, ICP, ST segments |

We currently down-sample to hourly buckets in the patient JSONs to keep LLM
prompts manageable. The underlying data is much denser if the model ever
needs it.

`nurseCharting` adds GCS total, urine output, mental-status flags, pain,
sedation, skin observations.

`note` adds free-text physician/nurse narrative (where present).

### 1.4 Can we predict 6 hours prior to the doctor's assessment?

**Yes — and it's a stricter benchmark than it sounds.**

Distribution of when sepsis was *coded* (hours from ICU admit):

| min | 25th | median | 75th | max |
|:-:|:-:|:-:|:-:|:-:|
| –10.3 h | 3.4 h | **27.5 h** | 92.3 h | 820.7 h |

What this means:
- Half of all sepsis diagnoses are coded > 27 h into the ICU stay — plenty of
  6-h-prior runway.
- Bottom 25% are coded within 3.4 h of admit — for these we cannot get a
  6-h-prior snapshot, so the cohort builder filters them out
  (`diagnosisoffset >= 360 min`).
- Negative offsets (–10 h) represent patients transferred in already
  diagnosed; also excluded.

> Coding lags clinical recognition by 2–6 h in practice (documentation,
> confirmatory cultures, attending sign-off). So our "6 h prior to coding"
> is closer to "6–12 h prior to clinical recognition" — the prediction
> task is actually *harder* than it sounds, not easier.

---

## 2. Cohort Quality Filters (target rules)

A patient is **kept** in the validation cohort only if **all** of the
following are true:

| # | Rule | Why |
|---|---|---|
| F1 | Age ≥ 18 (no `Age == 0`, no `Age < 18`) | Adult ICU only — neonates / paediatrics have different physiology and our guardrails are tuned for adults |
| F2 | Snapshot offset ≥ 6 h after ICU admit | We need a 6-h pre-onset trend window |
| F3 | ≥ 4 HR readings in the 6-h window | Trend needs a minimum of 4 hourly buckets |
| F4 | ≥ 4 Respiratory-rate readings | RR drives qSOFA & SIRS — non-negotiable |
| F5 | ≥ 3 vital types each with ≥ 4 readings | Robustness: pipeline shouldn't be reasoning from only HR + Temp |
| F6 | ≥ 2 distinct labs in 24-h pre-snapshot window | At least one organ-function indicator (lactate / WBC / creatinine / etc.) |
| F7 | Sepsis cohort: ICD code AND (abx ordered OR culture drawn) | Sepsis-3-style "suspicion of infection" cross-check |
| F8 | Control cohort: NO sepsis ICD code AND NO broad-spectrum abx in first 48 h | Pure non-sepsis controls |
| F9 | Control cohort: ICU stay ≥ 24 h | Long enough to have a stable 6-h window |
| F10 | `patient_notes` is not the "no notes available" placeholder | Discordance rules need at least some unstructured content to evaluate |

---

## 3. Demo Capacity — what 340 actually looks like

**Hard ceiling check (run on the demo files on disk, 2026-02-11):**

### Sepsis pool funnel
| Filter step | Patients |
|---|:--:|
| ICD-coded sepsis stays in demo | 322 |
| `+` onset ≥ 6 h after admit (F2) | 60 |
| `+` adult only (F1) | 60 |
| `+` Sepsis-3 cross-check: abx OR culture (F7) | 27 |
| `+` HR ≥ 4 AND RR ≥ 4 readings (F3, F4) | 26 |
| `+` ≥ 2 labs in 24 h pre-onset (F6) | **25** |

### Control pool funnel
| Filter step | Patients |
|---|:--:|
| Adult ICU stays ≥ 24 h, no sepsis ICD, no early abx (F1, F8, F9) | 1,162 |
| `+` vitals OK (F3–F5) | 960 |
| `+` labs OK (F6) | **915** |

### Hospital diversity
| | Hospitals represented |
|---|:--:|
| Sepsis pool | 17 |
| Control pool | 178 |

### Implication for the 340-patient target

> **The demo cannot deliver 100 sepsis patients.** With our quality filters,
> the absolute ceiling is **25** with the strict Sepsis-3 cross-check, or
> **34** with ICD-only ground truth and bucket-counted vitals (the actual
> v4 build). The control pool is fine — 915 available, more than enough
> for 116.

This is a constraint of the demo's selective sampling (it ships ~3.6% of the
full eICU-CRD by design), not of the methodology. The full eICU-CRD v2.0 has
~10,000+ confirmed sepsis cases — see Option C in §4.

---

## 4. Chosen Path: Option B — Demo, ICD-only Ground Truth

After reviewing the three options, the team chose **Option B**: stay within
the open demo dataset, use ICD-9 sepsis codes as ground truth without the
strict abx/culture cross-check. This avoids credentialing risk (Option C)
and gives us a publishable n while accepting slightly noisier ground truth.

| Option | Sepsis | Controls | Total | Compliance | Decision |
|---|:--:|:--:|:--:|---|:--:|
| A. Demo, strict Sepsis-3 cross-check | 25 | 60 | 85 | None — already open | Rejected — n too small |
| **B. Demo, ICD-only ground truth** | **34** | **116** | **150** | None — already open | ✅ **Selected (built)** |
| C. Full eICU-CRD v2.0 | 100 | 240 | 340 | Credentialed via PhysioNet | Deferred — same risk profile as MIMIC-IV |

> **What we built vs. what we planned.** The probe estimate (48 sepsis) used
> the 6-h pre-onset window for vital quality counts. The actual v4 build uses
> the more honest 6-h pre-snapshot window (T-12h to T-6h before onset) — i.e.
> the same data the model will actually see. Counting distinct hourly buckets
> in that window yields **34** sepsis patients with strong vital coverage.
> All 34 are adults from 30 distinct hospitals.

### 4.1 Why we accept the smaller cohort

- **Compliance certainty.** Demo data is fully open access, no DUA, no
  CITI training, no individual liability. Same risk envelope as our
  current Phase 1B baseline.
- **Methodology and code stay the same.** Only the cohort changes —
  pipeline, guardrails, prompts, infra are unaffected.
- **Pilot framing is honest.** This is positioned as a *pilot validation*,
  not a pivotal trial. The pivotal trial happens later, on real Red Rover
  data or full eICU.

### 4.2 Statistical impact of ~50 sepsis vs the original 100 target

The 95 % confidence-interval half-width on sensitivity scales as
1.96 × √(p (1-p) / n). Comparing what we have (n = 34) against what we'd
get from the full eICU-CRD (n = 100):

| Observed sensitivity | n = 34 (built) | n = 100 (original target) | Width loss |
|:-:|:-:|:-:|:-:|
| 85 % | ± 12.0 % | ± 7.0 % | + 5.0 pp |
| 90 % | ± 10.1 % | ± 5.9 % | + 4.2 pp |
| 95 % | ± 7.3 % | ± 4.3 % | + 3.0 pp |

The same applies to specificity on the control side:

| Observed specificity | n = 116 controls | n = 240 controls | Width loss |
|:-:|:-:|:-:|:-:|
| 70 % | ± 8.3 % | ± 5.8 % | + 2.5 pp |
| 80 % | ± 7.3 % | ± 5.1 % | + 2.2 pp |
| 90 % | ± 5.5 % | ± 3.8 % | + 1.7 pp |

#### What this means in plain English

> If the pipeline genuinely runs at, say, 88 % sensitivity, the headline
> we can defensibly publish becomes
> **"88 % sensitivity (95 % CI: 77 – 99 %)"** at n = 34, versus
> **"88 % (82 – 94 %)"** at n = 100. The point estimate is the same; the
> claim around it is roughly 5 percentage points looser at each end.

| | n ≈ 34 sepsis (built) | n ≈ 100 sepsis (original target) |
|---|---|---|
| Position | "Pilot validation, ICU demo dataset" | "Adequately powered single-dataset validation" |
| Suitable for | Internal sign-off · investor / pilot-customer review · publication as a pilot study | Regulatory submissions · pivotal-study-style claims |
| Risk of false confidence | Moderate — wide CI must always be quoted alongside any number | Low |
| Risk of false alarm | Same — methodology unchanged, only the precision of the estimate changes | Same |
| Sub-group analysis | Not advisable (each sub-group too small) | Limited (some sub-groups feasible) |

#### What ~34 sepsis cannot tell us

- **Sub-group performance** (e.g. by age band, ethnicity, admission
  unit). A sub-group of 8–10 patients has a CI of ± 25 %, which is
  essentially meaningless. We deliberately will *not* slice the headline
  metric by sub-group at this n.
- **Rare-but-important failure modes** (e.g. immunocompromised,
  post-arrest hypothermia, end-stage liver disease). At n = 34 we may see
  1–2 such cases at most. We will *describe* any we see qualitatively
  rather than report rates.
- **Calibration curves** (Hosmer-Lemeshow at decile bins). Each bin would
  hold 3–4 patients — too few to give a meaningful calibration plot.

### 4.3 Built v4 cohort sizes (actual)

| | Sepsis | Controls | Total | Ratio |
|---|:--:|:--:|:--:|:--:|
| Pool after F1, F2 | 60 | 1,162 | — | — |
| Pool after F3–F6 (bucket-counted) | 34 | ~600 | — | — |
| **Built v4 cohort** | **34** | **116** | **150** | **1 : 3.4** |
| Of which empty-notes (soft flag) | — | — | 6 | — |

The 1 : 3.4 ratio is slightly more control-heavy than the 1 : 2.4 we had
planned, because the sepsis pool is the binding constraint and we kept
controls near the original 120 target. The extra controls don't hurt —
they slightly tighten the specificity CI (see §4.2).

---

## 5. Audit Comparison: Phase 1B (v2) vs v4

### 5.1 Phase 1B v2 baseline (90 patients) — for reference

Audit run 2026-02-11 against `validation/eicu_cohort/`:

| | |
|---|:--:|
| Files inspected | 90 (one missing index — p00099 absent) |
| Will keep (no flags) | 52 |
| Will remove (≥ 1 flag) | **38** |
| Sepsis kept | 23 of 30 |
| Controls kept | 29 of 60 |

Removal reasons:

| Flag | Count |
|---|:--:|
| `VITAL_RESP_THIN` | 18 |
| `VITAL_TYPES_THIN` | 18 |
| `NOTES_EMPTY` | 17 |
| `LABS_MISSING` | 11 |
| `VITAL_HR_THIN` | 7 |
| `AGE_NEONATE` | 5 |
| `LABS_THIN` | 3 |

### 5.2 v4 cohort (150 patients) — current

Audit run 2026-02-11 against `validation/eicu_cohort_v4/`:

| | |
|---|:--:|
| Files inspected | **150** |
| Hard-flag rejects | **0** ✅ |
| Soft-flag (empty notes) | 6 (kept) |
| Sepsis kept | 34 of 34 |
| Controls kept | 116 of 116 |
| Hospitals represented | 95 |
| Sepsis hospitals | 30 |
| Age range / median | 22 – 90 / 67 |

> **Quality delta from v2 → v4:** v2 had 38/90 (42 %) patients with hard
> quality issues. v4 has **0/150** hard-fail patients — every patient
> passed F1, F2, F3, F4, F5, F6, F8 and F9. Six patients have empty
> nurse-note text (F10 soft flag); they are kept because the rest of the
> pipeline (vitals, labs, scoring, guardrails) still functions. The
> discordance rules will simply not fire for those six, which we will
> report as a separate sub-metric.

---

## 6. Cohort History

| Version | Date | Patients | Sepsis : Control | Filters | Notes / outcome |
|---|---|:--:|:--:|---|---|
| v1 (Phase 1A) | 2026-04-29 | 29 | 9 : 20 | basic, neonates not excluded | Initial pilot — exposed GCS-component bug |
| v2 (Phase 1B) | 2026-05-01 | 90 | 30 : 60 | Phase 1A + GCS fix + cohort-build determinism fix | Used in `Sepsis_GenAI_DeepDivev6` slide; 38 patients had quality issues |
| v3 (cleaned) | 2026-02-11 | 52 | 23 : 29 | v2 minus 38 quality-flagged patients | Skipped — superseded by v4 build |
| v4 (Option B, **built**) | 2026-02-11 | **150** | **34 : 116** | F1–F9 hard, F10 soft, bucket-counted vitals, ICD-only sepsis | ✅ Current cohort — 95 hospitals, ages 22–90, 6 empty-notes (kept). **Validation run complete** — sens 82.4 %, spec 44.8 %; see `EICU_VALIDATION_EXECUTION.md` |

---

## 7. Files in this work-stream

| Path | Purpose |
|---|---|
| `validation/eicu_demo/` | The raw eICU CSV.gz files (open access download) |
| `validation/eicu_cohort_builder.py` | v2 builder — kept for reference; has age-parsing fix from v4 work |
| `validation/eicu_cohort_builder_v4.py` | **v4 builder** — strict F1–F9 + bucket-counted vitals + ICD-only sepsis |
| `validation/finalize_cohort_v4.py` | Post-build audit-driven cleanup; renumbers files |
| `validation/eicu_cohort/` | Phase 1B v2 cohort (90 patients) — historical |
| `validation/eicu_cohort_v4/` | **Current v4 cohort (150 patients)** |
| `validation/eicu_cohort_v4/manifest.json` | Per-patient summary + filter rationale |
| `validation/eicu_cohort_v4/build_log.json` | Per-candidate accept/reject decision trail |
| `validation/eicu_cohort_v4/audit_report.json` | Latest quality audit (per-patient detail) |
| `validation/audit_eicu_cohort.py` | Audit script (set `COHORT_DIR=eicu_cohort_v4`) |
| `validation/probe_cohort_capacity.py` | Capacity analysis script |
| `validation/results/EICU_results_*.json` | Pipeline-execution results |
| `validation/docs/EICU_VALIDATION_EXECUTION.md` | Run log of validation experiments |

---

## 8. Open Questions / Decisions Pending

1. **SME spot-check (Paula).** Any ICD-only sepsis case the model
   classifies as "low risk" should be reviewed by Paula. That gives us a
   post-hoc safety net against ICD coding artefacts without forcing the
   strict F7 cross-check at build time. Estimate: ~5–10 cases at most.
2. **Empty-notes handling.** Six patients (4 % of cohort) have no
   free-text notes. Decision needed: do we report metrics with-and-without
   these six in the headline, or just include them as standard? Current
   plan is to include them and report a "discordance-rule eligible"
   sub-metric separately.
3. **Pivotal trial path** remains TBD — working assumption is real
   Red Rover data once the event-driven architecture is live; full
   eICU-CRD v2.0 is a fallback that reopens the credentialing decision.
