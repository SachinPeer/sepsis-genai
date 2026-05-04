# Sepsis GenAI — Project Tracker

> **Living document** — Update status and dates as work progresses.
> Last updated: **May 1, 2026** by Sachin

---

## Status Key

| Icon | Meaning |
|------|---------|
| ✅ | Done |
| 🔄 | In progress |
| ⏳ | Waiting on external dependency |
| ❌ | Not started |
| 🚫 | Blocked |
| 📌 | Decision needed |

## Owner Key

| Tag | Who |
|-----|-----|
| **US** | Sachin + AI |
| **NAR** | Narendra (CI/CD, Infra) |
| **SHA** | Shawn (Admin, Approvals) |
| **PAU** | Paula (Clinical SME) |
| **DAV** | David (Leadership) |
| **IBR** | Ibrahim (Associate) |
| **FE** | Frontend Team (TBD) |
| **RR** | Red Rover Team |
| **SAN** | Sanjay Sivakumar (Mongo / Data Platform) |

---

## Completed Work

| # | Task | Owner | Completed | Notes |
|---|------|-------|-----------|-------|
| C1 | Core 3-stage AI pipeline | US | Jan 2026 | Preprocess → LLM → Guardrail |
| C2 | Clinical guardrails (850+ params) | US | Jan 2026 | API-managed, hot-reload |
| C3 | Deterministic scores (qSOFA/SIRS/SOFA) | US | Jan 2026 | In guardrail_service.py |
| C4 | Confidence level + reasoning | US | Jan 2026 | LLM outputs confidence |
| C5 | Structured audit logging | US | Jan 2026 | JSON format, no PHI |
| C6 | EKS deployment | NAR | Feb 2026 | medbeacon-cluster, ALB routing |
| C7 | Bedrock throttle fix | US | Feb 2026 | Lightweight health check, Sonnet 4 |
| C8 | LLM model comparison | US | Feb 11, 2026 | 4 models tested, docs/LLM_MODEL_COMPARISON.md |
| C9 | Upgrade to Sonnet 4.5 (code) | US | Feb 11, 2026 | .env + genai_inference_service.py updated |
| C10 | Feedback analysis script | US | Feb 2026 | scripts/analyze_feedback.py |
| C11 | Feedback loop guide | US | Feb 2026 | docs/FEEDBACK_LOOP_GUIDE.md |
| C12 | Leadership presentation | US | Feb 11, 2026 | Medbeacon-branded + Deep Dive PPTXs |
| C13 | Execution plan | US | Feb 11, 2026 | docs/EXECUTION_PLAN.md |
| C14 | PhysioNet interim validation (R1–R7 + 8 experiments) | US | May 1, 2026 | 340 patients, 7 rounds, 8 out-of-band experiments; final: 62.86% sens / 52.50% spec (synthetic-notes floor); **system configuration LOCKED-IN**; see `validation/VALIDATION_EXECUTION.md` §19-§20 |
| C15 | Trend-ordering bug fix (newest-first) | US | Apr 30, 2026 | Cohort builder + preprocessor — pre-fix every round was interpreted through garbled data |
| C16 | Prompt v3.2 (decoupled priority + guardrail-aware) | US | Apr 30, 2026 | docs/prompt.md — biggest single specificity gain (+17.8 pts) |
| C17 | Full-trend preprocessor (knowledge/genai_proprocess.py rewrite) | US | Apr 30, 2026 | All 6 hourly readings now reach the LLM; CRITICAL FLAGS summary line added |
| C18 | Guardrail softening simulation (confirmed lock-in) | US | May 1, 2026 | Proved every early-detection relaxation trades sensitivity 1:1 — guardrails stay at 2-of-4 |
| C19 | Hidden TPs clinical evidence dossier | US | Apr 30, 2026 | validation/HIDDEN_TPS.md — 18 STRONG cases where AI caught what ground-truth label missed |
| C20 | eICU-CRD Demo Phase 1A baseline (real nurse notes) | US | May 1, 2026 | n=29 reproducible cohort; **Sens 81.82% / Spec 72.22%** (vs PhysioNet 62.86%/52.50% — +19 pts on both axes); GCS-extraction + DuckDB determinism bugs found and fixed; see `validation/EICU_VALIDATION_EXECUTION.md` |
| C21 | eICU-CRD Demo Phase 1B (n=90) | US | May 1, 2026 | Scaled to 30 sepsis + 60 controls; **Sens 73.33% (95% CI 55.6-85.8) / Spec 63.33% (95% CI 50.7-74.4)**; all 29 R3 verdicts reproduce byte-identically; **100% of FNs lack BP+Lactate** (data-quality bottleneck identified); 5 hidden-TP candidates among FPs |

---

## Locked System Configuration (baseline as of May 1, 2026)

| Component | Setting | File |
|---|---|---|
| Data window | 6h trend, snapshot 6h before clinical sepsis | `validation/select_cohort_v4.py` |
| Preprocessor | Full-trend serialization, CRITICAL FLAGS summary | `knowledge/genai_proprocess.py` |
| Prompt | v3.2 — decoupled priority, guardrail-aware | `docs/prompt.md` |
| Guardrails | Unchanged — 2-of-4 early-detection rule retained | `services/guardrail_service.py` + `knowledge/clinical_knowledge_structured.json` |
| Classifier | `risk_score ≥ 50 OR priority ∈ {High, Critical}` | `validation/run_validation.py` |
| LLM | Claude Sonnet 4.5 via Bedrock | `.env`, `services/genai_inference_service.py` |
| Floor performance (no notes) | Sens 62.86% / Spec 52.50% | `validation/VALIDATION_EXECUTION.md` §19 |
| Projected performance (with notes) | Sens 85–90% / Spec 55–60% | Pending Red Rover validation |

---

## Active Tasks

| # | Task | Owner | Status | Due | Blocker | Notes |
|---|------|-------|--------|-----|---------|-------|
| A1 | Sonnet 4.5 on EKS | NAR | ✅ | Apr 8 | — | Narendra updated env var on EKS |
| A2 | HTTPS/TLS on ALB | NAR | ⏳ | Feb 14 | — | ACM cert + port 443 listener + HTTP redirect |
| A3 | Red Rover SJSA sandbox | RR/SHA | ⏳ | TBD | Red Rover response | Meeting held Feb 11. Email sent to Shawn |
| A4 | Red Rover: confirm API endpoint | RR | ⏳ | TBD | Red Rover response | `/patients/orders` or `/encounters/{id}/events`? |
| A5 | Red Rover: full sepsis code list | RR | ⏳ | TBD | Red Rover response | 5 codes identified, need complete list |
| A6 | Hunt recent (post-2016, Sepsis-3 era) real patient data with nurse notes | US | 1-2 wks | — | Stopgap while Red Rover access is pending. Pre-2019 data risks Sepsis definition drift and weak clinician labels. Primary targets: MIMIC-IV + MIMIC-IV-Note (credentialed), MIMIC-IV Demo (open), Synthea synthetic cohorts. |

---

## Upcoming Tasks

| # | Task | Owner | Status | Target | Depends On | Notes |
|---|------|-------|--------|--------|------------|-------|
| U1 | SJSA integration code | US | ❌ | 1 wk after A3 | A3, A4, A5 | Parser + prompt update + guardrail update |
| U2 | Frontend: display AI output | FE | ❌ | TBD | 📌 DAV allocation | API contract ready (docs/GENAI_OUTPUT_EXPLAINED.md) |
| U3 | Frontend: feedback capture UI | FE | ❌ | TBD | 📌 DAV allocation | Schema ready (docs/FEEDBACK_LOOP_GUIDE.md) |
| U4 | Benchmarking: 50+ test cases | US/PAU | ❌ | 2 wks | Paula availability | Create cases → Paula reviews → measure accuracy |
| U5 | EKS CloudWatch logging | NAR | ❌ | Feb 21 | — | Fluent Bit / Container Insights |
| U6 | MongoDB instance for feedback | NAR | ❌ | TBD | 📌 Decision: Atlas vs self-hosted? | Collection schema ready |
| U7 | Prompt tuning from feedback | US | ❌ | After U3 | U3 + 50 feedbacks | Run analyze_feedback.py → update prompt |
| U8 | Retrospective validation study | US/PAU | ❌ | 1-3 months | Historical data from SHA/RR | Use Opus 4.6, measure lead time |
| U9 | SJSA vs AI lead time comparison | US | ❌ | After U1 | U1 + live data | Prove AI flags patients before SJSA |
| U10 | Multi-hospital config pilot | US/DAV | ❌ | 2-3 months | Hospital partnerships | Add hospital_id, per-hospital configs |
| U16 | Age-specific guardrail thresholds (Pediatric) | US/PAU | ❌ | TBD | Paula clinical review | Add pediatric age-band thresholds (neonate, infant, child, adolescent), pSOFA scoring, auto-select by patient age. Critical for mixed/rural units admitting children. |
| U17a | Hospital-specific guardrail config — Mongo collection + indexes | SAN | 🔄 | This sprint | Mongo `medbacon` DB access | Collection `hospital_guardrail_configs` per `docs/GUARDRAIL_CONFIG_MONGO_DESIGN.md`. Non-code; just Mongo provisioning. Code change deferred to U17c. |
| U17b | Seed baseline document from current guardrail JSON | US | ❌ | After U17a | U17a complete | Write `scripts/seed_baseline_config.py` and insert first `medbeacon_baseline` doc |
| U17c | Code change — read config from Mongo with container-JSON fallback | US/NAR | ❌ | After auth layer | U17a + auth | Phase C of design — Mongo-primary, container-fallback |
| U17d | Multi-pod broadcast reload on config change | US/NAR | ❌ | After U17c | U17c + SNS / Redis decision | Phase D — so PUT /guardrail/config reaches all pods |
| U11 | Real-time alert push | US | ❌ | After U2 | U2 working | SNS/WebSocket for critical alerts |
| U12 | Write-back to Cerner via Red Rover | US/RR | ❌ | 3-6 months | RR write API + clinical governance | AI writing into EHR — needs approval |
| U13 | Mobile/iOS alerts | FE | ❌ | 3-6 months | 📌 DAV decision | Push critical alerts to on-call docs |
| U14 | Expand to AKI/cardiac arrest | US/PAU | ❌ | 6+ months | 📌 DAV decision | New guardrails + prompts, same pipeline |
| U15 | FDA/regulatory assessment | Legal | ❌ | 6+ months | 📌 DAV decision | SaMD classification if needed |

---

## Decisions Pending

| # | Decision | Who Decides | Context | Status |
|---|----------|-------------|---------|--------|
| D1 | Frontend team allocation | DAV | Need devs for clinician dashboard + feedback UI | ⏳ Raised in leadership meeting |
| D2 | MongoDB: Atlas vs self-hosted vs existing? | SHA/NAR | Feedback storage needs a Mongo instance | ❌ Not yet discussed |
| D3 | Clinical validation study timeline | DAV/PAU | When to start? Need historical patient data | ❌ Not yet discussed |
| D4 | Which hospitals for pilot? | DAV | Multi-hospital config needs partner hospitals | ❌ Not yet discussed |
| D5 | Can AI write into EHR? | Clinical governance | Write-back to Cerner is powerful but regulated | ❌ Long-term |
| D6 | Pediatric threshold validation | PAU | Need Paula to review age-band thresholds (pSOFA, pediatric vitals norms) before implementation | ❌ Not yet discussed |

---

## Contacts

| Person | Role | Reach for |
|--------|------|-----------|
| Narendra | CI/CD & Infra | EKS, ALB, Docker, MongoDB, logging |
| Shawn | Admin | AWS permissions, Red Rover coordination, approvals |
| Paula | Clinical SME | Guardrail thresholds, test case validation, clinical review |
| David | Leadership | Team allocation, partnerships, budget, strategy |
| Ibrahim | Associate | Red Rover API testing, data exploration |
| Red Rover | External | SJSA data, API endpoints, sandbox, write-back API |

---

## Version History

| Date | Change | By |
|------|--------|----|
| Feb 11, 2026 | Created tracker with all tasks from execution plan | Sachin |
| Feb 11, 2026 | Added U16 (pediatric age-specific thresholds) and D6 (pediatric threshold validation) | Sachin |
| May 1, 2026 | Added C14–C19 (PhysioNet interim validation + locked system configuration section) | Sachin |
| May 1, 2026 | Added A6 (hunt recent Sepsis-3-era data with nurse notes) — next plan per leadership direction | Sachin |
| May 1, 2026 | Added C20 — eICU-CRD Demo Phase 1A baseline (Sens 81.82% / Spec 72.22%, n=29 reproducible); Phase 1B n=100 in flight | Sachin |
| May 1, 2026 | Added C21 — eICU-CRD Demo Phase 1B (n=90) Sens 73.33% / Spec 63.33%; FN root-cause: 100% missing BP+Lactate | Sachin |
| | | |

---

*To update: Edit this file, commit, and push. Keep the "Active Tasks" section current.*
