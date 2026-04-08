# Sepsis GenAI — Project Tracker

> **Living document** — Update status and dates as work progresses.
> Last updated: **Apr 8, 2026** by Sachin

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

---

## Active Tasks

| # | Task | Owner | Status | Due | Blocker | Notes |
|---|------|-------|--------|-----|---------|-------|
| A1 | Sonnet 4.5 on EKS | NAR | ⏳ | Feb 12 | — | Message sent to Narendra: `kubectl set env` |
| A2 | HTTPS/TLS on ALB | NAR | ⏳ | Feb 14 | — | ACM cert + port 443 listener + HTTP redirect |
| A3 | Red Rover SJSA sandbox | RR/SHA | ⏳ | TBD | Red Rover response | Meeting held Feb 11. Email sent to Shawn |
| A4 | Red Rover: confirm API endpoint | RR | ⏳ | TBD | Red Rover response | `/patients/orders` or `/encounters/{id}/events`? |
| A5 | Red Rover: full sepsis code list | RR | ⏳ | TBD | Red Rover response | 5 codes identified, need complete list |

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
| | | |
| | | |

---

*To update: Edit this file, commit, and push. Keep the "Active Tasks" section current.*
