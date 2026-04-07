# Sepsis GenAI — Execution Plan

**Created:** February 11, 2026
**Last Updated:** February 11, 2026
**Owner:** Sachin Jadhav

---

## Legend

| Tag | Meaning |
|-----|---------|
| 🟢 **US** | Sachin + AI assistant — fully within our control |
| 🔴 **EXT** | Externally dependent — blocked until someone else acts |
| 🟡 **HYBRID** | We do the work, but need input/approval from others |

---

## Phase 1: Immediate (This Week)

### 1.1 🔴 EXT — Upgrade EKS to Sonnet 4.5
| Item | Detail |
|------|--------|
| **What** | Change `BEDROCK_MODEL_ID` env var in Kubernetes deployment |
| **Depends on** | **Narendra** (CI/CD) |
| **Message sent** | Yes — asking him to run `kubectl set env deployment/sepsis-genai BEDROCK_MODEL_ID=us.anthropic.claude-sonnet-4-5-20250929-v1:0` |
| **Our status** | ✅ Code updated, git pushed, local `.env` updated |
| **Verification** | Hit `/health` endpoint after Narendra confirms — model field should show `sonnet-4-5` |

### 1.2 🔴 EXT — HTTPS/TLS on ALB
| Item | Detail |
|------|--------|
| **What** | Add HTTPS listener (port 443) with ACM certificate, redirect HTTP→HTTPS |
| **Depends on** | **Narendra** (infra) |
| **Message sent** | Yes — one-liner sent |
| **Why urgent** | HIPAA requires encryption in transit before real patient data flows |
| **Our action** | None — purely infrastructure |

### 1.3 🔴 EXT — SJSA Data: Red Rover Sandbox Setup
| Item | Detail |
|------|--------|
| **What** | Red Rover to load SJSA alert data for test patients (18552635, 18552637, 18564634) |
| **Depends on** | **Red Rover team** (via Shawn) |
| **Meeting held** | Yes — today (Feb 11). Asked for: endpoint clarification, full sepsis order codes, sandbox test data |
| **Email sent** | Yes — to Shawn requesting Red Rover coordination |
| **Our action** | Wait for sandbox data, then build parser |

### 1.4 🟢 US — Add older PPTXs to .gitignore
| Item | Detail |
|------|--------|
| **What** | Clean up untracked old PPTX files from repo |
| **Effort** | 5 minutes |

---

## Phase 2: Short Term (2-4 Weeks)

### 2.1 🟢 US — SJSA Integration Code
| Item | Detail |
|------|--------|
| **What** | Once Red Rover provides sandbox data, build the integration |
| **Blocked by** | Phase 1.3 (Red Rover sandbox) |
| **Work involved** | |
| | a. Parse SJSA alert orders from Red Rover `/api/v2/patients/orders` response |
| | b. Update `knowledge/genai_proprocess.py` to serialize alert data into the patient narrative |
| | c. Update `docs/prompt.md` to instruct the LLM on interpreting prior SJSA alerts |
| | d. Update `guardrail_service.py` to factor SJSA alerts into override logic |
| | e. Create test JSON with SJSA alert data for local testing |
| **Estimated effort** | 3-5 days of development |

### 2.2 🔴 EXT — Frontend: Display Sepsis AI Output
| Item | Detail |
|------|--------|
| **What** | Frontend team to build UI showing AI prediction results to clinicians |
| **Depends on** | **Frontend team** (allocation needed from David/Shawn) |
| **What we provide** | API contract (`POST /classify` response schema), documented in `docs/GENAI_OUTPUT_EXPLAINED.md` |
| **Fields to display** | Risk score, priority, alert color, clinical rationale, confidence level, confidence reasoning, qSOFA/SIRS/SOFA scores, guardrail override status |
| **API endpoint** | `POST /classify` — already production-ready on EKS |
| **Decision needed** | Where in the clinician workflow does this appear? Dashboard? Alert popup? EHR sidebar? |

### 2.3 🔴 EXT — Frontend: Doctor Feedback Capture UI
| Item | Detail |
|------|--------|
| **What** | Frontend team to build feedback form for doctors to agree/disagree with AI output |
| **Depends on** | **Frontend team** (same allocation as 2.2) |
| **What we provide** | Feedback schema (documented in `docs/FEEDBACK_LOOP_GUIDE.md`), MongoDB collection design |
| **Fields to capture** | |
| | - Doctor agrees/disagrees with risk score |
| | - Doctor's own risk assessment (if different) |
| | - Actual outcome (sepsis confirmed? when?) |
| | - Free-text comments |
| **Storage** | MongoDB (schema ready in `docs/FEEDBACK_LOOP_GUIDE.md`) |
| **Decision needed** | Frontend team allocation + MongoDB instance setup |

### 2.4 🟡 HYBRID — Model Benchmarking (50+ Synthetic Cases)
| Item | Detail |
|------|--------|
| **What** | Create 50+ synthetic patient scenarios with predetermined "correct" risk levels, run through Sonnet 4.5 |
| **Depends on** | **Paula** (clinical SME) to review and validate the "correct" answers |
| **Our work** | Create diverse test cases (high/medium/low risk, post-surgical, pediatric, elderly, septic shock, false alarms), run through API, document accuracy |
| **Output** | Accuracy report: % agreement with clinical expert assessment |
| **Estimated effort** | 1 week (case creation) + Paula review time |

### 2.5 🔴 EXT — EKS Logging Enablement
| Item | Detail |
|------|--------|
| **What** | Enable Fluent Bit / Container Insights on EKS for CloudWatch log forwarding |
| **Depends on** | **Narendra** (infra) |
| **Why** | Audit logs currently go to container stdout — need to confirm they land in CloudWatch |
| **Our action** | Verify log format once Narendra confirms logging is enabled |

### 2.6 🟢 US — MongoDB Setup for Feedback
| Item | Detail |
|------|--------|
| **What** | Set up MongoDB collection with proper indexes for feedback storage |
| **Blocked by** | Decision on MongoDB instance (existing? new? Atlas?) — **Shawn/Narendra** |
| **Our work** | Create collection schema, indexes, TTL policies per `docs/FEEDBACK_LOOP_GUIDE.md` |
| **Estimated effort** | 1 day once MongoDB instance is available |

---

## Phase 3: Medium Term (1-3 Months)

### 3.1 🟢 US — Feedback Analysis & Prompt Tuning
| Item | Detail |
|------|--------|
| **What** | Run `scripts/analyze_feedback.py` against collected doctor feedback, embed insights into prompt |
| **Blocked by** | Phase 2.3 (feedback UI) + sufficient feedback volume (50+ cases minimum) |
| **Our work** | Run analysis script → identify calibration patterns → update `docs/prompt.md` LEARNED CALIBRATIONS section |
| **Ongoing** | This is a recurring task — run monthly once feedback starts flowing |

### 3.2 🟡 HYBRID — Retrospective Clinical Validation Study
| Item | Detail |
|------|--------|
| **What** | Get 6-12 months of historical patient data with known sepsis outcomes, run our system against it |
| **Depends on** | **Shawn** (data access approval), **Red Rover** (historical data export), **Paula** (clinical oversight) |
| **Our work** | Build validation pipeline: take data at time T-6h, run prediction, compare with actual outcome at T |
| **Model** | Use **Opus 4.6** for this study (deepest reasoning, speed doesn't matter) |
| **Output** | Sensitivity, specificity, PPV, NPV, actual lead time measurement |
| **Why critical** | This is what validates the "6-hour" claim |
| **Estimated effort** | 2-3 weeks of engineering + clinical review |

### 3.3 🔴 EXT — SJSA Lead Time Comparison
| Item | Detail |
|------|--------|
| **What** | Compare our AI prediction timestamps against SJSA alert timestamps |
| **Depends on** | Phase 2.1 (SJSA integration complete) + live data flowing |
| **Question answered** | "Did our AI flag the patient BEFORE SJSA fired?" |
| **Why critical** | If yes → concrete proof of lead time advantage. Strongest evidence for leadership. |
| **Our work** | Log both timestamps, calculate delta, report distribution |

### 3.4 🟡 HYBRID — Multi-Hospital Configuration Pilot
| Item | Detail |
|------|--------|
| **What** | Deploy with different guardrail configs at 2 hospitals |
| **Depends on** | **David** (hospital partnerships), **Paula** (per-hospital threshold review) |
| **Our work** | Add hospital ID to API, load hospital-specific configs, store in S3 or DB |
| **Estimated effort** | 1 week engineering + clinical configuration per hospital |

### 3.5 🟢 US — Real-Time Alerting System
| Item | Detail |
|------|--------|
| **What** | Push high-risk predictions as real-time alerts (SNS/WebSocket) |
| **Blocked by** | Frontend integration (Phase 2.2) must be working first |
| **Our work** | Add SNS/WebSocket push when risk_score > threshold |

---

## Phase 4: Long Term (3-6+ Months)

### 4.1 🔴 EXT — Write-Back to Cerner via Red Rover
| Item | Detail |
|------|--------|
| **What** | Push AI-generated sepsis alerts back into Cerner as clinical events |
| **Depends on** | **Red Rover** (write API access), **Shawn** (approval), **Clinical governance** (what can AI write into EHR?) |
| **Regulatory consideration** | Significant — AI writing into EHR requires clinical governance approval |

### 4.2 🔴 EXT — Mobile/iOS Alert Dashboard
| Item | Detail |
|------|--------|
| **What** | Push critical alerts to mobile devices for on-call physicians |
| **Depends on** | **Frontend/Mobile team**, **David** (product decision) |

### 4.3 🟡 HYBRID — Expand to Other Conditions
| Item | Detail |
|------|--------|
| **What** | Apply same architecture to AKI (Acute Kidney Injury), cardiac arrest prediction |
| **Depends on** | **Paula** (clinical frameworks), **David** (business decision) |
| **Our work** | New guardrail configs, new prompts, same pipeline |

### 4.4 🔴 EXT — FDA/Regulatory Pathway
| Item | Detail |
|------|--------|
| **What** | Determine if system requires FDA clearance (SaMD — Software as Medical Device) |
| **Depends on** | **Legal/Regulatory team**, **David** (strategic decision) |
| **Note** | If system is "advisory only" (doctor always decides), regulatory burden is lower |

---

## Dependency Map

```
                    EXTERNAL DEPENDENCIES
    ┌─────────────────────────────────────────────┐
    │                                             │
    │  Narendra          Shawn           Paula    │
    │  ─────────         ──────          ─────    │
    │  EKS Sonnet 4.5    Red Rover       Case     │
    │  HTTPS/TLS         SJSA sandbox    review   │
    │  EKS logging       Data access     Guardrail│
    │  MongoDB infra     Approvals       Validate │
    │                                             │
    │  Frontend Team     Red Rover Team  David    │
    │  ──────────────    ──────────────  ─────    │
    │  Display output    API endpoints   Hospital │
    │  Feedback UI       Write-back      partners │
    │  Mobile app        Sandbox data    Budget   │
    │                                             │
    └──────────────────────┬──────────────────────┘
                           │
                           ▼
    ┌─────────────────────────────────────────────┐
    │           OUR WORK (Sachin + AI)            │
    │                                             │
    │  Phase 1: Sonnet 4.5 code ✅ Done           │
    │  Phase 2: SJSA parser, benchmark cases,     │
    │           MongoDB schema, prompt tuning     │
    │  Phase 3: Validation pipeline, multi-hospital│
    │           config, alerting system           │
    │  Phase 4: New conditions, write-back logic  │
    │                                             │
    └─────────────────────────────────────────────┘
```

---

## Quick Reference: Who to Contact for What

| Person | Role | Contact For |
|--------|------|-------------|
| **Narendra** | CI/CD & Infra | EKS deployments, ALB config, logging, MongoDB setup |
| **Shawn** | Admin/Manager | AWS permissions, Red Rover coordination, data access approvals |
| **Paula** | Clinical SME | Guardrail thresholds, test case validation, clinical accuracy review |
| **David** | Leadership | Hospital partnerships, team allocation, budget, strategic decisions |
| **Ibrahim** | Associate | Red Rover API testing, data exploration |
| **Frontend Team** | (TBD) | Clinician dashboard, feedback UI, mobile app |
| **Red Rover Team** | External | API endpoints, SJSA data, sandbox setup, write-back API |

---

## Current Status Summary

| Area | Status |
|------|--------|
| Core AI Pipeline | ✅ Production-ready on EKS |
| LLM Model Selection | ✅ Sonnet 4.5 selected, comparison documented |
| Clinical Guardrails | ✅ 850+ parameters, API-managed |
| Deterministic Scores | ✅ qSOFA, SIRS, SOFA implemented |
| Audit Logging | ✅ Structured JSON, no PHI |
| Feedback Analysis Script | ✅ Built (`scripts/analyze_feedback.py`) |
| Feedback Guide | ✅ Documented (`docs/FEEDBACK_LOOP_GUIDE.md`) |
| HTTPS/TLS | ⏳ Waiting on Narendra |
| SJSA Integration | ⏳ Waiting on Red Rover sandbox |
| Frontend Display | ❌ Not started — needs team allocation |
| Doctor Feedback UI | ❌ Not started — needs team allocation |
| Clinical Validation | ❌ Not started — needs historical data |
| Multi-Hospital Config | ❌ Not started — needs hospital partnerships |

---

*This plan is a living document. Update as dependencies are resolved and new tasks emerge.*
