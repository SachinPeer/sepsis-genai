# PHI Compliance & Data Handling — Sepsis GenAI System

**Date:** April 10, 2026
**Status:** Architecture-ready. BAA and formal compliance audit pending.

---

## 1. What We Mean by "PHI Must Stay Within AWS Boundary"

When we say this, we mean:

> Patient data (vitals, labs, nurse notes) never leaves the AWS cloud infrastructure during processing. It is not sent to Anthropic, Google, Microsoft, or any third-party server.

This is why we chose **AWS Bedrock** over alternatives like Azure OpenAI (GPT-4o) or Google (Gemini). With Bedrock, the Claude model runs **inside AWS data centers** — the data stays within our AWS account boundary.

```
Hospital EHR → Red Rover → [AWS BOUNDARY] → EKS → Bedrock → EKS → Response
                                ↑                                ↑
                          Data enters AWS                  Data stays in AWS
                          (encrypted in transit)           (never leaves)
```

---

## 2. What Is PHI?

**Protected Health Information (PHI)** under HIPAA includes any individually identifiable health data:

| PHI Element | Present in Our System? | How We Handle It |
|---|---|---|
| Patient name | No — we use `patient_id` only | Never sent to the LLM |
| Date of birth | No | Not part of our data model |
| Medical record number | No — only our internal `patient_id` | De-identified key |
| Vital signs (HR, BP, etc.) | **Yes** — core input | Processed in-memory, not stored permanently |
| Lab results (Lactate, WBC) | **Yes** — core input | Processed in-memory, not stored permanently |
| Nurse notes (free text) | **Yes** — may contain PHI | Processed in-memory, sent to Bedrock within AWS |
| Diagnosis codes | Not currently | Future: SJSA codes from Red Rover |
| Age, Gender | **Yes** — demographic context | Processed in-memory |

**Key point:** We receive clinical data, process it through the AI pipeline, return a risk assessment, and **do not permanently store the original patient data**. The data flows through, it doesn't stay.

---

## 3. How PHI Flows Through Our System

```
Step 1: Red Rover API sends patient vitals + notes
        → HTTPS/TLS encrypted in transit
        → Enters our EKS pod

Step 2: Preprocessor (Stage 1) converts raw data to clinical narrative
        → In-memory only
        → No disk write

Step 3: Narrative sent to Claude Sonnet 4.5 via AWS Bedrock
        → API call stays within AWS network (VPC → Bedrock endpoint)
        → Bedrock does NOT store input/output data (opt-out is default)
        → Bedrock does NOT use our data for model training

Step 4: AI response returned to our EKS pod
        → Risk score, rationale, clinical scores generated
        → Response sent back to the calling system

Step 5: Audit log written
        → Contains: request_id, timestamp, risk_score, model_version
        → Does NOT contain: patient vitals, notes, or any PHI
        → Safe for CloudWatch logging
```

---

## 4. AWS Bedrock & PHI — What AWS Guarantees

### 4.1 Data Not Used for Training

From AWS documentation:

> *"Amazon Bedrock does not use any inputs or outputs to train Amazon Bedrock base models or distribute them to third parties. Your data remains under your control."*

This means: Patient narratives sent to Sonnet 4.5 are **not used to improve Claude** and are **not accessible to Anthropic**.

### 4.2 Data Not Stored by Bedrock

By default, Bedrock does **not** store the prompts or completions. The data is processed and discarded. If model invocation logging is explicitly enabled by us, the logs go to **our own** S3 bucket or CloudWatch — still within our AWS account.

### 4.3 AWS BAA (Business Associate Agreement)

AWS offers a **HIPAA-eligible BAA** that covers Bedrock and other services. This is a legal agreement where AWS acknowledges its role as a Business Associate handling PHI.

| Requirement | Status |
|---|---|
| AWS BAA signed | **Pending** — must be executed by Shawn/Legal |
| Bedrock covered under BAA | Yes — AWS includes Bedrock in BAA-eligible services |
| EKS covered under BAA | Yes — included in BAA-eligible services |
| S3, CloudWatch covered | Yes — included in BAA-eligible services |

**Action needed:** Shawn or the legal/compliance team must sign the AWS BAA. This is a standard process through the AWS console under AWS Artifact.

---

## 5. HIPAA Compliance Checklist

### 5.1 Technical Safeguards (What We Control)

| Safeguard | Status | Detail |
|---|---|---|
| Encryption in transit | ✅ Done | All API calls use HTTPS/TLS 1.2+ |
| Encryption at rest | ✅ AWS default | EKS, S3, CloudWatch encrypted by default (AES-256) |
| No PHI in logs | ✅ Done | Audit logs contain request_id, scores, model version — no patient data |
| No PHI in error messages | ✅ Done | Error responses return generic messages, not patient data |
| Access control | ✅ Done | API key authentication on all endpoints |
| No PHI stored on disk | ✅ Done | All patient data processed in-memory, not written to container filesystem |
| Network isolation | ✅ EKS | Pods run in private VPC subnets |
| Bedrock data opt-out | ✅ Default | AWS does not store or train on our data |

### 5.2 Administrative Safeguards (What Organization Must Do)

| Safeguard | Status | Owner |
|---|---|---|
| Sign AWS BAA | ⏳ Pending | Shawn / Legal |
| HIPAA risk assessment | ⏳ Pending | Compliance team |
| Workforce training | ⏳ Pending | HR / Compliance |
| Incident response plan | ⏳ Pending | IT Security |
| Business Associate Agreements with Red Rover | ⏳ Pending | Legal / Red Rover |
| Data retention policy | ⏳ Pending | Compliance team |
| Access audit procedures | ⏳ Pending | IT Security |

### 5.3 Physical Safeguards

| Safeguard | Status | Detail |
|---|---|---|
| Data center security | ✅ AWS | AWS SOC 2, ISO 27001 certified data centers |
| No local data storage | ✅ Done | System runs in cloud, no on-premise PHI storage |

---

## 6. What We Do NOT Do (Important Boundaries)

| We Do NOT... | Why It Matters |
|---|---|
| Store patient vitals or notes permanently | No PHI at rest in our system |
| Send PHI outside AWS | No cross-cloud data transfer |
| Include PHI in audit logs | CloudWatch/logging is HIPAA-safe |
| Allow Bedrock to train on patient data | AWS default — data is not used for model improvement |
| Store PHI in `.env` files or config | Only API keys and model IDs in configuration |
| Write PHI to container filesystem | Stateless, in-memory processing only |
| Return PHI in error messages | Generic error responses only |

---

## 7. Future Considerations

### 7.1 Doctor Feedback Storage (MongoDB)

When we implement the doctor feedback loop, MongoDB will store:
- Doctor's agree/disagree with AI score
- The AI's risk score and rationale (which may reference clinical data indirectly)

**Action needed:** Decide whether feedback data constitutes PHI and select a HIPAA-compliant MongoDB deployment (Atlas with BAA, or self-hosted in AWS VPC).

### 7.2 SJSA Alert Integration

When Red Rover SJSA data is integrated, sepsis alert codes and order data will flow through the system. Same in-memory processing model applies — no permanent storage.

### 7.3 Write-Back to EHR

If we ever write AI assessments back to Cerner via Red Rover, that creates a new PHI flow requiring additional compliance review and clinical governance approval.

---

## 8. How to Sign the AWS BAA

For Shawn or the compliance team:

1. Log into the **AWS Management Console**
2. Go to **AWS Artifact** (search in console)
3. Under **Agreements**, find **AWS Business Associate Addendum**
4. Review and accept the agreement
5. This covers all BAA-eligible services in the account (including Bedrock, EKS, S3, CloudWatch)

No additional cost. The BAA is a legal agreement, not a paid feature.

---

## 9. Summary

| Question | Answer |
|----------|--------|
| Does PHI leave AWS? | **No** — all processing within AWS boundary |
| Does Anthropic see our data? | **No** — Bedrock runs Claude inside AWS |
| Is our data used for training? | **No** — AWS default is opt-out |
| Do we store PHI? | **No** — stateless, in-memory processing |
| Are our logs HIPAA-safe? | **Yes** — no PHI in audit logs |
| Is AWS BAA signed? | **Pending** — needs Shawn/Legal action |
| Is the system HIPAA-ready? | **Architecturally yes. Administratively pending.** |

---

*The system is built with HIPAA compliance by design. The technical safeguards are in place. The remaining items are administrative — BAA signing, risk assessment, and organizational policies — which require action from compliance and legal teams.*
