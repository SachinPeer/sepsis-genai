# Competitive Benchmark — AI Sepsis Prediction Systems

> **Last updated:** Feb 11, 2026
> **Purpose:** Compare our GenAI Sepsis Early Warning System against published and commercial alternatives.

---

## Market Landscape

| System | Developer | Type | Approach | Prediction Window | Status |
|--------|-----------|------|----------|-------------------|--------|
| **Epic Sepsis Model (ESM)** | Epic Systems | ML | Structured EHR data only (vitals, labs) | Real-time alerts | Widely deployed in Epic hospitals |
| **COMPOSER-LLM** | UC San Diego | ML + LLM hybrid | Structured data + LLM for borderline cases | Real-time | Published 2025, prospectively validated |
| **Sepsis ImmunoScore** | Prenosis | ML | 22 biomarker parameters | 24-hour prediction | FDA De Novo authorized (Apr 2024) |
| **NAVOY** | AlgoDx | ML | 19 clinical parameters, ICU-focused | 3-hour foresight | CE-marked, clinically validated |
| **VIOSync** | Aisthesis Medical | ML + Digital Twin | EHR + monitors + wearables + digital twin | Up to 48 hours | Commercial product |
| **Predictiv AI** | Predictiv AI Inc. | Small Language Models | Domain-specific SLMs for triage | TBD | Announced Apr 2026, patent filed |
| **Medbeacon Sepsis GenAI** | Medbeacon | GenAI (LLM) + Deterministic Guardrail | Full narrative intelligence + configurable safety layer | 6-hour prediction | EKS deployed, active testing |

---

## Detailed Comparison

### Epic Sepsis Model (ESM)

- **How it works:** Proprietary ML model built into Epic EHR. Uses structured vitals and lab data to generate sepsis risk scores.
- **Performance:** Highly variable across validations.
  - Best case: AUC 0.834, sensitivity 86%, specificity 80.8% (2023, academic trauma center)
  - Worst case: AUC 0.63 (Michigan Medicine, 2020) — missed **67% of sepsis patients**, generated alerts for 18% of hospitalizations
- **Limitations:**
  - No clinician notes analysis
  - No configurable thresholds — hospitals cannot adjust
  - Significant alert fatigue problem
  - Black-box model with no explainability
- **Source:** JAMA Internal Medicine (2021), JAMIA Open (2024)

### COMPOSER-LLM (UC San Diego)

- **How it works:** Enhances the COMPOSER ML model (2021) with an LLM layer. The LLM is invoked **only for borderline/uncertain predictions** to extract context from unstructured clinical notes (triage notes, progress notes, radiology reports).
- **Performance:** Evaluated on 2,074–2,500 patient encounters at UCSD Health.
  - Sensitivity: 70.3–72.1%
  - Positive Predictive Value: 31.9–52.9%
  - F-1 Score: 44.2–61.0%
  - False alarms: 0.0087–0.020 per patient hour
  - Manual chart review found 62% of false positives had bacterial infections
- **Strengths:** Closest architecture to ours; uses LLM for unstructured data; open-source
- **Limitations:**
  - LLM invoked only for borderline cases, not all patients
  - No deterministic guardrail layer
  - No configurable thresholds or hospital-specific settings
  - No fallback when LLM is unavailable
  - Uses open-source LLMs (Llama-3, Mixtral) — less capable than Claude Sonnet 4.5
- **Source:** npj Digital Medicine (2025), medRxiv, Pacific Symposium on Biocomputing (2025)

### Sepsis ImmunoScore (Prenosis)

- **How it works:** FDA-authorized ML diagnostic tool using 22 holistic parameters and biological data. Provides four risk categories predicting sepsis within 24 hours, in-hospital mortality, ICU transfer, and need for vasopressors/mechanical ventilation.
- **Strengths:** First FDA De Novo authorized AI sepsis diagnostic (April 2024); explainable AI showing parameter contributions
- **Limitations:**
  - No narrative intelligence — doesn't read clinician notes
  - No LLM reasoning
  - Fixed parameters, not hospital-configurable
  - Diagnostic (current state) rather than predictive (6-hour window)
- **Source:** FDA De Novo authorization (2024), prenosis.com

### NAVOY (AlgoDx)

- **How it works:** CE-marked ML software for ICU sepsis prediction. Uses 19 clinical parameters including vital signs, labs, and demographics. Integrates with EHR/PDMS systems.
- **Performance:** Validated on 60,000+ ICU stays; provides 3-hour predictive foresight
- **Limitations:**
  - ICU-only (not general ward or ED)
  - No unstructured data analysis
  - No configurability
  - 3-hour window (vs our 6-hour)
- **Source:** navoy.algodx.com

### VIOSync (Aisthesis Medical)

- **How it works:** AI-powered clinical decision support using explainable AI and digital patient twins. Integrates data from EHRs, monitors, labs, and wearables. Claims prediction up to 48 hours earlier.
- **Strengths:** Most ambitious prediction window; protocol-based guidance aligned with national sepsis bundles; clinician-friendly explanations
- **Limitations:**
  - No LLM-based narrative reasoning
  - Requires wearable/monitor integration (complex deployment)
  - No published peer-reviewed validation data found
- **Source:** aisthesismed.com

---

## Feature-by-Feature Comparison

| Capability | Epic ESM | COMPOSER-LLM | Prenosis | NAVOY | VIOSync | **Medbeacon** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| LLM narrative reasoning | — | Partial | — | — | — | **Full** |
| Every patient through LLM | — | Borderline only | — | — | — | **Yes** |
| Clinician notes analysis | — | Yes | — | — | — | **Yes** |
| Deterministic scores (qSOFA/SIRS/SOFA) | — | — | — | — | Partial | **Yes** |
| Configurable guardrails (API) | — | — | — | — | — | **Yes** |
| Hospital-specific config | — | — | — | — | — | **Planned** |
| Age-aware thresholds | — | — | — | — | Unknown | **Planned** |
| Graceful LLM degradation | N/A | — | N/A | N/A | N/A | **Yes** |
| Confidence level + reasoning | — | — | — | — | Partial | **Yes** |
| History-aware context checks | — | — | — | — | — | **Yes** |
| Early detection patterns | Basic | — | — | — | — | **Yes** |
| Discordance detection (silent sepsis) | — | — | — | — | — | **Yes** |
| SME-editable thresholds via API | — | — | — | — | — | **Yes** |
| Structured audit logging | Unknown | Unknown | Unknown | Unknown | Unknown | **Yes** |
| FDA authorization | — | — | **Yes** | CE-mark | — | Planned |

---

## Our Differentiators

### 1. Full Narrative Intelligence on Every Patient
COMPOSER-LLM only uses the LLM for borderline cases. We run every patient through Claude Sonnet 4.5 for comprehensive narrative reasoning — catching subtle patterns that ML models miss.

### 2. 3-Stage Architecture with Deterministic Safety Net
No other system separates AI reasoning from deterministic validation this cleanly:
- **Stage 1:** Preprocessor converts raw data to clinical narrative
- **Stage 2:** LLM provides reasoning, confidence, and risk assessment
- **Stage 3:** Guardrail applies 200+ configurable clinical rules independently

### 3. Configurable, Hospital-Specific Guardrails
No competing system offers API-managed, SME-editable clinical thresholds. Our guardrail config covers 11 organ systems with 200+ configurable parameters that can be adjusted per hospital without code changes.

### 4. Graceful Degradation
When the LLM is unavailable, our system still delivers deterministic qSOFA, SIRS, SOFA scores, early detection patterns, and threshold-based override alerts — in under 5ms. No other system offers this fallback.

### 5. History-Aware Context
Our guardrail adjusts interpretation based on patient medical history (chronic HTN, renal disease, seizure history, medications). No competing system has this capability.

### 6. Discordance Detection (Silent Sepsis)
We detect "silent sepsis" — cases where vitals appear stable but clinician notes reveal concerning signs (mottled skin, altered mental status, fluid refractory). This cross-references structured and unstructured data in the guardrail layer, not just the LLM.

---

## Gaps to Address

| Gap | Competitor Advantage | Our Plan | Tracker |
|-----|---------------------|----------|---------|
| FDA/regulatory clearance | Prenosis has FDA De Novo | Planned assessment | U15 |
| Peer-reviewed validation | COMPOSER-LLM published in npj Digital Medicine | Retrospective validation study planned | U8 |
| Prediction window | VIOSync claims 48 hours | Our 6-hour window is clinically actionable; longer windows have higher false positive rates | — |
| Wearable integration | VIOSync integrates wearables | Not planned for v1 | — |
| Pediatric thresholds | No competitor has this either | Planned (age-aware guardrails) | U16, D6 |
| Open-source availability | COMPOSER-LLM is open-source | Commercial product; not planned for open-source | — |

---

## Key Research References

1. **COMPOSER-LLM:** "Development and Prospective Implementation of a Large Language Model based System for Early Sepsis Prediction" — npj Digital Medicine, 2025
2. **Epic Sepsis Model critique:** "External Validation of a Widely Implemented Proprietary Sepsis Prediction Model in Hospitalized Patients" — JAMA Internal Medicine, 2021
3. **Prenosis ImmunoScore:** FDA De Novo Authorization, April 2024
4. **AI Sepsis CDSS Review:** "Patient Benefits in the Context of Sepsis-Related AI-Based Clinical Decision Support Systems: Scoping Review" — JMIR, 2026
5. **LLM Model Comparison for Sepsis:** "Llama-3 8B vs Mixtral 8x7B for sepsis prediction" — Pacific Symposium on Biocomputing, 2025

---

*This document should be updated as new competitors emerge or our capabilities evolve.*
