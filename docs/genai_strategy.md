# Strategic Roadmap: Multimodal Sepsis Early Warning System (2026)

## 1. Executive Vision
The primary objective of this MVP project is to develop and deploy an advanced **Sepsis Prediction System** within the AWS infrastructure to enable early clinical intervention for at-risk patients. By leveraging data from **RedRover APIs**—including physiological vital signs, critical laboratory values, and demographic parameters—fused with **unstructured clinician nursing notes**, the system ensures that high-velocity predictive power is blended with medical subject matter expertise, providing narrative-driven clinical explanations.

---

## 2. Production Data Flow: The 5-Minute Heartbeat
The architecture utilizes a "Streaming Batch" heartbeat to ensure 24/7 clinical monitoring without the overhead of manual infrastructure management.



1.  **Ingestion (AWS Lambda):** Automatically calls the Red Rover `GET /patients` API every 5 minutes to fetch the latest vitals/labs for 1,000 patients.
2.  **Multimodal Contextualizer:** A Python-based preprocessing layer that merges numeric data with unstructured nurse/physician notes.
3.  **Narrative Serialization:** Converts the raw fusion of data into a **"Medical Prose"** summary (The Sepsis Story).
4.  **Inference Engine (AWS Bedrock):** Submits the narratives in bulk via **Claude 3.5 Sonnet** using Batch Mode for deep reasoning.
5.  **Deterministic Guardrail:** A final Python validation layer (Stage 3) cross-checks the AI’s risk score against hard **Sepsis-3** thresholds (Lactate, MAP, SBP).
6.  **Alerting:** Validated risks are stored in **DynamoDB** and pushed to clinician mobile dashboards.

---

## 3. The Three-Stage Architecture

### **Stage 1: Narrative Serialization & Multimodal Fusion**
Instead of feeding raw JSON to the LLM, an in-house Python layer translates data into a verbose clinical narrative.
* **Method:** In-house Python (Lambda) using medical templates (Jinja2).
* **Discordance Analysis:** The system is weighted to escalate risk if vitals look stable but nursing notes indicate "Mottled skin," "Lethargy," or "Fluid bolus required." This identifies "Silent Sepsis" where physiological compensation masks organ failure.

### **Stage 2: Batch Inference (AWS Bedrock)**
Optimized for high-volume processing without rate-limiting or high on-demand costs.
* **Execution:** Processes 1,000 narratives per batch for 50% cost reduction and higher throughput stability.
* **Prompting:** Uses **Chain of Thought (CoT)**, forcing the LLM to identify the **Velocity of Change** (e.g., rising HR, dropping SBP) rather than just static values.

### **Stage 3: Deterministic Guardrails (Validation)**
**Location:** `/scripts/guardrail_service.py`
A "Hybrid" safety net ensures AI reasoning never bypasses hard medical facts.
* **Override Rule:** If the LLM predicts "Low Risk" but the patient meets critical Sepsis-3 criteria (e.g., SBP ≤ 90 or Lactate ≥ 2.0), the system automatically triggers a **"CRITICAL OVERRIDE"** alert.
* **Output:** Displays the Risk Score, the AI Clinical Rationale, and any specific Rules Triggered.

---

## 4. Implementation Workflow

| Step | Component | Technology |
| :--- | :--- | :--- |
| **1. Ingestion** | Red Rover API | AWS Lambda |
| **2. Serialization** | Multimodal Fusion | Python (Pre-processing) |
| **3. Inference** | Batch Prediction | AWS Bedrock (Claude 3.5) |
| **4. Validation** | **Safety Overrides** | **Python (Guardrail Service)** |
| **5. Delivery** | Dashboard Alerting | DynamoDB / AppSync |



---

## 5. Key Strategic Advantages
* **Clinical Accuracy:** Narrative serialization reduces "numerical confusion" in LLMs, allowing them to understand clinical trends over a 6-hour window.
* **Scalability:** AWS Bedrock Batch Inference handles 288,000 checks/day without the need for managing custom GPU clusters.
* **Hybrid Safety:** Combines GenAI's narrative reasoning with the "certainty" of deterministic rules to eliminate risk from hallucinations.

---

## 6. Maintenance & Quality Control
* **Self-Correction:** Monthly audits compare the AI’s 6-hour predictions against actual patient outcomes to "fine-tune" the system prompt.
* **Audit Trail:** Every "Guardrail Override" is logged to identify where the AI model may be underestimating specific clinical signs.

---
**Document Status:** Version 1.3 (Finalized for MVP)  
**Date:** February 9, 2026