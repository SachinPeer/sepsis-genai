"""
GenAI Sepsis Prediction Pipeline
Complete orchestration of the 3-Stage Architecture:
  Stage 1: Preprocessor (Narrative Serialization)
  Stage 2: LLM Inference (Azure OpenAI / GPT-4o)
  Stage 3: Guardrail (Deterministic Safety Validation)

This pipeline runs PARALLEL to the existing Hybrid ML model.
"""

import os
import sys
import json
import time
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

# Add preprocessing folder to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'preprocessing'))

from genai_inference_service import get_genai_service, GenAIInferenceService
from guardrail_service import SepsisSafetyGuardrail

# Import preprocessor from preprocessing folder
try:
    from preprocessing.genai_preprocess import SepsisPreprocessor
except ImportError:
    from genai_preprocess import SepsisPreprocessor

logger = logging.getLogger(__name__)


class GenAISepsisPipeline:
    """
    Complete GenAI-based sepsis prediction pipeline.
    
    This is the "Narrative Intelligence" pathway that:
    1. Converts raw vitals + notes into clinical prose
    2. Sends to GPT-4o for reasoning
    3. Validates with deterministic safety guardrails
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the pipeline with all three stages.
        
        Args:
            config_path: Optional path to guardrail config (uses defaults if not provided)
        """
        # Stage 1: Preprocessor
        self.preprocessor = SepsisPreprocessor()
        
        # Stage 2: LLM Inference Service (lazy loaded)
        self._llm_service: Optional[GenAIInferenceService] = None
        
        # Stage 3: Guardrail
        self.guardrail = SepsisSafetyGuardrail()
        
        # Pipeline metadata
        self.version = "1.0.0"
        self.pipeline_name = "GenAI Sepsis Early Warning System"
        
        logger.info(f"GenAI Pipeline initialized (v{self.version})")
    
    @property
    def llm_service(self) -> GenAIInferenceService:
        """Lazy-load the LLM service."""
        if self._llm_service is None:
            self._llm_service = get_genai_service()
        return self._llm_service
    
    def predict(
        self, 
        patient_vitals: Dict[str, Any], 
        clinician_notes: str = None,
        patient_id: str = None,
        patient_history: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run the full 3-stage prediction pipeline.
        
        Args:
            patient_vitals: Dictionary of vital signs and lab values
                           (supports Red Rover nested format or flat dict)
            clinician_notes: Optional unstructured nursing/physician notes
            patient_id: Optional patient identifier for logging
            patient_history: Optional dict with 'conditions' (list) and 'medications' (list)
            
        Returns:
            Complete prediction result with all stage outputs
        """
        start_time = time.time()
        request_id = f"genai_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{patient_id or 'unknown'}"
        
        logger.info(f"[{request_id}] Starting GenAI pipeline prediction")
        
        result = {
            "request_id": request_id,
            "patient_id": patient_id,
            "timestamp": datetime.now().isoformat(),
            "pipeline_version": self.version,
            "stages": {}
        }
        
        try:
            # ============================================
            # STAGE 1: Narrative Serialization
            # ============================================
            stage1_start = time.time()

            narrative = self.preprocessor.process(patient_vitals, clinician_notes)

            # Append deterministic clinical scores to the narrative so the LLM
            # has formal sepsis criteria as explicit anchors instead of being
            # forced to derive them from raw numbers. Discovered to reduce
            # LLM-driven false positives on the eICU v4 cohort. See
            # validation/docs/EICU_VALIDATION_EXECUTION.md §8.
            if os.getenv("INCLUDE_CLINICAL_SCORES_IN_NARRATIVE", "true").lower() == "true":
                flat_for_scores = self._flatten_vitals(patient_vitals)
                narrative_scores = self._build_clinical_scores_narrative(flat_for_scores)
                if narrative_scores:
                    narrative = f"{narrative}\n\n{narrative_scores}"

            result["stages"]["stage1_preprocessing"] = {
                "status": "success",
                "narrative_length": len(narrative),
                "processing_time_ms": (time.time() - stage1_start) * 1000,
                "narrative_preview": narrative[:500] + "..." if len(narrative) > 500 else narrative
            }

            logger.info(f"[{request_id}] Stage 1 complete: {len(narrative)} chars")
            
            # ============================================
            # STAGE 2: LLM Inference
            # ============================================
            stage2_start = time.time()
            
            try:
                llm_temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))
            except ValueError:
                llm_temperature = 0.1
            llm_response = self.llm_service.predict(narrative, temperature=llm_temperature)
            
            result["stages"]["stage2_llm_inference"] = {
                "status": "success" if "_metadata" not in llm_response or not llm_response.get("_metadata", {}).get("error") else "error",
                "processing_time_ms": (time.time() - stage2_start) * 1000,
                "raw_prediction": llm_response
            }
            
            logger.info(f"[{request_id}] Stage 2 complete: Risk={llm_response.get('prediction', {}).get('risk_score_0_100', 'N/A')}")
            
            # ============================================
            # STAGE 3: Guardrail Validation
            # ============================================
            stage3_start = time.time()
            
            # Flatten vitals for guardrail (handles nested Red Rover format)
            flat_vitals = self._flatten_vitals(patient_vitals)
            
            validated_output = self.guardrail.validate_prediction(
                llm_response,
                flat_vitals,
                nursing_notes=clinician_notes or "",
                patient_history=patient_history,
                raw_vitals_timeseries=patient_vitals,
            )
            
            result["stages"]["stage3_guardrail"] = {
                "status": "success",
                "processing_time_ms": (time.time() - stage3_start) * 1000,
                "override_applied": validated_output.get("logic_gate", {}).get("guardrail_override", False),
                "override_reasons": validated_output.get("logic_gate", {}).get("override_reasons", []),
                "early_warnings": validated_output.get("logic_gate", {}).get("early_warnings", []),
                "history_context": validated_output.get("logic_gate", {}).get("history_context", [])
            }
            
            logger.info(f"[{request_id}] Stage 3 complete: Override={validated_output.get('logic_gate', {}).get('guardrail_override', False)}")
            
            # ============================================
            # CLINICAL SCORES (Deterministic Calculation)
            # ============================================
            clinical_scores = self.guardrail.calculate_clinical_scores(flat_vitals)
            
            # ============================================
            # FINAL OUTPUT
            # ============================================
            result["final_prediction"] = validated_output.get("prediction", {})
            result["clinical_metrics"] = validated_output.get("clinical_metrics", {})
            result["logic_gate"] = validated_output.get("logic_gate", {})
            
            # Add deterministic clinical scores (guardrail-calculated, no LLM latency)
            result["deterministic_scores"] = clinical_scores
            
            # Merge confidence into final prediction (from LLM)
            if "confidence_level" in llm_response.get("prediction", {}):
                result["final_prediction"]["confidence_level"] = llm_response["prediction"]["confidence_level"]
            if "confidence_reasoning" in llm_response.get("prediction", {}):
                result["final_prediction"]["confidence_reasoning"] = llm_response["prediction"]["confidence_reasoning"]
            
            result["total_processing_time_ms"] = (time.time() - start_time) * 1000
            result["status"] = "success"
            
            # Add clinical summary for dashboard
            result["dashboard_summary"] = self._generate_dashboard_summary(validated_output, clinical_scores)
            
            # Audit logging (for compliance)
            self.log_audit(result, patient_vitals)
            
        except Exception as e:
            logger.error(f"[{request_id}] Pipeline error: {e}")
            result["status"] = "error"
            result["error"] = str(e)
            result["total_processing_time_ms"] = (time.time() - start_time) * 1000
            
            # Fail-safe: Return high priority alert on error
            result["final_prediction"] = {
                "risk_score_0_100": 70,
                "priority": "High",
                "sepsis_probability_6h": "Moderate",
                "clinical_rationale": f"[SYSTEM ERROR] Manual review required. Error: {str(e)[:100]}"
            }
        
        return result
    
    def predict_batch(
        self, 
        patients: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process multiple patients through the pipeline.
        
        Args:
            patients: List of dicts, each containing:
                      - vitals: Dict of vital signs
                      - notes: Optional clinician notes
                      - patient_id: Optional identifier
                      
        Returns:
            List of prediction results
        """
        results = []
        total = len(patients)
        
        logger.info(f"Starting batch prediction for {total} patients")
        
        for i, patient in enumerate(patients):
            logger.info(f"Processing patient {i+1}/{total}")
            
            result = self.predict(
                patient_vitals=patient.get("vitals", patient),
                clinician_notes=patient.get("notes"),
                patient_id=patient.get("patient_id", f"batch_{i}"),
                patient_history=patient.get("patient_history")
            )
            result["batch_index"] = i
            results.append(result)
        
        return results
    
    def _build_clinical_scores_narrative(self, flat_vitals: Dict[str, Any]) -> str:
        """Render qSOFA, SIRS and SOFA into a compact text block the LLM can
        anchor on. Uses the same deterministic rules the guardrail uses, so
        the LLM and guardrail can never disagree on the formal scores."""
        try:
            scores = self.guardrail.calculate_clinical_scores(flat_vitals)
        except Exception as e:
            logger.warning(f"Score precompute failed, sending narrative without scores: {e}")
            return ""

        qsofa = scores.get("qsofa", {}) or {}
        sirs  = scores.get("sirs", {})  or {}
        sofa  = scores.get("sofa", {})  or {}

        def _yn(b):
            return "yes" if b else "no"

        q_parts = []
        if qsofa.get("components"):
            c = qsofa["components"]
            q_parts.append(f"RR≥22: {_yn(c.get('respiratory_rate_ge_22'))}")
            q_parts.append(f"GCS<15 (altered mentation): {_yn(c.get('altered_mentation'))}")
            q_parts.append(f"SBP≤100: {_yn(c.get('systolic_bp_le_100'))}")

        s_parts = []
        if sirs.get("components"):
            c = sirs["components"]
            s_parts.append(f"Temp >38°C or <36°C: {_yn(c.get('temp_abnormal'))}")
            s_parts.append(f"HR >90: {_yn(c.get('hr_gt_90'))}")
            s_parts.append(f"RR >20 or PaCO2<32: {_yn(c.get('rr_gt_20_or_paco2_lt_32'))}")
            s_parts.append(f"WBC abnormal (>12, <4, or bands>10%): {_yn(c.get('wbc_abnormal'))}")

        sofa_total = sofa.get("estimated_score") or sofa.get("total_score")
        sofa_parts = []
        if sofa.get("components"):
            for organ in ("respiratory", "coagulation", "liver", "cardiovascular", "cns", "renal"):
                v = sofa["components"].get(organ)
                if v is not None:
                    sofa_parts.append(f"{organ}: {v}")

        lines = ["--- DETERMINISTIC CLINICAL SCORES ---"]
        lines.append(
            f"qSOFA: {qsofa.get('score', '?')}/3 — {qsofa.get('interpretation', 'n/a')}"
        )
        if q_parts:
            lines.append(f"  components: {'; '.join(q_parts)}")

        lines.append(
            f"SIRS: {sirs.get('criteria_met', '?')}/4 criteria met "
            f"({'positive' if sirs.get('sirs_positive') else 'negative'})"
        )
        if s_parts:
            lines.append(f"  components: {'; '.join(s_parts)}")

        if sofa_total is not None:
            lines.append(f"SOFA estimate: {sofa_total}/24")
            if sofa_parts:
                lines.append(f"  organ scores: {'; '.join(sofa_parts)}")

        lines.append(
            "(These scores are deterministic and computed from the same vitals "
            "you have above. Use them as anchors — disagreement between your "
            "narrative reasoning and these scores should be explicitly justified.)"
        )

        return "\n".join(lines)

    def _flatten_vitals(self, vitals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Flatten nested Red Rover format to simple key-value pairs.
        Takes the most recent value from any time series.
        """
        flat = {}
        for key, value in vitals.items():
            if isinstance(value, list) and len(value) > 0:
                # Red Rover format: [{"val": 118, "ts": "..."}, ...]
                if isinstance(value[0], dict) and "val" in value[0]:
                    flat[key] = value[0]["val"]
                else:
                    # Simple list: take first (most recent)
                    flat[key] = value[0]
            else:
                flat[key] = value
        return flat
    
    def _generate_dashboard_summary(self, prediction: Dict[str, Any], 
                                      clinical_scores: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a human-readable summary for the iOS dashboard.
        """
        pred = prediction.get("prediction", {})
        risk = pred.get("risk_score_0_100", 0)
        priority = pred.get("priority", "Standard")
        rationale = pred.get("clinical_rationale", "")
        confidence = pred.get("confidence_level", "Unknown")
        
        # Color coding
        if risk >= 80 or priority == "Critical":
            alert_color = "red"
            alert_level = "CRITICAL"
        elif risk >= 50 or priority == "High":
            alert_color = "orange"
            alert_level = "HIGH"
        else:
            alert_color = "green"
            alert_level = "STANDARD"
        
        # Override annotation
        if prediction.get("logic_gate", {}).get("guardrail_override"):
            override_note = " [GUARDRAIL OVERRIDE APPLIED]"
        else:
            override_note = ""
        
        # Build summary
        summary = {
            "alert_level": alert_level,
            "alert_color": alert_color,
            "headline": f"Sepsis Risk: {risk}% - {priority} Priority{override_note}",
            "summary": rationale[:200] + "..." if len(rationale) > 200 else rationale,
            "confidence": confidence,
            "action_required": "Immediate clinical review" if alert_level == "CRITICAL" else "Monitor closely" if alert_level == "HIGH" else "Routine monitoring"
        }
        
        # Add clinical scores summary if available
        if clinical_scores:
            qsofa = clinical_scores.get("qsofa", {})
            sirs = clinical_scores.get("sirs", {})
            sofa = clinical_scores.get("sofa", {})
            sepsis = clinical_scores.get("sepsis_criteria", {})
            
            summary["scores_summary"] = {
                "qSOFA": f"{qsofa.get('score', '?')}/3",
                "SIRS": f"{sirs.get('criteria_met', '?')}/4",
                "SOFA_estimated": f"{sofa.get('estimated_score', '?')}/24",
                "sepsis_3_met": sepsis.get("sepsis_3_met", False),
                "septic_shock_criteria": sepsis.get("septic_shock_criteria_met", False)
            }
            
            # Add warning if septic shock criteria met
            if sepsis.get("septic_shock_criteria_met"):
                summary["critical_warning"] = "SEPTIC SHOCK CRITERIA MET - Immediate intervention required"
            elif sepsis.get("sepsis_3_met"):
                summary["critical_warning"] = "Sepsis-3 criteria met - Close monitoring required"
        
        return summary
    
    def health_check(self) -> Dict[str, Any]:
        """Check health of all pipeline components."""
        return {
            "pipeline_version": self.version,
            "components": {
                "preprocessor": "healthy",
                "llm_service": self.llm_service.health_check(),
                "guardrail": "healthy"
            }
        }
    
    def _create_audit_log(self, result: Dict[str, Any], patient_vitals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a structured audit log entry for compliance and review.
        
        This log captures all information needed for:
        - Regulatory compliance (audit trail)
        - Clinical review of AI decisions
        - Model performance monitoring
        - Incident investigation
        """
        final_pred = result.get("final_prediction", {})
        logic_gate = result.get("logic_gate", {})
        deterministic = result.get("deterministic_scores", {})
        
        audit_entry = {
            "audit_version": "1.0",
            "timestamp": datetime.now().isoformat(),
            "request_id": result.get("request_id"),
            "patient_id": result.get("patient_id"),
            
            # Model information
            "model_info": {
                "pipeline_version": self.version,
                "llm_provider": self.llm_service.provider.provider_name if self._llm_service else "not_initialized",
                "llm_model": self.llm_service.provider.model_name if self._llm_service else "not_initialized"
            },
            
            # Input summary (no PHI in logs - just structure)
            "input_summary": {
                "vitals_provided": list(patient_vitals.keys()) if patient_vitals else [],
                "vitals_count": len(patient_vitals) if patient_vitals else 0,
                "notes_provided": result.get("stages", {}).get("stage1_preprocessing", {}).get("narrative_length", 0) > 100
            },
            
            # Prediction output
            "prediction": {
                "risk_score": final_pred.get("risk_score_0_100"),
                "confidence_level": final_pred.get("confidence_level"),
                "priority": final_pred.get("priority"),
                "sepsis_probability_6h": final_pred.get("sepsis_probability_6h")
            },
            
            # Clinical scores
            "clinical_scores": {
                "qsofa_score": deterministic.get("qsofa", {}).get("score"),
                "sirs_criteria_met": deterministic.get("sirs", {}).get("criteria_met"),
                "sofa_estimated": deterministic.get("sofa", {}).get("estimated_score"),
                "sepsis_3_met": deterministic.get("sepsis_criteria", {}).get("sepsis_3_met"),
                "septic_shock_met": deterministic.get("sepsis_criteria", {}).get("septic_shock_criteria_met")
            },
            
            # Guardrail actions
            "guardrail": {
                "override_applied": logic_gate.get("guardrail_override", False),
                "override_reasons": logic_gate.get("override_reasons", []),
                "discordance_detected": logic_gate.get("discordance_detected", False)
            },
            
            # Processing metrics
            "processing": {
                "total_time_ms": result.get("total_processing_time_ms"),
                "stage1_time_ms": result.get("stages", {}).get("stage1_preprocessing", {}).get("processing_time_ms"),
                "stage2_time_ms": result.get("stages", {}).get("stage2_llm_inference", {}).get("processing_time_ms"),
                "stage3_time_ms": result.get("stages", {}).get("stage3_guardrail", {}).get("processing_time_ms")
            },
            
            # Status
            "status": result.get("status"),
            "error": result.get("error")
        }
        
        return audit_entry
    
    def log_audit(self, result: Dict[str, Any], patient_vitals: Dict[str, Any]):
        """
        Log an audit entry for the prediction.
        
        In production, this could write to:
        - A dedicated audit log file
        - AWS CloudWatch Logs
        - A HIPAA-compliant audit database
        """
        audit_entry = self._create_audit_log(result, patient_vitals)
        
        # Log as structured JSON for easy parsing
        audit_logger = logging.getLogger("sepsis_audit")
        audit_logger.info(json.dumps(audit_entry, default=str))


# Singleton instance
_pipeline: Optional[GenAISepsisPipeline] = None

def get_genai_pipeline() -> GenAISepsisPipeline:
    """Get or create the GenAI pipeline singleton."""
    global _pipeline
    if _pipeline is None:
        _pipeline = GenAISepsisPipeline()
    return _pipeline


# --- CLI Testing ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    pipeline = GenAISepsisPipeline()
    
    # Test case: Red Rover style nested data
    test_patient = {
        "vitals": {
            "HR": [{"val": 118, "ts": "2026-02-07T18:30"}, {"val": 105, "ts": "2026-02-07T18:25"}],
            "SBP": [{"val": 85, "ts": "2026-02-07T18:30"}, {"val": 95, "ts": "2026-02-07T18:25"}],
            "Temp": [38.9],
            "Lactate": [3.2],
            "WBC": [15.8],
            "Creatinine": [2.1],
            "Age": 68,
            "Gender": 1
        },
        "notes": "Patient is complains of chills. Altered mental status noted by nursing staff. Skin appears mottled. Required 2L fluid bolus.",
        "patient_id": "TEST_001"
    }
    
    print("=" * 60)
    print("GenAI Sepsis Pipeline - Test Run")
    print("=" * 60)
    
    # Run prediction
    result = pipeline.predict(
        patient_vitals=test_patient["vitals"],
        clinician_notes=test_patient["notes"],
        patient_id=test_patient["patient_id"]
    )
    
    print("\n--- RESULT ---")
    print(json.dumps(result, indent=2, default=str))
    
    print("\n--- DASHBOARD SUMMARY ---")
    print(json.dumps(result.get("dashboard_summary", {}), indent=2))
