import json
import logging
import os
from typing import Dict, Any, Optional, List

class SepsisSafetyGuardrail:
    """
    Final validation layer to prevent AI hallucinations and 
    ensure Sepsis-3 clinical compliance before alerting.
    
    All thresholds are configurable via genai_clinical_guardrail.json
    Version 2.0 - Extended with comprehensive critical thresholds
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the guardrail with configurable thresholds.
        
        Args:
            config_path: Path to genai_clinical_guardrail.json. 
                         If None, searches in standard locations.
        """
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self._parse_thresholds()
        
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        search_paths = [
            config_path,
            "genai_clinical_guardrail.json",
            os.path.join(os.path.dirname(__file__), "genai_clinical_guardrail.json"),
            "/app/genai_clinical_guardrail.json"  # Docker path
        ]
        
        for path in search_paths:
            if path and os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        config = json.load(f)
                        self.logger.info(f"Loaded guardrail config from: {path}")
                        return config
                except Exception as e:
                    self.logger.warning(f"Failed to load config from {path}: {e}")
        
        self.logger.warning("No guardrail config found, using built-in defaults")
        return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration if JSON file not found."""
        return {
            "critical_thresholds": {
                "hemodynamic": {
                    "sbp_critical": {"value": 90, "condition": "<="},
                    "map_critical": {"value": 65, "condition": "<"},
                    "dbp_critical": {"value": 40, "condition": "<"}
                },
                "perfusion_markers": {
                    "lactate_elevated": {"value": 2.0, "condition": ">="},
                    "lactate_severe": {"value": 4.0, "condition": ">="}
                },
                "respiratory": {
                    "o2sat_critical": {"value": 88, "condition": "<"},
                    "resp_rate_critical": {"value": 30, "condition": ">="}
                },
                "temperature": {
                    "temp_hypothermia": {"value": 35.0, "condition": "<"},
                    "temp_severe_fever": {"value": 40.0, "condition": ">="}
                },
                "renal": {
                    "creatinine_critical": {"value": 3.5, "condition": ">="}
                },
                "hepatic": {
                    "bilirubin_critical": {"value": 4.0, "condition": ">="}
                },
                "hematologic": {
                    "platelets_critical": {"value": 50, "condition": "<"}
                },
                "metabolic": {
                    "ph_critical_low": {"value": 7.25, "condition": "<"}
                },
                "cardiac": {
                    "hr_critical_high": {"value": 140, "condition": ">="},
                    "hr_critical_low": {"value": 40, "condition": "<"}
                }
            },
            "override_logic": {
                "trigger_conditions": {"minimum_risk_for_critical": 80},
                "override_values": {
                    "forced_risk_score": 95,
                    "forced_priority": "Critical",
                    "forced_probability_6h": "High"
                }
            },
            "discordance_rules": {
                "enabled": True,
                "concerning_phrases": ["mottled skin", "altered mental status", "AMS"],
                "escalation_risk_score": 70,
                "escalation_priority": "High"
            }
        }
    
    def _parse_thresholds(self):
        """Parse thresholds from config into easy-to-use attributes."""
        thresholds = self.config.get("critical_thresholds", {})
        
        # ===== HEMODYNAMIC =====
        hemo = thresholds.get("hemodynamic", {})
        self.sbp_threshold = hemo.get("sbp_critical", {}).get("value", 90)
        self.map_threshold = hemo.get("map_critical", {}).get("value", 65)
        self.dbp_threshold = hemo.get("dbp_critical", {}).get("value", 40)
        
        # ===== PERFUSION MARKERS =====
        perf = thresholds.get("perfusion_markers", {})
        self.lactate_threshold = perf.get("lactate_elevated", {}).get("value", 2.0)
        self.lactate_severe_threshold = perf.get("lactate_severe", {}).get("value", 4.0)
        self.base_excess_threshold = perf.get("base_excess_critical", {}).get("value", -10)
        
        # ===== RESPIRATORY =====
        resp = thresholds.get("respiratory", {})
        self.o2sat_threshold = resp.get("o2sat_critical", {}).get("value", 88)
        self.resp_rate_threshold = resp.get("resp_rate_critical", {}).get("value", 30)
        self.pf_ratio_threshold = resp.get("pao2_fio2_ratio_critical", {}).get("value", 200)
        self.paco2_threshold = resp.get("paco2_critical_high", {}).get("value", 60)
        
        # ===== TEMPERATURE =====
        temp = thresholds.get("temperature", {})
        self.temp_hypothermia = temp.get("temp_hypothermia", {}).get("value", 35.0)
        self.temp_severe_fever = temp.get("temp_severe_fever", {}).get("value", 40.0)
        
        # ===== RENAL =====
        renal = thresholds.get("renal", {})
        self.creatinine_threshold = renal.get("creatinine_critical", {}).get("value", 3.5)
        self.bun_threshold = renal.get("bun_critical", {}).get("value", 80)
        self.urine_output_threshold = renal.get("urine_output_critical", {}).get("value", 0.3)
        
        # ===== HEPATIC =====
        hepatic = thresholds.get("hepatic", {})
        self.bilirubin_threshold = hepatic.get("bilirubin_critical", {}).get("value", 4.0)
        self.ast_threshold = hepatic.get("ast_critical", {}).get("value", 1000)
        self.alt_threshold = hepatic.get("alt_critical", {}).get("value", 1000)
        self.inr_threshold = hepatic.get("inr_critical", {}).get("value", 2.5)
        
        # ===== HEMATOLOGIC =====
        hema = thresholds.get("hematologic", {})
        self.platelets_threshold = hema.get("platelets_critical", {}).get("value", 50)
        self.wbc_high_threshold = hema.get("wbc_critical_high", {}).get("value", 30)
        self.wbc_low_threshold = hema.get("wbc_critical_low", {}).get("value", 2)
        self.hemoglobin_threshold = hema.get("hemoglobin_critical", {}).get("value", 7)
        self.ptt_threshold = hema.get("ptt_critical", {}).get("value", 80)
        self.fibrinogen_threshold = hema.get("fibrinogen_critical", {}).get("value", 100)
        self.d_dimer_threshold = hema.get("d_dimer_critical", {}).get("value", 10)
        
        # ===== METABOLIC =====
        metab = thresholds.get("metabolic", {})
        self.ph_low_threshold = metab.get("ph_critical_low", {}).get("value", 7.25)
        self.ph_high_threshold = metab.get("ph_critical_high", {}).get("value", 7.55)
        self.glucose_low_threshold = metab.get("glucose_critical_low", {}).get("value", 50)
        self.glucose_high_threshold = metab.get("glucose_critical_high", {}).get("value", 500)
        self.potassium_low_threshold = metab.get("potassium_critical_low", {}).get("value", 2.5)
        self.potassium_high_threshold = metab.get("potassium_critical_high", {}).get("value", 6.5)
        self.sodium_low_threshold = metab.get("sodium_critical_low", {}).get("value", 120)
        self.sodium_high_threshold = metab.get("sodium_critical_high", {}).get("value", 160)
        self.bicarbonate_threshold = metab.get("bicarbonate_critical_low", {}).get("value", 12)
        
        # ===== CARDIAC =====
        cardiac = thresholds.get("cardiac", {})
        self.hr_high_threshold = cardiac.get("hr_critical_high", {}).get("value", 140)
        self.hr_low_threshold = cardiac.get("hr_critical_low", {}).get("value", 40)
        self.troponin_threshold = cardiac.get("troponin_critical", {}).get("value", 1.0)
        self.bnp_threshold = cardiac.get("bnp_critical", {}).get("value", 1000)
        
        # ===== NEUROLOGIC =====
        neuro = thresholds.get("neurologic", {})
        self.gcs_threshold = neuro.get("gcs_critical", {}).get("value", 8)
        
        # ===== INFECTION MARKERS =====
        infection = thresholds.get("infection_markers", {})
        self.procalcitonin_threshold = infection.get("procalcitonin_critical", {}).get("value", 10)
        self.crp_threshold = infection.get("crp_critical", {}).get("value", 200)
        
        # ===== OVERRIDE LOGIC =====
        override = self.config.get("override_logic", {})
        trigger = override.get("trigger_conditions", {})
        self.min_risk_for_critical = trigger.get("minimum_risk_for_critical", 80)
        
        override_vals = override.get("override_values", {})
        self.forced_risk_score = override_vals.get("forced_risk_score", 95)
        self.forced_priority = override_vals.get("forced_priority", "Critical")
        self.forced_probability = override_vals.get("forced_probability_6h", "High")
        
        # ===== DISCORDANCE RULES =====
        discordance = self.config.get("discordance_rules", {})
        self.discordance_enabled = discordance.get("enabled", True)
        self.concerning_phrases = discordance.get("concerning_phrases", [])
        self.escalation_risk_score = discordance.get("escalation_risk_score", 70)
        self.escalation_priority = discordance.get("escalation_priority", "High")
        
        self.logger.info(f"Guardrail v2.0 initialized with {self._count_active_thresholds()} critical thresholds")

    def _count_active_thresholds(self) -> int:
        """Count the number of active thresholds for logging."""
        return len(self.config.get("critical_thresholds", {}))

    def reload_config(self, config_path: Optional[str] = None):
        """Hot-reload configuration without restarting."""
        self.config = self._load_config(config_path)
        self._parse_thresholds()
        self.logger.info("Guardrail configuration reloaded")

    def _get_vital(self, vitals: Dict, *keys) -> Optional[float]:
        """Helper to get vital value with multiple possible keys."""
        for key in keys:
            val = vitals.get(key)
            if val is not None:
                return val
        return None

    def validate_prediction(self, llm_output_json, raw_vitals: Dict[str, Any], 
                           nursing_notes: str = "") -> Dict[str, Any]:
        """
        Cross-references LLM risk scores with deterministic medical rules.
        
        Args:
            llm_output_json: LLM prediction output (dict or JSON string)
            raw_vitals: Dictionary of vital signs
            nursing_notes: Optional nursing notes for discordance detection
            
        Returns:
            Validated prediction with override flags if triggered
        """
        try:
            if isinstance(llm_output_json, str):
                prediction = json.loads(llm_output_json)
            else:
                prediction = llm_output_json.copy() if isinstance(llm_output_json, dict) else {}
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse LLM Output: {e}")
            return {"status": "error", "message": "Invalid LLM JSON format"}

        pred_data = prediction.get("prediction", {})
        risk_score = pred_data.get("risk_score_0_100", 0)
        overrides = []
        
        # =====================================================
        # HEMODYNAMIC CHECKS
        # =====================================================
        sbp = self._get_vital(raw_vitals, "SBP", "sbp", "systolic_bp")
        if sbp is not None and sbp <= self.sbp_threshold:
            overrides.append(f"Critical Hypotension (SBP {sbp} <= {self.sbp_threshold} mmHg)")
            
        map_val = self._get_vital(raw_vitals, "MAP", "map", "mean_arterial_pressure")
        if map_val is not None and map_val < self.map_threshold:
            overrides.append(f"Critical MAP ({map_val} < {self.map_threshold} mmHg)")
            
        dbp = self._get_vital(raw_vitals, "DBP", "dbp", "diastolic_bp")
        if dbp is not None and dbp < self.dbp_threshold:
            overrides.append(f"Critical DBP ({dbp} < {self.dbp_threshold} mmHg)")
        
        # =====================================================
        # PERFUSION MARKERS
        # =====================================================
        lactate = self._get_vital(raw_vitals, "Lactate", "lactate")
        if lactate is not None:
            if lactate >= self.lactate_severe_threshold:
                overrides.append(f"Severe Lactate ({lactate} >= {self.lactate_severe_threshold} mmol/L)")
            elif lactate >= self.lactate_threshold:
                overrides.append(f"Elevated Lactate ({lactate} >= {self.lactate_threshold} mmol/L)")
        
        base_excess = self._get_vital(raw_vitals, "BaseExcess", "base_excess", "BE")
        if base_excess is not None and base_excess <= self.base_excess_threshold:
            overrides.append(f"Severe Base Deficit (BE {base_excess} <= {self.base_excess_threshold} mEq/L)")
        
        # =====================================================
        # RESPIRATORY CHECKS
        # =====================================================
        o2sat = self._get_vital(raw_vitals, "O2Sat", "SaO2", "SpO2", "o2_saturation")
        if o2sat is not None and o2sat < self.o2sat_threshold:
            overrides.append(f"Critical Hypoxemia (O2Sat {o2sat} < {self.o2sat_threshold}%)")
            
        resp = self._get_vital(raw_vitals, "Resp", "resp_rate", "RR", "respiratory_rate")
        if resp is not None and resp >= self.resp_rate_threshold:
            overrides.append(f"Critical Tachypnea (RR {resp} >= {self.resp_rate_threshold}/min)")
        
        pf_ratio = self._get_vital(raw_vitals, "PaO2_FiO2", "pf_ratio", "P_F_ratio")
        if pf_ratio is not None and pf_ratio < self.pf_ratio_threshold:
            overrides.append(f"Moderate ARDS (P/F {pf_ratio} < {self.pf_ratio_threshold})")
        
        paco2 = self._get_vital(raw_vitals, "PaCO2", "paco2")
        if paco2 is not None and paco2 >= self.paco2_threshold:
            overrides.append(f"Severe Hypercapnia (PaCO2 {paco2} >= {self.paco2_threshold} mmHg)")
            
        # =====================================================
        # TEMPERATURE CHECKS
        # =====================================================
        temp = self._get_vital(raw_vitals, "Temp", "temperature", "Temperature")
        if temp is not None:
            if temp < self.temp_hypothermia:
                overrides.append(f"Critical Hypothermia ({temp} < {self.temp_hypothermia}°C)")
            elif temp >= self.temp_severe_fever:
                overrides.append(f"Severe Hyperthermia ({temp} >= {self.temp_severe_fever}°C)")
        
        # =====================================================
        # RENAL CHECKS
        # =====================================================
        creatinine = self._get_vital(raw_vitals, "Creatinine", "creatinine")
        if creatinine is not None and creatinine >= self.creatinine_threshold:
            overrides.append(f"Critical AKI (Creatinine {creatinine} >= {self.creatinine_threshold} mg/dL)")
        
        bun = self._get_vital(raw_vitals, "BUN", "bun", "blood_urea_nitrogen")
        if bun is not None and bun >= self.bun_threshold:
            overrides.append(f"Critical BUN ({bun} >= {self.bun_threshold} mg/dL)")
        
        urine_output = self._get_vital(raw_vitals, "UrineOutput", "urine_output", "UO")
        if urine_output is not None and urine_output < self.urine_output_threshold:
            overrides.append(f"Severe Oliguria (UO {urine_output} < {self.urine_output_threshold} mL/kg/hr)")
        
        # =====================================================
        # HEPATIC CHECKS
        # =====================================================
        bilirubin = self._get_vital(raw_vitals, "Bilirubin_total", "Bilirubin_direct", "bilirubin", "TotalBilirubin")
        if bilirubin is not None and bilirubin >= self.bilirubin_threshold:
            overrides.append(f"Critical Liver Dysfunction (Bilirubin {bilirubin} >= {self.bilirubin_threshold} mg/dL)")
        
        ast = self._get_vital(raw_vitals, "AST", "ast", "SGOT")
        if ast is not None and ast >= self.ast_threshold:
            overrides.append(f"Shock Liver (AST {ast} >= {self.ast_threshold} U/L)")
        
        alt = self._get_vital(raw_vitals, "ALT", "alt", "SGPT")
        if alt is not None and alt >= self.alt_threshold:
            overrides.append(f"Severe Hepatic Injury (ALT {alt} >= {self.alt_threshold} U/L)")
        
        inr = self._get_vital(raw_vitals, "INR", "inr")
        if inr is not None and inr >= self.inr_threshold:
            overrides.append(f"Severe Coagulopathy (INR {inr} >= {self.inr_threshold})")
        
        # =====================================================
        # HEMATOLOGIC CHECKS
        # =====================================================
        platelets = self._get_vital(raw_vitals, "Platelets", "platelets", "PLT")
        if platelets is not None and platelets < self.platelets_threshold:
            overrides.append(f"Critical Thrombocytopenia (Platelets {platelets} < {self.platelets_threshold} K)")
        
        wbc = self._get_vital(raw_vitals, "WBC", "wbc", "white_blood_cells")
        if wbc is not None:
            if wbc >= self.wbc_high_threshold:
                overrides.append(f"Severe Leukocytosis (WBC {wbc} >= {self.wbc_high_threshold} K)")
            elif wbc < self.wbc_low_threshold:
                overrides.append(f"Severe Leukopenia (WBC {wbc} < {self.wbc_low_threshold} K)")
        
        hemoglobin = self._get_vital(raw_vitals, "Hgb", "hemoglobin", "Hemoglobin", "HGB")
        if hemoglobin is not None and hemoglobin < self.hemoglobin_threshold:
            overrides.append(f"Critical Anemia (Hgb {hemoglobin} < {self.hemoglobin_threshold} g/dL)")
        
        ptt = self._get_vital(raw_vitals, "PTT", "ptt", "aPTT")
        if ptt is not None and ptt >= self.ptt_threshold:
            overrides.append(f"Severe PTT Prolongation ({ptt} >= {self.ptt_threshold} sec)")
        
        fibrinogen = self._get_vital(raw_vitals, "Fibrinogen", "fibrinogen")
        if fibrinogen is not None and fibrinogen < self.fibrinogen_threshold:
            overrides.append(f"Critical Fibrinogen ({fibrinogen} < {self.fibrinogen_threshold} mg/dL - DIC?)")
        
        d_dimer = self._get_vital(raw_vitals, "D_Dimer", "d_dimer", "DDimer")
        if d_dimer is not None and d_dimer >= self.d_dimer_threshold:
            overrides.append(f"Markedly Elevated D-Dimer ({d_dimer} >= {self.d_dimer_threshold} µg/mL)")
        
        # =====================================================
        # METABOLIC CHECKS
        # =====================================================
        ph = self._get_vital(raw_vitals, "pH", "ph", "arterial_pH")
        if ph is not None:
            if ph < self.ph_low_threshold:
                overrides.append(f"Severe Acidosis (pH {ph} < {self.ph_low_threshold})")
            elif ph >= self.ph_high_threshold:
                overrides.append(f"Severe Alkalosis (pH {ph} >= {self.ph_high_threshold})")
        
        glucose = self._get_vital(raw_vitals, "Glucose", "glucose", "blood_glucose")
        if glucose is not None:
            if glucose < self.glucose_low_threshold:
                overrides.append(f"Critical Hypoglycemia (Glucose {glucose} < {self.glucose_low_threshold} mg/dL)")
            elif glucose >= self.glucose_high_threshold:
                overrides.append(f"Critical Hyperglycemia (Glucose {glucose} >= {self.glucose_high_threshold} mg/dL)")
        
        potassium = self._get_vital(raw_vitals, "Potassium", "potassium", "K")
        if potassium is not None:
            if potassium < self.potassium_low_threshold:
                overrides.append(f"Critical Hypokalemia (K+ {potassium} < {self.potassium_low_threshold} mEq/L)")
            elif potassium >= self.potassium_high_threshold:
                overrides.append(f"Critical Hyperkalemia (K+ {potassium} >= {self.potassium_high_threshold} mEq/L)")
        
        sodium = self._get_vital(raw_vitals, "Sodium", "sodium", "Na")
        if sodium is not None:
            if sodium < self.sodium_low_threshold:
                overrides.append(f"Critical Hyponatremia (Na+ {sodium} < {self.sodium_low_threshold} mEq/L)")
            elif sodium >= self.sodium_high_threshold:
                overrides.append(f"Critical Hypernatremia (Na+ {sodium} >= {self.sodium_high_threshold} mEq/L)")
        
        bicarbonate = self._get_vital(raw_vitals, "HCO3", "bicarbonate", "Bicarbonate")
        if bicarbonate is not None and bicarbonate < self.bicarbonate_threshold:
            overrides.append(f"Severe Bicarbonate Deficit (HCO3 {bicarbonate} < {self.bicarbonate_threshold} mEq/L)")
        
        # =====================================================
        # CARDIAC CHECKS
        # =====================================================
        hr = self._get_vital(raw_vitals, "HR", "heart_rate", "HeartRate", "pulse")
        if hr is not None:
            if hr >= self.hr_high_threshold:
                overrides.append(f"Critical Tachycardia (HR {hr} >= {self.hr_high_threshold} bpm)")
            elif hr < self.hr_low_threshold:
                overrides.append(f"Critical Bradycardia (HR {hr} < {self.hr_low_threshold} bpm)")
        
        troponin = self._get_vital(raw_vitals, "Troponin", "troponin", "TroponinI", "TroponinT")
        if troponin is not None and troponin >= self.troponin_threshold:
            overrides.append(f"Myocardial Injury (Troponin {troponin} >= {self.troponin_threshold} ng/mL)")
        
        bnp = self._get_vital(raw_vitals, "BNP", "bnp", "NT_proBNP")
        if bnp is not None and bnp >= self.bnp_threshold:
            overrides.append(f"Significant Heart Failure (BNP {bnp} >= {self.bnp_threshold} pg/mL)")
        
        # =====================================================
        # NEUROLOGIC CHECKS
        # =====================================================
        gcs = self._get_vital(raw_vitals, "GCS", "gcs", "glasgow_coma_scale")
        if gcs is not None and gcs <= self.gcs_threshold:
            overrides.append(f"Coma (GCS {gcs} <= {self.gcs_threshold})")
        
        # =====================================================
        # INFECTION MARKERS
        # =====================================================
        procalcitonin = self._get_vital(raw_vitals, "Procalcitonin", "procalcitonin", "PCT")
        if procalcitonin is not None and procalcitonin >= self.procalcitonin_threshold:
            overrides.append(f"Severe Bacterial Infection (PCT {procalcitonin} >= {self.procalcitonin_threshold} ng/mL)")
        
        crp = self._get_vital(raw_vitals, "CRP", "crp", "C_reactive_protein")
        if crp is not None and crp >= self.crp_threshold:
            overrides.append(f"Severe Inflammation (CRP {crp} >= {self.crp_threshold} mg/L)")

        # =====================================================
        # SEPTIC SHOCK DETECTION (Combined Criteria)
        # =====================================================
        hypotension = (sbp is not None and sbp <= self.sbp_threshold) or \
                      (map_val is not None and map_val < self.map_threshold)
        hyperlactatemia = lactate is not None and lactate >= self.lactate_threshold
        
        if hypotension and hyperlactatemia:
            overrides.append("Septic Shock Criteria Met (Hypotension + Hyperlactatemia)")
        
        # DIC Detection (Combined Criteria)
        dic_signs = sum([
            platelets is not None and platelets < self.platelets_threshold,
            inr is not None and inr >= self.inr_threshold,
            fibrinogen is not None and fibrinogen < self.fibrinogen_threshold,
            d_dimer is not None and d_dimer >= self.d_dimer_threshold
        ])
        if dic_signs >= 3:
            overrides.append(f"Probable DIC ({dic_signs}/4 criteria met)")

        # =====================================================
        # DISCORDANCE CHECK (Silent Sepsis Detection)
        # =====================================================
        if self.discordance_enabled and nursing_notes:
            notes_lower = nursing_notes.lower()
            detected_concerns = [
                phrase for phrase in self.concerning_phrases 
                if phrase.lower() in notes_lower
            ]
            if detected_concerns and risk_score < self.escalation_risk_score:
                prediction.setdefault("logic_gate", {})
                prediction["logic_gate"]["discordance_detected"] = True
                prediction["logic_gate"]["discordance_phrases"] = detected_concerns
                
                if not overrides:
                    pred_data["risk_score_0_100"] = max(risk_score, self.escalation_risk_score)
                    pred_data["priority"] = self.escalation_priority
                    self.logger.warning(f"Discordance escalation: {detected_concerns}")

        # =====================================================
        # APPLY OVERRIDE LOGIC
        # =====================================================
        if overrides and risk_score < self.min_risk_for_critical:
            prediction["prediction"]["risk_score_0_100"] = self.forced_risk_score
            prediction["prediction"]["priority"] = self.forced_priority
            prediction["prediction"]["sepsis_probability_6h"] = self.forced_probability
            
            prediction.setdefault("logic_gate", {})
            prediction["logic_gate"]["guardrail_override"] = True
            prediction["logic_gate"]["override_reasons"] = overrides
            prediction["logic_gate"]["original_risk_score"] = risk_score
            prediction["logic_gate"]["override_count"] = len(overrides)
            
            rationale = prediction["prediction"].get("clinical_rationale", "")
            prediction["prediction"]["clinical_rationale"] = \
                f"{rationale} [GUARDRAIL OVERRIDE: {', '.join(overrides[:3])}{'...' if len(overrides) > 3 else ''}]"
            
            self.logger.warning(f"Guardrail OVERRIDE triggered: {len(overrides)} critical findings")
        else:
            prediction.setdefault("logic_gate", {})
            prediction["logic_gate"]["guardrail_override"] = False
            prediction["logic_gate"]["override_reasons"] = []
            
            audit = self.config.get("audit_settings", {})
            near_miss_threshold = audit.get("near_miss_threshold", 75)
            if audit.get("log_near_misses", True) and self.min_risk_for_critical - 5 <= risk_score < self.min_risk_for_critical:
                self.logger.info(f"Near-miss detected: risk_score={risk_score}, potential_overrides={overrides}")

        return prediction
    
    def get_full_config(self) -> Dict[str, Any]:
        """Return the complete configuration including all clinical details."""
        return self.config.copy()
    
    def update_config(self, section: str, updates: Dict[str, Any], 
                      save_to_file: bool = True) -> Dict[str, Any]:
        """
        Update a specific section of the configuration.
        
        Args:
            section: Dot-notation path to section (e.g., "critical_thresholds.hemodynamic")
            updates: Dictionary of updates to merge into the section
            save_to_file: Whether to persist changes to the JSON file
            
        Returns:
            Dictionary with updated values
        """
        # Navigate to the target section
        keys = section.split(".")
        target = self.config
        
        # Navigate to parent of target section
        for key in keys[:-1]:
            if key not in target:
                raise ValueError(f"Section '{key}' not found in configuration")
            target = target[key]
        
        # Get the final key
        final_key = keys[-1] if keys else None
        
        if final_key:
            if final_key not in target:
                raise ValueError(f"Section '{final_key}' not found in configuration")
            
            # Deep merge updates into the target section
            self._deep_merge(target[final_key], updates)
            updated_section = target[final_key]
        else:
            # Update root level
            self._deep_merge(target, updates)
            updated_section = target
        
        # Re-parse thresholds to apply changes
        self._parse_thresholds()
        
        # Save to file if requested
        if save_to_file:
            self._save_config()
        
        self.logger.info(f"Configuration section '{section}' updated")
        
        return {"updated_values": updates}
    
    def _deep_merge(self, target: Dict, source: Dict):
        """Deep merge source dict into target dict."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = value
    
    def _save_config(self):
        """Save current configuration to the JSON file."""
        config_paths = [
            "genai_clinical_guardrail.json",
            os.path.join(os.path.dirname(__file__), "genai_clinical_guardrail.json"),
            "/app/genai_clinical_guardrail.json"
        ]
        
        for path in config_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'w') as f:
                        json.dump(self.config, f, indent=2)
                    self.logger.info(f"Configuration saved to: {path}")
                    return
                except PermissionError:
                    self.logger.warning(f"Cannot write to {path} (read-only filesystem). Use export endpoint or update source file.")
                    raise ValueError(
                        f"Cannot save to {path} - filesystem is read-only. "
                        "Changes are applied in-memory only. Use GET /guardrail/config/export "
                        "to download the updated configuration, then update the source file."
                    )
                except Exception as e:
                    self.logger.error(f"Failed to save config to {path}: {e}")
                    raise
        
        raise ValueError("Could not find configuration file to save")
    
    def export_config(self) -> str:
        """Export current configuration as JSON string for download."""
        return json.dumps(self.config, indent=2)

    def get_current_thresholds(self) -> Dict[str, Any]:
        """Return current thresholds for API exposure."""
        return {
            "hemodynamic": {
                "sbp_threshold": self.sbp_threshold,
                "map_threshold": self.map_threshold,
                "dbp_threshold": self.dbp_threshold
            },
            "perfusion": {
                "lactate_threshold": self.lactate_threshold,
                "lactate_severe_threshold": self.lactate_severe_threshold,
                "base_excess_threshold": self.base_excess_threshold
            },
            "respiratory": {
                "o2sat_threshold": self.o2sat_threshold,
                "resp_rate_threshold": self.resp_rate_threshold,
                "pf_ratio_threshold": self.pf_ratio_threshold,
                "paco2_threshold": self.paco2_threshold
            },
            "temperature": {
                "hypothermia_threshold": self.temp_hypothermia,
                "hyperthermia_threshold": self.temp_severe_fever
            },
            "renal": {
                "creatinine_threshold": self.creatinine_threshold,
                "bun_threshold": self.bun_threshold,
                "urine_output_threshold": self.urine_output_threshold
            },
            "hepatic": {
                "bilirubin_threshold": self.bilirubin_threshold,
                "ast_threshold": self.ast_threshold,
                "alt_threshold": self.alt_threshold,
                "inr_threshold": self.inr_threshold
            },
            "hematologic": {
                "platelets_threshold": self.platelets_threshold,
                "wbc_high_threshold": self.wbc_high_threshold,
                "wbc_low_threshold": self.wbc_low_threshold,
                "hemoglobin_threshold": self.hemoglobin_threshold,
                "ptt_threshold": self.ptt_threshold,
                "fibrinogen_threshold": self.fibrinogen_threshold,
                "d_dimer_threshold": self.d_dimer_threshold
            },
            "metabolic": {
                "ph_low_threshold": self.ph_low_threshold,
                "ph_high_threshold": self.ph_high_threshold,
                "glucose_low_threshold": self.glucose_low_threshold,
                "glucose_high_threshold": self.glucose_high_threshold,
                "potassium_low_threshold": self.potassium_low_threshold,
                "potassium_high_threshold": self.potassium_high_threshold,
                "sodium_low_threshold": self.sodium_low_threshold,
                "sodium_high_threshold": self.sodium_high_threshold,
                "bicarbonate_threshold": self.bicarbonate_threshold
            },
            "cardiac": {
                "hr_high_threshold": self.hr_high_threshold,
                "hr_low_threshold": self.hr_low_threshold,
                "troponin_threshold": self.troponin_threshold,
                "bnp_threshold": self.bnp_threshold
            },
            "neurologic": {
                "gcs_threshold": self.gcs_threshold
            },
            "infection_markers": {
                "procalcitonin_threshold": self.procalcitonin_threshold,
                "crp_threshold": self.crp_threshold
            },
            "override_settings": {
                "min_risk_for_critical": self.min_risk_for_critical,
                "forced_risk_score": self.forced_risk_score,
                "forced_priority": self.forced_priority
            }
        }
