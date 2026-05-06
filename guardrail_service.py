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

    # ---------- C1: Reasoning-aware guardrail suppression ----------
    # Phrases below indicate the LLM has *explicitly identified a non-sepsis
    # cause* for the patient's abnormal vitals (e.g. post-op stress, GI bleed,
    # DKA, neurologic indication, physiologic recovery). When such language is
    # present AND the LLM's own risk score is below the suppression threshold,
    # the guardrail respects the LLM verdict instead of bumping it. This stops
    # the guardrail from contradicting reasoning the LLM already articulated.
    #
    # Discovered empirically on the eICU v4 cohort (2026-02-11): 72% of false
    # positives had non-sepsis explanations in the LLM rationale that the
    # guardrail was ignoring. See validation/docs/EICU_VALIDATION_EXECUTION.md
    # §8 for the full analysis.
    _C1_DENIAL_PATTERNS = [
        r"more\s+consistent\s+with\s+(?!sepsis|infection|septic|infectious)",
        r"pattern\s+(more\s+)?consistent\s+with\s+(?!sepsis|infection|septic|infectious)",
        r"does\s+not\s+indicate\s+sepsis",
        r"\bnot\s+sepsis\b",
        r"argues?\s+against\s+(evolving\s+)?(sepsis|infection)",
        r"unlikely\s+(to\s+be\s+)?(sepsis|septic|infection)",
        r"no\s+(clear\s+)?(infection|sepsis)\s+(signal|criteria|source|features?)",
        r"physiologic\s+(recovery|response)",
        r"alternative\s+diagnos",
        r"emergence\s+from\s+(sedation|anesthesia)",
        r"surgical\s+stress\s+response",
        r"residual\s+(anesthet|sedat)",
        r"reassuring\s+(context|features?|presentation)",
        r"(typical|expected)\s+post[- ]?(operat|cardiac\s+surgery|surg|procedur)",
        r"absent\s+(infection|sepsis|septic)",
        r"non[- ]?infectious\s+(cause|aetiolog|etiolog)",
        r"likely\s+(prophylact|surgical|mechanical|cardiac|gi|trauma|withdraw)",
        r"is\s+typical\s+(post[- ]?(operat|surg|procedur))",
        r"isolated\s+finding",
    ]
    _C1_SUPPRESSION_RISK_MAX = 50  # only suppress if LLM's own risk < this
    _C1_DENIAL_REGEX = None  # lazy compile

    @classmethod
    def _c1_regex(cls):
        if cls._C1_DENIAL_REGEX is None:
            import re as _re
            cls._C1_DENIAL_REGEX = _re.compile(
                "|".join(cls._C1_DENIAL_PATTERNS), _re.IGNORECASE
            )
        return cls._C1_DENIAL_REGEX

    def _llm_denies_sepsis(self, prediction: Dict[str, Any]) -> tuple:
        """
        C1 check: does the LLM's clinical_rationale explicitly identify a
        non-sepsis cause? Returns (denies_bool, list_of_matched_phrases).
        """
        pred_data = (prediction or {}).get("prediction", {})
        text = pred_data.get("clinical_rationale", "") or ""
        regex = self._c1_regex()
        hits = sorted({m.group(0).lower() for m in regex.finditer(text)})
        return (len(hits) > 0, hits[:3])

    def _c1_should_suppress(self, prediction: Dict[str, Any], llm_risk_score: float) -> tuple:
        """
        Returns (suppress_bool, list_of_matched_phrases).
        Suppresses only if ALL of:
          1. C1 enabled (env var ENABLE_C1_SUPPRESSION, default true)
          2. LLM's own risk score < self._C1_SUPPRESSION_RISK_MAX
          3. LLM rationale contains a denial phrase
        """
        if os.getenv("ENABLE_C1_SUPPRESSION", "true").lower() != "true":
            return (False, [])
        if llm_risk_score is None or llm_risk_score >= self._C1_SUPPRESSION_RISK_MAX:
            return (False, [])
        denies, hits = self._llm_denies_sepsis(prediction)
        return (denies, hits)

    def _c1_align_priority_with_risk(self, prediction: Dict[str, Any]) -> None:
        """
        After C1 suppression we want the alert's *priority* to be consistent
        with the (now low) risk score. Otherwise an LLM that returned
        "risk=35, priority=High" still drives downstream alerting even though
        the model has explicitly explained the abnormality away. This forces
        priority back to Standard whenever we suppress, so the audit trail
        and the dashboard agree with each other.
        """
        pred_data = prediction.get("prediction", {}) or {}
        try:
            r = float(pred_data.get("risk_score_0_100", 0))
        except Exception:
            r = 0.0
        if r < 50 and pred_data.get("priority") in ("High", "Critical"):
            pred_data["original_llm_priority"] = pred_data.get("priority")
            pred_data["priority"] = "Standard"
        # And keep the 6-h prob field consistent if present
        if r < 50 and pred_data.get("sepsis_probability_6h") in ("High", "Moderate"):
            pred_data["original_llm_probability_6h"] = pred_data.get("sepsis_probability_6h")
            pred_data["sepsis_probability_6h"] = "Low"

    # ---------- C2: FP-pattern reasoning-aware suppression ----------
    # C2 extends C1 by recognising five specific FP archetypes that the v6
    # validation surfaced (eICU v4 cohort, 2026-02-11) - see
    # validation/docs/SPECIFICITY_PATH_TO_60PCT.md for the full analysis.
    #
    # Where C1 catches "the LLM literally said 'not sepsis'", C2 catches:
    #   br0  - override fired only on alkalosis (pH>=7.55) (never sepsis)
    #   br1  - qSOFA=0 + SIRS<=1 + LLM<50 + no biomarker rescue
    #   br2  - early-detection bump on patient with no biomarker rescue at all
    #   br4  - single weak override + non-infectious context + LLM<40
    #   br5  - LLM voted positive + non-infectious context + LLM<50
    #   br6  - early-detection bump + non-infectious context
    #   br7  - early-detection bump + stable/improving language + no rescue
    #
    # Each branch is gated by a "rescue signals" check that protects every
    # one of the 28 v6 true positives - simulation showed 0 sensitivity loss
    # while 27 of 70 false positives were correctly suppressed.

    # ----- C2 pattern lists -----
    # Phrases the LLM uses when it has identified a NON-INFECTIOUS aetiology
    # for the patient's vital sign abnormalities. These are concrete clinical
    # alternatives to sepsis (cardiogenic, GI bleed, post-op, drug, etc.).
    _C2_NON_INFECTIOUS_CONTEXT_PATTERNS = [
        r"acute\s+coronary\s+syndrome",
        r"cardiogenic\s+shock",
        r"acute\s+MI\b",
        r"acute\s+myocardial",
        r"cardiac\s+catheterization",
        r"cardiac\s+pathology",
        r"cardiac\s+(?:rather\s+than|not)\s+(?:septic|sepsis|infection)",
        r"GI\s+bleed",
        r"active\s+(?:GI|gastrointestinal)",
        r"gastrointestinal\s+bleed",
        r"variceal",
        r"gi\s+source",
        r"post[\s-]?operative",
        r"post[\s-]?op\b",
        r"surgical\s+stress",
        r"cardiac\s+surgery",
        r"emergence\s+from\s+(?:sedation|anesthesia)",
        r"drug\s+overdose",
        r"medication\s+overdose",
        r"toxic\s+ingestion",
        r"\bDKA\b",
        r"diabetic\s+ketoacidos",
        r"\bHHS\b",
        r"COPD\s+exacerbation",
        r"asthma\s+exacerbation",
        r"status\s+asthmaticus",
        r"hepatic\s+encephalopathy",
        r"cirrhotic",
        r"chronic\s+liver",
        r"residual\s+(?:anesthet|sedat)",
        r"transfusion\s+reaction",
        r"alcoholic\s+ketoacidosis",
        r"non[- ]?infectious",
        r"(?:rather\s+than|not)\s+(?:septic|sepsis|infection)",
        r"trauma\s+patient",
        r"post[- ]?traumatic",
        r"bone\s+marrow\s+(?:suppression|consumption)",
        r"flash\s+pulmonary\s+edema",
        r"anesthetic\s+effect",
        r"sedation\s+effect",
    ]

    # Phrases the LLM uses when it sees a recovering or stable patient
    # (the patient is improving, not decompensating into sepsis).
    _C2_STABLE_IMPROVING_PATTERNS = [
        r"stable[- ]to[- ]improving",
        r"improving\s+(?:vital|trajectory|hemodynamic|trend|mental|respiratory)",
        r"dramatic\s+(?:neurological\s+)?recovery",
        r"physiologic\s+recovery",
        r"hemodynamic\s+stabili[zs]ation",
    ]

    # Phrases that NEGATE the meaning of subsequent septic-shock language.
    # E.g. "argues against septic shock" should NOT be treated as the
    # LLM asserting septic shock.
    _C2_NEGATION_PATTERNS = [
        r"argues?\s+against",
        r"rule\s+out",
        r"rather\s+than",
        r"not\s+(?:fulminant|septic|sepsis|imminent)",
        r"absent\s+(?:fever|signs|hallmarks)",
        r"lacks?\s+(?:sepsis|infectious)",
        r"vs\.?\s+septic",
        r"differential\s+(?:between|includes)",
        r"non[- ]?(?:infectious|septic)",
        r"argues?\s+for\s+(?:cardiogenic|hemorrhagic|other)",
    ]

    # Positive assertions of septic shock by the LLM (a sepsis-protective
    # rescue signal). Negation-aware: the actual check examines preceding
    # text via _c2_has_septic_shock_assertion.
    _C2_SEPTIC_SHOCK_PATTERNS = [
        r"(?:already\s+|patient\s+is\s+|currently\s+|established\s+|in\s+)"
        r"(?:in\s+|with\s+)?(?:septic\s+shock|established\s+septic|established\s+sepsis)",
        r"distributive\s+shock",
        r"silent\s+sepsis(?:\s+pattern)?",
        r"classic\s+'?silent\s+sepsis",
        r"sepsis\s+with\s+(?:respiratory|organ|multi)",
        r"septic\s+process(?:\s+despite|\s+with)",
        r"established\s+septic\s+process",
        r"existing\s+(?:septic|shock)\s+state",
    ]

    # Lactate / GCS extraction (text-based fallback if structured value missing)
    _C2_LACTATE_RE = None  # lazy compile
    _C2_GCS_RE = None      # lazy compile
    _C2_NON_INFECT_RE = None
    _C2_STABLE_RE = None
    _C2_NEGATION_RE = None
    _C2_SEPTIC_SHOCK_RE = None
    _C2_OVERRIDE_TRIGGER_RE = None

    # Override-trigger labels considered "weak" (single non-specific finding).
    # Single-trigger overrides matching this set are candidates for Br4
    # suppression when accompanied by a non-infectious context.
    _C2_WEAK_OVERRIDE_TOKENS = (
        "Elevated Lactate", "Critical Anemia", "Severe Acidosis",
        "Severe Alkalosis", "Critical Tachypnea", "Critical Tachycardia",
        "Severe Leukocytosis", "Critical Thrombocytopenia",
    )

    # Risk thresholds (hospital-tunable in future iteration)
    _C2_BR4_LLM_RISK_MAX = 40   # br4 LLM risk ceiling (override + non-infect)
    _C2_BR5_LLM_RISK_MAX = 50   # br5 LLM risk ceiling (LLM-only + non-infect)
    _C2_BR0_LLM_RISK_MAX = 70   # br0 LLM risk ceiling (alkalosis-only override)

    @classmethod
    def _c2_compile_patterns(cls):
        if cls._C2_LACTATE_RE is None:
            import re as _re
            cls._C2_LACTATE_RE = _re.compile(
                r"lactate\s*(?:of\s*|=|:|\()?\s*(\d+\.?\d*)", _re.IGNORECASE)
            cls._C2_GCS_RE = _re.compile(
                r"GCS\s*(?:of\s*|=|:|\()?\s*(\d+)(?:\s*[-\u2014\u2013]\s*(\d+))?",
                _re.IGNORECASE)
            cls._C2_NON_INFECT_RE = _re.compile(
                "|".join(cls._C2_NON_INFECTIOUS_CONTEXT_PATTERNS), _re.IGNORECASE)
            cls._C2_STABLE_RE = _re.compile(
                "|".join(cls._C2_STABLE_IMPROVING_PATTERNS), _re.IGNORECASE)
            cls._C2_NEGATION_RE = _re.compile(
                "|".join(cls._C2_NEGATION_PATTERNS), _re.IGNORECASE)
            cls._C2_SEPTIC_SHOCK_RE = _re.compile(
                "|".join(cls._C2_SEPTIC_SHOCK_PATTERNS), _re.IGNORECASE)
            cls._C2_OVERRIDE_TRIGGER_RE = _re.compile(
                r"\[GUARDRAIL OVERRIDE:\s*([^\]]+)\]")

    # ----- C2 helpers -----
    def _c2_extract_lactate_from_text(self, rationale: str) -> Optional[float]:
        """Highest lactate value mentioned in the LLM rationale, or None.
        Ignores values < 2.5 (we treat 2-2.4 as non-specific)."""
        if not rationale:
            return None
        self._c2_compile_patterns()
        highest = None
        for m in self._C2_LACTATE_RE.finditer(rationale):
            try:
                v = float(m.group(1))
                if v >= 2.5 and (highest is None or v > highest):
                    highest = v
            except Exception:
                continue
        return highest

    def _c2_extract_gcs_from_text(self, rationale: str) -> Optional[int]:
        """Lowest GCS value mentioned in the LLM rationale.
        Returns the value if < 10 (severe altered mentation), else None."""
        if not rationale:
            return None
        self._c2_compile_patterns()
        for m in self._C2_GCS_RE.finditer(rationale):
            try:
                lo = int(m.group(1))
                if lo < 10:
                    return lo
            except Exception:
                continue
        return None

    def _c2_has_non_infectious_context(self, rationale: str) -> bool:
        if not rationale:
            return False
        self._c2_compile_patterns()
        return bool(self._C2_NON_INFECT_RE.search(rationale))

    def _c2_has_stable_improving(self, rationale: str) -> bool:
        if not rationale:
            return False
        self._c2_compile_patterns()
        return bool(self._C2_STABLE_RE.search(rationale))

    def _c2_has_septic_shock_assertion(self, rationale: str) -> bool:
        """True only if a septic-shock phrase appears WITHOUT a negation
        in the 60 characters preceding it."""
        if not rationale:
            return False
        self._c2_compile_patterns()
        for m in self._C2_SEPTIC_SHOCK_RE.finditer(rationale):
            start = max(0, m.start() - 60)
            preceding = rationale[start:m.start()]
            if self._C2_NEGATION_RE.search(preceding):
                continue  # negated, not a real assertion
            return True
        return False

    def _c2_get_override_triggers(self, rationale: str) -> List[str]:
        """Parse the '[GUARDRAIL OVERRIDE: ...]' tail of a rationale into
        a list of trigger labels."""
        if not rationale:
            return []
        self._c2_compile_patterns()
        m = self._C2_OVERRIDE_TRIGGER_RE.search(rationale)
        if not m:
            return []
        return [t.strip() for t in m.group(1).split(",") if t.strip()]

    def _c2_is_single_weak_override(self, triggers: List[str]) -> bool:
        if len(triggers) != 1:
            return False
        return any(tok in triggers[0] for tok in self._C2_WEAK_OVERRIDE_TOKENS)

    @staticmethod
    def _c2_lab_extreme(vitals: Dict[str, Any], key: str, agg: str = "max") -> Optional[float]:
        """Return max/min of a vital that may be a list of {val, ts} dicts,
        a list of scalars, or a single scalar. None if absent/empty."""
        v = vitals.get(key)
        if v is None:
            return None
        nums: List[float] = []
        if isinstance(v, list):
            for x in v:
                if isinstance(x, dict) and x.get("val") is not None:
                    try:
                        nums.append(float(x["val"]))
                    except Exception:
                        continue
                elif x not in (None, "", []):
                    try:
                        nums.append(float(x))
                    except Exception:
                        continue
        else:
            try:
                nums.append(float(v))
            except Exception:
                return None
        if not nums:
            return None
        return max(nums) if agg == "max" else min(nums)

    def _c2_has_strong_rescue(self, prediction: Dict[str, Any],
                               raw_vitals: Dict[str, Any],
                               clinical_scores: Optional[Dict[str, Any]]) -> tuple:
        """Sepsis-specific rescue signals - if any fires, C2 will not suppress.
        Returns (has_strong_rescue, label_str)."""
        rationale = (prediction.get("prediction", {}) or {}).get("clinical_rationale", "") or ""

        lact_text = self._c2_extract_lactate_from_text(rationale)
        if lact_text is not None:
            return True, f"lactate-text>=2.5 ({lact_text})"

        gcs_text = self._c2_extract_gcs_from_text(rationale)
        if gcs_text is not None:
            return True, f"GCS<10 ({gcs_text})"

        if self._c2_has_septic_shock_assertion(rationale):
            return True, "septic-shock-asserted"

        lact_struct = self._c2_lab_extreme(raw_vitals, "Lactate", "max")
        if lact_struct is None:
            lact_struct = self._c2_lab_extreme(raw_vitals, "lactate", "max")
        if lact_struct is not None and lact_struct >= 2.5:
            return True, f"Lact>=2.5 ({lact_struct})"

        if clinical_scores:
            qsofa = (clinical_scores.get("qsofa") or {}).get("score")
            sofa = (clinical_scores.get("sofa") or {}).get("score")
            try:
                if qsofa is not None and int(qsofa) >= 2:
                    return True, f"qSOFA>=2 ({qsofa})"
            except Exception:
                pass
            try:
                if sofa is not None and int(sofa) >= 4:
                    return True, f"SOFA>=4 ({sofa})"
            except Exception:
                pass

        return False, ""

    def _c2_find_rescue_signals(self, prediction: Dict[str, Any],
                                 raw_vitals: Dict[str, Any],
                                 clinical_scores: Optional[Dict[str, Any]]) -> List[str]:
        """All rescue signals (strong + soft). Used by branches that require
        any rescue absence (br1, br2, br7)."""
        sigs: List[str] = []
        strong, msg = self._c2_has_strong_rescue(prediction, raw_vitals, clinical_scores)
        if strong:
            sigs.append(msg)
        wbc_max = self._c2_lab_extreme(raw_vitals, "WBC", "max") or self._c2_lab_extreme(raw_vitals, "wbc", "max")
        wbc_min = self._c2_lab_extreme(raw_vitals, "WBC", "min") or self._c2_lab_extreme(raw_vitals, "wbc", "min")
        if wbc_max is not None and wbc_max > 15:
            sigs.append(f"WBC>15 ({wbc_max})")
        if wbc_min is not None and wbc_min < 4:
            sigs.append(f"WBC<4 ({wbc_min})")
        cr_max = self._c2_lab_extreme(raw_vitals, "Creatinine", "max") or self._c2_lab_extreme(raw_vitals, "creatinine", "max")
        if cr_max is not None and cr_max >= 3.0:
            sigs.append(f"Cr>=3 ({cr_max})")
        map_min = self._c2_lab_extreme(raw_vitals, "MAP", "min") or self._c2_lab_extreme(raw_vitals, "map", "min")
        sbp_min = self._c2_lab_extreme(raw_vitals, "SBP", "min") or self._c2_lab_extreme(raw_vitals, "sbp", "min")
        if map_min is not None and map_min < 65:
            sigs.append(f"MAP<65 ({map_min})")
        if sbp_min is not None and sbp_min < 90:
            sigs.append(f"SBP<90 ({sbp_min})")
        hco3_min = self._c2_lab_extreme(raw_vitals, "HCO3", "min") or self._c2_lab_extreme(raw_vitals, "bicarbonate", "min")
        if hco3_min is not None and hco3_min < 20:
            sigs.append(f"HCO3<20 ({hco3_min})")
        return sigs

    def _c2_should_suppress(self, prediction: Dict[str, Any],
                              raw_vitals: Dict[str, Any],
                              llm_original_risk: float,
                              early_detection_fired: bool,
                              override_fired: bool,
                              override_triggers: List[str],
                              clinical_scores: Optional[Dict[str, Any]]) -> tuple:
        """C2 main decision. Returns (should_suppress, audit_dict).
        audit_dict has keys: branch, reason, rescues_checked, llm_risk."""
        if os.getenv("ENABLE_C2_SUPPRESSION", "false").lower() != "true":
            return (False, {})

        pred_data = prediction.get("prediction", {}) or {}
        rationale = pred_data.get("clinical_rationale", "") or ""

        try:
            llm_risk = float(llm_original_risk) if llm_original_risk is not None else None
        except Exception:
            llm_risk = None
        if llm_risk is None:
            return (False, {})

        # Pull qSOFA / SIRS from clinical_scores (deterministic guardrail computation)
        qs = 0
        ss = 0
        if clinical_scores:
            try:
                qs = int((clinical_scores.get("qsofa") or {}).get("score") or 0)
            except Exception:
                qs = 0
            try:
                ss = int((clinical_scores.get("sirs") or {}).get("criteria_met") or 0)
            except Exception:
                ss = 0

        rescues = self._c2_find_rescue_signals(prediction, raw_vitals, clinical_scores)
        strong, _ = self._c2_has_strong_rescue(prediction, raw_vitals, clinical_scores)
        has_non_infect = self._c2_has_non_infectious_context(rationale)
        has_stable = self._c2_has_stable_improving(rationale)
        is_llm_only = (not override_fired) and (not early_detection_fired)

        audit_base = {
            "llm_risk": llm_risk,
            "qsofa": qs, "sirs_met": ss,
            "rescues_found": rescues,
            "non_infect_context": has_non_infect,
            "stable_improving": has_stable,
            "override_triggers": override_triggers,
        }

        # Br0 - alkalosis-only override is never a sepsis criterion
        if (override_fired and override_triggers
                and all("Alkalosis" in t for t in override_triggers)
                and llm_risk < self._C2_BR0_LLM_RISK_MAX):
            return (True, {**audit_base, "branch": "br0",
                           "reason": "alkalosis-only override + LLM<70"})

        # Br4 - single weak override + non-infectious context + LLM<40
        if (override_fired and self._c2_is_single_weak_override(override_triggers)
                and has_non_infect and llm_risk < self._C2_BR4_LLM_RISK_MAX
                and not strong):
            return (True, {**audit_base, "branch": "br4",
                           "reason": "single-weak-override + non-infect + LLM<40"})

        # Br5 - LLM voted positive on its own + non-infectious context + LLM<50
        if (is_llm_only and has_non_infect
                and llm_risk < self._C2_BR5_LLM_RISK_MAX
                and not strong and qs <= 1 and ss <= 2):
            return (True, {**audit_base, "branch": "br5",
                           "reason": "LLM-only + non-infect + LLM<50"})

        # Br6 - early-detection bump + non-infectious context
        if (early_detection_fired and has_non_infect
                and not strong and qs <= 1):
            return (True, {**audit_base, "branch": "br6",
                           "reason": "early-detection + non-infect"})

        # Br7 - early-detection bump + stable/improving + no rescue at all
        if (early_detection_fired and has_stable
                and not rescues and qs <= 1):
            return (True, {**audit_base, "branch": "br7",
                           "reason": "early-detection + stable/improving + no rescue"})

        # Br1 - formal criteria negative + LLM<50 + no rescue
        if (qs == 0 and ss <= 1 and llm_risk < 50 and not rescues):
            return (True, {**audit_base, "branch": "br1",
                           "reason": "qSOFA=0, SIRS<=1, LLM<50, no rescue"})

        # Br2 - early-detection bump + mild formal criteria + no rescue
        if (early_detection_fired and qs <= 1 and ss <= 2 and not rescues):
            return (True, {**audit_base, "branch": "br2",
                           "reason": "early-detection bump, no rescue"})

        return (False, {})

    def _c2_apply_suppression(self, prediction: Dict[str, Any],
                                llm_original_risk: float,
                                audit: Dict[str, Any]) -> None:
        """Restore the LLM's original verdict and force priority/probability
        to Standard/Low. Records branch + reason in logic_gate."""
        pred_data = prediction.setdefault("prediction", {})
        try:
            r = float(llm_original_risk) if llm_original_risk is not None else None
        except Exception:
            r = None

        # Restore LLM's original risk score (undo any guardrail bump)
        if r is not None:
            current = pred_data.get("risk_score_0_100")
            if current is None or float(current) > r:
                pred_data["risk_score_0_100"] = r

        # Force priority + probability consistent with low risk
        if pred_data.get("priority") in ("High", "Critical"):
            pred_data["original_llm_priority"] = pred_data.get("priority")
        pred_data["priority"] = "Standard"
        if pred_data.get("sepsis_probability_6h") in ("High", "Moderate"):
            pred_data["original_llm_probability_6h"] = pred_data.get("sepsis_probability_6h")
        pred_data["sepsis_probability_6h"] = "Low"

        # Undo override flag (since C2 is overruling it)
        lg = prediction.setdefault("logic_gate", {})
        lg["guardrail_override"] = False
        lg["c2_suppression_applied"] = True
        lg["c2_branch"] = audit.get("branch")
        lg["c2_reason"] = audit.get("reason")
        lg["c2_llm_risk"] = audit.get("llm_risk")
        lg["c2_qsofa"] = audit.get("qsofa")
        lg["c2_sirs_met"] = audit.get("sirs_met")
        lg["c2_rescues_checked"] = audit.get("rescues_found", [])
        lg["c2_non_infect_context"] = audit.get("non_infect_context", False)
        lg["c2_stable_improving"] = audit.get("stable_improving", False)

    def validate_prediction(self, llm_output_json, raw_vitals: Dict[str, Any],
                           nursing_notes: str = "",
                           patient_history: Optional[Dict[str, Any]] = None,
                           raw_vitals_timeseries: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Cross-references LLM risk scores with deterministic medical rules.

        Args:
            llm_output_json: LLM prediction output (dict or JSON string)
            raw_vitals: Dictionary of vital signs (typically already flattened
                        to most-recent values per metric)
            nursing_notes: Optional nursing notes for discordance detection
            patient_history: Optional dict with 'conditions' (list of str) and 'medications' (list of str)
            raw_vitals_timeseries: Optional dictionary of vital signs in the
                        original time-series format (list of {val, ts}). Used
                        by C2 to look at min/max over the snapshot window
                        (e.g. MAP nadir, WBC peak). When omitted, C2 falls
                        back to raw_vitals.

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

        # Snapshot the LLM's true initial risk + priority before any guardrail
        # action mutates them. The simulator and the audit log both rely on
        # this to recover what the LLM said vs. what the guardrails did.
        prediction.setdefault("logic_gate", {})
        if "llm_initial_risk_score" not in prediction["logic_gate"]:
            try:
                prediction["logic_gate"]["llm_initial_risk_score"] = float(risk_score)
            except Exception:
                prediction["logic_gate"]["llm_initial_risk_score"] = risk_score
        if "llm_initial_priority" not in prediction["logic_gate"]:
            prediction["logic_gate"]["llm_initial_priority"] = pred_data.get("priority")

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
        # EARLY DETECTION PATTERNS (Combination Checks)
        # =====================================================
        early_warnings = self._check_early_detection_patterns(raw_vitals)
        
        # =====================================================
        # HISTORY-AWARE CONTEXT CHECKS
        # =====================================================
        context_flags = []
        if patient_history:
            overrides, context_flags = self._check_history_context(
                raw_vitals, patient_history, overrides
            )

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
        # EARLY DETECTION ESCALATION  (with C1 reasoning-aware suppression)
        # =====================================================
        if early_warnings and not overrides and risk_score < self.escalation_risk_score:
            c1_suppress, c1_hits = self._c1_should_suppress(prediction, risk_score)
            if c1_suppress:
                # LLM explicitly identified a non-sepsis cause AND its own risk
                # score is below the suppression threshold; respect that verdict.
                prediction.setdefault("logic_gate", {})
                prediction["logic_gate"]["c1_suppression_applied"] = True
                prediction["logic_gate"]["c1_suppression_path"] = "early_detection"
                prediction["logic_gate"]["c1_suppression_phrases"] = c1_hits
                prediction["logic_gate"]["c1_original_risk_score"] = risk_score
                prediction["logic_gate"]["c1_would_have_been_bumped_to"] = self.escalation_risk_score
                self._c1_align_priority_with_risk(prediction)
                self.logger.info(
                    f"C1 suppression (early-detection path): risk={risk_score} "
                    f"hits={c1_hits} warnings={early_warnings}"
                )
            else:
                pred_data["risk_score_0_100"] = max(risk_score, self.escalation_risk_score)
                pred_data["priority"] = self.escalation_priority
                self.logger.warning(f"Early detection escalation: {early_warnings}")

        # =====================================================
        # APPLY OVERRIDE LOGIC  (with C1 reasoning-aware suppression)
        # =====================================================
        if overrides and risk_score < self.min_risk_for_critical:
            c1_suppress, c1_hits = self._c1_should_suppress(prediction, risk_score)
            if c1_suppress:
                # LLM has explicitly explained the abnormal vitals as non-sepsis;
                # respect that and skip the forced override.
                prediction.setdefault("logic_gate", {})
                prediction["logic_gate"]["guardrail_override"] = False
                prediction["logic_gate"]["c1_suppression_applied"] = True
                prediction["logic_gate"]["c1_suppression_path"] = "override_logic"
                prediction["logic_gate"]["c1_suppression_phrases"] = c1_hits
                prediction["logic_gate"]["c1_original_risk_score"] = risk_score
                prediction["logic_gate"]["c1_suppressed_overrides"] = overrides
                prediction["logic_gate"]["override_reasons"] = []
                self._c1_align_priority_with_risk(prediction)
                self.logger.info(
                    f"C1 suppression (override path): risk={risk_score} "
                    f"hits={c1_hits} suppressed_overrides={overrides[:3]}"
                )
            else:
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

        # Add early warnings and history context to output
        prediction["logic_gate"]["early_warnings"] = early_warnings
        prediction["logic_gate"]["history_context"] = context_flags

        # =====================================================
        # C2 - Reasoning-aware FP-pattern suppression
        # Runs AFTER override/early-detection logic. May undo a guardrail
        # bump or LLM-driven priority="High" if the LLM rationale clearly
        # describes a non-sepsis aetiology / stable trajectory and no
        # sepsis-specific rescue signals are present.
        # =====================================================
        if os.getenv("ENABLE_C2_SUPPRESSION", "false").lower() == "true":
            try:
                # Compute deterministic clinical scores once for C2 to consume.
                clinical_scores_for_c2 = None
                try:
                    clinical_scores_for_c2 = self.calculate_clinical_scores(raw_vitals)
                except Exception as cs_err:
                    self.logger.warning(f"C2: clinical_scores unavailable ({cs_err})")

                # Recover the override-trigger labels: prefer the audit list,
                # fall back to parsing the [GUARDRAIL OVERRIDE: ...] tail.
                override_triggers = list(prediction.get("logic_gate", {}).get("override_reasons", []) or [])
                if not override_triggers:
                    override_triggers = self._c2_get_override_triggers(
                        (prediction.get("prediction", {}) or {}).get("clinical_rationale", "") or ""
                    )

                early_detection_fired = bool(early_warnings)
                override_fired = bool(prediction.get("logic_gate", {}).get("guardrail_override"))

                # C2 prefers the time-series vitals so it can compute MAP-nadir,
                # WBC-peak, etc. Falls back to flattened raw_vitals if absent.
                vitals_for_c2 = raw_vitals_timeseries if raw_vitals_timeseries else raw_vitals

                c2_suppress, c2_audit = self._c2_should_suppress(
                    prediction=prediction,
                    raw_vitals=vitals_for_c2,
                    llm_original_risk=risk_score,
                    early_detection_fired=early_detection_fired,
                    override_fired=override_fired,
                    override_triggers=override_triggers,
                    clinical_scores=clinical_scores_for_c2,
                )
                if c2_suppress:
                    self._c2_apply_suppression(
                        prediction=prediction,
                        llm_original_risk=risk_score,
                        audit=c2_audit,
                    )
                    self.logger.info(
                        f"C2 suppression {c2_audit.get('branch')}: "
                        f"{c2_audit.get('reason')} | "
                        f"llm_risk={c2_audit.get('llm_risk')} | "
                        f"qSOFA={c2_audit.get('qsofa')} SIRS={c2_audit.get('sirs_met')}"
                    )
            except Exception as e:
                self.logger.error(f"C2 suppression layer failed (non-fatal): {e}")

        return prediction
    
    def _check_early_detection_patterns(self, vitals: Dict[str, Any]) -> List[str]:
        """
        Check combination patterns from early_detection_patterns config.
        These catch early/impending sepsis before individual critical thresholds are breached.
        """
        warnings = []
        patterns = self.config.get("early_detection_patterns", {})
        
        # Pattern 1: Earliest reliable lab combo (low lymph + high neutrophils + high bands)
        lab_pattern = patterns.get("earliest_reliable_lab", {})
        if lab_pattern:
            criteria = lab_pattern.get("criteria", {})
            lymph = self._get_vital(vitals, "Lymphocytes", "lymphocytes", "ALC")
            neut = self._get_vital(vitals, "Neutrophils", "neutrophils", "ANC_pct", "Neut_pct")
            bands = self._get_vital(vitals, "Bands", "bands", "band_neutrophils")
            
            matches = []
            if lymph is not None and lymph < criteria.get("lymphocytes", {}).get("value", 1.0):
                matches.append(f"Lymphocytes {lymph} < {criteria['lymphocytes']['value']} K/µL")
            if neut is not None and neut > criteria.get("neutrophils", {}).get("value", 80):
                matches.append(f"Neutrophils {neut}% > {criteria['neutrophils']['value']}%")
            if bands is not None and bands >= criteria.get("bands", {}).get("value", 10):
                matches.append(f"Bands {bands}% >= {criteria['bands']['value']}%")
            
            requires_all = lab_pattern.get("requires_all", True)
            if requires_all and len(matches) == 3:
                warnings.append(f"Early Detection — Bacterial infection pattern: {', '.join(matches)}")
            elif not requires_all and len(matches) >= 2:
                warnings.append(f"Early Detection — Bacterial infection pattern: {', '.join(matches)}")
        
        # Pattern 2: Early sepsis vitals combo (2+ of: HR>=90, RR>=22, temp abnormal, WBC abnormal)
        vitals_pattern = patterns.get("early_sepsis_vitals", {})
        if vitals_pattern:
            criteria = vitals_pattern.get("criteria", {})
            hr = self._get_vital(vitals, "HR", "heart_rate", "HeartRate", "pulse")
            rr = self._get_vital(vitals, "Resp", "resp_rate", "RR", "respiratory_rate")
            temp = self._get_vital(vitals, "Temp", "temperature", "Temperature")
            wbc = self._get_vital(vitals, "WBC", "wbc", "white_blood_cells")
            
            matches = []
            if hr is not None and hr >= criteria.get("hr", {}).get("value", 90):
                matches.append(f"HR {hr} >= 90")
            if rr is not None and rr >= criteria.get("rr", {}).get("value", 22):
                matches.append(f"RR {rr} >= 22")
            if temp is not None and temp >= criteria.get("temp_high", {}).get("value", 38.0):
                matches.append(f"Temp {temp} >= 38.0°C")
            if temp is not None and temp < criteria.get("temp_low", {}).get("value", 36.0):
                matches.append(f"Temp {temp} < 36.0°C")
            if wbc is not None and wbc >= criteria.get("wbc_high", {}).get("value", 12):
                matches.append(f"WBC {wbc} >= 12")
            if wbc is not None and wbc <= criteria.get("wbc_low", {}).get("value", 4):
                matches.append(f"WBC {wbc} <= 4")
            
            if len(matches) >= 2:
                warnings.append(f"Early Detection — Early sepsis vitals: {', '.join(matches)}")
        
        return warnings

    def _check_history_context(self, vitals: Dict[str, Any], 
                                patient_history: Dict[str, Any],
                                overrides: List[str]) -> tuple:
        """
        Evaluate patient history and medications to modify guardrail interpretation.
        
        - Chronic HTN → use stricter MAP threshold (75 instead of 65)
        - Renal/hepatic/hematologic history → flag that elevated values may be baseline
        - Seizure history → lactate may be from seizure, not sepsis
        - Antipyretics → normal temp may be masked fever
        - Anticoagulants → elevated INR may be medication, not DIC
        
        Returns (modified_overrides, context_flags)
        """
        context_flags = []
        conditions = [c.lower() for c in patient_history.get("conditions", [])]
        medications = [m.lower() for m in patient_history.get("medications", [])]
        conditions_text = " ".join(conditions)
        medications_text = " ".join(medications)
        
        history_config = self.config.get("history_context_checks", {})
        
        # --- Cardiovascular: Chronic HTN → stricter MAP threshold ---
        cv_config = history_config.get("cardiovascular_history", {})
        htn_terms = [c.lower() for c in cv_config.get("conditions_to_check", 
                     ["chronic hypertension", "htn", "high blood pressure"])]
        has_htn = any(term in conditions_text for term in htn_terms)
        
        if has_htn:
            map_val = self._get_vital(vitals, "MAP", "map", "mean_arterial_pressure")
            sbp = self._get_vital(vitals, "SBP", "sbp", "systolic_bp")
            dbp = self._get_vital(vitals, "DBP", "dbp", "diastolic_bp")
            if map_val is None and sbp is not None and dbp is not None:
                map_val = (sbp + 2 * dbp) / 3
            
            htn_threshold = self.config.get("critical_thresholds", {}).get(
                "hemodynamic", {}).get("map_critical_htn_patient", {}).get("value", 75)
            
            if map_val is not None and self.map_threshold <= map_val < htn_threshold:
                overrides.append(
                    f"Critical MAP for HTN patient (MAP {map_val:.0f} < {htn_threshold} mmHg — "
                    f"standard threshold {self.map_threshold} would miss this)"
                )
            context_flags.append(f"Chronic HTN detected: using stricter MAP threshold (< {htn_threshold} mmHg)")
        
        # --- Renal history ---
        renal_config = history_config.get("renal_history", {})
        renal_terms = [c.lower() for c in renal_config.get("conditions_to_check",
                       ["renal disease", "chronic kidney disease", "ckd", "esrf", "dialysis"])]
        has_renal = any(term in conditions_text for term in renal_terms)
        
        creatinine = self._get_vital(vitals, "Creatinine", "creatinine")
        if creatinine is not None and creatinine >= 2.0:
            if has_renal:
                context_flags.append(
                    f"Renal history present: Creatinine {creatinine} may be patient's baseline")
            else:
                context_flags.append(
                    f"NO renal history: Creatinine {creatinine} — possible sepsis-related AKI")
        
        # --- Hepatic history ---
        hepatic_config = history_config.get("hepatic_history", {})
        hepatic_terms = [c.lower() for c in hepatic_config.get("conditions_to_check",
                         ["liver disease", "hepatitis", "cirrhosis"])]
        has_hepatic = any(term in conditions_text for term in hepatic_terms)
        
        bilirubin = self._get_vital(vitals, "Bilirubin_total", "Bilirubin_direct", "bilirubin", "TotalBilirubin")
        if bilirubin is not None and bilirubin >= 2.0:
            if has_hepatic:
                context_flags.append(
                    f"Hepatic history present: Bilirubin {bilirubin} may be patient's baseline")
            else:
                context_flags.append(
                    f"NO hepatic history: Bilirubin {bilirubin} — possible sepsis-related hepatic injury")
        
        # --- Hematologic history ---
        hema_config = history_config.get("hematologic_history", {})
        hema_terms = [c.lower() for c in hema_config.get("conditions_to_check",
                      ["thrombocytopenia", "itp", "leukemia", "chemotherapy"])]
        has_hema = any(term in conditions_text for term in hema_terms)
        
        platelets = self._get_vital(vitals, "Platelets", "platelets", "PLT")
        if platelets is not None and platelets < 150:
            if has_hema:
                context_flags.append(
                    f"Hematologic history present: Platelets {platelets} may be patient's baseline")
            else:
                context_flags.append(
                    f"NO hematologic history: Platelets {platelets} — possible sepsis-related thrombocytopenia")
        
        # --- Diabetes history ---
        diabetes_config = history_config.get("diabetes_history", {})
        diabetes_terms = [c.lower() for c in diabetes_config.get("conditions_to_check",
                          ["diabetes", "dm", "type 1 diabetes", "type 2 diabetes", "insulin dependent"])]
        has_diabetes = any(term in conditions_text for term in diabetes_terms)
        
        glucose = self._get_vital(vitals, "Glucose", "glucose", "blood_glucose")
        if glucose is not None and glucose >= 180 and has_diabetes:
            context_flags.append(
                f"Diabetic patient: Glucose {glucose} — compare to patient's baseline")
        
        # --- Seizure history → lactate may not indicate sepsis ---
        seizure_config = history_config.get("seizure_history", {})
        seizure_terms = [c.lower() for c in seizure_config.get("conditions_to_check",
                         ["seizure disorder", "epilepsy", "recent seizure"])]
        has_seizure = any(term in conditions_text for term in seizure_terms)
        
        lactate = self._get_vital(vitals, "Lactate", "lactate")
        if lactate is not None and lactate >= 2.0 and has_seizure:
            context_flags.append(
                f"Seizure history: Lactate {lactate} may be from seizure activity, not sepsis")
        
        # --- Neurologic history → AMS must be NEW ONSET ---
        neuro_config = history_config.get("neurologic_history", {})
        neuro_terms = [c.lower() for c in neuro_config.get("conditions_affecting_mental_status",
                       ["alzheimer", "dementia", "baseline confusion", "cognitive dysfunction"])]
        has_neuro = any(term in conditions_text for term in neuro_terms)
        
        if has_neuro:
            context_flags.append(
                "Neurologic history: Altered mental status must be NEW ONSET to count for sepsis scoring")
        
        # --- Medication checks ---
        med_config = history_config.get("medications_affecting_labs", {})
        
        # Corticosteroids → can elevate glucose
        steroid_terms = ["corticosteroid", "prednisone", "dexamethasone", "methylprednisolone", "hydrocortisone"]
        if any(term in medications_text for term in steroid_terms):
            if glucose is not None and glucose >= 180:
                context_flags.append(
                    f"On corticosteroids: Glucose {glucose} may be medication-related, not infection")
        
        # Antipyretics → can mask fever
        antipyretic_meds = [m.lower() for m in med_config.get("antipyretics", {}).get(
            "medications", ["acetaminophen", "ibuprofen", "naproxen", "aspirin"])]
        if any(med in medications_text for med in antipyretic_meds):
            temp = self._get_vital(vitals, "Temp", "temperature", "Temperature")
            if temp is not None and 36.0 <= temp <= 38.0:
                context_flags.append(
                    f"On antipyretics: Temp {temp}°C appears normal but fever may be masked")
        
        # Anticoagulants → INR may be medication-related
        anticoag_terms = ["warfarin", "coumadin", "heparin", "eliquis", "xarelto", "anticoagulant", "enoxaparin", "lovenox"]
        if any(term in medications_text for term in anticoag_terms):
            inr = self._get_vital(vitals, "INR", "inr")
            if inr is not None and inr >= 2.5:
                context_flags.append(
                    f"On anticoagulants: INR {inr} may be medication-related, not DIC")
        
        return overrides, context_flags

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
        """Return all active configurable settings used by the guardrail."""
        
        # Read early detection patterns from config
        patterns = self.config.get("early_detection_patterns", {})
        lab_pattern = patterns.get("earliest_reliable_lab", {})
        lab_criteria = lab_pattern.get("criteria", {})
        vitals_pattern = patterns.get("early_sepsis_vitals", {})
        vitals_criteria = vitals_pattern.get("criteria", {})
        
        # Read combined criteria from config
        override_logic = self.config.get("override_logic", {})
        shock = override_logic.get("shock_criteria", {})
        dic = override_logic.get("dic_criteria", {})
        
        # Read discordance from config
        discordance = self.config.get("discordance_rules", {})
        
        # Read qSOFA from config
        qsofa_config = self.config.get("qsofa_criteria", {})
        
        # Read history-aware HTN threshold
        htn_map = self.config.get("critical_thresholds", {}).get(
            "hemodynamic", {}).get("map_critical_htn_patient", {})
        
        return {
            "critical_thresholds": {
                "hemodynamic": {
                    "sbp_threshold": self.sbp_threshold,
                    "map_threshold": self.map_threshold,
                    "map_threshold_htn_patient": htn_map.get("value", 75),
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
                }
            },
            "combined_criteria": {
                "septic_shock": {
                    "description": shock.get("description", "Hypotension + Hyperlactatemia"),
                    "requires_both": shock.get("requires_both", True),
                    "condition_1": f"SBP <= {self.sbp_threshold} OR MAP < {self.map_threshold}",
                    "condition_2": f"Lactate >= {self.lactate_threshold}"
                },
                "dic_detection": {
                    "description": dic.get("description", "Disseminated Intravascular Coagulation"),
                    "requires_n_of_4": 3,
                    "criteria": [
                        f"Platelets < {self.platelets_threshold}",
                        f"INR >= {self.inr_threshold}",
                        f"Fibrinogen < {self.fibrinogen_threshold}",
                        f"D-Dimer >= {self.d_dimer_threshold}"
                    ]
                }
            },
            "early_detection_patterns": {
                "bacterial_infection_combo": {
                    "description": lab_pattern.get("description", "Earliest reliable lab combination"),
                    "requires_all": lab_pattern.get("requires_all", True),
                    "lymphocytes_lt": lab_criteria.get("lymphocytes", {}).get("value", 1.0),
                    "neutrophils_gt": lab_criteria.get("neutrophils", {}).get("value", 80),
                    "bands_gte": lab_criteria.get("bands", {}).get("value", 10)
                },
                "early_sepsis_vitals": {
                    "description": vitals_pattern.get("description", "Early vital sign pattern"),
                    "requires_n_or_more": 2,
                    "hr_gte": vitals_criteria.get("hr", {}).get("value", 90),
                    "rr_gte": vitals_criteria.get("rr", {}).get("value", 22),
                    "temp_high_gte": vitals_criteria.get("temp_high", {}).get("value", 38.0),
                    "temp_low_lt": vitals_criteria.get("temp_low", {}).get("value", 36.0),
                    "wbc_high_gte": vitals_criteria.get("wbc_high", {}).get("value", 12),
                    "wbc_low_lte": vitals_criteria.get("wbc_low", {}).get("value", 4)
                }
            },
            "scoring_thresholds": {
                "qSOFA": {
                    "rr_gte": qsofa_config.get("respiratory_rate", {}).get("threshold", 22),
                    "sbp_lte": qsofa_config.get("systolic_bp", {}).get("threshold", 100),
                    "altered_mentation": "GCS < 15 or documented new-onset AMS",
                    "positive_score": ">=2 of 3"
                },
                "SIRS": {
                    "temp_gt": 38.0,
                    "temp_lt": 36.0,
                    "hr_gt": 90,
                    "rr_gt": 20,
                    "paco2_lt": 32,
                    "wbc_gt": 12,
                    "wbc_lt": 4,
                    "bands_gt": 10,
                    "positive_criteria": ">=2 of 4"
                },
                "SOFA": {
                    "respiratory_pf_bands": ">=400:0, >=300:1, >=200:2, >=100:3, <100:4",
                    "coagulation_platelets_bands": ">=150:0, >=100:1, >=50:2, >=20:3, <20:4",
                    "liver_bilirubin_bands": "<1.2:0, <2.0:1, <6.0:2, <12.0:3, >=12.0:4",
                    "cardiovascular": "MAP>=70:0, MAP>=65:1, MAP<65:2, vasopressors:3-4",
                    "cns_gcs_bands": "15:0, 13-14:1, 10-12:2, 6-9:3, <6:4",
                    "renal_creatinine_bands": "<1.2:0, <2.0:1, <3.5:2, <5.0:3, >=5.0:4",
                    "max_score": 24
                }
            },
            "discordance_detection": {
                "enabled": self.discordance_enabled,
                "escalation_risk_score": self.escalation_risk_score,
                "escalation_priority": self.escalation_priority,
                "concerning_phrase_categories": {
                    "perfusion": len(discordance.get("perfusion_concerning_phrases", [])),
                    "mental_status": len(discordance.get("mental_status_phrases", {}).get("phrases", [])),
                    "fluid_response": len(discordance.get("fluid_response_phrases", [])),
                    "vasopressor": len(discordance.get("vasopressor_phrases", [])),
                    "urine_output": len(discordance.get("urine_output_phrases", [])),
                    "respiratory_distress": len(discordance.get("respiratory_distress_phrases", []))
                },
                "total_phrases_monitored": len(self.concerning_phrases)
            },
            "override_settings": {
                "min_risk_for_critical": self.min_risk_for_critical,
                "forced_risk_score": self.forced_risk_score,
                "forced_priority": self.forced_priority,
                "forced_probability_6h": self.forced_probability
            },
            "history_aware_checks": {
                "cardiovascular_htn_map_threshold": htn_map.get("value", 75),
                "monitored_conditions": [
                    "chronic hypertension", "renal disease", "liver disease",
                    "hematologic disorders", "diabetes", "seizure disorder",
                    "neurologic conditions"
                ],
                "monitored_medications": [
                    "corticosteroids", "antipyretics", "anticoagulants"
                ]
            },
            "audit_settings": {
                "log_all_overrides": self.config.get("audit_settings", {}).get("log_all_overrides", True),
                "log_near_misses": self.config.get("audit_settings", {}).get("log_near_misses", True),
                "near_miss_threshold": self.config.get("audit_settings", {}).get("near_miss_threshold", 75)
            }
        }

    def calculate_clinical_scores(self, vitals: Dict[str, Any], gcs: Optional[int] = None, 
                                   on_vasopressors: bool = False) -> Dict[str, Any]:
        """
        Calculate qSOFA, SIRS, and estimated SOFA scores from vital signs.
        
        Args:
            vitals: Dictionary of vital signs and lab values
            gcs: Glasgow Coma Scale (if available)
            on_vasopressors: Whether patient is on vasopressor support
            
        Returns:
            Dictionary containing all calculated scores with components
        """
        qsofa = self._calculate_qsofa(vitals, gcs)
        sirs = self._calculate_sirs(vitals)
        sofa = self._calculate_sofa(vitals, gcs, on_vasopressors)
        sepsis_criteria = self._evaluate_sepsis_criteria(vitals, qsofa, sofa)
        
        return {
            "qsofa": qsofa,
            "sirs": sirs,
            "sofa": sofa,
            "sepsis_criteria": sepsis_criteria
        }
    
    def _calculate_qsofa(self, vitals: Dict[str, Any], gcs: Optional[int] = None) -> Dict[str, Any]:
        """
        Calculate qSOFA (Quick SOFA) score.
        qSOFA >= 2 with suspected infection suggests high risk.
        
        Components:
        - Respiratory rate >= 22/min (1 point)
        - Altered mentation (GCS < 15) (1 point)
        - Systolic BP <= 100 mmHg (1 point)
        """
        resp = self._get_vital(vitals, "Resp", "resp_rate", "RR", "respiratory_rate")
        sbp = self._get_vital(vitals, "SBP", "sbp", "systolic_bp")
        gcs_val = gcs or self._get_vital(vitals, "GCS", "gcs", "glasgow_coma_scale")
        
        components = {
            "respiratory_rate_ge_22": resp is not None and resp >= 22,
            "altered_mentation": gcs_val is not None and gcs_val < 15,
            "systolic_bp_le_100": sbp is not None and sbp <= 100
        }
        
        score = sum(1 for v in components.values() if v)
        
        return {
            "score": score,
            "max_score": 3,
            "components": components,
            "interpretation": "High risk" if score >= 2 else "Lower risk",
            "sepsis_suspected": score >= 2
        }
    
    def _calculate_sirs(self, vitals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate SIRS (Systemic Inflammatory Response Syndrome) criteria.
        SIRS >= 2 criteria suggests systemic inflammatory response.
        
        Components:
        - Temperature > 38°C or < 36°C (1 point)
        - Heart rate > 90 bpm (1 point)
        - Respiratory rate > 20/min OR PaCO2 < 32 mmHg (1 point)
        - WBC > 12K or < 4K or > 10% bands (1 point)
        """
        temp = self._get_vital(vitals, "Temp", "temperature", "Temperature")
        hr = self._get_vital(vitals, "HR", "heart_rate", "HeartRate", "pulse")
        resp = self._get_vital(vitals, "Resp", "resp_rate", "RR", "respiratory_rate")
        paco2 = self._get_vital(vitals, "PaCO2", "paco2")
        wbc = self._get_vital(vitals, "WBC", "wbc", "white_blood_cells")
        bands = self._get_vital(vitals, "Bands", "bands", "band_neutrophils")
        
        components = {
            "temp_abnormal": temp is not None and (temp > 38.0 or temp < 36.0),
            "hr_gt_90": hr is not None and hr > 90,
            "rr_gt_20_or_paco2_lt_32": (resp is not None and resp > 20) or (paco2 is not None and paco2 < 32),
            "wbc_abnormal": (wbc is not None and (wbc > 12 or wbc < 4)) or (bands is not None and bands > 10)
        }
        
        score = sum(1 for v in components.values() if v)
        
        return {
            "criteria_met": score,
            "max_criteria": 4,
            "components": components,
            "sirs_positive": score >= 2
        }
    
    def _calculate_sofa(self, vitals: Dict[str, Any], gcs: Optional[int] = None,
                        on_vasopressors: bool = False) -> Dict[str, Any]:
        """
        Calculate estimated SOFA (Sequential Organ Failure Assessment) score.
        Each organ system scored 0-4 based on dysfunction severity.
        Total score 0-24.
        
        Note: This is an ESTIMATE - some components require clinical context
        that may not be available in vitals alone.
        """
        components = {}
        
        # Respiratory: PaO2/FiO2 ratio
        pf_ratio = self._get_vital(vitals, "PaO2_FiO2", "pf_ratio", "P_F_ratio")
        o2sat = self._get_vital(vitals, "O2Sat", "SaO2", "SpO2", "o2_saturation")
        if pf_ratio is not None:
            if pf_ratio >= 400: components["respiratory"] = 0
            elif pf_ratio >= 300: components["respiratory"] = 1
            elif pf_ratio >= 200: components["respiratory"] = 2
            elif pf_ratio >= 100: components["respiratory"] = 3
            else: components["respiratory"] = 4
        elif o2sat is not None:
            # Rough estimate from SpO2 (less accurate)
            if o2sat >= 96: components["respiratory"] = 0
            elif o2sat >= 92: components["respiratory"] = 1
            elif o2sat >= 88: components["respiratory"] = 2
            else: components["respiratory"] = 3
        else:
            components["respiratory"] = None
        
        # Coagulation: Platelets
        platelets = self._get_vital(vitals, "Platelets", "platelets", "PLT")
        if platelets is not None:
            if platelets >= 150: components["coagulation"] = 0
            elif platelets >= 100: components["coagulation"] = 1
            elif platelets >= 50: components["coagulation"] = 2
            elif platelets >= 20: components["coagulation"] = 3
            else: components["coagulation"] = 4
        else:
            components["coagulation"] = None
        
        # Liver: Bilirubin
        bilirubin = self._get_vital(vitals, "Bilirubin_total", "Bilirubin_direct", "bilirubin", "TotalBilirubin")
        if bilirubin is not None:
            if bilirubin < 1.2: components["liver"] = 0
            elif bilirubin < 2.0: components["liver"] = 1
            elif bilirubin < 6.0: components["liver"] = 2
            elif bilirubin < 12.0: components["liver"] = 3
            else: components["liver"] = 4
        else:
            components["liver"] = None
        
        # Cardiovascular: MAP and vasopressors
        map_val = self._get_vital(vitals, "MAP", "map", "mean_arterial_pressure")
        sbp = self._get_vital(vitals, "SBP", "sbp", "systolic_bp")
        dbp = self._get_vital(vitals, "DBP", "dbp", "diastolic_bp")
        
        # Calculate MAP if not provided but SBP/DBP available
        if map_val is None and sbp is not None and dbp is not None:
            map_val = (sbp + 2 * dbp) / 3
        
        if on_vasopressors:
            components["cardiovascular"] = 3  # At least 3 if on vasopressors
        elif map_val is not None:
            if map_val >= 70: components["cardiovascular"] = 0
            elif map_val >= 65: components["cardiovascular"] = 1
            else: components["cardiovascular"] = 2
        else:
            components["cardiovascular"] = None
        
        # CNS: GCS
        gcs_val = gcs or self._get_vital(vitals, "GCS", "gcs", "glasgow_coma_scale")
        if gcs_val is not None:
            if gcs_val == 15: components["cns"] = 0
            elif gcs_val >= 13: components["cns"] = 1
            elif gcs_val >= 10: components["cns"] = 2
            elif gcs_val >= 6: components["cns"] = 3
            else: components["cns"] = 4
        else:
            components["cns"] = None
        
        # Renal: Creatinine or Urine Output
        creatinine = self._get_vital(vitals, "Creatinine", "creatinine")
        urine_output = self._get_vital(vitals, "UrineOutput", "urine_output", "UO")
        
        if creatinine is not None:
            if creatinine < 1.2: components["renal"] = 0
            elif creatinine < 2.0: components["renal"] = 1
            elif creatinine < 3.5: components["renal"] = 2
            elif creatinine < 5.0: components["renal"] = 3
            else: components["renal"] = 4
        elif urine_output is not None:
            if urine_output >= 0.5: components["renal"] = 0
            elif urine_output >= 0.3: components["renal"] = 2
            else: components["renal"] = 4
        else:
            components["renal"] = None
        
        # Calculate total score (only from available components)
        available_scores = [v for v in components.values() if v is not None]
        total_score = sum(available_scores) if available_scores else None
        
        # Estimate missing components as 0 for total estimate
        estimated_total = sum(v if v is not None else 0 for v in components.values())
        
        return {
            "score": total_score,
            "estimated_score": estimated_total,
            "max_score": 24,
            "components": components,
            "components_available": len(available_scores),
            "interpretation": self._interpret_sofa(estimated_total)
        }
    
    def _interpret_sofa(self, score: int) -> str:
        """Interpret SOFA score severity."""
        if score is None:
            return "Unable to calculate"
        elif score <= 1:
            return "Minimal organ dysfunction"
        elif score <= 5:
            return "Mild organ dysfunction"
        elif score <= 10:
            return "Moderate organ dysfunction"
        elif score <= 15:
            return "Severe organ dysfunction"
        else:
            return "Very severe organ dysfunction"
    
    def _evaluate_sepsis_criteria(self, vitals: Dict[str, Any], qsofa: Dict, sofa: Dict) -> Dict[str, Any]:
        """
        Evaluate Sepsis-3 criteria.
        
        Sepsis = Suspected infection + SOFA increase >= 2 (or qSOFA >= 2 for screening)
        Septic Shock = Sepsis + Vasopressors needed for MAP >= 65 + Lactate > 2 despite fluids
        """
        lactate = self._get_vital(vitals, "Lactate", "lactate")
        map_val = self._get_vital(vitals, "MAP", "map", "mean_arterial_pressure")
        sbp = self._get_vital(vitals, "SBP", "sbp", "systolic_bp")
        
        criteria_met = []
        
        # qSOFA screening
        if qsofa["score"] >= 2:
            criteria_met.append("qSOFA >= 2 (sepsis screening positive)")
        
        # SOFA-based sepsis (assuming baseline SOFA of 0 for acute presentation)
        if sofa["estimated_score"] >= 2:
            criteria_met.append(f"SOFA >= 2 (estimated score: {sofa['estimated_score']})")
        
        # Hypotension
        if sbp is not None and sbp <= 90:
            criteria_met.append(f"Hypotension (SBP {sbp} <= 90 mmHg)")
        elif map_val is not None and map_val < 65:
            criteria_met.append(f"Hypotension (MAP {map_val} < 65 mmHg)")
        
        # Hyperlactatemia
        if lactate is not None and lactate >= 2:
            criteria_met.append(f"Elevated lactate ({lactate} >= 2 mmol/L)")
        
        # Determine sepsis status
        sepsis_3_met = qsofa["score"] >= 2 or sofa["estimated_score"] >= 2
        
        # Septic shock: hypotension + lactate > 2
        hypotension = (sbp is not None and sbp <= 90) or (map_val is not None and map_val < 65)
        hyperlactatemia = lactate is not None and lactate >= 2
        septic_shock_criteria = sepsis_3_met and hypotension and hyperlactatemia
        
        return {
            "sepsis_3_met": sepsis_3_met,
            "septic_shock_criteria_met": septic_shock_criteria,
            "criteria_details": criteria_met,
            "requires_immediate_action": septic_shock_criteria or (qsofa["score"] >= 2 and hyperlactatemia)
        }
