import json
import re

# Vitals where the full multi-hour trend should be serialized into the narrative
# (these drive sepsis judgment and benefit from trend visibility).
TREND_VITALS = ("HR", "SBP", "DBP", "MAP", "Temp", "Resp", "O2Sat")

# Vitals/labs where we only show the most-recent value (single-shot or
# slow-changing). The numeric current value still goes through.
SINGLE_VALUE_KEYS = (
    "Lactate", "WBC", "Creatinine", "Platelets", "Bilirubin_total",
    "BUN", "Glucose", "pH", "BaseExcess", "HCO3", "PaCO2", "AST",
    "Potassium", "Hgb", "PTT", "Fibrinogen", "TroponinI", "FiO2",
)


def _extract_series(value):
    """Given a vital field that may be a list of {val, ts} dicts, a list of
    scalars, or a scalar, return:
      - current: the latest value (or None)
      - series: list of {val, ts} (oldest -> newest), or empty list

    The input is assumed NEWEST-FIRST (the v4 cohort fix and the documented
    Red Rover convention).
    """
    if isinstance(value, list) and len(value) > 0:
        if isinstance(value[0], dict) and "val" in value[0]:
            # newest-first; keep oldest->newest for display
            ordered = list(reversed(value))
            current = ordered[-1].get("val")
            return current, ordered
        else:
            # plain list of scalars (single-element lab arrays mostly)
            return value[0], [{"val": v, "ts": None} for v in reversed(value)]
    if value is None:
        return None, []
    return value, []


class SepsisPreprocessor:
    def __init__(self):
        self.thresholds = {
            "hr_high": 100, "temp_high": 38.3, "temp_low": 36.0,
            "resp_high": 22, "sbp_low": 100, "lactate_high": 2.0,
            "map_low": 65, "o2sat_low": 92,
        }

    def normalize_red_rover(self, rr_data):
        """
        Flatten Red Rover's nested time-series format into:
          - flattened[key]:        latest scalar (current)
          - flattened[key + '_prev']: 1-hour-prior scalar (kept for backward compat)
          - flattened[key + '_series']: ordered list of {val, ts} oldest -> newest
                                         (only for trend-relevant vitals)
        """
        flattened = {}
        for key, value in rr_data.items():
            current, series = _extract_series(value)
            if current is None and not series:
                continue
            flattened[key] = current
            if len(series) > 1:
                flattened[f"{key}_prev"] = series[-2].get("val") if isinstance(series[-2], dict) else series[-2]
                if key in TREND_VITALS:
                    flattened[f"{key}_series"] = series
        return flattened

    def clean_notes(self, text):
        if not text:
            return "No clinical notes available for this window."
        expansions = {
            r'\bSOB\b': 'shortness of breath',
            r'\bAMS\b': 'altered mental status',
            r'\bpt\b': 'patient',
            r'\bbolus\b': 'fluid bolus',
            r'\bc/o\b': 'complains of',
        }
        for pattern, replacement in expansions.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text.strip()

    @staticmethod
    def _format_trend(vital_label, current, series, unit=""):
        """Build a one-liner like:
          'HR (bpm): 78 -> 92 -> 105 -> 110 -> 116 -> 116 (current)'
        Falls back to current-only if no series available.
        """
        if current is None:
            return None
        if len(series) <= 1:
            return f"{vital_label}{(' ' + unit) if unit else ''}: {current} (current)"
        vals = [pt.get("val") if isinstance(pt, dict) else pt for pt in series]
        # Round numeric vals to 1 decimal for compactness
        def _fmt(v):
            if v is None:
                return "—"
            try:
                f = float(v)
                if abs(f - round(f)) < 0.05:
                    return f"{int(round(f))}"
                return f"{f:.1f}"
            except Exception:
                return str(v)
        sequence = " -> ".join(_fmt(v) for v in vals)
        return f"{vital_label}{(' ' + unit) if unit else ''}: {sequence} (current)"

    def serialize_vitals(self, data):
        narrative = []
        critical_flags = []

        # ---- Build trend lines for the major vitals ----
        trend_units = {
            "HR": "bpm", "SBP": "mmHg", "DBP": "mmHg", "MAP": "mmHg",
            "Temp": "°C", "Resp": "/min", "O2Sat": "%",
        }
        trend_lines = []
        for vk in TREND_VITALS:
            current = data.get(vk)
            series = data.get(f"{vk}_series", [])
            line = self._format_trend(vk, current, series, trend_units.get(vk, ""))
            if line:
                trend_lines.append(line)

        if trend_lines:
            narrative.append("Vital trends (oldest -> newest, last value is current):")
            narrative.extend(["  " + ln for ln in trend_lines])

        # ---- Critical-trend flags (legacy semantics for clarity) ----
        hr = data.get("HR"); hr_prev = data.get("HR_prev")
        sbp = data.get("SBP"); sbp_prev = data.get("SBP_prev")
        map_v = data.get("MAP"); map_prev = data.get("MAP_prev")
        temp = data.get("Temp")
        resp = data.get("Resp")
        o2 = data.get("O2Sat")

        if hr is not None and hr > self.thresholds["hr_high"]:
            trend = "rising" if hr_prev is not None and hr > hr_prev else (
                "falling" if hr_prev is not None and hr < hr_prev else "stable"
            )
            critical_flags.append(f"tachycardia (HR {hr}, {trend})")
        if sbp is not None and sbp <= self.thresholds["sbp_low"]:
            trend = "dropping" if sbp_prev is not None and sbp < sbp_prev else "persistently low"
            critical_flags.append(f"hypotension (SBP {sbp}, {trend})")
        if map_v is not None and map_v < self.thresholds["map_low"]:
            trend = "falling" if map_prev is not None and map_v < map_prev else "persistently low"
            critical_flags.append(f"low MAP ({map_v}, {trend})")
        if temp is not None:
            if temp >= self.thresholds["temp_high"]:
                critical_flags.append(f"febrile (Temp {temp}°C)")
            elif temp < self.thresholds["temp_low"]:
                critical_flags.append(f"hypothermic (Temp {temp}°C)")
        if resp is not None and resp >= self.thresholds["resp_high"]:
            critical_flags.append(f"tachypnea (RR {resp})")
        if o2 is not None and o2 < self.thresholds["o2sat_low"]:
            critical_flags.append(f"hypoxemia (SpO2 {o2}%)")

        if critical_flags:
            narrative.insert(0, "CRITICAL FLAGS: " + ", ".join(critical_flags) + ".")

        # ---- Lactate is special — single value but key for sepsis ----
        lactate = data.get("Lactate")
        if lactate is not None:
            comment = "critically elevated" if lactate >= self.thresholds["lactate_high"] else "within normal limits"
            narrative.append(f"Lactate: {lactate} mmol/L ({comment}).")

        # ---- Other labs / single-value parameters ----
        other_pairs = []
        for k in SINGLE_VALUE_KEYS:
            if k == "Lactate":
                continue  # already handled
            v = data.get(k)
            if v is not None:
                other_pairs.append(f"{k}: {v}")
        if other_pairs:
            narrative.append("Other labs: " + ", ".join(other_pairs[:15]) + ".")

        # ---- Demographics ----
        age = data.get("Age")
        gender = data.get("Gender")
        if age or gender:
            demo = []
            if age:
                demo.append(f"Age {age}")
            if gender:
                demo.append(f"Gender {gender}")
            narrative.append("Patient: " + ", ".join(demo) + ".")

        return "\n".join(narrative)

    def process(self, red_rover_json, clinician_notes=None):
        patient_data = self.normalize_red_rover(red_rover_json)
        structured_part = self.serialize_vitals(patient_data)
        unstructured_part = self.clean_notes(clinician_notes)
        return (
            f"--- NUMERIC TRENDS ---\n{structured_part}\n\n"
            f"--- CLINICIAN NOTES ---\n{unstructured_part}"
        )


# --- Local self-test ---
if __name__ == "__main__":
    preprocessor = SepsisPreprocessor()
    red_rover_packet = {
        "HR": [{"val": 118, "ts": "2026-02-07T18:30"}, {"val": 110, "ts": "2026-02-07T17:30"}, {"val": 105, "ts": "2026-02-07T16:30"}, {"val": 95, "ts": "2026-02-07T15:30"}, {"val": 88, "ts": "2026-02-07T14:30"}, {"val": 82, "ts": "2026-02-07T13:30"}],
        "SBP": [{"val": 95, "ts": "2026-02-07T18:30"}, {"val": 110, "ts": "2026-02-07T17:30"}, {"val": 118, "ts": "2026-02-07T16:30"}],
        "MAP": [{"val": 60, "ts": "2026-02-07T18:30"}, {"val": 70, "ts": "2026-02-07T17:30"}],
        "Temp": [{"val": 38.5, "ts": "2026-02-07T18:30"}, {"val": 37.9, "ts": "2026-02-07T17:30"}],
        "Resp": [{"val": 24, "ts": "2026-02-07T18:30"}, {"val": 20, "ts": "2026-02-07T17:30"}],
        "O2Sat": [{"val": 91, "ts": "2026-02-07T18:30"}, {"val": 95, "ts": "2026-02-07T17:30"}],
        "Lactate": [2.4],
        "Age": 72,
        "Gender": "Male",
    }
    notes = "Pt is c/o chills. AMS noted by nursing staff."
    print(preprocessor.process(red_rover_packet, notes))
