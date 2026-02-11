import json
import re

class SepsisPreprocessor:
    def __init__(self):
        self.thresholds = {
            "hr_high": 100, "temp_high": 38.3, "temp_low": 36.0,
            "resp_high": 22, "sbp_low": 100, "lactate_high": 2.0
        }

    def normalize_red_rover(self, rr_data):
        """
        Red Rover often nests vitals. This flattens them and extracts 
        the 'latest' value for the narrative while keeping history for trends.
        """
        flattened = {}
        for key, value in rr_data.items():
            if isinstance(value, list) and len(value) > 0:
                # Take the most recent reading (Red Rover typically orders by timestamp desc)
                flattened[key] = value[0].get('val') if isinstance(value[0], dict) else value[0]
                # Store the previous value to check the trend
                if len(value) > 1:
                    flattened[f"{key}_prev"] = value[1].get('val') if isinstance(value[1], dict) else value[1]
            else:
                flattened[key] = value
        return flattened

    def clean_notes(self, text):
        if not text: return "No clinical notes available for this window."
        expansions = {
            r'\bSOB\b': 'shortness of breath',
            r'\bAMS\b': 'altered mental status',
            r'\bpt\b': 'patient',
            r'\bbolus\b': 'fluid bolus',
            r'\bc/o\b': 'complains of'
        }
        for pattern, replacement in expansions.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text.strip()

    def serialize_vitals(self, data):
        hr = data.get("HR")
        hr_prev = data.get("HR_prev")
        sbp = data.get("SBP")
        sbp_prev = data.get("SBP_prev")
        
        narrative = []
        
        # Trend Reasoning (Critical for 6-hour prediction)
        vitals_status = []
        
        # Heart Rate Analysis
        if hr:
            status = f"{hr} bpm"
            if hr > self.thresholds["hr_high"]:
                trend = "rising" if hr_prev and hr > hr_prev else "stable"
                vitals_status.append(f"tachycardia ({status}, {trend})")
            narrative.append(f"Heart Rate is {hr} bpm.")

        # Blood Pressure Analysis
        if sbp:
            if sbp <= self.thresholds["sbp_low"]:
                trend = "dropping" if sbp_prev and sbp < sbp_prev else "persistently low"
                vitals_status.append(f"hypotension ({sbp} mmHg, {trend})")
            else:
                narrative.append(f"Blood pressure is currently stable at {sbp} mmHg.")

        if vitals_status:
            narrative.insert(0, f"CRITICAL TRENDS: The patient shows signs of {', '.join(vitals_status)}.")

        # Laboratory Summary
        lactate = data.get("Lactate")
        if lactate:
            comment = "critically elevated" if lactate >= self.thresholds["lactate_high"] else "within normal limits"
            narrative.append(f"Lactate is {lactate} mmol/L, which is {comment}.")

        # Background parameters (The other 30+ fields)
        excluded = ['HR', 'HR_prev', 'SBP', 'SBP_prev', 'Temp', 'Resp', 'Lactate', 'Age', 'Gender']
        others = [f"{k}: {v}" for k, v in data.items() if k not in excluded and v is not None]
        narrative.append(f"\nOther telemetry: {', '.join(others[:15])}...") # Limit to avoid token bloat
        
        return " ".join(narrative)

    def process(self, red_rover_json, clinician_notes=None):
        # Step 1: Flatten the Red Rover nested structure
        patient_data = self.normalize_red_rover(red_rover_json)
        
        # Step 2: Create structured and unstructured components
        structured_part = self.serialize_vitals(patient_data)
        unstructured_part = self.clean_notes(clinician_notes)
        
        return (
            f"IDENTIFICATION: Patient Age {patient_data.get('Age', 'N/A')}\n"
            f"--- NUMERIC TRENDS ---\n{structured_part}\n\n"
            f"--- CLINICIAN NOTES ---\n{unstructured_part}"
        )

# --- Red Rover Style Batch Execution ---
if __name__ == "__main__":
    preprocessor = SepsisPreprocessor()
    
    # Simulating Red Rover's time-series nested format
    red_rover_packet = {
        "HR": [{"val": 118, "ts": "2026-02-07T18:30"}, {"val": 105, "ts": "2026-02-07T18:25"}],
        "SBP": [{"val": 95, "ts": "2026-02-07T18:30"}, {"val": 110, "ts": "2026-02-07T18:25"}],
        "Lactate": [2.4],
        "Age": 72,
        "Gender": 1,
        "PaO2": [85]
    }
    
    notes = "Pt is c/o chills. AMS noted by nursing staff."
    
    print(preprocessor.process(red_rover_packet, notes))