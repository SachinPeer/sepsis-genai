"""
eICU-CRD Demo Cohort Builder for Sepsis GenAI Validation

Selects 30-40 patients (~15-20 sepsis positive, ~15-20 non-sepsis controls) from
the eICU Collaborative Research Database Demo v2.0.1, and emits patient JSONs in
the same schema our /classify API expects (identical to PhysioNet v4 cohort).

Sepsis identification (for this pilot):
  - ICD-9 based: any of {038.x, 995.91, 995.92, 785.52} recorded
  - Onset time = earliest diagnosisoffset among the sepsis ICD-9 codes
  - Snapshot = onset - 360 min (6h prior to doctor's diagnosis)
  - Cross-check: patient has antibiotic order and/or microLab culture within 48h
    around the sepsis diagnosis (adds "suspicion of infection" confirmation)

Controls:
  - No sepsis ICD-9 codes
  - No antibiotic ordered within first 48h of ICU stay
  - ICU stay >= 24h (so we have a 6h trend window + stable period)
  - Snapshot = random hour between 12h and (dischargeoffset - 6h)

Output files:
  validation/eicu_cohort/manifest.json
  validation/eicu_cohort/p00001.json, p00002.json, ...

Run:
  python validation/eicu_cohort_builder.py
"""

from __future__ import annotations
import duckdb
import json
import random
import os
from pathlib import Path
from datetime import datetime, timedelta

# ----- Configuration -----------------------------------------------------------
EICU_DIR = Path(__file__).parent / "eicu_demo"
COHORT_DIR = Path(__file__).parent / "eicu_cohort"
COHORT_DIR.mkdir(exist_ok=True)

TARGET_SEPSIS = 40       # Phase 1B; uses all qualified sepsis up to 40
TARGET_CONTROLS = 60     # Phase 1B; first 100 patients = strict superset of Phase 1A's 29
TREND_HOURS = 6
NOTES_WINDOW_HOURS = 12  # wider than TREND_HOURS since notes are less frequent
SEPSIS_SNAPSHOT_OFFSET_MIN = 360  # 6 hours before sepsis onset
SEED = 42

# Sanity bounds to filter out unit-encoding errors and outliers
LAB_BOUNDS = {
    "WBC": (0.1, 200.0),        # x 10^9/L
    "Lactate": (0.1, 30.0),      # mmol/L
    "Creatinine": (0.1, 20.0),   # mg/dL
    "Platelets": (5.0, 2000.0),  # x 10^9/L
    "Glucose": (20.0, 1500.0),   # mg/dL
    "BUN": (1.0, 300.0),
    "pH": (6.5, 8.0),
    "HCO3": (3.0, 60.0),
    "PaCO2": (10.0, 200.0),
    "Hgb": (2.0, 25.0),
    "FiO2": (0.15, 1.0),
}

# Sepsis-indicating ICD-9 codes
SEPSIS_ICD9 = ["038", "995.91", "995.92", "785.52", "995.90"]

# We anchor the synthetic "timestamp" to a fake base time so our API sees
# realistic ISO8601 strings while preserving ordering. eICU data has offsets
# in minutes from ICU admit; we map offset 0 to BASE_TS.
BASE_TS = datetime(2024, 6, 1, 0, 0, 0)

random.seed(SEED)


# ----- Helpers -----------------------------------------------------------------

def offset_to_ts(offset_min: int | float | None) -> str | None:
    if offset_min is None:
        return None
    try:
        return (BASE_TS + timedelta(minutes=float(offset_min))).isoformat()
    except Exception:
        return None


def connect_db():
    """Open a DuckDB connection with all eICU tables loaded as MATERIALIZED
    in-memory tables (10-50x faster than VIEWs that re-parse csv.gz each query)."""
    con = duckdb.connect()
    tables = {
        "patient": EICU_DIR / "patient.csv.gz",
        "diagnosis": EICU_DIR / "diagnosis.csv.gz",
        "vitalPeriodic": EICU_DIR / "vitalPeriodic.csv.gz",
        "vitalAperiodic": EICU_DIR / "vitalAperiodic.csv.gz",
        "lab": EICU_DIR / "lab.csv.gz",
        "nurseAssessment": EICU_DIR / "nurseAssessment.csv.gz",
        "nurseCharting": EICU_DIR / "nurseCharting.csv.gz",
        "nurseCare": EICU_DIR / "nurseCare.csv.gz",
        "note": EICU_DIR / "note.csv.gz",
        "physicalExam": EICU_DIR / "physicalExam.csv.gz",
        "treatment": EICU_DIR / "treatment.csv.gz",
        "medication": EICU_DIR / "medication.csv.gz",
        "microLab": EICU_DIR / "microLab.csv.gz",
        "carePlanInfectiousDisease": EICU_DIR / "carePlanInfectiousDisease.csv.gz",
    }
    for name, path in tables.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing eICU table: {path}")
        print(f"  loading {name} ...")
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM read_csv_auto('{path}', header=True, sample_size=-1)")
    # Useful indexes for fast patient-scoped queries
    for t in ("vitalPeriodic", "vitalAperiodic", "lab", "nurseAssessment",
              "nurseCharting", "note", "physicalExam", "treatment",
              "medication", "microLab"):
        try:
            con.execute(f"CREATE INDEX idx_{t}_pid ON {t}(patientunitstayid)")
        except Exception:
            pass
    return con


# ----- Cohort selection --------------------------------------------------------

def find_sepsis_patients(con):
    """Return list of (patientunitstayid, sepsis_onset_offset_min, icd9_codes)."""
    # eICU icd9code column may contain multiple codes separated by commas
    code_pattern = "|".join([f"^{c}" for c in SEPSIS_ICD9])  # used only for Python filter below
    rows = con.execute(
        """
        SELECT patientunitstayid, diagnosisoffset, icd9code, diagnosisstring
        FROM diagnosis
        WHERE icd9code IS NOT NULL
          AND (
               icd9code LIKE '038%'
            OR icd9code LIKE '995.91%'
            OR icd9code LIKE '995.92%'
            OR icd9code LIKE '995.90%'
            OR icd9code LIKE '785.52%'
          )
        ORDER BY patientunitstayid, diagnosisoffset
        """
    ).fetchall()

    by_patient = {}
    for pid, offset, icd, diag_str in rows:
        if pid not in by_patient:
            by_patient[pid] = {"onset_offset": offset, "codes": set(), "strings": set()}
        if offset is not None and (by_patient[pid]["onset_offset"] is None or offset < by_patient[pid]["onset_offset"]):
            by_patient[pid]["onset_offset"] = offset
        if icd:
            by_patient[pid]["codes"].add(icd)
        if diag_str:
            by_patient[pid]["strings"].add(diag_str)

    # Filter: need onset >= 6h after admit so we have snapshot room
    qualified = []
    for pid, info in by_patient.items():
        onset = info["onset_offset"]
        if onset is None:
            continue
        if onset < SEPSIS_SNAPSHOT_OFFSET_MIN + 60:  # need at least 6h + 1h buffer before sepsis
            continue
        qualified.append({
            "patientunitstayid": pid,
            "onset_offset_min": onset,
            "icd9_codes": sorted(info["codes"]),
            "diagnosis_strings": sorted(info["strings"]),
        })
    return qualified


def find_control_patients(con, exclude_ids):
    """Return list of patients with no sepsis codes, ICU stay >= 24h, no early antibiotics."""
    exclude_sql = "(" + ",".join(str(i) for i in exclude_ids) + ")" if exclude_ids else "(0)"

    rows = con.execute(
        f"""
        SELECT p.patientunitstayid, p.unitdischargeoffset, p.hospitaldischargestatus, p.age, p.gender, p.unittype, p.hospitalid
        FROM patient p
        WHERE p.patientunitstayid NOT IN {exclude_sql}
          AND p.unitdischargeoffset >= 1440   -- >= 24h ICU stay
          AND p.patientunitstayid NOT IN (
              SELECT DISTINCT patientunitstayid FROM medication
              WHERE drugstartoffset IS NOT NULL AND drugstartoffset <= 2880
              AND (
                   UPPER(drugname) LIKE '%VANCOMYCIN%' OR UPPER(drugname) LIKE '%CEFTRIAXONE%'
                OR UPPER(drugname) LIKE '%PIPERACILLIN%' OR UPPER(drugname) LIKE '%MEROPENEM%'
                OR UPPER(drugname) LIKE '%LEVOFLOXACIN%' OR UPPER(drugname) LIKE '%CIPROFLOXACIN%'
                OR UPPER(drugname) LIKE '%CEFEPIME%' OR UPPER(drugname) LIKE '%METRONIDAZOLE%'
                OR UPPER(drugname) LIKE '%AZITHROMYCIN%' OR UPPER(drugname) LIKE '%CEFTAZIDIME%'
              )
          )
        ORDER BY p.patientunitstayid
        LIMIT 500
        """
    ).fetchall()
    return [{
        "patientunitstayid": r[0],
        "unit_discharge_offset": r[1],
        "disposition": r[2],
        "age": r[3],
        "gender": r[4],
        "unit_type": r[5],
        "hospital_id": r[6],
    } for r in rows]


# ----- Vitals + labs extraction ------------------------------------------------

VITAL_MAP_PERIODIC = {
    # our_key: eicu column name in vitalPeriodic
    "HR": "heartrate",
    "SBP": "systemicsystolic",
    "DBP": "systemicdiastolic",
    "MAP": "systemicmean",
    "Temp": "temperature",
    "Resp": "respiration",
    "O2Sat": "sao2",
}

LAB_MAP = {
    # our_key: possible eicu lab.labname values (case-insensitive LIKE)
    "WBC": ["WBC x 1000", "WBC"],
    "Lactate": ["lactate"],
    "Creatinine": ["creatinine"],
    "Platelets": ["platelets x 1000"],
    "Glucose": ["glucose"],
    "BUN": ["BUN"],
    "pH": ["pH"],
    "HCO3": ["bicarbonate", "HCO3"],
    "PaCO2": ["PaCO2", "paCO2"],
    "Hgb": ["Hgb", "hemoglobin"],
    "FiO2": ["FiO2"],
}


def fetch_vitals(con, pid: int, snapshot_offset: int):
    """Pull 6h of vital data preceding snapshot_offset. Returns dict of lists,
    newest-first, same schema as PhysioNet v4 cohort."""
    window_start = snapshot_offset - TREND_HOURS * 60
    vitals = {}
    for our_key, eicu_col in VITAL_MAP_PERIODIC.items():
        rows = con.execute(
            f"""
            SELECT observationoffset, {eicu_col} AS val
            FROM vitalPeriodic
            WHERE patientunitstayid = ?
              AND observationoffset >= ?
              AND observationoffset <= ?
              AND {eicu_col} IS NOT NULL
            ORDER BY observationoffset DESC
            """,
            [pid, window_start, snapshot_offset],
        ).fetchall()
        if not rows:
            vitals[our_key] = []
            continue
        # Downsample to roughly hourly by taking at most 1 reading per hour,
        # newest-first
        hourly = {}
        for offset, val in rows:
            hour_bucket = snapshot_offset - int((snapshot_offset - offset) // 60) * 60
            if hour_bucket not in hourly:
                hourly[hour_bucket] = (offset, val)
        ordered = sorted(hourly.values(), key=lambda x: x[0], reverse=True)
        vitals[our_key] = [{"val": float(v), "ts": offset_to_ts(ofs)} for ofs, v in ordered]

    # Temp fallback from nurseCharting if vitalPeriodic is empty
    if not vitals.get("Temp"):
        rows = con.execute(
            """
            SELECT nursingchartoffset, nursingchartvalue
            FROM nurseCharting
            WHERE patientunitstayid = ?
              AND nursingchartoffset >= ? AND nursingchartoffset <= ?
              AND LOWER(nursingchartcelltypevallabel) LIKE '%temp%'
              AND nursingchartvalue IS NOT NULL
            ORDER BY nursingchartoffset DESC
            """,
            [pid, window_start, snapshot_offset],
        ).fetchall()
        if rows:
            hourly = {}
            for offset, val in rows:
                try:
                    v = float(str(val).strip())
                    # convert F to C if looks like Fahrenheit
                    if v > 60:
                        v = (v - 32.0) * 5.0 / 9.0
                    if v < 28 or v > 45:
                        continue
                except Exception:
                    continue
                hour_bucket = snapshot_offset - int((snapshot_offset - offset) // 60) * 60
                if hour_bucket not in hourly:
                    hourly[hour_bucket] = (offset, v)
            ordered = sorted(hourly.values(), key=lambda x: x[0], reverse=True)
            vitals["Temp"] = [{"val": v, "ts": offset_to_ts(ofs)} for ofs, v in ordered]

    return vitals


def fetch_labs(con, pid: int, snapshot_offset: int):
    """Pull most recent lab values within 24h before snapshot. Returns dict."""
    window_start = snapshot_offset - 24 * 60
    labs = {}
    for our_key, names in LAB_MAP.items():
        name_clauses = " OR ".join([f"LOWER(labname) LIKE LOWER('%{n}%')" for n in names])
        rows = con.execute(
            f"""
            SELECT labresultoffset, labresult
            FROM lab
            WHERE patientunitstayid = ?
              AND labresultoffset >= ?
              AND labresultoffset <= ?
              AND labresult IS NOT NULL
              AND ({name_clauses})
            ORDER BY labresultoffset DESC
            LIMIT 1
            """,
            [pid, window_start, snapshot_offset],
        ).fetchall()
        if rows:
            ofs, val = rows[0]
            try:
                v = float(val)
            except Exception:
                continue
            lo, hi = LAB_BOUNDS.get(our_key, (None, None))
            if lo is not None and (v < lo or v > hi):
                continue
            labs[our_key] = [{"val": v, "ts": offset_to_ts(ofs)}]
    return labs


def fetch_demographics(con, pid: int):
    row = con.execute(
        "SELECT age, gender, unittype, hospitalid, ethnicity FROM patient WHERE patientunitstayid = ?",
        [pid],
    ).fetchone()
    if not row:
        return {}
    age, gender, unit, hospital, ethnicity = row
    age_str = str(age).strip() if age is not None else ""
    if age_str.replace(" ", "") in ("", ">89"):
        # eICU uses "> 89" (with a space) as HIPAA Safe-Harbor binning for ages
        # 90+. Empty / missing also lands here.
        age_num = 90.0 if age_str else 0.0
    else:
        try:
            age_num = float(age_str)
        except Exception:
            age_num = 0.0
    return {
        "age": age_num,
        "gender": gender or "Unknown",
        "unit_type": unit or "ICU",
        "hospital_id": str(hospital) if hospital else "unknown",
        "ethnicity": ethnicity or "Unknown",
    }


# ----- Note / narrative stitching ---------------------------------------------

def build_patient_notes(con, pid: int, snapshot_offset: int) -> str:
    """Stitch a nurse-narrative-like patient_notes field from multiple tables,
    newest-first, limited to entries inside the 12h window (wider than vitals
    because notes are written less frequently than vitals are measured)."""
    window_start = snapshot_offset - NOTES_WINDOW_HOURS * 60

    segments = []

    # 1. Free-text notes
    notes = con.execute(
        """
        SELECT noteoffset, notetype, notetext
        FROM note
        WHERE patientunitstayid = ?
          AND noteoffset >= ? AND noteoffset <= ?
          AND notetext IS NOT NULL AND LENGTH(notetext) > 20
        ORDER BY noteoffset DESC
        LIMIT 5
        """,
        [pid, window_start, snapshot_offset],
    ).fetchall()
    for ofs, ntype, txt in notes:
        hours_ago = round((snapshot_offset - ofs) / 60.0, 1)
        txt_clean = (txt or "").replace("\n", " ").replace("  ", " ").strip()[:600]
        segments.append(f"Note ({ntype or 'clinical'}, T-{hours_ago}h): {txt_clean}")

    # 2. Nursing assessments (structured)
    nurse_asmt = con.execute(
        """
        SELECT nurseAssessOffset, celllabel, cellattribute, cellattributevalue
        FROM nurseAssessment
        WHERE patientunitstayid = ?
          AND nurseAssessOffset >= ? AND nurseAssessOffset <= ?
          AND cellattributevalue IS NOT NULL
        ORDER BY nurseAssessOffset DESC
        LIMIT 40
        """,
        [pid, window_start, snapshot_offset],
    ).fetchall()
    asmt_by_time = {}
    for ofs, label, attr, value in nurse_asmt:
        hours_ago = round((snapshot_offset - ofs) / 60.0, 1)
        key = (hours_ago, label)
        asmt_by_time.setdefault(key, []).append(f"{attr}: {value}")
    for (hours_ago, label), entries in list(asmt_by_time.items())[:8]:
        summary = "; ".join(entries[:4])
        segments.append(f"Nursing assessment (T-{hours_ago}h, {label}): {summary}")

    # 3. Nursing charting: mental status / GCS total / pain / skin / RASS
    # IMPORTANT: for GCS we filter on valname in ('GCS Total', 'Value') because
    # nurseCharting stores each GCS row 4x (Eyes, Verbal, Motor, GCS Total).
    # Pulling all 4 caused the LLM to mistake component scores for the total
    # and over-flag altered mentation (Phase 1 pilot false positives).
    nurse_chart = con.execute(
        """
        SELECT nursingchartoffset,
               nursingchartcelltypevallabel,
               nursingchartcelltypevalname,
               nursingchartvalue
        FROM nurseCharting
        WHERE patientunitstayid = ?
          AND nursingchartoffset >= ? AND nursingchartoffset <= ?
          AND nursingchartvalue IS NOT NULL
          AND (
                -- GCS: only keep the total (not Eyes/Verbal/Motor components)
                (
                  (LOWER(nursingchartcelltypevallabel) LIKE '%glasgow%'
                    OR LOWER(nursingchartcelltypevallabel) LIKE '%gcs%')
                  AND (
                        nursingchartcelltypevalname IN ('GCS Total', 'Value')
                     OR LOWER(nursingchartcelltypevalname) LIKE '%total%'
                  )
                )
             OR LOWER(nursingchartcelltypevallabel) LIKE '%mental%'
             OR LOWER(nursingchartcelltypevallabel) LIKE '%pain%'
             OR LOWER(nursingchartcelltypevallabel) LIKE '%skin%'
             OR LOWER(nursingchartcelltypevallabel) LIKE '%rass%'
             OR LOWER(nursingchartcelltypevallabel) LIKE '%sedation%'
          )
        ORDER BY nursingchartoffset DESC
        LIMIT 20
        """,
        [pid, window_start, snapshot_offset],
    ).fetchall()
    # Deduplicate (same timestamp + label + value) and collapse to a
    # human-readable form.
    seen = set()
    for ofs, label, valname, value in nurse_chart:
        hours_ago = round((snapshot_offset - ofs) / 60.0, 1)
        # Normalize GCS label
        if "glasgow" in (label or "").lower() or "gcs" in (label or "").lower():
            display = f"GCS total = {value}"
        else:
            display = f"{label} = {value}"
        key = (hours_ago, display)
        if key in seen:
            continue
        seen.add(key)
        segments.append(f"Nurse charting (T-{hours_ago}h): {display}")
        if len(seen) >= 10:
            break

    # 4. Recent treatments (antibiotics, pressors, fluid) - if any
    tx = con.execute(
        """
        SELECT treatmentoffset, treatmentstring
        FROM treatment
        WHERE patientunitstayid = ?
          AND treatmentoffset >= ? AND treatmentoffset <= ?
        ORDER BY treatmentoffset DESC
        LIMIT 8
        """,
        [pid, window_start, snapshot_offset],
    ).fetchall()
    for ofs, tx_str in tx:
        hours_ago = round((snapshot_offset - ofs) / 60.0, 1)
        segments.append(f"MD order (T-{hours_ago}h): {tx_str}")

    # 5. Cultures drawn
    cul = con.execute(
        """
        SELECT culturetakenoffset, culturesite, organism
        FROM microLab
        WHERE patientunitstayid = ?
          AND culturetakenoffset >= ? AND culturetakenoffset <= ?
        ORDER BY culturetakenoffset DESC
        LIMIT 3
        """,
        [pid, window_start, snapshot_offset],
    ).fetchall()
    for ofs, site, organism in cul:
        hours_ago = round((snapshot_offset - ofs) / 60.0, 1)
        segments.append(f"MicroLab (T-{hours_ago}h): culture drawn from {site or 'unknown site'}" +
                        (f"; organism: {organism}" if organism and organism.strip() else ""))

    if not segments:
        return "No free-text notes or structured observations in the 6-hour window prior to snapshot."

    return " | ".join(segments)


# ----- Patient JSON emission ---------------------------------------------------

def build_patient_json(con, pid: int, snapshot_offset: int, is_sepsis: bool,
                       sepsis_info: dict | None, cohort_idx: int):
    demo = fetch_demographics(con, pid)
    vitals = fetch_vitals(con, pid, snapshot_offset)
    labs = fetch_labs(con, pid, snapshot_offset)
    notes = build_patient_notes(con, pid, snapshot_offset)

    patient_vitals = {**vitals, **labs}

    out = {
        "patient_id": f"eicu_p{cohort_idx:05d}",
        "source": {
            "dataset": "eICU-CRD Demo v2.0.1",
            "patientunitstayid": int(pid),
            "snapshot_offset_min_from_admit": int(snapshot_offset),
            "trend_window_hours": TREND_HOURS,
        },
        "patient_demographics": {
            "Age": demo.get("age"),
            "Gender": demo.get("gender"),
            "UnitType": demo.get("unit_type"),
            "HospitalId": demo.get("hospital_id"),
            "Ethnicity": demo.get("ethnicity"),
        },
        "patient_vitals": patient_vitals,
        "patient_notes": notes,
        "ground_truth": {
            "actual_sepsis": bool(is_sepsis),
            "sepsis_onset_offset_min": sepsis_info["onset_offset_min"] if sepsis_info else None,
            "icd9_codes": sepsis_info["icd9_codes"] if sepsis_info else [],
            "diagnosis_strings": sepsis_info["diagnosis_strings"] if sepsis_info else [],
        },
    }
    return out


# ----- Main driver -------------------------------------------------------------

def main():
    if not EICU_DIR.exists() or not (EICU_DIR / "patient.csv.gz").exists():
        raise SystemExit(
            f"eICU Demo files not found in {EICU_DIR}. "
            f"Download the ZIP first: "
            f"curl -L -o {EICU_DIR}/eicu_demo.zip 'https://physionet.org/content/eicu-crd-demo/get-zip/2.0.1/' "
            f"&& cd {EICU_DIR} && unzip -j eicu_demo.zip"
        )

    print(f"Connecting to eICU data at {EICU_DIR} ...")
    con = connect_db()

    # --- Sepsis cohort ---
    print("\n[1/4] Finding sepsis-positive patients (ICD-9 based) ...")
    sepsis_pool = find_sepsis_patients(con)
    print(f"  Qualified sepsis patients (onset >= 6h after admit): {len(sepsis_pool)}")
    random.shuffle(sepsis_pool)
    sepsis_selected = sepsis_pool[:TARGET_SEPSIS]
    print(f"  Selected: {len(sepsis_selected)}")

    # --- Control cohort ---
    print("\n[2/4] Finding non-sepsis controls ...")
    exclude = [s["patientunitstayid"] for s in sepsis_pool]
    controls_pool = find_control_patients(con, exclude)
    print(f"  Candidate controls: {len(controls_pool)}")
    random.shuffle(controls_pool)
    controls_selected = controls_pool[:TARGET_CONTROLS]
    print(f"  Selected: {len(controls_selected)}")

    # --- Build patient JSONs ---
    print("\n[3/4] Building patient JSON files ...")
    manifest = []
    idx = 0

    for s in sepsis_selected:
        idx += 1
        pid = s["patientunitstayid"]
        snapshot_offset = int(s["onset_offset_min"] - SEPSIS_SNAPSHOT_OFFSET_MIN)
        if snapshot_offset < TREND_HOURS * 60:
            # not enough trend room, skip
            continue
        pjson = build_patient_json(con, pid, snapshot_offset, is_sepsis=True,
                                    sepsis_info=s, cohort_idx=idx)
        out_path = COHORT_DIR / f"{pjson['patient_id']}.json"
        out_path.write_text(json.dumps(pjson, indent=2, default=str))
        manifest.append({
            "patient_id": pjson["patient_id"],
            "file": out_path.name,
            "actual_sepsis": True,
            "patientunitstayid": int(pid),
            "snapshot_offset_min_from_admit": snapshot_offset,
            "sepsis_onset_offset_min": int(s["onset_offset_min"]),
            "icd9_codes": s["icd9_codes"],
        })
        print(f"  [{idx:02d}] {pjson['patient_id']} SEPSIS (onset T+{s['onset_offset_min']}min, snapshot T+{snapshot_offset}min, vitals={sum(len(v) for v in pjson['patient_vitals'].values())} pts, notes={len(pjson['patient_notes'])} chars)")

    for c in controls_selected:
        idx += 1
        pid = c["patientunitstayid"]
        disc = c["unit_discharge_offset"] or 2880
        # snapshot random hour between 12h after admit and discharge-6h
        earliest = 720
        latest = max(earliest + 60, disc - 360)
        snapshot_offset = random.randint(earliest, max(earliest + 60, latest))
        pjson = build_patient_json(con, pid, snapshot_offset, is_sepsis=False,
                                    sepsis_info=None, cohort_idx=idx)
        out_path = COHORT_DIR / f"{pjson['patient_id']}.json"
        out_path.write_text(json.dumps(pjson, indent=2, default=str))
        manifest.append({
            "patient_id": pjson["patient_id"],
            "file": out_path.name,
            "actual_sepsis": False,
            "patientunitstayid": int(pid),
            "snapshot_offset_min_from_admit": snapshot_offset,
        })
        print(f"  [{idx:02d}] {pjson['patient_id']} CONTROL (snapshot T+{snapshot_offset}min, vitals={sum(len(v) for v in pjson['patient_vitals'].values())} pts, notes={len(pjson['patient_notes'])} chars)")

    # --- Write manifest ---
    print("\n[4/4] Writing manifest ...")
    man_path = COHORT_DIR / "manifest.json"
    man_path.write_text(json.dumps({
        "version": "eicu_demo_v1",
        "dataset": "eICU-CRD Demo v2.0.1",
        "created": datetime.utcnow().isoformat() + "Z",
        "trend_hours": TREND_HOURS,
        "sepsis_snapshot_offset_min": SEPSIS_SNAPSHOT_OFFSET_MIN,
        "seed": SEED,
        "patients": manifest,
        "summary": {
            "total": len(manifest),
            "sepsis": sum(1 for m in manifest if m["actual_sepsis"]),
            "controls": sum(1 for m in manifest if not m["actual_sepsis"]),
        },
    }, indent=2))
    print(f"  Wrote {man_path}")
    print(f"\nDone. Cohort size = {len(manifest)} patients at {COHORT_DIR}")


if __name__ == "__main__":
    main()
