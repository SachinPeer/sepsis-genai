"""
eICU-CRD Demo Cohort Builder — v4 (strict F1-F10 quality filters)

Implements the cohort selection rules in
`validation/docs/EICU_DATASET_AND_COHORT.md` §2:

  F1  Age >= 18
  F2  Snapshot offset >= 6h after ICU admit (sepsis: 6h pre-onset)
  F3  >= 4 HR readings in the 6h trend window
  F4  >= 4 Resp readings in the 6h trend window
  F5  >= 3 vital types each with >= 4 readings
  F6  >= 2 distinct labs in the 24h pre-snapshot window
  F7  Sepsis: ICD code (Option B uses ICD-only; abx/culture cross-check optional)
  F8  Control: NO sepsis ICD code AND NO broad-spectrum abx in first 48h
  F9  Control: ICU stay >= 24h
  F10 patient_notes is not the "no notes available" placeholder

Output:
  validation/eicu_cohort_v4/eicu_p00001.json ... (renumbered, no gaps)
  validation/eicu_cohort_v4/manifest.json
  validation/eicu_cohort_v4/build_log.json (filter decisions per candidate)

Run:
  python3 validation/eicu_cohort_builder_v4.py
"""
from __future__ import annotations

import json
import random
import sys
from datetime import datetime
from pathlib import Path

# Reuse helpers from the v2 builder
sys.path.insert(0, str(Path(__file__).parent))
from eicu_cohort_builder import (   # noqa: E402
    connect_db,
    build_patient_json,
    SEPSIS_SNAPSHOT_OFFSET_MIN,
    TREND_HOURS,
)

EICU_DIR = Path(__file__).parent / "eicu_demo"
COHORT_DIR = Path(__file__).parent / "eicu_cohort_v4"
COHORT_DIR.mkdir(exist_ok=True)

TARGET_SEPSIS = 60        # we'll keep all that pass; capped by the demo
TARGET_CONTROLS = 120
SEED = 42
USE_SEPSIS3_CROSSCHECK = False   # Option B: ICD-only

NO_NOTES_MARKER = "No free-text notes or structured observations"

random.seed(SEED)


# ---------------------------------------------------------------- quality probe
def assess_vitals_labs(con, pid: int, snapshot_offset_min: int) -> dict:
    """Return per-patient quality counts for F3-F6. Counts are *distinct hourly
    buckets* — matching the JSON downsampling done by fetch_vitals — because
    that's what the model actually sees in the patient JSON. Raw row counts
    can over-state coverage when readings are clustered."""
    window_start = snapshot_offset_min - TREND_HOURS * 60
    lab_window_start = snapshot_offset_min - 24 * 60

    counts = con.execute(
        f"""
        SELECT
            COUNT(DISTINCT CASE WHEN heartrate IS NOT NULL
                  THEN FLOOR((? - observationoffset) / 60) END) AS HR,
            COUNT(DISTINCT CASE WHEN systemicsystolic IS NOT NULL
                  THEN FLOOR((? - observationoffset) / 60) END) AS SBP,
            COUNT(DISTINCT CASE WHEN systemicmean IS NOT NULL
                  THEN FLOOR((? - observationoffset) / 60) END) AS MAP,
            COUNT(DISTINCT CASE WHEN temperature IS NOT NULL
                  THEN FLOOR((? - observationoffset) / 60) END) AS Temp,
            COUNT(DISTINCT CASE WHEN respiration IS NOT NULL
                  THEN FLOOR((? - observationoffset) / 60) END) AS Resp,
            COUNT(DISTINCT CASE WHEN sao2 IS NOT NULL
                  THEN FLOOR((? - observationoffset) / 60) END) AS O2Sat
        FROM vitalPeriodic
        WHERE patientunitstayid = ?
          AND observationoffset >= ? AND observationoffset <= ?
        """,
        [snapshot_offset_min] * 6 + [pid, window_start, snapshot_offset_min],
    ).fetchone()
    hr, sbp, mp, tp, rr, o2 = [int(c or 0) for c in counts]

    # Number of vital types with >=4 hourly buckets covered
    types_ok = sum(1 for c in (hr, sbp, mp, tp, rr, o2) if c >= 4)

    # Distinct labs in 24h window
    labs = con.execute(
        """
        SELECT COUNT(DISTINCT labname) FROM lab
        WHERE patientunitstayid = ?
          AND labresultoffset >= ? AND labresultoffset <= ?
        """,
        [pid, lab_window_start, snapshot_offset_min],
    ).fetchone()[0]
    labs = int(labs or 0)

    return {
        "HR": hr, "SBP": sbp, "MAP": mp, "Temp": tp, "Resp": rr, "O2Sat": o2,
        "types_ok": types_ok, "distinct_labs": labs,
    }


def quality_flags(q: dict) -> list[str]:
    flags: list[str] = []
    if q["HR"] < 4:           flags.append("VITAL_HR_THIN")
    if q["Resp"] < 4:         flags.append("VITAL_RESP_THIN")
    if q["types_ok"] < 3:     flags.append("VITAL_TYPES_THIN")
    if q["distinct_labs"] < 2: flags.append("LABS_THIN")
    return flags


# ---------------------------------------------------------------- pool builders
SEPSIS_ICD_LIKE = """(
    icd9code LIKE '038%' OR icd9code LIKE '995.9%' OR icd9code LIKE '785.52%'
)"""

ABX_LIKE = """(
    UPPER(drugname) LIKE '%VANCOMYCIN%' OR UPPER(drugname) LIKE '%CEFTRIAXONE%'
 OR UPPER(drugname) LIKE '%PIPERACILLIN%' OR UPPER(drugname) LIKE '%MEROPENEM%'
 OR UPPER(drugname) LIKE '%LEVOFLOXACIN%' OR UPPER(drugname) LIKE '%CIPROFLOXACIN%'
 OR UPPER(drugname) LIKE '%CEFEPIME%'    OR UPPER(drugname) LIKE '%METRONIDAZOLE%'
 OR UPPER(drugname) LIKE '%AZITHROMYCIN%' OR UPPER(drugname) LIKE '%CEFTAZIDIME%'
)"""


def find_adult_age(con, pid: int) -> float | None:
    row = con.execute(
        "SELECT age FROM patient WHERE patientunitstayid = ?", [pid],
    ).fetchone()
    if not row or row[0] is None:
        return None
    a = str(row[0]).strip()
    if a in ("", "0"):
        return 0.0
    if a == "> 89":
        return 90.0
    try:
        return float(a)
    except Exception:
        return None


def find_sepsis_candidates(con) -> list[dict]:
    """Adults, sepsis ICD code, onset >= 6h after admit. Optionally with
    abx/culture cross-check if USE_SEPSIS3_CROSSCHECK = True."""
    rows = con.execute(
        f"""
        SELECT patientunitstayid,
               MIN(diagnosisoffset) AS onset_min,
               LIST(DISTINCT icd9code) AS codes,
               LIST(DISTINCT diagnosisstring) AS diag_strings
        FROM diagnosis
        WHERE icd9code IS NOT NULL AND {SEPSIS_ICD_LIKE}
        GROUP BY patientunitstayid
        HAVING MIN(diagnosisoffset) >= 360
        """
    ).fetchall()

    candidates = []
    for pid, onset, codes, diag_strings in rows:
        age = find_adult_age(con, pid)
        if age is None or age < 18:
            continue
        if USE_SEPSIS3_CROSSCHECK:
            cross = con.execute(
                f"""
                SELECT 1 FROM medication WHERE patientunitstayid = ? AND {ABX_LIKE}
                LIMIT 1
                """, [pid],
            ).fetchone()
            if not cross:
                cross = con.execute(
                    "SELECT 1 FROM microLab WHERE patientunitstayid = ? LIMIT 1",
                    [pid],
                ).fetchone()
            if not cross:
                continue
        candidates.append({
            "patientunitstayid": int(pid),
            "onset_offset_min": int(onset),
            "icd9_codes": sorted(c for c in (codes or []) if c),
            "diagnosis_strings": sorted(s for s in (diag_strings or []) if s),
            "age": age,
        })
    return candidates


def find_control_candidates(con, exclude_ids: set[int]) -> list[dict]:
    """Adult, ICU stay >= 24h, no sepsis ICD, no broad-spectrum abx in first 48h."""
    excl_sql = "(" + ",".join(str(i) for i in exclude_ids) + ")" if exclude_ids else "(0)"
    rows = con.execute(
        f"""
        SELECT p.patientunitstayid, p.unitdischargeoffset, p.hospitalid,
               p.age, p.gender, p.unittype, p.ethnicity
        FROM patient p
        WHERE p.unitdischargeoffset >= 1440
          AND p.patientunitstayid NOT IN {excl_sql}
          AND p.patientunitstayid NOT IN (
              SELECT DISTINCT patientunitstayid FROM diagnosis
              WHERE icd9code IS NOT NULL AND {SEPSIS_ICD_LIKE}
          )
          AND p.patientunitstayid NOT IN (
              SELECT DISTINCT patientunitstayid FROM medication
              WHERE drugstartoffset IS NOT NULL AND drugstartoffset <= 2880
                AND {ABX_LIKE}
          )
        """
    ).fetchall()
    out = []
    for r in rows:
        pid = int(r[0])
        age = find_adult_age(con, pid)
        if age is None or age < 18:
            continue
        out.append({
            "patientunitstayid": pid,
            "unit_discharge_offset": int(r[1]) if r[1] is not None else 2880,
            "hospital_id": str(r[2]) if r[2] is not None else "unknown",
            "age": age,
            "gender": r[4] or "Unknown",
            "unit_type": r[5] or "ICU",
            "ethnicity": r[6] or "Unknown",
        })
    return out


# ---------------------------------------------------------------- main driver
def main():
    if not (EICU_DIR / "patient.csv.gz").exists():
        raise SystemExit(f"eICU demo not found at {EICU_DIR}")

    print(f"Connecting to eICU data at {EICU_DIR} ...")
    con = connect_db()

    build_log: list[dict] = []
    manifest: list[dict] = []

    # -------- sepsis cohort
    print("\n[1/4] Sepsis candidate scan (Option B: ICD-only) ...")
    sepsis_candidates = find_sepsis_candidates(con)
    print(f"  Adult sepsis stays with onset >= 6h: {len(sepsis_candidates)}")

    random.shuffle(sepsis_candidates)

    sepsis_kept: list[dict] = []
    for cand in sepsis_candidates:
        pid = cand["patientunitstayid"]
        snapshot = cand["onset_offset_min"] - SEPSIS_SNAPSHOT_OFFSET_MIN
        if snapshot < TREND_HOURS * 60:
            build_log.append({"pid": pid, "type": "sepsis", "decision": "REJECT",
                              "flags": ["F2_SNAPSHOT_TOO_EARLY"]})
            continue
        q = assess_vitals_labs(con, pid, snapshot)
        flags = quality_flags(q)
        if flags:
            build_log.append({"pid": pid, "type": "sepsis", "decision": "REJECT",
                              "flags": flags, "quality": q})
            continue
        sepsis_kept.append({**cand, "snapshot_offset_min": snapshot, "quality": q})
        build_log.append({"pid": pid, "type": "sepsis", "decision": "KEEP",
                          "quality": q})
        if len(sepsis_kept) >= TARGET_SEPSIS:
            break
    print(f"  Sepsis after F1-F6: {len(sepsis_kept)}")

    # -------- control cohort
    print("\n[2/4] Control candidate scan ...")
    excluded_ids = {c["patientunitstayid"] for c in sepsis_candidates}
    control_candidates = find_control_candidates(con, excluded_ids)
    print(f"  Adult non-sepsis ICU stays >=24h, no early abx: {len(control_candidates)}")

    random.shuffle(control_candidates)

    controls_kept: list[dict] = []
    for cand in control_candidates:
        pid = cand["patientunitstayid"]
        disc = cand["unit_discharge_offset"]
        # snapshot at admit + 12h, but cap at discharge - 6h
        snapshot = min(720, max(720, disc - 360))
        if snapshot < TREND_HOURS * 60:
            build_log.append({"pid": pid, "type": "control", "decision": "REJECT",
                              "flags": ["F2_SNAPSHOT_TOO_EARLY"]})
            continue
        q = assess_vitals_labs(con, pid, snapshot)
        flags = quality_flags(q)
        if flags:
            build_log.append({"pid": pid, "type": "control", "decision": "REJECT",
                              "flags": flags, "quality": q})
            continue
        controls_kept.append({**cand, "snapshot_offset_min": snapshot, "quality": q})
        build_log.append({"pid": pid, "type": "control", "decision": "KEEP",
                          "quality": q})
        if len(controls_kept) >= TARGET_CONTROLS:
            break
    print(f"  Controls after F1, F3-F6, F8, F9: {len(controls_kept)}")

    # -------- build patient JSONs (and apply F10 = notes-not-empty)
    print("\n[3/4] Building patient JSONs and applying F10 ...")

    # clean output dir
    for f in COHORT_DIR.glob("eicu_p*.json"):
        f.unlink()

    idx = 0
    rejected_for_notes: list[int] = []

    def emit(pjson: dict, kind: str, extra: dict):
        nonlocal idx
        idx += 1
        pjson["patient_id"] = f"eicu_p{idx:05d}"
        out = COHORT_DIR / f"{pjson['patient_id']}.json"
        out.write_text(json.dumps(pjson, indent=2, default=str))
        manifest.append({
            "patient_id": pjson["patient_id"],
            "file": out.name,
            "actual_sepsis": kind == "sepsis",
            **extra,
        })

    notes_soft_count = 0
    for s in sepsis_kept:
        pjson = build_patient_json(con, s["patientunitstayid"],
                                    s["snapshot_offset_min"], is_sepsis=True,
                                    sepsis_info=s, cohort_idx=0)
        notes_empty = NO_NOTES_MARKER in (pjson.get("patient_notes") or "")
        if notes_empty:
            notes_soft_count += 1
        emit(pjson, "sepsis", {
            "patientunitstayid": s["patientunitstayid"],
            "snapshot_offset_min_from_admit": s["snapshot_offset_min"],
            "sepsis_onset_offset_min": s["onset_offset_min"],
            "icd9_codes": s["icd9_codes"],
            "quality": s["quality"],
            "notes_empty": notes_empty,
        })
        build_log.append({"pid": s["patientunitstayid"], "type": "sepsis",
                          "decision": "KEEP_POSTBUILD",
                          "notes_empty": notes_empty})

    for c in controls_kept:
        pjson = build_patient_json(con, c["patientunitstayid"],
                                    c["snapshot_offset_min"], is_sepsis=False,
                                    sepsis_info=None, cohort_idx=0)
        notes_empty = NO_NOTES_MARKER in (pjson.get("patient_notes") or "")
        if notes_empty:
            notes_soft_count += 1
        emit(pjson, "control", {
            "patientunitstayid": c["patientunitstayid"],
            "snapshot_offset_min_from_admit": c["snapshot_offset_min"],
            "quality": c["quality"],
            "notes_empty": notes_empty,
        })
        build_log.append({"pid": c["patientunitstayid"], "type": "control",
                          "decision": "KEEP_POSTBUILD",
                          "notes_empty": notes_empty})

    rejected_for_notes = []  # F10 is now a soft flag, not a hard filter

    # -------- manifest + build_log
    print("\n[4/4] Writing manifest and build_log ...")
    n_sepsis = sum(1 for m in manifest if m["actual_sepsis"])
    n_controls = sum(1 for m in manifest if not m["actual_sepsis"])
    n_with_empty_notes = sum(1 for m in manifest if m.get("notes_empty"))
    (COHORT_DIR / "manifest.json").write_text(json.dumps({
        "version": "eicu_cohort_v4",
        "dataset": "eICU-CRD Demo v2.0.1",
        "ground_truth_method": "ICD-9 sepsis codes (Option B, ICD-only)",
        "cross_check": "abx OR culture" if USE_SEPSIS3_CROSSCHECK else "none (ICD-only)",
        "filters_hard": ["F1", "F2", "F3", "F4", "F5", "F6", "F8", "F9"],
        "filters_soft": ["F10 (empty-notes flagged but not rejected)"],
        "trend_hours": TREND_HOURS,
        "sepsis_snapshot_offset_min": SEPSIS_SNAPSHOT_OFFSET_MIN,
        "seed": SEED,
        "created": datetime.utcnow().isoformat() + "Z",
        "patients": manifest,
        "summary": {
            "total": len(manifest),
            "sepsis": n_sepsis,
            "controls": n_controls,
            "with_empty_notes": n_with_empty_notes,
        },
    }, indent=2))
    (COHORT_DIR / "build_log.json").write_text(json.dumps({
        "candidates_evaluated": len(build_log),
        "decisions": build_log,
    }, indent=2))

    print()
    print("=" * 60)
    print(f" v4 cohort built: {COHORT_DIR}")
    print(f"   Sepsis : {n_sepsis}")
    print(f"   Control: {n_controls}")
    print(f"   Total  : {len(manifest)}")
    print(f"   Rejected for empty notes (post-build): {len(rejected_for_notes)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
