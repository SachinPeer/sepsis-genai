"""
Probe the eICU demo to see how many patients qualify under STRICT filters.
Used as a pre-flight check before scaling the cohort to 340.

Strict filters (must satisfy ALL):
  - Age >= 18 and Age != 0 (no neonates / pediatric)
  - Snapshot offset >= 6h after ICU admit (so we have a 6h trend window)
  - >= 4 HR readings in the 6h pre-snapshot window
  - >= 4 Resp readings in the 6h pre-snapshot window
  - >= 3 vital types each with >= 4 readings (HR, SBP, MAP, Temp, Resp, O2Sat)
  - >= 2 labs in the 24h pre-snapshot window (any of WBC, Lactate, Creat,
    Platelets, Glucose, BUN, pH, HCO3, PaCO2, Hgb)
  - For sepsis: ICD-9 code present AND broad-spectrum abx ordered within
    +/- 24h of diagnosis (Sepsis-3 style cross-check)
  - For controls: NO sepsis ICD-9 codes AND NO broad-spectrum abx in first 48h

For sepsis patients, snapshot is at (sepsis_onset - 360 minutes).
For controls, snapshot is taken at (admit + 12h) — far enough in to have a
trend window but not so deep that controls are systematically older than
sepsis snapshots.
"""
from __future__ import annotations

import duckdb
from pathlib import Path

EICU_DIR = Path(__file__).parent / "eicu_demo"

VITAL_COLS = {
    "HR": "heartrate",
    "SBP": "systemicsystolic",
    "MAP": "systemicmean",
    "Temp": "temperature",
    "Resp": "respiration",
    "O2Sat": "sao2",
}

ABX_LIKE = """(
    UPPER(drugname) LIKE '%VANCOMYCIN%' OR UPPER(drugname) LIKE '%CEFTRIAXONE%'
 OR UPPER(drugname) LIKE '%PIPERACILLIN%' OR UPPER(drugname) LIKE '%MEROPENEM%'
 OR UPPER(drugname) LIKE '%LEVOFLOXACIN%' OR UPPER(drugname) LIKE '%CIPROFLOXACIN%'
 OR UPPER(drugname) LIKE '%CEFEPIME%' OR UPPER(drugname) LIKE '%METRONIDAZOLE%'
 OR UPPER(drugname) LIKE '%AZITHROMYCIN%' OR UPPER(drugname) LIKE '%CEFTAZIDIME%'
)"""


def main():
    con = duckdb.connect()
    for name in ("patient", "diagnosis", "vitalPeriodic", "lab",
                 "medication"):
        con.execute(
            f"CREATE TABLE {name} AS SELECT * FROM "
            f"read_csv_auto('{EICU_DIR/(name + '.csv.gz')}', sample_size=-1, all_varchar=true)"
        )

    # Cast common columns
    con.execute("CREATE OR REPLACE VIEW pat AS SELECT patientunitstayid, hospitalid, "
                "TRY_CAST(unitdischargeoffset AS DOUBLE) AS los_min, "
                "CASE WHEN age = '> 89' THEN 90 ELSE TRY_CAST(age AS DOUBLE) END AS age_num "
                "FROM patient")

    # ---- Sepsis pool ----
    con.execute("""
    CREATE OR REPLACE TABLE sepsis_idx AS
    SELECT patientunitstayid,
           MIN(TRY_CAST(diagnosisoffset AS DOUBLE)) AS sepsis_onset_min
    FROM diagnosis
    WHERE icd9code LIKE '038%' OR icd9code LIKE '995.9%' OR icd9code LIKE '785.52%'
    GROUP BY 1
    """)
    n_sepsis_raw = con.execute("SELECT COUNT(*) FROM sepsis_idx").fetchone()[0]

    # Filter: onset >= 360 min so we have the 6h trend window
    n_sepsis_window = con.execute(
        "SELECT COUNT(*) FROM sepsis_idx WHERE sepsis_onset_min >= 360"
    ).fetchone()[0]

    # Adult only
    n_sepsis_adult = con.execute("""
        SELECT COUNT(*)
        FROM sepsis_idx s
        JOIN pat p USING (patientunitstayid)
        WHERE s.sepsis_onset_min >= 360 AND p.age_num >= 18
    """).fetchone()[0]

    # Add abx cross-check (Sepsis-3 style)
    n_sepsis_with_abx = con.execute(f"""
        SELECT COUNT(*)
        FROM sepsis_idx s
        JOIN pat p USING (patientunitstayid)
        WHERE s.sepsis_onset_min >= 360 AND p.age_num >= 18
          AND EXISTS (
            SELECT 1 FROM medication m
            WHERE m.patientunitstayid = s.patientunitstayid
              AND {ABX_LIKE}
          )
    """).fetchone()[0]

    # Vital coverage in the 6h pre-onset window
    print("Computing vital coverage on sepsis cohort...")
    con.execute("""
    CREATE OR REPLACE TABLE sepsis_eligible AS
    SELECT s.patientunitstayid, s.sepsis_onset_min,
           p.hospitalid, p.age_num
    FROM sepsis_idx s
    JOIN pat p USING (patientunitstayid)
    WHERE s.sepsis_onset_min >= 360 AND p.age_num >= 18
      AND EXISTS (
        SELECT 1 FROM medication m
        WHERE m.patientunitstayid = s.patientunitstayid AND %s
      )
    """ % ABX_LIKE)

    # For each sepsis patient count distinct hour-buckets per vital in window
    sepsis_vital_q = """
    SELECT s.patientunitstayid,
        SUM(CASE WHEN v.heartrate         IS NOT NULL THEN 1 ELSE 0 END) AS hr_n,
        SUM(CASE WHEN v.systemicsystolic  IS NOT NULL THEN 1 ELSE 0 END) AS sbp_n,
        SUM(CASE WHEN v.systemicmean      IS NOT NULL THEN 1 ELSE 0 END) AS map_n,
        SUM(CASE WHEN v.temperature       IS NOT NULL THEN 1 ELSE 0 END) AS temp_n,
        SUM(CASE WHEN v.respiration       IS NOT NULL THEN 1 ELSE 0 END) AS rr_n,
        SUM(CASE WHEN v.sao2              IS NOT NULL THEN 1 ELSE 0 END) AS o2_n
    FROM sepsis_eligible s
    JOIN vitalPeriodic v
      ON v.patientunitstayid = s.patientunitstayid
     AND TRY_CAST(v.observationoffset AS DOUBLE)
         BETWEEN s.sepsis_onset_min - 360 AND s.sepsis_onset_min
    GROUP BY 1
    """
    con.execute(f"CREATE OR REPLACE TABLE sepsis_vital AS {sepsis_vital_q}")

    n_sepsis_vital_ok = con.execute("""
        SELECT COUNT(*) FROM sepsis_vital
        WHERE hr_n >= 4 AND rr_n >= 4
          AND ((hr_n>=4)::INT + (sbp_n>=4)::INT + (map_n>=4)::INT
             + (temp_n>=4)::INT + (rr_n>=4)::INT + (o2_n>=4)::INT) >= 3
    """).fetchone()[0]

    # Labs check
    n_sepsis_lab_ok = con.execute("""
    SELECT COUNT(*)
    FROM sepsis_vital sv
    JOIN sepsis_eligible se USING (patientunitstayid)
    WHERE sv.hr_n>=4 AND sv.rr_n>=4
      AND ((sv.hr_n>=4)::INT + (sv.sbp_n>=4)::INT + (sv.map_n>=4)::INT
         + (sv.temp_n>=4)::INT + (sv.rr_n>=4)::INT + (sv.o2_n>=4)::INT) >= 3
      AND (
        SELECT COUNT(DISTINCT labname)
        FROM lab l
        WHERE l.patientunitstayid = sv.patientunitstayid
          AND TRY_CAST(l.labresultoffset AS DOUBLE)
              BETWEEN se.sepsis_onset_min - 1440 AND se.sepsis_onset_min
      ) >= 2
    """).fetchone()[0]

    print()
    print("=" * 70)
    print("  SEPSIS POOL (demo)")
    print("=" * 70)
    print(f"  ICD-coded sepsis stays                         : {n_sepsis_raw:>4}")
    print(f"  + onset >= 6h after admit                      : {n_sepsis_window:>4}")
    print(f"  + adult only (age >= 18)                       : {n_sepsis_adult:>4}")
    print(f"  + Sepsis-3 cross-check (abx ordered)           : {n_sepsis_with_abx:>4}")
    print(f"  + vitals OK (HR>=4, RR>=4, 3+ types)           : {n_sepsis_vital_ok:>4}")
    print(f"  + labs OK (>= 2 labs in 24h pre-onset)         : {n_sepsis_lab_ok:>4}  <-- usable sepsis pool")

    # ---- Control pool ----
    # Snapshot for controls = admit + 12h ; we apply same vital/lab logic
    print()
    print("Computing vital coverage on control cohort...")
    con.execute("""
    CREATE OR REPLACE TABLE control_eligible AS
    SELECT p.patientunitstayid, p.hospitalid, p.age_num,
           720.0 AS snapshot_min  -- 12h after admit
    FROM pat p
    WHERE p.age_num >= 18
      AND p.los_min >= 1440        -- ICU stay >= 24h
      AND p.patientunitstayid NOT IN (SELECT patientunitstayid FROM sepsis_idx)
      AND p.patientunitstayid NOT IN (
        SELECT m.patientunitstayid
        FROM medication m
        WHERE TRY_CAST(m.drugstartoffset AS DOUBLE) <= 2880
          AND %s
      )
    """ % ABX_LIKE)

    n_ctrl_raw = con.execute("SELECT COUNT(*) FROM control_eligible").fetchone()[0]

    con.execute("""
    CREATE OR REPLACE TABLE control_vital AS
    SELECT c.patientunitstayid,
        SUM(CASE WHEN v.heartrate         IS NOT NULL THEN 1 ELSE 0 END) AS hr_n,
        SUM(CASE WHEN v.systemicsystolic  IS NOT NULL THEN 1 ELSE 0 END) AS sbp_n,
        SUM(CASE WHEN v.systemicmean      IS NOT NULL THEN 1 ELSE 0 END) AS map_n,
        SUM(CASE WHEN v.temperature       IS NOT NULL THEN 1 ELSE 0 END) AS temp_n,
        SUM(CASE WHEN v.respiration       IS NOT NULL THEN 1 ELSE 0 END) AS rr_n,
        SUM(CASE WHEN v.sao2              IS NOT NULL THEN 1 ELSE 0 END) AS o2_n
    FROM control_eligible c
    JOIN vitalPeriodic v
      ON v.patientunitstayid = c.patientunitstayid
     AND TRY_CAST(v.observationoffset AS DOUBLE)
         BETWEEN c.snapshot_min - 360 AND c.snapshot_min
    GROUP BY 1
    """)

    n_ctrl_vital_ok = con.execute("""
        SELECT COUNT(*) FROM control_vital
        WHERE hr_n >= 4 AND rr_n >= 4
          AND ((hr_n>=4)::INT + (sbp_n>=4)::INT + (map_n>=4)::INT
             + (temp_n>=4)::INT + (rr_n>=4)::INT + (o2_n>=4)::INT) >= 3
    """).fetchone()[0]

    n_ctrl_lab_ok = con.execute("""
    SELECT COUNT(*)
    FROM control_vital cv
    JOIN control_eligible ce USING (patientunitstayid)
    WHERE cv.hr_n>=4 AND cv.rr_n>=4
      AND ((cv.hr_n>=4)::INT + (cv.sbp_n>=4)::INT + (cv.map_n>=4)::INT
         + (cv.temp_n>=4)::INT + (cv.rr_n>=4)::INT + (cv.o2_n>=4)::INT) >= 3
      AND (
        SELECT COUNT(DISTINCT labname)
        FROM lab l
        WHERE l.patientunitstayid = cv.patientunitstayid
          AND TRY_CAST(l.labresultoffset AS DOUBLE)
              BETWEEN ce.snapshot_min - 1440 AND ce.snapshot_min
      ) >= 2
    """).fetchone()[0]

    print()
    print("=" * 70)
    print("  CONTROL POOL (demo)")
    print("=" * 70)
    print(f"  No-sepsis-no-early-abx adult ICU stays >=24h   : {n_ctrl_raw:>4}")
    print(f"  + vitals OK                                    : {n_ctrl_vital_ok:>4}")
    print(f"  + labs OK                                      : {n_ctrl_lab_ok:>4}  <-- usable control pool")

    # ---- Hospital diversity in usable pools ----
    h_sep = con.execute("""
        SELECT COUNT(DISTINCT se.hospitalid)
        FROM sepsis_eligible se
        JOIN sepsis_vital sv USING (patientunitstayid)
        WHERE sv.hr_n>=4 AND sv.rr_n>=4
    """).fetchone()[0]
    h_ctrl = con.execute("""
        SELECT COUNT(DISTINCT ce.hospitalid)
        FROM control_eligible ce
        JOIN control_vital cv USING (patientunitstayid)
        WHERE cv.hr_n>=4 AND cv.rr_n>=4
    """).fetchone()[0]
    print()
    print(f"Hospitals represented (sepsis pool):  {h_sep}")
    print(f"Hospitals represented (control pool): {h_ctrl}")

    print()
    print("=" * 70)
    print(f"  TARGET: 100 sepsis + 240 controls = 340 patients")
    print(f"  Achievable from demo? sepsis: "
          f"{'YES' if n_sepsis_lab_ok >= 100 else f'NO ({n_sepsis_lab_ok}/100)'}, "
          f"controls: {'YES' if n_ctrl_lab_ok >= 240 else f'NO ({n_ctrl_lab_ok}/240)'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
