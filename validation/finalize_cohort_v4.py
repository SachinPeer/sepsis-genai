"""
Post-process the eICU v4 cohort: remove any patient with a HARD flag
according to the audit (NOTES_EMPTY is a soft flag and is kept), then
renumber remaining files + rewrite manifest. Idempotent.

Usage:
  python3 validation/finalize_cohort_v4.py
"""
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

COHORT = Path(__file__).parent / "eicu_cohort_v4"

# NOTES_EMPTY is informational, not a removal reason
HARD_FLAGS = {
    "AGE_NEONATE", "AGE_PEDIATRIC", "AGE_OUTLIER",
    "VITAL_HR_THIN", "VITAL_RESP_THIN", "VITAL_TYPES_THIN",
    "LABS_MISSING", "LABS_THIN",
}


def main():
    audit = json.loads((COHORT / "audit_report.json").read_text())
    manifest = json.loads((COHORT / "manifest.json").read_text())

    # Map patient_id -> hard-flag list
    hard_by_pid: dict[str, list[str]] = {}
    for p in audit["patients"]:
        hard = [f for f in p["flags"] if f in HARD_FLAGS]
        if hard:
            hard_by_pid[p["patient_id"]] = hard

    print(f"Hard-flag patients to remove: {len(hard_by_pid)}")
    for pid, flags in list(hard_by_pid.items())[:20]:
        print(f"  {pid}: {flags}")

    # Build keep-set from manifest, preserving order
    kept: list[dict] = []
    for m in manifest["patients"]:
        if m["patient_id"] in hard_by_pid:
            (COHORT / m["file"]).unlink(missing_ok=True)
            continue
        kept.append(m)

    # Renumber: assign new sequential ids and rename files
    renumbered: list[dict] = []
    for i, m in enumerate(kept, start=1):
        new_id = f"eicu_p{i:05d}"
        new_file = f"{new_id}.json"
        old_path = COHORT / m["file"]
        new_path = COHORT / new_file
        if old_path.exists():
            data = json.loads(old_path.read_text())
            data["patient_id"] = new_id
            new_path.write_text(json.dumps(data, indent=2, default=str))
            if old_path != new_path:
                old_path.unlink()
        m_new = {**m, "patient_id": new_id, "file": new_file}
        renumbered.append(m_new)

    # Final summary
    n_sepsis = sum(1 for m in renumbered if m["actual_sepsis"])
    n_ctrl = sum(1 for m in renumbered if not m["actual_sepsis"])
    n_empty_notes = sum(1 for m in renumbered if m.get("notes_empty"))

    manifest["patients"] = renumbered
    manifest["finalized_at"] = datetime.utcnow().isoformat() + "Z"
    manifest["finalize_removed"] = len(hard_by_pid)
    manifest["summary"] = {
        "total": len(renumbered),
        "sepsis": n_sepsis,
        "controls": n_ctrl,
        "with_empty_notes": n_empty_notes,
    }
    (COHORT / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print()
    print("=" * 60)
    print(f"  Finalized v4 cohort: {COHORT}")
    print(f"    Total       : {len(renumbered)}")
    print(f"    Sepsis      : {n_sepsis}")
    print(f"    Controls    : {n_ctrl}")
    print(f"    Empty notes : {n_empty_notes} (soft, kept)")
    print(f"    Removed     : {len(hard_by_pid)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
