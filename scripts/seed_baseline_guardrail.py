"""
Seed (or update) the Phase-B baseline guardrail document in MongoDB.

This writes the *current* `genai_clinical_guardrail.json` (which is the v7
production config we just validated and shipped to ECR) into the
`hospital_guardrail_configs` collection on the `medbacon` database, as the
canonical baseline that all hospitals fall back to.

Schema follows docs/architecture/GUARDRAIL_CONFIG_MONGO_DESIGN.md.

Behaviour:
  - Idempotent: if no baseline exists, inserts config_version=1, status=active.
  - Otherwise, inserts a NEW document at config_version=N+1, status=active,
    and flips the previous active baseline to status=archived. Older versions
    are kept untouched (full audit trail).
  - Always (re)creates the four design indexes.

Auth & target:
  - Reads MONGO_URI from env (preferred) or .env (skipping placeholder).
  - --db / --collection flags override the design defaults if Sanjay used
    different names. Run inspect_mongo_guardrail.py first to find out.

Usage:
    # dry-run first to see what would happen
    MONGO_URI='...' python scripts/seed_baseline_guardrail.py --dry-run

    # commit the seed
    MONGO_URI='...' python scripts/seed_baseline_guardrail.py \
        --change-reason 'Initial seed from v7 (T=0, C1+C2)'

    # if Sanjay used a different collection name:
    MONGO_URI='...' python scripts/seed_baseline_guardrail.py \
        --collection guardrail_configs --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Confirmed live (May 2026) — see .env for source-of-truth.
DEFAULT_DB = "medbeacon_dev_db_rr"
DEFAULT_COLLECTION = "hospital_guardrail_configs"
HOSPITAL_ID = "medbeacon_baseline"
GUARDRAIL_JSON = ROOT / "genai_clinical_guardrail.json"
DEFAULT_AUTHOR = "sachin.jadhav@medbeacon.ai"

# Document schema — aligned with Sanjay's seeded shape (May 2026):
#   { hospital_id, status, config_version, is_baseline,
#     parameters: <full guardrail JSON>,
#     created_at, updated_at,
#     created_by, change_reason, based_on_version }
PAYLOAD_FIELD = "parameters"  # the JSON config lives under this key

DESIGN_INDEXES = [
    {"name": "ix_hospital_status",  "keys": [("hospital_id", 1), ("status", 1)]},
    {"name": "ix_hospital_version", "keys": [("hospital_id", 1), ("config_version", -1)]},
    {"name": "ix_unique_version",
     "keys": [("hospital_id", 1), ("config_version", 1)],
     "unique": True},
    {"name": "ix_baseline",
     "keys": [("is_baseline", 1), ("status", 1)],
     "partial": {"is_baseline": True}},
]


def load_mongo_uri() -> str:
    uri = os.getenv("MONGO_URI")
    if not uri:
        env_file = ROOT / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.strip().startswith("MONGO_URI="):
                    uri = line.split("=", 1)[1].strip()
                    break
    if not uri or "<ROTATED_PASSWORD>" in uri or "<password>" in uri:
        print("ERROR: live MONGO_URI not available. Set it in your shell:")
        print("       export MONGO_URI='mongodb+srv://...:<password>@.../...'")
        sys.exit(2)
    return uri


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1] if __doc__ else "")
    p.add_argument("--db", default=DEFAULT_DB,
                   help=f"Mongo database (default: {DEFAULT_DB})")
    p.add_argument("--collection", default=DEFAULT_COLLECTION,
                   help=f"Mongo collection (default: {DEFAULT_COLLECTION})")
    p.add_argument("--hospital-id", default=HOSPITAL_ID,
                   help=f"hospital_id field (default: {HOSPITAL_ID})")
    p.add_argument("--author", default=DEFAULT_AUTHOR,
                   help="created_by / updated_by metadata")
    p.add_argument("--change-reason", required=True,
                   help="REQUIRED — SME-readable rationale for this version "
                        "(e.g. 'Initial v7 baseline seed (T=0, C1+C2)' or "
                        "'Lower lactate critical_high to 3.5 per Paula 2026-05-15'). "
                        "Stored as the change_reason audit field on the new doc.")
    p.add_argument("--dry-run", action="store_true",
                   help="print what would be written but don't write")
    p.add_argument("--ensure-indexes-only", action="store_true",
                   help="just create design indexes, don't seed")
    return p.parse_args()


def ensure_indexes(col, dry: bool) -> None:
    existing = {idx["name"] for idx in col.list_indexes()}
    for spec in DESIGN_INDEXES:
        if spec["name"] in existing:
            print(f"  index already present: {spec['name']}")
            continue
        if dry:
            print(f"  WOULD CREATE index: {spec['name']}  keys={spec['keys']}"
                  + (" UNIQUE" if spec.get("unique") else "")
                  + (f"  partial={spec['partial']}" if spec.get("partial") else ""))
            continue
        kwargs = {"name": spec["name"]}
        if spec.get("unique"):
            kwargs["unique"] = True
        if spec.get("partial"):
            kwargs["partialFilterExpression"] = spec["partial"]
        col.create_index(spec["keys"], **kwargs)
        print(f"  created index: {spec['name']}")


def main() -> int:
    args = parse_args()

    if not GUARDRAIL_JSON.exists():
        print(f"ERROR: {GUARDRAIL_JSON} not found")
        return 2

    config_payload = json.loads(GUARDRAIL_JSON.read_text())

    try:
        from pymongo import MongoClient, ASCENDING
    except ImportError:
        print("ERROR: pymongo not installed. Run: pip install pymongo")
        return 2

    uri = load_mongo_uri()
    client = MongoClient(uri, serverSelectionTimeoutMS=10000)
    try:
        client.admin.command("ping")
    except Exception as e:
        print(f"ERROR: cannot reach Mongo: {type(e).__name__}: {e}")
        return 3

    db = client[args.db]
    if args.collection not in db.list_collection_names():
        print(f"ERROR: collection {args.db}.{args.collection} does not exist.")
        print(f"       Run inspect_mongo_guardrail.py first to find the right name.")
        return 4

    col = db[args.collection]
    print(f"Target: {args.db}.{args.collection}")
    print(f"        documents currently: {col.estimated_document_count()}")

    print("\n--- Indexes ---")
    ensure_indexes(col, args.dry_run)

    if args.ensure_indexes_only:
        return 0

    # Find latest version for this hospital_id
    latest = col.find_one(
        {"hospital_id": args.hospital_id},
        sort=[("config_version", -1)],
    )
    next_version = (latest["config_version"] + 1) if latest else 1
    based_on = latest["config_version"] if latest else 0

    now = datetime.now(timezone.utc)
    new_doc = {
        "hospital_id": args.hospital_id,
        "status": "active",
        "config_version": next_version,
        "is_baseline": True,
        PAYLOAD_FIELD: config_payload,
        "created_at": now,
        "updated_at": now,
        "created_by": args.author,
        "updated_by": args.author,
        "change_reason": args.change_reason,
        "based_on_version": based_on,
    }

    print(f"\n--- Seed plan ---")
    print(f"  hospital_id           : {args.hospital_id}")
    print(f"  next config_version   : {next_version}")
    print(f"  based_on_version      : {based_on}")
    print(f"  config size           : ~{len(json.dumps(config_payload))} bytes,"
          f" {len(config_payload)} top-level keys")
    print(f"  config top-level keys : {list(config_payload.keys())}")
    print(f"  change_reason         : {args.change_reason}")
    print(f"  author                : {args.author}")

    if args.dry_run:
        print("\nDRY RUN — nothing written. Re-run without --dry-run to commit.")
        return 0

    # Atomic-ish: archive old active baselines, then insert the new one.
    # Backfill any missing audit fields on the previous doc so the audit log
    # is complete from config_version=1, even if the original doc (e.g. Sanjay's
    # sample) was inserted before audit fields were standardised.
    if latest and latest.get("status") == "active":
        archive_set = {
            "status": "archived",
            "updated_at": now,
            "updated_by": args.author,
        }
        if "created_by" not in latest:
            archive_set["created_by"] = "sanjay+system@medbeacon.ai"
        if "change_reason" not in latest:
            archive_set["change_reason"] = (
                "Phase-B provisioning sample (placeholder schema, "
                "pre-audit-fields). Backfilled during v7 baseline seed."
            )
        if "based_on_version" not in latest:
            archive_set["based_on_version"] = 0
        col.update_one({"_id": latest["_id"]}, {"$set": archive_set})
        backfilled = [k for k in archive_set if k not in ("status", "updated_at", "updated_by")]
        print(f"\n  archived previous active baseline "
              f"(version {latest['config_version']})"
              + (f"  [backfilled: {backfilled}]" if backfilled else ""))

    inserted = col.insert_one(new_doc)
    print(f"  inserted new baseline _id={inserted.inserted_id} "
          f"(version {next_version}, status=active)")

    print("\n--- Verify ---")
    active = col.find_one({"hospital_id": args.hospital_id, "status": "active"})
    if not active:
        print("  ERROR: no active baseline after seed — investigate manually")
        return 5
    payload = active.get(PAYLOAD_FIELD, {})
    print(f"  active baseline now at config_version={active['config_version']} "
          f"with {len(payload)} top-level keys in '{PAYLOAD_FIELD}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
