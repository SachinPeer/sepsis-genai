"""
Inspect the MongoDB cluster for the hospital-guardrail collection Sanjay set up.

Reports:
  - what databases exist,
  - what collections exist on the medbacon (and sepsis_genai) DB,
  - sample document count + first 1-2 sample documents (truncated) for any
    collection whose name looks like guardrail / hospital / config,
  - which design indexes are present vs missing,
  - whether the Phase-B baseline document seems to be in place yet.

Auth:
  Reads MONGO_URI from env (preferred) or falls back to .env file.
  If the .env URI still has '<ROTATED_PASSWORD>', the script exits early with
  instructions on how to provide the live password.

Usage:
    # Set the live password ad-hoc, then run:
    MONGO_URI='mongodb+srv://medbeacon_dev_db_user:<password>@medbeacondevcluster.bi9ewl6.mongodb.net/?retryWrites=true&w=majority' \
        python scripts/inspect_mongo_guardrail.py

    # Or, if you've already exported MONGO_URI in your shell:
    python scripts/inspect_mongo_guardrail.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent

# Load MONGO_URI from .env if not in env, but skip the placeholder
mongo_uri = os.getenv("MONGO_URI")
if not mongo_uri:
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.strip().startswith("MONGO_URI="):
                mongo_uri = line.split("=", 1)[1].strip()
                break

if not mongo_uri or "<ROTATED_PASSWORD>" in mongo_uri or "<password>" in mongo_uri:
    print("ERROR: live MONGO_URI not available.")
    print("       The .env file has a placeholder.")
    print("       Provide the active password by exporting MONGO_URI in your shell:")
    print()
    print("    export MONGO_URI='mongodb+srv://medbeacon_dev_db_user:<PASSWORD>@\\")
    print("        medbeacondevcluster.bi9ewl6.mongodb.net/?retryWrites=true&w=majority'")
    print()
    print("    python scripts/inspect_mongo_guardrail.py")
    sys.exit(2)

# Hide password before printing the URI
parsed = urlparse(mongo_uri)
safe_host = parsed.hostname or "unknown"

try:
    from pymongo import MongoClient
except ImportError:
    print("ERROR: pymongo not installed. Run: pip install pymongo")
    sys.exit(2)


GUARDRAIL_NAME_HINTS = ("guardrail", "hospital", "config")
DESIGN_INDEXES = {
    "ix_hospital_status":  [("hospital_id", 1), ("status", 1)],
    "ix_hospital_version": [("hospital_id", 1), ("config_version", -1)],
    "ix_unique_version":   [("hospital_id", 1), ("config_version", 1)],
    "ix_baseline":         [("is_baseline", 1), ("status", 1)],
}


def truncate(s: str, n: int = 240) -> str:
    return s if len(s) <= n else s[:n] + f"... <{len(s)-n} more chars>"


def show_doc(d: dict) -> None:
    """Pretty-print a guardrail config doc, summarising the big nested config."""
    summary = {}
    for k, v in d.items():
        if k == "config" and isinstance(v, dict):
            summary[k] = f"<nested config dict, {len(v)} top-level keys: {list(v.keys())[:8]}>"
        elif k == "_id":
            summary[k] = str(v)
        elif isinstance(v, (dict, list)) and len(json.dumps(v, default=str)) > 400:
            summary[k] = f"<{type(v).__name__}, summarised>"
        else:
            summary[k] = v
    print(json.dumps(summary, indent=2, default=str))


def main() -> int:
    print(f"Connecting to MongoDB at {safe_host} ...")
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=10000)
    try:
        client.admin.command("ping")
    except Exception as e:
        print(f"ERROR: cannot reach Mongo: {type(e).__name__}: {e}")
        return 3
    print("Connected.\n")

    print("=" * 70)
    print("Databases visible to this user")
    print("=" * 70)
    for name in sorted(client.list_database_names()):
        if name in ("admin", "config", "local"):
            continue
        print(f"  - {name}")
    print()

    # Probe (a) the env-configured DB first, (b) any DB whose name contains
    # 'medbeacon'/'medbacon'. Skip system DBs.
    visible = [
        n for n in client.list_database_names()
        if n not in ("admin", "config", "local")
    ]
    env_db = os.getenv("MONGO_DB_NAME")
    candidate_dbs = []
    if env_db and env_db in visible:
        candidate_dbs.append(env_db)
    for n in visible:
        if n in candidate_dbs:
            continue
        if "medbeacon" in n.lower() or "medbacon" in n.lower():
            candidate_dbs.append(n)

    if not candidate_dbs:
        print("WARN: no candidate DBs found.")
        print(f"      MONGO_DB_NAME env: {env_db!r}")
        print(f"      Visible: {visible}")
        return 4

    print(f"Probing DBs: {candidate_dbs}\n")

    for db_name in candidate_dbs:
        db = client[db_name]
        print("=" * 70)
        print(f"DATABASE: {db_name}")
        print("=" * 70)
        cols = sorted(db.list_collection_names())
        for col_name in cols:
            count = db[col_name].estimated_document_count()
            mark = (
                "  <-- guardrail-related"
                if any(h in col_name.lower() for h in GUARDRAIL_NAME_HINTS)
                else ""
            )
            print(f"  - {col_name:<40s}  ({count} docs){mark}")
        print()

        # Detailed view for any guardrail-looking collection
        for col_name in cols:
            if not any(h in col_name.lower() for h in GUARDRAIL_NAME_HINTS):
                continue
            print("-" * 70)
            print(f"COLLECTION: {db_name}.{col_name}")
            print("-" * 70)
            col = db[col_name]
            count = col.estimated_document_count()
            print(f"  documents: {count}")

            # Indexes vs design
            existing = {idx["name"]: idx.get("key", []) for idx in col.list_indexes()}
            print(f"  indexes ({len(existing)}):")
            for name, key in existing.items():
                print(f"    - {name:<25s}  key={key}")
            missing = [n for n in DESIGN_INDEXES if n not in existing]
            if missing:
                print(f"  MISSING design indexes: {missing}")
            else:
                print("  All design indexes present.")

            # Sample documents
            print(f"\n  Sample documents (up to 2):")
            for i, doc in enumerate(col.find().limit(2)):
                print(f"\n  --- document {i+1} ---")
                show_doc(doc)

            # Phase-B baseline check
            baseline = col.find_one({"is_baseline": True, "status": "active"})
            if baseline:
                v = baseline.get("config_version", "?")
                hid = baseline.get("hospital_id", "?")
                print(f"\n  Phase-B baseline: PRESENT  "
                      f"(hospital_id={hid}, config_version={v})")
            else:
                print(f"\n  Phase-B baseline: MISSING  "
                      f"(no doc with is_baseline=true && status=active)")
            print()

    print("=" * 70)
    print("Inspection complete.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
