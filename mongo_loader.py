"""
Hospital-config loader from MongoDB Atlas.

Phase C of the hospital-config migration (see
docs/architecture/GUARDRAIL_CONFIG_MONGO_DESIGN.md).

Single-tenant interim:
  - The pod looks up exactly one hospital per deployment via the env var
    HOSPITAL_ID (default: 'medbeacon_baseline').
  - When SSO / auth lands, that env var becomes the *fallback* hospital id
    used when no auth claim is present. The Mongo load logic does not
    change — only the lookup key does. See §8.1 of the design doc.

Failure mode:
  - Any failure (Mongo unreachable, doc not found, connection timeout,
    schema mismatch, etc.) raises MongoConfigLoadError. Callers SHOULD
    catch this and fall back to the on-disk JSON. Pods must continue to
    serve even when Mongo is broken — the disk JSON is the safety floor.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class MongoConfigLoadError(RuntimeError):
    """Raised when the Mongo-backed guardrail config cannot be loaded.
    Callers should catch this and fall back to the on-disk JSON."""


# ---------- public API -------------------------------------------------
def load_active_baseline(
    hospital_id: Optional[str] = None,
    *,
    mongo_uri: Optional[str] = None,
    db_name: Optional[str] = None,
    collection_name: Optional[str] = None,
    server_selection_timeout_ms: int = 5000,
) -> Dict[str, Any]:
    """Fetch the currently-active config doc for `hospital_id` from Mongo.

    Returns a dict with the structure:

        {
            "config":          <the full guardrail JSON, equivalent to
                                what genai_clinical_guardrail.json
                                would yield from disk>,
            "metadata": {
                "hospital_id":      str,
                "config_version":   int,
                "based_on_version": int,
                "created_at":       ISO datetime str,
                "updated_at":       ISO datetime str,
                "created_by":       str,
                "change_reason":    str,
                "loaded_at":        ISO datetime str,
                "loaded_in_ms":     float,
                "source":           "mongo",
                "db":               str,
                "collection":       str,
            }
        }

    Resolves args from env if not passed:
        hospital_id      <- HOSPITAL_ID            (default: 'medbeacon_baseline')
        mongo_uri        <- MONGO_URI              (REQUIRED — raises if unset)
        db_name          <- MONGO_DB_NAME          (default: 'medbeacon_dev_db_rr')
        collection_name  <- MONGO_GUARDRAIL_COLLECTION
                                                   (default: 'hospital_guardrail_configs')
    """
    hospital_id = hospital_id or os.getenv("HOSPITAL_ID", "medbeacon_baseline")
    mongo_uri = mongo_uri or os.getenv("MONGO_URI")
    db_name = db_name or os.getenv("MONGO_DB_NAME", "medbeacon_dev_db_rr")
    collection_name = collection_name or os.getenv(
        "MONGO_GUARDRAIL_COLLECTION", "hospital_guardrail_configs"
    )

    if not mongo_uri:
        raise MongoConfigLoadError(
            "MONGO_URI is not set; cannot load guardrail config from Mongo."
        )

    try:
        from pymongo import MongoClient
        from pymongo.errors import PyMongoError
    except ImportError as e:
        raise MongoConfigLoadError(
            f"pymongo not installed: {e}. Add 'pymongo>=4.6.0' to requirements."
        )

    t0 = time.time()
    try:
        client = MongoClient(
            mongo_uri,
            serverSelectionTimeoutMS=server_selection_timeout_ms,
            appname="sepsis-genai",
        )
        # Force connection so we fail fast on bad URIs / network problems.
        client.admin.command("ping")
        coll = client[db_name][collection_name]
        doc = coll.find_one({"hospital_id": hospital_id, "status": "active"})
    except PyMongoError as e:
        raise MongoConfigLoadError(
            f"Mongo error fetching {db_name}.{collection_name} "
            f"hospital_id={hospital_id!r}: {type(e).__name__}: {e}"
        )

    if doc is None:
        # Fallback chain inside Mongo itself: if the requested hospital has
        # no active doc, try the baseline. This is the right behaviour for
        # multi-hospital later, and is harmless today (hospital_id already
        # defaults to 'medbeacon_baseline').
        if hospital_id != "medbeacon_baseline":
            logger.warning(
                "No active config for hospital_id=%r; falling back to "
                "medbeacon_baseline within Mongo.", hospital_id
            )
            doc = coll.find_one(
                {"hospital_id": "medbeacon_baseline", "status": "active"}
            )
        if doc is None:
            raise MongoConfigLoadError(
                f"No active baseline doc in {db_name}.{collection_name} "
                f"for hospital_id={hospital_id!r} or 'medbeacon_baseline'."
            )

    payload = doc.get("parameters")
    if not isinstance(payload, dict):
        raise MongoConfigLoadError(
            f"Doc _id={doc.get('_id')} hospital_id={doc.get('hospital_id')} "
            f"has no 'parameters' dict; got {type(payload).__name__}."
        )

    elapsed_ms = (time.time() - t0) * 1000
    return {
        "config": payload,
        "metadata": {
            "source": "mongo",
            "db": db_name,
            "collection": collection_name,
            "hospital_id": doc.get("hospital_id"),
            "config_version": doc.get("config_version"),
            "based_on_version": doc.get("based_on_version"),
            "created_at": _iso(doc.get("created_at")),
            "updated_at": _iso(doc.get("updated_at")),
            "created_by": doc.get("created_by"),
            "change_reason": doc.get("change_reason"),
            "loaded_at": _iso(time.time()),
            "loaded_in_ms": round(elapsed_ms, 1),
        },
    }


def _iso(value: Any) -> Optional[str]:
    """Normalise a Mongo datetime / float epoch / None into an ISO string."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        # Treat as epoch seconds.
        from datetime import datetime, timezone
        return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
    try:
        return value.isoformat()
    except Exception:
        return str(value)
