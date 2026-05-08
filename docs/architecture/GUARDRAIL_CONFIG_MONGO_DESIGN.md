# Hospital-Specific Guardrail Config — MongoDB Design

**Status:** Design complete, implementation deferred until auth layer is ready. MongoDB collection to be created now by Sanjay so we have the target in place.
**Owner:** Sachin (design) + Sanjay (Mongo provisioning) + Narendra (deployment integration later)
**Date:** May 1, 2026

---

## 1. Why we need this

**Current state (risk):**
- The guardrail config (`genai_clinical_guardrail.json`) is baked into the Docker image and lives on the ephemeral container filesystem.
- `PUT /guardrail/config` writes survive only until the pod restarts.
- Multi-pod deployments diverge silently — a PUT only hits one pod.
- No audit trail of who changed what, when, or why.
- Every hospital today runs the same config.

**Target state:**
- Per-hospital config stored in MongoDB, versioned, auditable.
- Containers still boot from the baked JSON as a **safe default** — if Mongo is unreachable or no hospital-specific config exists, the container-native config is used.
- SME edits via `PUT /guardrail/config` write a new version to Mongo, never mutate previous versions.
- Rollback is "set `status = active` on an older version."

## 2. Transition approach

| Phase | Guardrail source | Auth | Timeline |
|---|---|---|---|
| **Phase A (today)** | Container-native JSON only | None | Current |
| **Phase B (this sprint)** | Mongo collection exists + populated with a "baseline" document; code still reads from container | Not required yet | Mongo provisioning only |
| **Phase C (next)** | Code reads from Mongo on startup + on `POST /guardrail/reload`; falls back to container JSON if Mongo unavailable | Hospital ID inferred from API key or JWT claim | Needs auth layer |
| **Phase D (prod)** | PUT /guardrail/config writes new version to Mongo; ConfigMap or SNS-based broadcast triggers reload across all pods | RBAC: only `sme_admin` role can PUT | After Phase C |

Today we're provisioning for Phase B. No code changes yet.

## 3. Collection design

**Database:** `medbeacon_dev_db_rr` (dev) — confirmed live, May 2026
**Collection:** `hospital_guardrail_configs`

> **Schema-history note (May 2026):** The original draft of this design used
> `config` for the JSON payload field and nested all audit fields under a
> `metadata` sub-document. When Sanjay provisioned the collection in Phase B,
> he used `parameters` (top-level) and flat `created_at` / `updated_at`. We
> reviewed both shapes against multi-hospital, long-term needs (audit, query
> ergonomics, regulator-readability, no-rewrite-later) and adopted a
> **flat-Sanjay shape with the design's audit fields kept (also flat)**.
> See `scripts/seed_baseline_guardrail.py` for the source of truth.

### Document schema (canonical)

```json
{
  "_id": "ObjectId (auto)",

  "hospital_id":     "string  — REQUIRED; e.g. 'medbeacon_baseline', 'hosp_mercy_001'",
  "status":          "string  — 'active' | 'draft' | 'archived'; exactly one 'active' per hospital_id",
  "config_version":  "int     — REQUIRED; monotonically increasing per hospital (1, 2, 3, ...)",
  "is_baseline":     "bool    — true only for 'medbeacon_baseline' fallback doc",

  "parameters": {
    "comment": "The full nested guardrail JSON — identical to genai_clinical_guardrail.json",
    "guardrail_config":          { "...": "..." },
    "critical_thresholds":       { "...": "..." },
    "early_detection_patterns":  { "...": "..." },
    "history_context_checks":    { "...": "..." },
    "differential_diagnosis":    { "...": "..." },
    "override_logic":            { "...": "..." },
    "discordance_rules":         { "...": "..." },
    "audit_settings":            { "...": "..." }
  },

  "created_at":        "ISODate — when this version was first written",
  "updated_at":        "ISODate — last status / metadata change on this version",

  "created_by":        "string  — email of the SME or system actor that produced this version",
  "updated_by":        "string  — email of the SME or system actor that last touched it",
  "change_reason":     "string  — SME-readable rationale (e.g. 'Lower lactate critical_high to 3.5 per Paula 2026-05-15')",
  "based_on_version":  "int     — previous config_version this was forked from (0 if seed)"
}
```

#### Why this shape (vs the earlier draft)

| Dimension | Earlier draft | Adopted shape | Why |
|---|---|---|---|
| Payload field | `config` | `parameters` | Already in DB; pure aesthetics, no benefit to flipping |
| Timestamps | `metadata.created_at` / `metadata.updated_at` | top-level `created_at` / `updated_at` | Easier indexing + simpler queries (`sort({updated_at: -1})`) and UI |
| Audit identity (`created_by`, `updated_by`, `change_reason`, `based_on_version`) | nested under `metadata.*` | top-level (flat) | Same query / index ergonomics as above; identical information content |

The audit fields are **mandatory** (not optional) — without them we cannot answer the regulator-grade question *"who changed this for hospital X, when, and why?"*.

### Example documents

#### Active baseline (canonical example, what `seed_baseline_guardrail.py` produces)

```json
{
  "hospital_id": "medbeacon_baseline",
  "status": "active",
  "config_version": 2,
  "is_baseline": true,

  "parameters": { /* full genai_clinical_guardrail.json — v7 (T=0, C1+C2) */ },

  "created_at": "2026-05-08T11:30:00Z",
  "updated_at": "2026-05-08T11:30:00Z",

  "created_by": "sachin.jadhav@medbeacon.ai",
  "updated_by": "sachin.jadhav@medbeacon.ai",
  "change_reason": "Initial v7 baseline seed (T=0, C1+C2). Replaces Phase-B placeholder.",
  "based_on_version": 1
}
```

#### Hospital-specific override (forked from baseline) — illustrative

```json
{
  "hospital_id": "hosp_mercy_001",
  "status": "active",
  "config_version": 1,
  "is_baseline": false,

  "parameters": { /* same shape as baseline, but with mercy-specific tweaks */ },

  "created_at": "2026-09-15T14:22:00Z",
  "updated_at": "2026-09-15T14:22:00Z",

  "created_by": "paula.smith@medbeacon.ai",
  "updated_by": "paula.smith@medbeacon.ai",
  "change_reason": "Lowered lactate critical_high from 4.0 to 3.5 mmol/L per Mercy ICU protocol; everything else inherits baseline v2.",
  "based_on_version": 2
}
```

### Common audit queries

```javascript
// "What did Paula change last week for hospital_mercy_001?"
db.hospital_guardrail_configs.find({
  hospital_id: "hosp_mercy_001",
  created_by: "paula.smith@medbeacon.ai",
  created_at: { $gte: ISODate("2026-05-01T00:00:00Z") }
}).sort({ config_version: -1 });

// "Show every active config across all hospitals."
db.hospital_guardrail_configs.find({ status: "active" })
  .project({ hospital_id: 1, config_version: 1, change_reason: 1, updated_at: 1 });

// "Walk the provenance chain back to the baseline for v23."
db.hospital_guardrail_configs.find({
  hospital_id: "hosp_mercy_001",
  config_version: { $lte: 23 }
}).sort({ config_version: -1 });
```

## 4. Required indexes

| Index | Fields | Type | Rationale |
|---|---|---|---|
| `ix_hospital_status` | `{ hospital_id: 1, status: 1 }` | Compound | Hot path — fetch the active config for a given hospital |
| `ix_hospital_version` | `{ hospital_id: 1, config_version: -1 }` | Compound | Version history listing (newest first) |
| `ix_unique_version` | `{ hospital_id: 1, config_version: 1 }` | **Unique** | Prevent duplicate version numbers |
| `ix_baseline` | `{ is_baseline: 1, status: 1 }` | Partial (where `is_baseline=true`) | Fast fallback to baseline |

### MongoDB shell commands

```javascript
use medbeacon_dev_db_rr;

db.createCollection("hospital_guardrail_configs");

db.hospital_guardrail_configs.createIndex(
  { hospital_id: 1, status: 1 },
  { name: "ix_hospital_status" }
);

db.hospital_guardrail_configs.createIndex(
  { hospital_id: 1, config_version: -1 },
  { name: "ix_hospital_version" }
);

db.hospital_guardrail_configs.createIndex(
  { hospital_id: 1, config_version: 1 },
  { name: "ix_unique_version", unique: true }
);

db.hospital_guardrail_configs.createIndex(
  { is_baseline: 1, status: 1 },
  { name: "ix_baseline", partialFilterExpression: { is_baseline: true } }
);
```

## 5. Access patterns

| Operation | Frequency | Read/Write | Example query |
|---|---|---|---|
| Fetch active config for hospital | Per pod startup + per `POST /guardrail/reload` | Read | `find({ hospital_id: "xyz", status: "active" })` |
| Fall back to baseline | When no hospital-specific config exists | Read | `find({ is_baseline: true, status: "active" })` |
| List version history | SME UI (rare) | Read | `find({ hospital_id: "xyz" }).sort({ config_version: -1 }).limit(50)` |
| Insert new version | SME `PUT /guardrail/config` (rare — maybe 1-5/day across all hospitals) | Write | Insert new doc with status=draft, then flip previous active → archived, new draft → active |
| Rollback | Rare | Write | Flip `status` on a historic doc |

## 6. Sizing estimate

| Dimension | Estimate |
|---|---|
| Document size | 30–50 KB (850+ nested parameters) |
| Hospitals (year 1) | 5–10 |
| Versions per hospital (year 1) | 10–20 |
| Total documents | ~200 |
| Total collection size | < 10 MB |
| Read QPS | < 1 (config is loaded on pod startup / reload, not per prediction) |
| Write QPS | < 0.001 (SME edits are human-paced) |

No special sharding, no TTL, no archival strategy needed in year 1.

## 7. Security / compliance

- **No PHI in this collection.** Config is parameter thresholds only. This collection is *not* subject to PHI handling rules — but still inherits MongoDB Atlas TLS + encryption-at-rest by default.
- **BAA not required specifically for this collection** (no PHI), but the existing Atlas BAA covers us either way.
- **Read access**: to be restricted to the sepsis-genai EKS service account only (Phase C). Current state: all `medbeacon_dev_db_rr` users can read.
- **Write access**: restricted to a dedicated `sepsis_config_writer` role when we add the auth layer (Phase D).
- **Audit**: every document is immutable once `status != "draft"`; update = insert new version. Top-level audit fields (`created_by`, `updated_by`, `change_reason`, `based_on_version`) capture actor + reason. The seed script (`scripts/seed_baseline_guardrail.py`) refuses to write without an explicit `--change-reason`.

## 8. Open items for later

- Connection string config via env var — already standardised on `MONGO_URI` (see `.env`); same pattern as the feedback analyzer.
- Decide on broadcast mechanism for multi-pod reload (SNS vs Redis pub-sub vs direct HTTP broadcast).
- Hospital ID inference — will come from auth layer (JWT claim or API-key-to-hospital mapping).
- Phase C code change: load the active doc from Mongo on pod startup + `POST /guardrail/reload`, fall back to container-native JSON if Mongo is unreachable.

## 9. Next actions

| # | Action | Owner | Status |
|---|---|---|---|
| 1 | Create `hospital_guardrail_configs` collection + indexes on `medbeacon_dev_db_rr` | Sanjay | ✅ Done (May 2026) |
| 2 | Reconcile schema with Sanjay's actual seed (flat shape, `parameters` field) + add audit fields | Sachin | ✅ Done — see §3 schema-history note |
| 3 | Seed v7 baseline document from current `genai_clinical_guardrail.json` (archives Sanjay's v1 placeholder, inserts v2 active) | Sachin | ⏳ Ready to run — `scripts/seed_baseline_guardrail.py --change-reason "..."` |
| 4 | Update PROJECT_TRACKER.md with new task (Phase B complete) | Sachin | ⏳ Pending |
| 5 | Phase C code changes — deferred until auth layer lands | Sachin + Narendra | Deferred |
