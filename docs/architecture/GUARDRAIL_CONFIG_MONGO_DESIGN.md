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

**Database:** `medbacon`
**Collection:** `hospital_guardrail_configs`

### Document schema

```json
{
  "_id": "ObjectId (auto)",
  "hospital_id": "string — REQUIRED; e.g., 'medbeacon_baseline', 'hosp_mercy_001', 'hosp_partners_mgh'",
  "config_version": "int — REQUIRED; monotonically increasing per hospital (1, 2, 3, ...)",
  "status": "string — 'active' | 'draft' | 'archived'; exactly one 'active' per hospital at a time",
  "is_baseline": "bool — true only for the 'medbeacon_baseline' seed doc used when no hospital-specific config exists",
  "config": {
    "comment": "The full nested guardrail JSON — identical structure to current genai_clinical_guardrail.json",
    "version": "string (tracked inside the config itself)",
    "critical_thresholds": { "...": "..." },
    "override_logic": { "...": "..." },
    "discordance_rules": { "...": "..." },
    "audit_settings": { "...": "..." }
  },
  "metadata": {
    "created_by": "string — email or user_id of the SME who created this version",
    "created_at": "ISODate",
    "updated_by": "string",
    "updated_at": "ISODate",
    "change_reason": "string — SME-provided rationale for the change",
    "based_on_version": "int — previous config_version this was forked from (0 if seed)"
  }
}
```

### Example seed document (this is the one to insert first)

```json
{
  "hospital_id": "medbeacon_baseline",
  "config_version": 1,
  "status": "active",
  "is_baseline": true,
  "config": { /* full contents of genai_clinical_guardrail.json */ },
  "metadata": {
    "created_by": "sachin.jadhav@medbeacon.ai",
    "created_at": "2026-05-01T00:00:00Z",
    "updated_by": "sachin.jadhav@medbeacon.ai",
    "updated_at": "2026-05-01T00:00:00Z",
    "change_reason": "Initial seed from container-native genai_clinical_guardrail.json v1.0",
    "based_on_version": 0
  }
}
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
use medbacon;

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
- **Read access**: to be restricted to the sepsis-genai EKS service account only (Phase C). Current state: all `medbacon` DB users can read.
- **Write access**: restricted to a dedicated `sepsis_config_writer` role when we add the auth layer (Phase D).
- **Audit**: every document is immutable once `status != "draft"`; update = insert new version. Metadata block captures actor + reason.

## 8. Open items for later

- Connection string config via env var — align with how feedback analyzer connects (`MONGODB_URI` pattern)
- Decide on broadcast mechanism for multi-pod reload (SNS vs Redis pub-sub vs direct HTTP broadcast)
- Hospital ID inference — will come from auth layer (JWT claim or API-key-to-hospital mapping)
- Migration script: `scripts/seed_baseline_config.py` to write the first baseline doc from the committed JSON

## 9. Next actions

| # | Action | Owner |
|---|---|---|
| 1 | Create `hospital_guardrail_configs` collection + indexes on `medbacon` DB | Sanjay |
| 2 | Seed baseline document from current `genai_clinical_guardrail.json` | Sachin (after collection exists) |
| 3 | Update PROJECT_TRACKER.md with new task (Phase B complete) | Sachin |
| 4 | Phase C code changes — deferred until auth layer lands | Sachin + Narendra |
