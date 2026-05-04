# Sample Payloads

Small example JSON files used for manual / smoke testing of the pipeline. These are **not** part of the runtime image — they are developer utilities.

| File | Shape | Used by |
|---|---|---|
| `genai_test_patients.json` | Full patient object (demographics + vitals + labs + note). | Manual `curl`/Postman calls to `/classify`. Also shared with Red Rover as the schema example for their `POST /observation` endpoint. |
| `test_notes.json` | Minimal `notes`-only payload. | Quick narrative-only tests. |
| `test_trend.json` | Vitals trend array example. | Trend serialization smoke tests. |

## Quick test

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d @samples/genai_test_patients.json
```

> **Runtime config note:** `genai_clinical_guardrail.json` at the repo root is **not** a sample — it is the live guardrail config loaded by the container. It stays at root because `Dockerfile` copies it from there (`COPY genai_clinical_guardrail.json .`). See `docs/architecture/GUARDRAIL_CONFIG_MONGO_DESIGN.md` for the planned move to MongoDB.
