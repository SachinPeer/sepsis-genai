# Docs

Project documentation organized by purpose.

## `architecture/` — how the system works

| File | What it covers |
|---|---|
| `GENAI_OUTPUT_EXPLAINED.md` | Detailed walkthrough of the `/classify` response: `risk_score`, `priority`, `alert_level`, `guardrail_override`, `early_warnings`. |
| `prompt.md` | The current LLM system prompt (v3.2 — guardrail-aware, decoupled priority vs risk score). |
| `PHI_COMPLIANCE_AND_DATA_HANDLING.md` | HIPAA / PHI stance, BAAs (AWS + MongoDB), encryption, retention. |
| `SCALABILITY_AND_THROUGHPUT.md` | Throughput, latency, scaling behaviour. |
| `GUARDRAIL_CONFIG_MONGO_DESIGN.md` | Design doc: migrating hospital-specific guardrail configs from container to MongoDB. |

## `planning/` — project management

| File | What it covers |
|---|---|
| `PROJECT_TRACKER.md` | Live task tracker (owners, statuses, blockers). |
| `EXECUTION_PLAN.md` | Overall execution roadmap. |
| `VALIDATION_STUDY_PLAN.md` | Statistical design for the validation study (sample size, endpoints). |
| `COMPETITIVE_BENCHMARK.md` | Competitive landscape analysis. |
| `doc_feedback.md` | Reviewer feedback on docs. |

## `guides/` — how-to material

| File | What it covers |
|---|---|
| `FEEDBACK_LOOP_GUIDE.md` | How the human-in-the-loop feedback system is intended to work. |

## `llm-comparison/` — multi-LLM bake-offs

| File | What it covers |
|---|---|
| `LLM_MODEL_COMPARISON.md` | Narrative comparison of Claude / other LLMs on our task. |
| `LLM_Model_Comparison_Results.csv`, `Test_Results_For_Paula.csv`, `Sepsis_GenAI_Test_Results.xlsx` | Raw bake-off outputs. |
| `legacy/` | Scripts (`compare_models.py`, `run_model_comparison.py`) and the `model_comparison_results.json` used to produce the above. Kept for reference; internal paths are historical. |

## Utility scripts (stay in `docs/` root)

| File | What it does |
|---|---|
| `generate_pptx.py` | Builds `presentations/current/Medbeacon_Executive_Overview.pptx` programmatically. |
| `generate_storyboard.py` | Builds `presentations/current/Medbeacon_Clinical_Storyboard.pptx` programmatically. |
