# LLM-comparison — legacy scripts

Historical scripts used to produce the results in `docs/llm-comparison/LLM_Model_Comparison_Results.csv` and `Sepsis_GenAI_Test_Results.xlsx`.

**Status:** not actively maintained. Relative paths inside these scripts (`BASE_DIR`, `test_trend.json`, `test_notes.json`, `genai_test_patients.json`) were relative to the old repo root. Those sample files now live in `samples/` — if you want to re-run these, either copy samples next to the scripts or update the scripts to read from `samples/`.

| File | What it was |
|---|---|
| `compare_models.py` | CLI wrapper to call multiple LLMs with the same payload and diff outputs. |
| `run_model_comparison.py` | Full bake-off runner that produced the leadership-facing comparison CSV. |
| `model_comparison_results.json` | Raw output from the last run. |
