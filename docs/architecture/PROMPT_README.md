# LLM System Prompt — operational canonical

This directory holds two prompt files. **Only `prompt.md` is loaded at runtime.**

## `prompt.md` — operational canonical (loaded by the runtime)

This is the **active system prompt** sent to the LLM (Claude Sonnet 4.5 via Bedrock) for every `/classify` call.

- **Length:** 32 lines / ~1,400 characters
- **Status:** validated; produces v7.1 KPIs of **82.4% sensitivity, 62.9% specificity** on the 150-patient eICU cohort (`validation/results/EICU_results_v7_t0_c1_scores_c2_latest.json`)
- **Source of truth:** this file *and* the `_get_default_prompt()` fallback in `genai_inference_service.py` hold byte-identical content. The fallback exists as a safety floor in case the file is missing at container startup.

### Why it is intentionally minimal

We tested two longer alternatives in May 2026 and both regressed sensitivity:

| Variant | Sensitivity | Specificity | Decision |
|---|---|---|---|
| **Current operational** | **82.4%** | **62.9%** | **Keep** |
| Canonical v3.2 (268 lines, see `prompt_canonical_v3.md`) | 61.8% | 81.9% | Rejected — −20.6pp sensitivity |
| Synthesis v7.3 (134 lines) | pilot 4/7 — would have failed full run | n/a | Rejected at pilot stage |

The full reasoning is captured in `validation/docs/PROMPT_PATH_BUG_AND_REVALIDATION_PLAN.md`.

The 33-line minimal prompt has the **inversion property**: shorter prompts lose sensitivity (drop the role / specialty / silent-sepsis cues and v7's discordance catches drop), while longer prompts also lose sensitivity (additional rules, ICU-context guidance, or "trust the guardrail" framing all push the LLM toward under-calling on borderline sepsis cases).

### Editing rules

**Do not modify `prompt.md` without re-validating** against the 150-patient eICU cohort. Specifically:

1. Run `validation/run_eicu_validation.py --cohort-dir validation/eicu_cohort_v4 --tag <new-version>`
2. Compare against `validation/results/EICU_results_v7_t0_c1_scores_c2_latest.json`
3. Acceptance criteria:
   - Sensitivity ≥ 82.4% (no TP regression — non-negotiable)
   - Specificity ≥ 62.9% **OR** specificity gain ≥ +5pp
   - 9-case demo smoke test (`scripts/demo_smoke_test.py`) all pass
   - No new failure modes
4. If accepted: also update `_get_default_prompt()` in `genai_inference_service.py` so the fallback stays bit-identical to disk.
5. Bump the version tag in any deployment manifests / image tags.

## `prompt_canonical_v3.md` — aspirational version (NOT loaded)

The 268-line clinical prompt the team originally authored as the "ideal" system prompt. Preserved here for SME review and as a reference for future fine-tuning data labelling.

This prompt was tested against the 150-patient cohort in May 2026 and produced sensitivity 61.8% (vs current 82.4%) — losing 7 true-positive sepsis cases including 5 v7-Critical. It is retained as a record of what the LLM does when it is given explicit "trust the guardrail" framing, ICU-context anti-patterns, and a hard-rule decoupling of risk score from priority. The takeaway is that those instructions cannot be safely composed with a short, pattern-detection-friendly prompt — see `validation/docs/PROMPT_PATH_BUG_AND_REVALIDATION_PLAN.md` §11 for details.

## Runtime load path

`genai_inference_service._load_system_prompt()` searches in order:

1. `<package_dir>/docs/architecture/prompt.md` ← preferred (this file)
2. `<package_dir>/docs/prompt.md` ← legacy
3. `/app/docs/architecture/prompt.md` ← container path
4. `/app/docs/prompt.md` ← legacy container path
5. **Fallback:** `_get_default_prompt()` inline string (bit-identical to #1)

On success, the pod logs:
```
INFO genai_inference_service: Loaded LLM system prompt from: <path> (1426 chars)
```

If the disk file goes missing for any reason, the pod still boots cleanly off the inline fallback — no behavioural change, only a `WARNING` in the log that should trigger investigation.
