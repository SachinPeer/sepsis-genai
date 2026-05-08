# Prompt-Path Bug — Material Differences & Re-Validation Plan

**Discovery date:** 2026-05-08 (during v7.1 EKS rollout verification)
**Severity:** Latent — present in every image since v4
**Patient-facing impact:** None observed (v7 KPIs were achieved with the misconfigured state)
**Decision needed:** Whether to switch to the canonical prompt and re-validate, or formally adopt the current inline default as the canonical going forward

---

## 1. The bug

`genai_inference_service.py:377-379` looks for the LLM system prompt at:

```python
prompt_path = os.path.join(os.path.dirname(__file__), "docs", "prompt.md")
```

Inside the container this resolves to `/app/docs/prompt.md`. The actual canonical prompt (`docs/architecture/prompt.md`) lives at `/app/docs/architecture/prompt.md`.

`open()` raises `FileNotFoundError`, the warning is logged, and the code falls back to `_get_default_prompt()` — an inline 33-line / ~1.5 KB Python string at `genai_inference_service.py:392-426`.

This warning has been firing in every image we've shipped (v4, v5, v6, v7, v7.1).

**Implication:** every validation run, every demo, every live request — including all the v7 results we have been quoting (sens 82.4% / spec 62.9% on 150 eICU patients) — has been produced by the **inline default prompt**, NOT by the canonical `prompt.md`.

---

## 2. Side-by-side comparison

| Dimension | Inline default (in-flight today) | Canonical `docs/architecture/prompt.md` |
|---|---|---|
| Length | ~33 lines / ~1.5 KB | 268 lines / ~16 KB |
| Version label | none | `v3.2 — guardrail-aware judgment layer` |
| "Not the last line of defense" framing | absent | explicit (lines 19–35) |
| Output discipline (risk_score vs priority decoupling) | absent | explicit hard rule (lines 39–52) |
| Discordance analysis with note examples | absent | detailed (lines 59–69) |
| Velocity-of-change reasoning | absent | detailed (lines 71–83) |
| Sepsis-3 / qSOFA / SIRS criteria walkthrough | absent | full (lines 85–94) |
| Confidence-grade rubric | "use sparingly" only | full High/Medium/Low criteria |
| `priority_justification` JSON field | **absent** | **present** |
| Internal consistency check (pre-emit chain-of-thought) | absent | 3 explicit questions (lines 156–166) |
| Priority assignment criteria | absent | full criteria for Critical / High / Standard |
| **Anti-patterns ("Do NOT assign High for…")** | absent | **explicit list of 6 anti-patterns** |
| ICU-context awareness ("isolated abnormal vitals are common") | absent | explicit section |
| Risk-score calibration bands (0–19, 20–39, …, 80–100) | absent | explicit |
| Worked examples | none | 5 (silent sepsis, trending, low conf, decoupled, justified critical) |
| Calibrated-escalation instruction ("alert fatigue erodes trust") | absent | explicit |

### What the inline default *does* have

- Role definition (1 line)
- JSON output structure (without `priority_justification`)
- 5 short guidelines: be conservative with High confidence, watch trends, watch silent sepsis

That's it.

---

## 3. Why this matters

The canonical prompt was clearly written **after** observing v4/v5/v6 false-positive patterns. It contains exactly the kind of guidance we ended up implementing in C1 and C2 deterministic guardrails:

- "Trust the downstream guardrail; don't over-escalate" → mirrors C1's design intent
- "Do NOT assign High for isolated tachycardia / mild hypoxemia / leukocytosis on post-op" → mirrors C2 branches
- "ICU patients commonly have abnormal vitals from non-sepsis causes" → mirrors C2's "non-infectious context" branch
- 5 worked examples including one explicit "decoupled risk vs priority" case → exactly the kind of scenario v7 still gets wrong

In other words: **the canonical prompt and the C1/C2 guardrail layers are doing the same job, partially in duplicate.** If the canonical prompt had been live during v6/v7 development, we likely would have:
- Built C1/C2 differently (or maybe not at all)
- Gotten a different sensitivity/specificity trade-off curve
- Tuned different regex patterns for C1

The current state is internally consistent (v7 KPIs are real, and the 9-case demo cohort passes end-to-end against v7.1), but the *system we think we have* and the *system we actually have* are not identical.

---

## 4. Risks of switching to the canonical prompt without re-validation

| Risk | Why it matters | Likelihood |
|---|---|---|
| **JSON schema mismatch** — canonical adds `priority_justification` field | Our downstream code may or may not handle unknown fields gracefully; need to audit `genai_pipeline.py` and `guardrail_service.py` for strict-key access | Low (Python dict access is forgiving) |
| **C1 regex patterns become brittle** — C1 matches LLM rationale strings; canonical prompt produces different vocabulary | C1 was tuned to inline-default's hedged-rationale style; canonical may produce more formal medical phrasing (or less) | Medium |
| **C2 branch logic becomes brittle** — same reason; C2 archetypes were tuned to v6/v7 LLM outputs | Some C2 branches may suppress less (or more) under canonical prompt | Medium |
| **Sensitivity drop** — canonical actively dampens over-escalation ("trust the guardrail") | LLM may under-call true positives that v7 currently catches via direct High/Critical priority | Medium-High |
| **Specificity gain** — canonical has explicit anti-patterns, ICU-context calibration, decoupled-priority discipline | LLM should produce fewer false High/Critical priorities | High |
| **Net KPI shift** unknown without an A/B re-run on the same 150-patient eICU cohort | We cannot quote KPIs to Paula or to a hospital until we re-validate | Certain |

---

## 5. Decision tree

```
Discovered: inline default ≠ canonical prompt
              │
              ▼
   ┌──────────────────────────────────┐
   │ Re-validate canonical on 150 pt  │
   │   eICU cohort under v7.2 image   │
   └──────────────────────────────────┘
              │
        ┌─────┴─────┐
        ▼           ▼
    BETTER     WORSE / SIDEWAYS
        │           │
        ▼           ▼
  Roll v7.2     Keep v7.1, formally adopt
  + update      `_get_default_prompt()` as
  prompt.md     the canonical, replace
  + fix path    `docs/architecture/prompt.md`
                with that text, fix path
```

Both branches end at the same place: the path bug is fixed and `prompt.md` becomes the unambiguous source of truth. The only question is *which prompt text* gets enshrined.

---

## 6. Re-validation plan (Step 3 of action plan)

### 6.1 Branch & image

```
Branch :  feat/v7.2-canonical-prompt
Image  :  sepsis-genai:v7.2-canonical-prompt-<commit>
Change :  fix prompt-path to docs/architecture/prompt.md
          (single line in genai_inference_service.py)
```

### 6.2 Validation cohort

Same 150-patient eICU cohort used for v7 (`validation/results/EICU_results_v7_t0_c1_scores_c2_*.csv`). Apples-to-apples: same patients, same MongoDB-backed C1+C2 settings, T=0, same Claude Sonnet 4.5 model.

### 6.3 Metrics to compare

For each of v7.1 (current) and v7.2 (canonical-prompt):

| Metric | v7.1 (baseline) | v7.2 | Δ | Notes |
|---|---|---|---|---|
| Sensitivity | 82.4% (28/34) | ? | | Headline: do we still catch true sepsis? |
| Specificity | 62.9% (73/116) | ? | | Headline: fewer false alarms? |
| F1 | 0.53 | ? | | Combined |
| FAR | 37.1% (43/116) | ? | | Operational impact |
| C1 fire rate | n / 150 | ? | | Did the canonical prompt reduce LLM rationale that C1 was suppressing? |
| C2 fire rate | n / 150 | ? | | Same question for C2 |
| LLM avg risk | x.x | ? | | Did the canonical prompt make the LLM more or less aggressive? |
| Cases where v7.2 caught a TP that v7.1 missed | — | n | | Headline win for canonical |
| Cases where v7.2 missed a TP that v7.1 caught | — | n | | Headline risk |

### 6.4 Acceptance criteria for switching to v7.2

Roll v7.2 only if **all** of these hold:

1. Sensitivity ≥ 82.4% (no TP regression vs v7) — non-negotiable
2. Specificity ≥ 62.9% (no FP regression) OR specificity gain ≥ +5pp
3. No new failure mode in the 9-case demo smoke test
4. C1 + C2 still fire on at least 80% of the cases they fired on in v7 (so we don't have to re-tune them in the same release)

If criterion 1 fails: **stop**. Sensitivity is sacrosanct in this domain. Keep v7.1.

If criterion 1 passes but 2 doesn't: discuss with Paula whether the trade-off is acceptable.

### 6.5 Effort estimate

- Code change: 5 minutes (single line)
- Image build + push: 10 minutes
- Run validation against 150 pt cohort: ~45 minutes (LLM-bound; same as v7 run took)
- Compare results, write up: 30 minutes
- Total: ~1.5 hours of focused work

---

## 7. Status & ownership

| # | Action | Owner | Status |
|---|---|---|---|
| 1 | Do nothing on running image; v7.1 stays live as-is | — | ✅ done |
| 2 | Diff the two prompts; document material differences | assistant | ✅ done |
| 3 | Cut v7.2-canonical-prompt branch; re-run validation on 150 pt cohort | assistant | ✅ done — see §8 |
| 4 | Decide v7.2 vs v7.1 based on validation deltas | Sachin + assistant | ✅ done — **keep v7.1** |
| 5 | **Synthesis attempt: build v7.3 (operational base + ICU-context)** | assistant | ✅ done — pilot failed — see §11 |
| 6 | Fix prompt-path so warning stops; enshrine winning prompt as canonical | assistant | pending — see §9 |
| 7 | Mention to Paula in next SME review | Sachin | pending |

---

## 8. Re-validation results (Step 3) — 2026-05-08

**Branch:** `feat/v7.2-canonical-prompt`
**Code change:** single function — `genai_inference_service._load_system_prompt()` now searches `docs/architecture/prompt.md` first (and several other candidate paths) before falling back to the inline default.
**Cohort:** `validation/eicu_cohort_v4/` — exact same 150 patients (34 sepsis, 116 controls) used for v7.
**Run:** `validation/results/EICU_results_v7.2_canonical_prompt_latest.json`
**Stack:** identical to v7.1 — Mongo-backed config v2, C1 + C2 enabled, T = 0, Claude Sonnet 4.5.

### 8.1 Headline metrics

| Metric | v7 (current) | v7.2 (canonical prompt) | Δ |
|---|---:|---:|---:|
| **Sensitivity** | **82.4%** (28/34) | **61.8%** (21/34) | **−20.6pp** ❌ |
| Specificity | 62.9% (73/116) | 81.9% (95/116) | +19.0pp ✅ |
| PPV (precision) | 39.4% | 50.0% | +10.6pp ✅ |
| NPV | 92.4% | 88.0% | −4.4pp ❌ |
| F1 | 53.3% | 55.3% | +1.9pp ≈ |
| False alarm rate | 37.1% | 18.1% | −19.0pp ✅ |
| TP / FN / FP / TN | 28 / 6 / 43 / 73 | 21 / **13** / 21 / 95 | **+7 sepsis missed** |

### 8.2 What v7.2 lost — 7 true-positive sepsis cases

All 7 lost TPs collapse to the **same uniform verdict**: `risk = 42, priority = Standard`. This is a single repeatable failure mode of the canonical prompt:

| patient | v7 verdict | v7.2 verdict | Severity of regression |
|---|---|---|---|
| eicu_p00005 | risk 70 / High | risk 42 / Standard | High → missed |
| eicu_p00006 | risk 95 / **Critical** | risk 42 / Standard | **Critical → missed** |
| eicu_p00007 | risk 72 / **Critical** | risk 42 / Standard | **Critical → missed** |
| eicu_p00009 | risk 95 / **Critical** | risk 42 / Standard | **Critical → missed** |
| eicu_p00010 | risk 42 / High | risk 42 / Standard | High → missed |
| eicu_p00014 | risk 95 / **Critical** | risk 42 / Standard | **Critical → missed** |
| eicu_p00017 | risk 95 / **Critical** | risk 42 / Standard | **Critical → missed** |

**5 of the 7 were `Critical` under v7.** These are not borderline calls — these are patients the current system flags as the most urgent category, and the canonical prompt is silently demoting them to Standard.

### 8.3 What v7.2 gained — 26 false-positive suppressions

v7.2 correctly suppressed 26 false positives that v7 had been raising. Notably, several of these were also `Critical` under v7 and got demoted to Standard under v7.2 — same mechanism as the TP losses, just acting on the right patients. Examples:
- p00042: risk 95/Critical → risk 35/Standard ✅ (correctly suppressed)
- p00080: risk 92/Critical → risk 45/Standard ✅ (correctly suppressed)

### 8.4 Failure-mode diagnosis

The canonical prompt's two strongest design decisions are working **too well**:

1. **"Trust the guardrail; don't pre-emptively classify as septic"** — the LLM is now over-deferring. On any narrative where the criteria aren't textbook-clean, it converges to the safe-but-low default of risk 42, priority Standard.

2. **"Risk score and priority are decoupled; don't auto-elevate priority just because risk is high"** — eliminated the priority bumps that v7's pipeline relied on. Several of the lost TPs had v7-priority `High` despite a relatively low risk score; v7.2 keeps the risk score similar but sets priority to Standard, which falls below our `risk≥50 OR priority∈{High,Critical}` alert threshold.

The C1 and C2 deterministic guardrails were **not** able to rescue these 7 cases because the LLM-rationale signals C1 looks for (explicit non-sepsis vocabulary like "post-op stress", "compensated state") were absent — the canonical prompt produces brief, conservative rationales without those anchor phrases. C2's branch matchers had nothing to fire on either.

### 8.5 Acceptance-criteria evaluation

| Crit. | Threshold | v7.2 result | Status |
|---|---|---|---|
| 1 | Sensitivity ≥ 82.4% (no TP regression) | 61.8% | **FAIL** (non-negotiable) |
| 2 | Spec ≥ 62.9% OR spec gain ≥ +5pp | 81.9% (Δ +19.0pp) | PASS |
| 3 | 9-case demo smoke test all pass | 9/9 | PASS |
| 4 | C1 + C2 fire on ≥80% of v7's firings | not measured (criterion 1 already failed; not worth measuring) | n/a |

**Decision: DO NOT ROLL v7.2. Keep v7.1 in production.** The sensitivity regression is severe (−20.6pp, 7 lost sepsis cases including 5 v7-Critical cases) and is non-recoverable without changes to the canonical prompt that would essentially turn it back into the inline default.

### 8.6 Why this is the right call (not just risk-aversion)

In a sepsis-prediction system, the cost of a missed alert (FN) is qualitatively worse than the cost of a false alarm (FP):
- **FN cost:** patient-facing harm; mortality risk increases ~7%/hour of treatment delay.
- **FP cost:** alert fatigue; nursing time; trust erosion. Real but recoverable.

A 19pp specificity gain is meaningful only if sensitivity is held. Trading 7 sepsis catches for 26 fewer false alarms would be the wrong trade for any clinical product, and it would be impossible to defend in front of Paula or in any FDA-style review.

---

## 9. Step 5 — fix the prompt-path bug; enshrine winning prompt as canonical

Now that v7.1 is the validated winner:

1. **`genai_inference_service.py`**: keep the multi-candidate path search from the v7.2 branch (it's strictly better than the broken hard-coded path), so the inline-default code-path becomes a real fallback rather than the silent default.
2. **`docs/architecture/prompt.md`**: replace the contents with a markdown wrapper around the inline default — the prompt that has produced our v7 KPIs. Add a header noting it is the *operational* prompt; preserve the canonical *aspirational* prompt as `prompt_canonical_v3.md` for SME reference and a future re-attempt if Paula wants to revisit the trade-off explicitly.
3. **No image rebuild needed urgently** — v7.1 is correct and live. The path-fix can ride along with the next material change.
4. **Update `validation/docs/EICU_VALIDATION_COMPARISON.md`** with the v7.2 row showing the failed re-validation, so future readers know we tested it.

---

## 10. What this means for the SME conversation

When you brief Paula:

- v7.1 is validated and live; KPIs (82.4% / 62.9%) are real and reproducible.
- Independent of the rollout, we discovered that the LLM has been driven by a 33-line operational prompt rather than the 268-line aspirational prompt the team wrote.
- We re-ran the full 150-patient validation against the aspirational prompt as a proper A/B. It improves specificity (+19pp) but at the cost of catastrophic sensitivity loss (−20.6pp / 7 TPs missed including 5 Critical cases).
- The trade-off is not acceptable for a sepsis product. We are formally adopting the operational prompt as canonical and preserving the aspirational version for SME review.
- This is a useful artifact: it shows v7.1 is not just "lucky"; the lift from C1/C2 + the operational prompt vocabulary is structural.

---

---

## 11. Synthesis attempt — v7.3 (operational base + selective canonical elements)

**Hypothesis:** Take v7's 33-line operational prompt as the structural base (preserves 82.4% sensitivity), then layer in *only* the elements of the canonical prompt that improve specificity through ICU context (anti-patterns, ICU-baseline awareness, calibration bands, worked examples) — explicitly excluding the "trust the guardrail" framing that drove v7.2's −20.6pp regression.

**Draft prompt:** `docs/architecture/prompt_v7.3_synthesis_DRAFT.md` (134 lines / 8.4 KB — between v7's 33 lines and v7.2's 268 lines).

### 11.1 Pilot design

Run v7.3 against the **7 sepsis patients v7.2 lost** (p00005, p00006, p00007, p00009, p00010, p00014, p00017). Tripwire: catch ≥ 6 / 7 to proceed to full 150-patient re-run. Else abort and revert.

Rationale for the tripwire: v7.3 needed to recover most of the 7 losses to have any chance of passing the 82.4% sensitivity floor on the full cohort. Math: best case for v7.3 = (21 v7.2-caught) + (7 lost-TPs recovered) = 28 / 34 = 82.4% — exactly at threshold. Anything less than 6/7 in the pilot makes the full run mathematically incapable of passing.

### 11.2 Pilot result

| Patient | v7 | v7.2 | v7.3 |
|---|---|---|---|
| eicu_p00005 | risk 70 / High ✅ | risk 42 / Standard ❌ | risk 32 / Standard ❌ |
| eicu_p00006 | risk 95 / Critical ✅ | risk 42 / Standard ❌ | risk 42 / Standard ❌ |
| eicu_p00007 | risk 72 / Critical ✅ | risk 42 / Standard ❌ | risk 72 / High ✅ |
| eicu_p00009 | risk 95 / Critical ✅ | risk 42 / Standard ❌ | risk 95 / Critical ✅ |
| eicu_p00010 | risk 42 / High ✅ | risk 42 / Standard ❌ | risk 42 / Standard ❌ |
| eicu_p00014 | risk 95 / Critical ✅ | risk 42 / Standard ❌ | risk 95 / Critical ✅ |
| eicu_p00017 | risk 95 / Critical ✅ | risk 42 / Standard ❌ | risk 72 / Standard ✅ (alert HIGH via guardrail) |
| **Caught** | **7 / 7** | **0 / 7** | **4 / 7** |

**Pilot result: 4/7 — TRIPWIRE FAILED.** 

### 11.3 Why this failed

The same failure mode as v7.2, just attenuated:
- p00005, p00006, p00010 all returned with **rationales explicitly noting non-sepsis explanations** ("no infection signals", "elderly with chronic renal dysfunction", "vitals currently stable"). These exact phrasings closely match what the canonical prompt's ICU-context section trains the LLM to produce.
- The **ICU CONTEXT section was still the active force** suppressing escalation, even after we removed the "trust the guardrail" framing and the "expected for risk 60-75 with priority Standard" instruction.
- v7's operational prompt has no ICU-context guidance at all, which is precisely why it catches these patients — the LLM doesn't know to apply the "common ICU patterns" filter, so it escalates based on the raw vitals/notes signal.

### 11.4 Why the math forced the abort

Even in the best case (4 already-caught + perfect on remaining 3 = 28 / 34 = 82.4%), v7.3 would only just tie v7. In practice it would also lose some of v7's other 21 TPs to the same suppression mechanism, putting the full-cohort sensitivity well below 82.4%. The pilot is a reliable signal here and proceeding to a full run would be wasted compute.

### 11.5 Conclusion of the synthesis experiment

**The canonical prompt's specificity gain and v7's sensitivity are not independently composable.** The same instructions that suppress v7's 26 false positives also suppress v7's catches on cases like p00005/p00006/p00010 — the LLM cannot reliably distinguish between "this control patient has incidental abnormal vitals" and "this sepsis patient has subtle compensated-shock signs", and the canonical-style ICU-context guidance pushes both directions toward the same Standard/risk≈42 verdict.

**The 19pp specificity gap between v7.1 (62.9%) and v7.2 (81.9%) is real but cannot be closed via prompt synthesis alone.** Future paths to bridge it:
- **Fine-tuning** on a labelled corpus where the LLM learns the boundary explicitly
- **Retrieval-augmented prompting** with case-similar exemplars at inference time
- **Continued C2-branch development** (deterministic post-LLM rules, where we have ground-truth labels and can be precise)
- **Confidence-stratified routing** — route low-confidence LLM verdicts to a different prompt or to human review

Of these, **continued C2-branch development is the only one we can safely pursue without re-validation effort** — it operates after the LLM and we already have the eICU run data to mine for new branch patterns.

### 11.6 Status: reverted

| Action | Status |
|---|---|
| Restore working tree to main (drop v7.3 path-fix code, drop modified prompt.md) | ✅ done |
| Local API state matches v7.1 production | ✅ verified |
| Production (EKS) untouched throughout entire experiment | ✅ verified |
| Draft v7.3 prompt preserved at `docs/architecture/prompt_v7.3_synthesis_DRAFT.md` | ✅ kept for record |
| v3.2 canonical preserved at `docs/architecture/prompt_canonical_v3.md` | ✅ kept for SME reference |
| v7.2 full-run results preserved at `validation/results/EICU_results_v7.2_canonical_prompt_*.json` | ✅ kept for record |
| Pilot script preserved at `scripts/v7_3_pilot.py` | ✅ kept for future re-attempts |

---

*Document version: 1.2 (synthesis attempt + pilot abort landed)  •  Last updated: 2026-05-08*
