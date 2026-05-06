"""
Guardrail-replay simulator.
=================================================================

Takes a baseline validation run (e.g. v6_t0_c1_scores) and replays every
patient through the *current* `SepsisSafetyGuardrail` code WITHOUT calling
the LLM again. Lets us predict the metrics impact of code changes to C1 / C2
in seconds instead of running a full live validation (~40 min).

Crucially, this simulator uses the runner's **actual** positive-classification
rule:

    predicted_sepsis = (risk_score >= 50) OR (priority in {High, Critical})
                       OR (alert_level in {HIGH, CRITICAL})

Earlier ad-hoc simulations used a "suppression == negative" shortcut, which
hid threshold-trapped cases where C2 dropped the priority but kept risk >=
50. This script does NOT make that mistake.

Usage:
    python validation/simulate_guardrail_change.py \\
        --baseline v6_t0_c1_scores \\
        --compare-with v7_t0_c1_scores_c2

Pass --baseline as either a result tag (looked up in validation/results/) or
a full path. --compare-with is optional and lets you sanity-check the
simulator against a known production run (the simulator should agree with
production to within ~1-2 patients due to LLM near-determinism at T=0).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).parent
REPO_ROOT = ROOT.parent
COHORT_DIR = ROOT / "eicu_cohort_v4"
RES = ROOT / "results"

# Make sure we import the project's guardrail (not anything else on the path).
sys.path.insert(0, str(REPO_ROOT))

# Default flags ON for the simulator. Caller can override via env before
# invoking this script.
os.environ.setdefault("ENABLE_C1_SUPPRESSION", "true")
os.environ.setdefault("ENABLE_C2_SUPPRESSION", "true")

from guardrail_service import SepsisSafetyGuardrail  # noqa: E402


# ---------------------------------------------------------------------------
# Runner classification rule
# (mirrors validation/run_eicu_validation.py:classify_prediction)
# ---------------------------------------------------------------------------
def runner_classify(risk: Any, priority: Any, alert_level: Any = None) -> bool:
    try:
        r = float(risk) if risk is not None else 0.0
    except Exception:
        r = 0.0
    if r >= 50:
        return True
    if priority in ("High", "Critical"):
        return True
    if alert_level in ("HIGH", "CRITICAL"):
        return True
    return False


# ---------------------------------------------------------------------------
# Reconstruct an "LLM-output-shaped" prediction from a saved baseline row.
#
# A baseline row has (post-C1, pre-C2) state - C1 may have already
# downgraded priority to Standard if it fired. For the simulator's purpose
# (replay through current guardrail code) we treat the row as the input to
# the C2 step, which is the right framing as long as the *baseline* run had
# C1 enabled and C2 disabled.
# ---------------------------------------------------------------------------
def reconstruct_prediction(row: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
    """
    Returns (prediction_dict, llm_original_risk).

    llm_original_risk is the risk score the LLM itself returned, BEFORE any
    guardrail action (override, early-detection bump, C1 priority alignment).

    Recovery priority:
      1. row['llm_initial_risk_score']   - exact, available on runs from
         2026-02-11+ where guardrail_service stores it in logic_gate
      2. row['original_risk_score']      - set when guardrail OVERRIDE fires
      3. heuristic for Early-Detection:  if early_warnings is non-empty AND
         risk_score is within ED-escalation range (~70), the LLM's true risk
         was below the escalation threshold. We mark as approximate.
      4. otherwise: row['risk_score']    - LLM-only path (no bump)
    """
    risk_post = row.get("risk_score")
    approximate = False

    if row.get("llm_initial_risk_score") is not None:
        llm_risk = row["llm_initial_risk_score"]
    elif row.get("guardrail_override") and row.get("original_risk_score") is not None:
        llm_risk = row["original_risk_score"]
    elif row.get("early_warnings") and risk_post is not None:
        # ED-bump path on a baseline that didn't save the LLM's true initial
        # risk. The LLM was below escalation_risk_score (default 70). We use
        # the post-bump value but mark approximate so callers know.
        try:
            llm_risk = float(risk_post)
            if llm_risk >= 70:
                # rough estimate: LLM was probably hedging in the 30-49 range
                llm_risk = 35.0
                approximate = True
        except Exception:
            llm_risk = 35.0
            approximate = True
    else:
        llm_risk = risk_post

    pred = {
        "prediction": {
            "risk_score_0_100": risk_post,
            "priority": row.get("priority"),
            "sepsis_probability_6h": row.get("sepsis_probability_6h"),
            "clinical_rationale": row.get("reasoning") or "",
        },
        "logic_gate": {
            "guardrail_override": bool(row.get("guardrail_override")),
            "override_reasons": [],
            "original_risk_score": row.get("original_risk_score"),
            "early_warnings": [row.get("early_warnings")] if row.get("early_warnings") else [],
            "_sim_llm_risk_approximate": approximate,
        },
    }
    try:
        return pred, float(llm_risk) if llm_risk is not None else 0.0
    except Exception:
        return pred, 0.0


def flatten_vitals(vitals_ts: Dict[str, Any]) -> Dict[str, Any]:
    """Same logic genai_pipeline._flatten_vitals uses (most-recent value)."""
    flat: Dict[str, Any] = {}
    for k, v in vitals_ts.items():
        if isinstance(v, list) and v:
            first = v[0]
            if isinstance(first, dict) and "val" in first:
                flat[k] = first["val"]
            else:
                flat[k] = first
        else:
            flat[k] = v
    return flat


# ---------------------------------------------------------------------------
# Simulate one patient
# ---------------------------------------------------------------------------
def simulate_row(row: Dict[str, Any],
                 vitals_ts: Dict[str, Any],
                 gr: SepsisSafetyGuardrail) -> Dict[str, Any]:
    pred, llm_risk = reconstruct_prediction(row)
    rationale = pred["prediction"]["clinical_rationale"]

    flat = flatten_vitals(vitals_ts)
    try:
        clinical_scores = gr.calculate_clinical_scores(flat)
    except Exception:
        clinical_scores = {
            "qsofa": {"score": row.get("qsofa") or 0},
            "sirs": {"criteria_met": row.get("sirs_met") or 0},
            "sofa": {"score": row.get("sofa") or 0},
        }

    override_fired = bool(row.get("guardrail_override"))
    early_detection_fired = bool(row.get("early_warnings"))

    triggers = gr._c2_get_override_triggers(rationale)
    if not triggers and override_fired:
        triggers = ["__unknown_override__"]

    suppressed = False
    audit: Dict[str, Any] = {}
    if os.getenv("ENABLE_C2_SUPPRESSION", "false").lower() == "true":
        suppressed, audit = gr._c2_should_suppress(
            prediction=pred,
            raw_vitals=vitals_ts,
            llm_original_risk=llm_risk,
            early_detection_fired=early_detection_fired,
            override_fired=override_fired,
            override_triggers=triggers,
            clinical_scores=clinical_scores,
        )
        if suppressed:
            gr._c2_apply_suppression(pred, llm_original_risk=llm_risk, audit=audit)

    out = dict(row)
    out["risk_score"] = pred["prediction"]["risk_score_0_100"]
    out["priority"] = pred["prediction"]["priority"]
    out["sepsis_probability_6h"] = pred["prediction"]["sepsis_probability_6h"]
    out["c2_fired"] = bool(suppressed)
    out["c2_branch"] = audit.get("branch") if suppressed else None
    out["c2_reason"] = audit.get("reason") if suppressed else None
    out["llm_risk_approximate"] = pred["logic_gate"].get("_sim_llm_risk_approximate", False)
    # When C2 suppresses, the original alert_level (HIGH/CRITICAL) is stale.
    # Recompute from priority so the runner's third clause doesn't carry over
    # a pre-suppression alert level. (Production sets alert_level fresh per
    # request; the simulator must do the same.)
    if suppressed:
        out["alert_level"] = "LOW"
    out["predicted_sepsis"] = runner_classify(
        out["risk_score"], out["priority"], out.get("alert_level"))
    return out


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------
def confusion(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    succ = [r for r in rows if r.get("status") == "success"
            and r.get("predicted_sepsis") in (True, False)]
    tp = sum(1 for r in succ if r["actual_sepsis"] and r["predicted_sepsis"])
    fn = sum(1 for r in succ if r["actual_sepsis"] and not r["predicted_sepsis"])
    fp = sum(1 for r in succ if not r["actual_sepsis"] and r["predicted_sepsis"])
    tn = sum(1 for r in succ if not r["actual_sepsis"] and not r["predicted_sepsis"])
    return {
        "n": len(succ), "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "sensitivity": tp / (tp + fn) if (tp + fn) else 0.0,
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "ppv": tp / (tp + fp) if (tp + fp) else 0.0,
        "npv": tn / (tn + fn) if (tn + fn) else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0,
    }


def fmt(m: Dict[str, Any]) -> str:
    return (f"TP={m['tp']:>2} FN={m['fn']:>2} FP={m['fp']:>2} TN={m['tn']:>2}  "
            f"sens={100*m['sensitivity']:5.1f}%  spec={100*m['specificity']:5.1f}%  "
            f"PPV={100*m['ppv']:5.1f}%  F1={m['f1']:.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def resolve_results(spec: Optional[str]) -> Optional[Path]:
    if not spec:
        return None
    p = Path(spec)
    if p.exists():
        return p
    p = RES / f"EICU_results_{spec}_latest.json"
    if p.exists():
        return p
    return None


def load_vitals_index() -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    for jf in COHORT_DIR.glob("eicu_p*.json"):
        try:
            payload = json.loads(jf.read_text())
            pid = payload.get("patient_id") or jf.stem
            idx[pid] = payload.get("patient_vitals", {})
        except Exception:
            continue
    return idx


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", required=True,
                    help="Tag (v6_t0_c1_scores) or full path to a baseline results JSON")
    ap.add_argument("--compare-with", default=None,
                    help="Optional production results tag/path to compare against")
    ap.add_argument("--out", default=None,
                    help="Output JSON path (default: validation/results/sim_<baseline>.json)")
    args = ap.parse_args()

    base_path = resolve_results(args.baseline)
    if not base_path:
        print(f"ERROR: cannot find baseline '{args.baseline}'")
        sys.exit(1)

    payload = json.loads(base_path.read_text())
    rows = payload["results"]
    print(f"Loaded baseline: {base_path.name} ({len(rows)} patients)")

    vitals_idx = load_vitals_index()
    print(f"Loaded vitals time-series for {len(vitals_idx)} patients")

    gr = SepsisSafetyGuardrail()
    print(f"C1 enabled: {os.getenv('ENABLE_C1_SUPPRESSION', 'false')}")
    print(f"C2 enabled: {os.getenv('ENABLE_C2_SUPPRESSION', 'false')}")

    new_rows = []
    for r in rows:
        if r.get("status") != "success":
            new_rows.append(r)
            continue
        ts = vitals_idx.get(r["patient_id"], {})
        new_rows.append(simulate_row(r, ts, gr))

    base_m = confusion(rows)
    new_m = confusion(new_rows)
    print()
    print(f"Baseline:  {fmt(base_m)}")
    print(f"Simulated: {fmt(new_m)}")
    print(f"  Δ sens: {100*(new_m['sensitivity']-base_m['sensitivity']):+.1f} pp")
    print(f"  Δ spec: {100*(new_m['specificity']-base_m['specificity']):+.1f} pp")

    flips = []
    for old, new in zip(rows, new_rows):
        if old.get("predicted_sepsis") != new.get("predicted_sepsis"):
            flips.append({
                "patient_id": old["patient_id"],
                "actual": old.get("actual_sepsis"),
                "old_pred": old.get("predicted_sepsis"),
                "new_pred": new.get("predicted_sepsis"),
                "old_risk": old.get("risk_score"),
                "new_risk": new.get("risk_score"),
                "old_priority": old.get("priority"),
                "new_priority": new.get("priority"),
                "c2_branch": new.get("c2_branch"),
                "c2_reason": new.get("c2_reason"),
            })

    print(f"\n{len(flips)} patients flipped (showing up to 20):")
    for f in flips[:20]:
        sym = "✓" if (
            (f["actual"] and f["new_pred"]) or
            (not f["actual"] and not f["new_pred"])) else "✗"
        print(f"  {sym} {f['patient_id']}  act={int(bool(f['actual']))}  "
              f"{int(bool(f['old_pred']))}->{int(bool(f['new_pred']))}  "
              f"risk={f['old_risk']}->{f['new_risk']}  "
              f"pri={f['old_priority']}->{f['new_priority']}  "
              f"c2={f['c2_branch']}")

    # Sanity-check vs a known production run
    if args.compare_with:
        prod_path = resolve_results(args.compare_with)
        if prod_path:
            prod_rows = json.loads(prod_path.read_text())["results"]
            prod_idx = {r["patient_id"]: r for r in prod_rows}
            sim_idx = {r["patient_id"]: r for r in new_rows}
            disagreements = []
            for pid, sim_r in sim_idx.items():
                pr = prod_idx.get(pid)
                if not pr:
                    continue
                if pr.get("predicted_sepsis") != sim_r.get("predicted_sepsis"):
                    disagreements.append((pid, sim_r.get("predicted_sepsis"),
                                          pr.get("predicted_sepsis"),
                                          sim_r.get("c2_branch"),
                                          pr.get("risk_score"),
                                          pr.get("priority"),
                                          sim_r.get("risk_score"),
                                          sim_r.get("priority")))
            prod_m = confusion(prod_rows)
            print(f"\nProduction ({prod_path.name}): {fmt(prod_m)}")
            print(f"Simulator vs production disagreements: {len(disagreements)}")
            for d in disagreements[:10]:
                print(f"  {d[0]}  sim={d[1]} prod={d[2]}  "
                      f"sim(risk={d[6]}, pri={d[7]})  prod(risk={d[4]}, pri={d[5]})  "
                      f"c2_branch={d[3]}")

    out_path = Path(args.out) if args.out else (
        RES / f"sim_{base_path.stem.replace('EICU_results_', '').replace('_latest', '')}.json")
    out_path.write_text(json.dumps({
        "baseline": str(base_path),
        "metrics": new_m,
        "rows": new_rows,
    }, indent=2, default=str))
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
