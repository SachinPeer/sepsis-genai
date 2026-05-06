"""
Verify that every clinically-meaningful field in genai_clinical_guardrail.json
appears in the generated PDF (Medbeacon_Guardrail_Review_v<ver>.pdf).

Approach:
  1. Extract plain text from the PDF using pdftotext.
  2. Walk the JSON and assemble a list of "must-appear" strings:
       - threshold values + units + condition operators
       - description, clinical_rationale, sme_notes, context_check
       - qsofa, shock, DIC criteria
       - early-detection patterns
       - history context conditions, impact, alert_action
       - discordance phrases
  3. For each must-appear string, normalise whitespace/quotes/Unicode and
     check it is present in the PDF text.
  4. Print a coverage report — counts, missing items, and a final PASS/FAIL.

Usage:
    python3 scripts/verify_guardrail_pdf.py

Exits with code 1 if any required fragment is missing.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
JSON_PATH = REPO / "genai_clinical_guardrail.json"
PDF_PATH = REPO / "docs" / "clinical-review" / "Medbeacon_Guardrail_Review_v3_0.pdf"


# --------------------------------------------------------------------- helpers
def normalise(s: str) -> str:
    """Make text comparable: NFKD-normalise, replace fancy unicode with ASCII,
    collapse whitespace, lowercase. Used for lenient containment check."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", str(s))
    repl = {
        "≥": ">=", "≤": "<=", "≠": "!=",
        "–": "-", "—": "-",
        "“": '"', "”": '"', "’": "'", "‘": "'",
        "·": " ", "•": " ", "→": "->",
        "µ": "u",  # micro
        "\u2009": " ", "\u00a0": " ", "\u200b": " ",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def is_present(needle: str, hay_norm: str) -> bool:
    """Lenient containment: return True if normalised needle appears in
    normalised PDF text. For long sentences accepts a 70-char prefix.
    For 'value unit' pairs (e.g. '50 thousand/uL') accepts proximity match
    within a 60-char window — necessary because PDF tables split value and
    unit onto separate lines."""
    n = normalise(needle)
    if not n:
        return True
    if n in hay_norm:
        return True
    # Long-sentence fallback: 70-char prefix
    if len(n) > 80:
        if n[:70] in hay_norm:
            return True
    # Value+unit proximity fallback: split on first space and look for both
    # tokens within a small window in the haystack. Only useful for short
    # 2-token strings like "50 thousand/ul" or "2.0 mmol/l".
    parts = n.split(" ", 1)
    if len(parts) == 2 and len(n) <= 30:
        a, b = parts
        # find all positions of `a` and check if `b` appears within 60 chars
        start = 0
        while True:
            idx = hay_norm.find(a, start)
            if idx < 0:
                break
            window = hay_norm[idx: idx + 60 + len(a)]
            if b in window:
                return True
            start = idx + 1
    return False


def fmt_value_unit(spec: dict) -> list[str]:
    """Yield numeric-value-with-unit strings that should appear verbatim."""
    out = []
    val = spec.get("value", spec.get("threshold"))
    unit = (spec.get("unit") or "").strip()
    if val is None:
        return out
    if isinstance(val, list):
        for v in val:
            out.append(f"{v} {unit}".strip())
    else:
        out.append(f"{val} {unit}".strip())
    return out


# --------------------------------------------------------------------- harvest
def harvest_must_appear(g: dict) -> list[tuple[str, str]]:
    """Return (label, needle) pairs we expect in the PDF."""
    items: list[tuple[str, str]] = []

    # -------- meta
    cfg = g.get("guardrail_config", {})
    for k in ("version", "last_updated", "maintainer", "sme_review_date"):
        if cfg.get(k):
            items.append((f"meta.{k}", str(cfg[k])))

    # -------- critical thresholds: every value, unit, description,
    # clinical_rationale, sme_notes, context_check
    for cat_key, cat in g.get("critical_thresholds", {}).items():
        if cat_key.startswith("_") or not isinstance(cat, dict):
            continue
        for p_key, p in cat.items():
            if p_key.startswith("_") or not isinstance(p, dict):
                continue
            base = f"critical_thresholds.{cat_key}.{p_key}"
            # parameter human name
            items.append((f"{base}.name",
                          p_key.replace("_", " ")))
            for vu in fmt_value_unit(p):
                items.append((f"{base}.value", vu))
            for fld in ("description", "clinical_rationale",
                        "sme_notes", "context_check",
                        "action", "alert_action"):
                if p.get(fld):
                    items.append((f"{base}.{fld}", p[fld]))

    # -------- early detection patterns
    for pat_key, pat in g.get("early_detection_patterns", {}).items():
        if pat_key.startswith("_") or not isinstance(pat, dict):
            continue
        base = f"early_detection.{pat_key}"
        if pat.get("description"):
            items.append((f"{base}.description", pat["description"]))
        if pat.get("action"):
            items.append((f"{base}.action", pat["action"]))
        crits = pat.get("criteria", {})
        if isinstance(crits, dict):
            for cr_k, cr_v in crits.items():
                if isinstance(cr_v, dict):
                    for vu in fmt_value_unit(cr_v):
                        items.append((f"{base}.criteria.{cr_k}", vu))

    # -------- history context checks
    for hk, hv in g.get("history_context_checks", {}).items():
        if hk.startswith("_") or not isinstance(hv, dict):
            continue
        base = f"history.{hk}"
        for cond in hv.get("conditions_to_check", []) or []:
            items.append((f"{base}.cond", cond))
        for fld in ("impact", "alert_action"):
            if hv.get(fld):
                items.append((f"{base}.{fld}", hv[fld]))
        # nested medications
        if hk == "medications_affecting_labs":
            for med, info in hv.items():
                if not isinstance(info, dict):
                    continue
                for fld in ("impact", "alert_action"):
                    if info.get(fld):
                        items.append((f"{base}.{med}.{fld}", info[fld]))

    # -------- override logic
    ol = g.get("override_logic", {})
    trig = ol.get("trigger_conditions", {})
    if trig.get("minimum_risk_for_critical") is not None:
        items.append(("override.min_risk", str(trig["minimum_risk_for_critical"])))
    ov = ol.get("override_values", {})
    for fld in ("forced_risk_score", "forced_priority", "forced_probability_6h"):
        if ov.get(fld) is not None:
            items.append((f"override.{fld}", str(ov[fld])))
    sh = ol.get("shock_criteria", {})
    for fld in ("description", "condition_1", "condition_2", "action"):
        if sh.get(fld):
            items.append((f"shock.{fld}", sh[fld]))
    dic = ol.get("dic_criteria", {})
    for fld in ("description", "action"):
        if dic.get(fld):
            items.append((f"dic.{fld}", dic[fld]))
    for c in dic.get("criteria", []) or []:
        items.append(("dic.criterion", c))

    # -------- qSOFA
    qs = g.get("qsofa_criteria", {})
    for k in ("respiratory_rate", "systolic_bp"):
        v = qs.get(k, {})
        for vu in fmt_value_unit(v):
            items.append((f"qsofa.{k}", vu))
    if qs.get("altered_mentation", {}).get("description"):
        items.append(("qsofa.altered_mentation",
                      qs["altered_mentation"]["description"]))

    # -------- discordance rules
    dr = g.get("discordance_rules", {})
    for fld in ("perfusion_concerning_phrases", "fluid_response_phrases",
                "vasopressor_phrases", "urine_output_phrases",
                "respiratory_distress_phrases"):
        for ph in dr.get(fld, []) or []:
            items.append((f"discordance.{fld}", ph))
    ms = dr.get("mental_status_phrases", {})
    for ph in ms.get("phrases", []) or []:
        items.append(("discordance.mental_status_phrase", ph))
    if dr.get("escalation_risk_score") is not None:
        items.append(("discordance.escalation_risk_score",
                      str(dr["escalation_risk_score"])))
    if dr.get("escalation_priority"):
        items.append(("discordance.escalation_priority",
                      dr["escalation_priority"]))

    return items


# --------------------------------------------------------------------- main
def main() -> int:
    if not JSON_PATH.exists():
        print(f"ERROR: JSON not found: {JSON_PATH}", file=sys.stderr)
        return 1
    if not PDF_PATH.exists():
        print(f"ERROR: PDF not found: {PDF_PATH}", file=sys.stderr)
        return 1

    g = json.loads(JSON_PATH.read_text())

    # Extract PDF text in reading order (no -layout). This concatenates
    # column text into single lines, which gives the most lenient match
    # for contiguous strings split across table columns.
    pdf_reading = subprocess.run(
        ["pdftotext", str(PDF_PATH), "-"],
        check=True, capture_output=True, text=True,
    ).stdout
    # Also extract with -layout, which is better for tabular value+unit
    # pairs that sit close to each other on the same row.
    pdf_layout = subprocess.run(
        ["pdftotext", "-layout", str(PDF_PATH), "-"],
        check=True, capture_output=True, text=True,
    ).stdout
    hay_reading = normalise(pdf_reading)
    hay_layout = normalise(pdf_layout)
    # Combined haystack — needle is "present" if found in EITHER extraction.
    hay = hay_reading + " || " + hay_layout

    items = harvest_must_appear(g)

    missing: list[tuple[str, str]] = []
    by_category_total: dict[str, int] = {}
    by_category_missing: dict[str, int] = {}
    for label, needle in items:
        cat = label.split(".", 1)[0]
        by_category_total[cat] = by_category_total.get(cat, 0) + 1
        if not is_present(needle, hay):
            missing.append((label, needle))
            by_category_missing[cat] = by_category_missing.get(cat, 0) + 1

    total = len(items)
    found = total - len(missing)
    pct = 100.0 * found / total if total else 100.0

    print("=" * 72)
    print(" GUARDRAIL JSON → PDF CONTENT COVERAGE CHECK")
    print("=" * 72)
    print(f" JSON   : {JSON_PATH.relative_to(REPO)}")
    print(f" PDF    : {PDF_PATH.relative_to(REPO)}")
    print(f" Fields checked : {total}")
    print(f" Found in PDF   : {found}  ({pct:.1f}%)")
    print(f" Missing        : {len(missing)}")
    print()
    print(" By section:")
    print(f"   {'section':28s}  {'total':>6s}  {'missing':>8s}")
    for cat in sorted(by_category_total):
        miss = by_category_missing.get(cat, 0)
        flag = "  OK" if miss == 0 else "  <-- check"
        print(f"   {cat:28s}  {by_category_total[cat]:6d}  {miss:8d}{flag}")
    print()

    if missing:
        print(" Missing items (first 30):")
        for label, needle in missing[:30]:
            short = (needle[:90] + "...") if len(needle) > 90 else needle
            print(f"   - {label:50s} :: {short}")
        if len(missing) > 30:
            print(f"   ... and {len(missing) - 30} more")
        print()
        print(" RESULT: FAIL — please re-check missing fields above")
        return 1

    print(" RESULT: PASS — every required field is present in the PDF")
    return 0


if __name__ == "__main__":
    sys.exit(main())
