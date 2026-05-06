"""
Generate a clinically-friendly PDF of the Sepsis Guardrail config for SME review.

Reads `genai_clinical_guardrail.json` from the repo root and produces a
nicely-formatted PDF organized by clinical category (hemodynamic, respiratory,
renal, etc.) plus override logic, qSOFA, early-detection patterns, history
context, and discordance rules.

Output: docs/clinical-review/Medbeacon_Guardrail_Review_v<version>.pdf

Usage:
    python3 scripts/generate_guardrail_review_pdf.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER

# --- Paths ---
REPO_ROOT = Path(__file__).resolve().parent.parent
GUARDRAIL_JSON = REPO_ROOT / "genai_clinical_guardrail.json"
OUTPUT_DIR = REPO_ROOT / "docs" / "clinical-review"

# --- Brand palette ---
BRAND_NAVY = colors.HexColor("#0E2841")
BRAND_BLUE = colors.HexColor("#156082")
BRAND_TEAL = colors.HexColor("#506370")
BRAND_GREY = colors.HexColor("#454545")
BRAND_LIGHT_GREY = colors.HexColor("#898A87")
BRAND_BG = colors.HexColor("#F4F6F8")
BRAND_ROW = colors.HexColor("#FAFBFC")
BRAND_ACCENT = colors.HexColor("#E97132")  # warm orange (matches Winter brand)

# --- Layout constants ---
PAGE_W, PAGE_H = letter
LEFT, RIGHT = 0.7 * inch, 0.7 * inch
TOP, BOTTOM = 0.9 * inch, 0.7 * inch
USABLE_W = PAGE_W - LEFT - RIGHT


# ======================================================================
# Helpers
# ======================================================================
def make_styles():
    base = getSampleStyleSheet()
    return {
        "TitleMain": ParagraphStyle(
            "TitleMain", parent=base["Title"],
            fontSize=24, leading=28, textColor=BRAND_NAVY,
            spaceAfter=6, alignment=TA_LEFT, fontName="Helvetica-Bold",
        ),
        "Subtitle": ParagraphStyle(
            "Subtitle", parent=base["Normal"],
            fontSize=12, leading=16, textColor=BRAND_TEAL,
            spaceAfter=18, fontName="Helvetica",
        ),
        "H1": ParagraphStyle(
            "H1", parent=base["Heading1"],
            fontSize=16, leading=20, textColor=BRAND_NAVY,
            spaceBefore=14, spaceAfter=8, fontName="Helvetica-Bold",
            keepWithNext=True,
        ),
        "H2": ParagraphStyle(
            "H2", parent=base["Heading2"],
            fontSize=12, leading=15, textColor=BRAND_BLUE,
            spaceBefore=10, spaceAfter=4, fontName="Helvetica-Bold",
            keepWithNext=True,
        ),
        "Body": ParagraphStyle(
            "Body", parent=base["Normal"],
            fontSize=10, leading=14, textColor=BRAND_GREY,
            spaceAfter=6, fontName="Helvetica",
        ),
        "Note": ParagraphStyle(
            "Note", parent=base["Normal"],
            fontSize=9, leading=13, textColor=BRAND_LIGHT_GREY,
            spaceAfter=4, fontName="Helvetica-Oblique",
        ),
        "Cell": ParagraphStyle(
            "Cell", parent=base["Normal"],
            fontSize=9, leading=12, textColor=BRAND_GREY,
            fontName="Helvetica",
        ),
        "CellBold": ParagraphStyle(
            "CellBold", parent=base["Normal"],
            fontSize=9, leading=12, textColor=BRAND_NAVY,
            fontName="Helvetica-Bold",
        ),
        "CellMono": ParagraphStyle(
            "CellMono", parent=base["Normal"],
            fontSize=9, leading=12, textColor=BRAND_NAVY,
            fontName="Courier-Bold",
        ),
        "Tag": ParagraphStyle(
            "Tag", parent=base["Normal"],
            fontSize=8, leading=10, textColor=BRAND_TEAL,
            fontName="Helvetica-Bold",
        ),
    }


def header_footer(canvas, doc):
    """Page header + footer drawn on every page."""
    canvas.saveState()
    # Footer
    canvas.setStrokeColor(colors.HexColor("#E0E0E0"))
    canvas.setLineWidth(0.5)
    canvas.line(LEFT, BOTTOM - 0.15 * inch, PAGE_W - RIGHT, BOTTOM - 0.15 * inch)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(BRAND_LIGHT_GREY)
    canvas.drawString(LEFT, BOTTOM - 0.32 * inch,
                      "Confidential · Sepsis GenAI · Medbeacon · For Clinical SME Review")
    canvas.drawRightString(PAGE_W - RIGHT, BOTTOM - 0.32 * inch,
                           f"Page {canvas.getPageNumber()}")
    # Top mark on non-cover pages
    if doc.page > 1:
        canvas.setFillColor(BRAND_ACCENT)
        canvas.setFont("Helvetica-Bold", 9)
        canvas.drawString(LEFT, PAGE_H - 0.55 * inch, "MEDBEACON")
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(BRAND_LIGHT_GREY)
        canvas.drawString(LEFT + 0.95 * inch, PAGE_H - 0.55 * inch,
                          "Sepsis GenAI — Clinical Guardrail Configuration")
    canvas.restoreState()


def thr_table(rows, col_widths):
    """Standard threshold table with brand styling."""
    t = Table(rows, colWidths=col_widths, hAlign="LEFT", repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BRAND_NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
        ("TOPPADDING", (0, 0), (-1, 0), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BRAND_ROW]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E0E0E0")),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 1), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return t


def section_band(title, S):
    """Coloured band before a major section."""
    band = Table([[Paragraph(f"<b>{title}</b>", S["TitleMain"])]],
                 colWidths=[USABLE_W])
    band.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BRAND_BG),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LINEBEFORE", (0, 0), (0, -1), 4, BRAND_ACCENT),
    ]))
    return band


def fmt_threshold(spec):
    """Render '{value: 90, condition: '<=', unit: 'mmHg'}' as '<= 90 mmHg'."""
    cond = spec.get("condition", "")
    val = spec.get("value", spec.get("threshold", ""))
    unit = spec.get("unit", "")
    if isinstance(val, list):
        val = " or ".join(str(x) for x in val)
    if cond and val != "":
        return f"{cond} {val} {unit}".strip()
    return str(val or "—")


def threshold_interpretation(spec, S):
    """Compose a rich Paragraph combining description, clinical rationale, SME notes,
    and context-check from a threshold spec dict — exactly what an SME wants to see."""
    parts = []
    if spec.get("description"):
        parts.append(short(spec["description"]))
    if spec.get("clinical_rationale"):
        parts.append(f"<b>Clinical rationale:</b> {short(spec['clinical_rationale'])}")
    if spec.get("sme_notes"):
        parts.append(f"<b>SME notes:</b> <i>{short(spec['sme_notes'])}</i>")
    if spec.get("context_check"):
        parts.append(f"<b>Consider also:</b> {short(spec['context_check'])}")
    if spec.get("action") or spec.get("alert_action"):
        parts.append(f"<b>Action:</b> {short(spec.get('action') or spec.get('alert_action'))}")
    text = "<br/>".join(parts) if parts else "—"
    return Paragraph(text, S["Cell"])


def short(text, n=None):
    if not text:
        return "—"
    s = str(text).strip()
    if n and len(s) > n:
        s = s[: n - 1].rstrip() + "…"
    return s


# ======================================================================
# Section builders
# ======================================================================
def build_cover(g, S, story):
    cfg = g.get("guardrail_config", {})
    today = datetime.today().strftime("%B %d, %Y")
    story.append(Spacer(1, 1.2 * inch))
    story.append(Paragraph("MEDBEACON", ParagraphStyle(
        "logoText", parent=S["Body"], fontSize=11, textColor=BRAND_ACCENT,
        fontName="Helvetica-Bold", spaceAfter=2)))
    story.append(Paragraph(
        "Sepsis GenAI · Clinical Decision Support",
        ParagraphStyle("BrandSub", parent=S["Body"], fontSize=10,
                       textColor=BRAND_TEAL, spaceAfter=40)))

    story.append(Paragraph("Clinical Guardrail Configuration", S["TitleMain"]))
    story.append(Paragraph(
        f"Version {cfg.get('version', '—')} · For Clinical SME Review",
        S["Subtitle"]))

    meta_rows = [
        ["Document purpose", cfg.get("purpose", "")],
        ["Configuration version", cfg.get("version", "")],
        ["Last updated", cfg.get("last_updated", "")],
        ["Maintainer", cfg.get("maintainer", "")],
        ["Last SME review", cfg.get("sme_review_date", "")],
        ["This review prepared for", "Paula"],
        ["Prepared on", today],
    ]
    table = Table([
        [Paragraph(f"<b>{k}</b>", S["Cell"]), Paragraph(v, S["Cell"])]
        for k, v in meta_rows
    ], colWidths=[1.7 * inch, USABLE_W - 1.7 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), BRAND_BG),
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor("#E0E0E0")),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(table)
    story.append(Spacer(1, 30))

    story.append(Paragraph("How to use this document", S["H2"]))
    story.append(Paragraph(
        "This document captures the deterministic clinical guardrail rules that wrap the Sepsis "
        "GenAI pipeline. The AI model produces a risk score; these rules are evaluated "
        "<b>after</b> the AI output and can <b>override</b> the AI when critical thresholds are "
        "breached, ensuring no clinically dangerous picture is ever down-rated.",
        S["Body"]))
    story.append(Paragraph(
        "Please review each threshold for clinical appropriateness. Mark up freely — "
        "track-changes welcome, or annotate the PDF directly. We can adjust any value or rule "
        "and reload the configuration without redeploying the system.",
        S["Body"]))

    cl = cfg.get("change_log")
    if cl:
        story.append(Spacer(1, 8))
        story.append(Paragraph("Recent changes", S["H2"]))
        story.append(Paragraph(cl, S["Note"]))

    story.append(PageBreak())


def build_overview(g, S, story):
    story.append(section_band("1. Overview", S))
    story.append(Spacer(1, 6))

    story.append(Paragraph("Where guardrails sit in the pipeline", S["H1"]))
    story.append(Paragraph(
        "The Sepsis GenAI system is a three-stage pipeline:", S["Body"]))
    pipeline = Table([[
        Paragraph("<b>1. Preprocess</b><br/>Vitals · labs · nurse notes are normalised and "
                  "trended into a clinical narrative.", S["Cell"]),
        Paragraph("<b>2. AI Reasoning</b><br/>Claude Sonnet 4.5 reads the narrative and "
                  "produces a risk score, priority, and rationale.", S["Cell"]),
        Paragraph("<b>3. Clinical Guardrails</b><br/>The rules in this document are evaluated. "
                  "If a critical threshold is breached they override the AI output.", S["Cell"]),
    ]], colWidths=[USABLE_W / 3] * 3)
    pipeline.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BRAND_BG),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#D9D9D9")),
        ("LINEAFTER", (0, 0), (1, -1), 0.5, colors.HexColor("#D9D9D9")),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(pipeline)
    story.append(Spacer(1, 14))

    story.append(Paragraph("Override logic in plain English", S["H1"]))
    ol = g.get("override_logic", {})
    trig = ol.get("trigger_conditions", {})
    story.append(Paragraph(
        f"<b>Rule:</b> if the AI risk score is below "
        f"<b>{trig.get('minimum_risk_for_critical', 80)}</b> but any critical threshold below "
        f"is breached, the system forces the output to:", S["Body"]))
    ov = ol.get("override_values", {})
    forced = Table([[
        Paragraph(f"<b>Risk score</b><br/>{ov.get('forced_risk_score', '—')}", S["Cell"]),
        Paragraph(f"<b>Priority</b><br/>{ov.get('forced_priority', '—')}", S["Cell"]),
        Paragraph(f"<b>6-h probability</b><br/>{ov.get('forced_probability_6h', '—')}", S["Cell"]),
    ]], colWidths=[USABLE_W / 3] * 3)
    forced.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("BOX", (0, 0), (-1, -1), 0.5, BRAND_ACCENT),
        ("LINEAFTER", (0, 0), (1, -1), 0.5, colors.HexColor("#E0E0E0")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(forced)

    expert = g.get("expert_notes", {})
    if expert:
        story.append(Spacer(1, 14))
        story.append(Paragraph("Key principles", S["H1"]))
        principles = expert.get("key_principles", [])
        for p in principles:
            story.append(Paragraph(f"• {p}", S["Body"]))
        if expert.get("caution"):
            story.append(Spacer(1, 6))
            story.append(Paragraph(f"<i>Caution:</i> {expert['caution']}", S["Note"]))


def build_critical_thresholds(g, S, story):
    story.append(PageBreak())
    story.append(section_band("2. Critical Thresholds", S))
    story.append(Spacer(1, 4))
    desc = g["critical_thresholds"].get("_description", "")
    if desc:
        story.append(Paragraph(desc, S["Note"]))
    story.append(Spacer(1, 6))

    # Display order + nicer human names
    category_order = [
        ("hemodynamic",       "Hemodynamic"),
        ("perfusion_markers", "Perfusion markers"),
        ("respiratory",       "Respiratory"),
        ("temperature",       "Temperature"),
        ("renal",             "Renal"),
        ("hepatic",           "Hepatic"),
        ("hematologic",       "Hematologic"),
        ("metabolic",         "Metabolic"),
        ("cardiac",           "Cardiac"),
        ("neurologic",        "Neurologic"),
        ("infection_markers", "Infection markers"),
    ]

    for key, display in category_order:
        cat = g["critical_thresholds"].get(key)
        if not cat:
            continue
        story.append(Paragraph(display, S["H2"]))

        rows = [["Parameter", "Threshold", "Clinical interpretation"]]
        for p_key, p in cat.items():
            if p_key.startswith("_") or not isinstance(p, dict):
                continue
            param_name = p.get("name") or p_key.replace("_", " ").title()
            rows.append([
                Paragraph(f"<b>{param_name}</b>", S["Cell"]),
                Paragraph(fmt_threshold(p), S["CellMono"]),
                threshold_interpretation(p, S),
            ])
        story.append(thr_table(rows, [1.4 * inch, 1.1 * inch, USABLE_W - 2.5 * inch]))
        story.append(Spacer(1, 10))


def build_qsofa_and_shock(g, S, story):
    story.append(PageBreak())
    story.append(section_band("3. qSOFA and Shock Criteria", S))
    story.append(Spacer(1, 6))

    story.append(Paragraph("qSOFA (Quick SOFA)", S["H1"]))
    qs = g.get("qsofa_criteria", {})
    if qs.get("_description"):
        story.append(Paragraph(qs["_description"], S["Note"]))
    rows = [["Parameter", "Criterion", "Notes"]]
    for k in ("respiratory_rate", "altered_mentation", "systolic_bp"):
        v = qs.get(k, {})
        if not isinstance(v, dict):
            continue
        crit = fmt_threshold(v) if v.get("threshold") is not None else v.get("description", "—")
        rows.append([
            Paragraph(f"<b>{k.replace('_', ' ').title()}</b>", S["Cell"]),
            Paragraph(crit, S["CellMono"]),
            Paragraph(v.get("notes", v.get("description", "—")), S["Cell"]),
        ])
    story.append(thr_table(rows, [1.6 * inch, 1.6 * inch, 3.7 * inch]))
    si = qs.get("score_interpretation", {})
    if si:
        story.append(Spacer(1, 6))
        story.append(Paragraph("Score interpretation", S["H2"]))
        for k, v in si.items():
            story.append(Paragraph(f"<b>{k}</b> — {v}", S["Body"]))
    story.append(Spacer(1, 12))

    ol = g.get("override_logic", {})
    sh = ol.get("shock_criteria", {})
    if sh:
        story.append(Paragraph("Septic shock criteria", S["H1"]))
        if sh.get("description"):
            story.append(Paragraph(sh["description"], S["Note"]))
        story.append(Paragraph(
            f"<b>Both</b> conditions must be met:", S["Body"]))
        story.append(Paragraph(f"1. {sh.get('condition_1', '—')}", S["Body"]))
        story.append(Paragraph(f"2. {sh.get('condition_2', '—')}", S["Body"]))
        if sh.get("action"):
            story.append(Paragraph(f"<b>Action:</b> {sh['action']}", S["Body"]))
        story.append(Spacer(1, 10))

    dic = ol.get("dic_criteria", {})
    if dic:
        story.append(Paragraph("DIC (Disseminated Intravascular Coagulation)", S["H1"]))
        if dic.get("description"):
            story.append(Paragraph(dic["description"], S["Note"]))
        story.append(Paragraph(
            f"<b>{'Three or more' if dic.get('requires_three_or_more') else 'Any'}</b> of:",
            S["Body"]))
        for c in dic.get("criteria", []):
            story.append(Paragraph(f"• {c}", S["Body"]))
        if dic.get("action"):
            story.append(Paragraph(f"<b>Action:</b> {dic['action']}", S["Body"]))


def build_early_detection(g, S, story):
    story.append(PageBreak())
    story.append(section_band("4. Early-Detection Patterns", S))
    story.append(Spacer(1, 4))
    edp = g.get("early_detection_patterns", {})
    if edp.get("_description"):
        story.append(Paragraph(edp["_description"], S["Note"]))
    story.append(Spacer(1, 6))

    for key, p in edp.items():
        if key.startswith("_") or not isinstance(p, dict):
            continue
        title = key.replace("_", " ").title()
        story.append(Paragraph(title, S["H2"]))
        if p.get("description"):
            story.append(Paragraph(p["description"], S["Body"]))
        rule = "All of the below must be true" if p.get("requires_all") else \
               ("Any two or more of the below" if p.get("requires_two_or_more") else
                "Match the criteria below")
        story.append(Paragraph(f"<i>Rule:</i> {rule}", S["Note"]))
        crits = p.get("criteria", {})
        if isinstance(crits, dict):
            rows = [["Parameter", "Criterion"]]
            for cr_k, cr_v in crits.items():
                if isinstance(cr_v, dict):
                    rows.append([
                        Paragraph(f"<b>{cr_k.replace('_', ' ').title()}</b>", S["Cell"]),
                        Paragraph(fmt_threshold(cr_v) or cr_v.get("description", "—"),
                                  S["CellMono"]),
                    ])
                else:
                    rows.append([
                        Paragraph(f"<b>{cr_k.replace('_', ' ').title()}</b>", S["Cell"]),
                        Paragraph(str(cr_v), S["Cell"]),
                    ])
            story.append(thr_table(rows, [2.0 * inch, USABLE_W - 2.0 * inch]))
        if p.get("action"):
            story.append(Spacer(1, 4))
            story.append(Paragraph(f"<b>Action:</b> {p['action']}", S["Body"]))
        story.append(Spacer(1, 10))


def build_history_context(g, S, story):
    story.append(PageBreak())
    story.append(section_band("5. Patient-History Context Checks", S))
    story.append(Spacer(1, 4))
    hc = g.get("history_context_checks", {})
    if hc.get("_description"):
        story.append(Paragraph(hc["_description"], S["Note"]))
    story.append(Spacer(1, 6))

    rows = [["History area", "Conditions to flag", "Clinical impact", "Alert action"]]
    # Keys other than `medications_affecting_labs` (which is rendered separately
    # below) AND that carry an `impact` field — this picks up
    # neurologic_history (conditions_affecting_mental_status) and any other
    # history block that uses a non-standard list key.
    LIST_FIELD_FALLBACKS = (
        "conditions_to_check",
        "conditions_affecting_mental_status",
    )
    for k, v in hc.items():
        if k.startswith("_") or not isinstance(v, dict):
            continue
        if k == "medications_affecting_labs":
            continue
        if "impact" not in v:
            continue
        conds = []
        for fld in LIST_FIELD_FALLBACKS:
            if isinstance(v.get(fld), list):
                conds = v[fld]
                break
        title = k.replace("_", " ").title()
        rows.append([
            Paragraph(f"<b>{title}</b>", S["Cell"]),
            Paragraph(", ".join(conds) or "—", S["Cell"]),
            Paragraph(v.get("impact", "—"), S["Cell"]),
            Paragraph(v.get("alert_action", "—"), S["Cell"]),
        ])
    story.append(thr_table(rows, [1.1 * inch, 1.8 * inch, 2.1 * inch, 1.9 * inch]))

    meds = hc.get("medications_affecting_labs", {})
    if meds:
        story.append(Spacer(1, 14))
        story.append(Paragraph("Medications that affect interpretation", S["H2"]))
        for cls, info in meds.items():
            if not isinstance(info, dict):
                continue
            story.append(Paragraph(f"<b>{cls.title()}</b>", S["Body"]))
            if info.get("impact"):
                story.append(Paragraph(info["impact"], S["Note"]))
            if info.get("alert_action"):
                story.append(Paragraph(f"<i>Action:</i> {info['alert_action']}", S["Note"]))


def build_discordance(g, S, story):
    story.append(PageBreak())
    story.append(section_band("6. Discordance Rules — Silent Sepsis", S))
    story.append(Spacer(1, 4))
    dr = g.get("discordance_rules", {})
    if dr.get("_description"):
        story.append(Paragraph(dr["_description"], S["Note"]))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Trigger logic", S["H2"]))
    story.append(Paragraph(
        f"If concerning phrases are detected in the nurse notes <b>and</b> the AI risk score "
        f"is below 60, the system escalates the priority to "
        f"<b>{dr.get('escalation_priority', 'High')}</b> with risk score "
        f"<b>{dr.get('escalation_risk_score', 70)}</b>.",
        S["Body"]))
    story.append(Spacer(1, 8))

    phrase_groups = [
        ("Perfusion concerns", dr.get("perfusion_concerning_phrases", [])),
        ("Fluid response", dr.get("fluid_response_phrases", [])),
        ("Vasopressor mention", dr.get("vasopressor_phrases", [])),
        ("Urine output", dr.get("urine_output_phrases", [])),
        ("Respiratory distress", dr.get("respiratory_distress_phrases", [])),
    ]
    rows = [["Category", "Trigger phrases (any of)"]]
    for label, phrases in phrase_groups:
        if not phrases:
            continue
        rows.append([
            Paragraph(f"<b>{label}</b>", S["Cell"]),
            Paragraph(", ".join(phrases), S["Cell"]),
        ])
    ms = dr.get("mental_status_phrases", {})
    if ms:
        rows.append([
            Paragraph("<b>Mental status (new onset)</b>", S["Cell"]),
            Paragraph(
                ", ".join(ms.get("phrases", [])) +
                (f" <br/><i>Qualifier:</i> {ms.get('qualifier', '')}" if ms.get("qualifier") else ""),
                S["Cell"]),
        ])
    story.append(thr_table(rows, [1.5 * inch, USABLE_W - 1.5 * inch]))


def build_review_signoff(S, story):
    story.append(PageBreak())
    story.append(section_band("7. SME Review & Sign-off", S))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Reviewer", S["H2"]))

    rows = [
        ["Name", "Paula"],
        ["Role", ""],
        ["Date reviewed", ""],
        ["Approve as-is?", "[  ] Yes        [  ] With changes (see annotations)"],
        ["Re-review required?", "[  ] No         [  ] Yes — date __________"],
        ["Signature", ""],
    ]
    table = Table([
        [Paragraph(f"<b>{k}</b>", S["Cell"]), Paragraph(v, S["Cell"])]
        for k, v in rows
    ], colWidths=[1.6 * inch, USABLE_W - 1.6 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), BRAND_BG),
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor("#E0E0E0")),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(table)

    story.append(Spacer(1, 14))
    story.append(Paragraph("Free-text comments", S["H2"]))
    box = Table([[Paragraph("&nbsp;", S["Cell"])]], colWidths=[USABLE_W],
                rowHeights=[2.4 * inch])
    box.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#C0C0C0")),
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
    ]))
    story.append(box)


# ======================================================================
# Main
# ======================================================================
def build():
    if not GUARDRAIL_JSON.exists():
        print(f"ERROR: {GUARDRAIL_JSON} not found", file=sys.stderr)
        sys.exit(1)
    g = json.loads(GUARDRAIL_JSON.read_text())
    cfg = g.get("guardrail_config", {})
    version = cfg.get("version", "0.0").replace(".", "_")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"Medbeacon_Guardrail_Review_v{version}.pdf"

    doc = BaseDocTemplate(
        str(out),
        pagesize=letter,
        leftMargin=LEFT, rightMargin=RIGHT,
        topMargin=TOP, bottomMargin=BOTTOM,
        title=f"Medbeacon Guardrail Review v{cfg.get('version', '')}",
        author="Sepsis GenAI Team",
        subject="Clinical guardrail configuration for SME review",
    )
    frame = Frame(LEFT, BOTTOM, USABLE_W, PAGE_H - TOP - BOTTOM, id="main")
    doc.addPageTemplates([PageTemplate(id="all", frames=[frame], onPage=header_footer)])

    S = make_styles()
    story = []
    build_cover(g, S, story)
    build_overview(g, S, story)
    build_critical_thresholds(g, S, story)
    build_qsofa_and_shock(g, S, story)
    build_early_detection(g, S, story)
    build_history_context(g, S, story)
    build_discordance(g, S, story)
    build_review_signoff(S, story)

    doc.build(story)
    print(f"PDF generated: {out}")
    print(f"Size: {out.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    build()
