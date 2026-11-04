"""Feature 3: the line-clearance record as a PDF, the way a batch-record entry is kept and signed.

Built from the last check (the same data as the HTML record): verdict, reference and current images, changes with
risk, zone results, candidates rejected as false alarms, operator sign-off, and the audit ids that trace every
decision back to data/events.jsonl. Standard fonts only, so it renders on every laptop.
"""

from __future__ import annotations

import io
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

INK, MUTED, LINE = colors.HexColor("#16181D"), colors.HexColor("#5D6370"), colors.HexColor("#D0D0CA")
OK, BAD = colors.HexColor("#137333"), colors.HexColor("#B3261E")

base = ParagraphStyle("b", fontName="Helvetica", fontSize=9.5, leading=13, textColor=INK)
small = ParagraphStyle("s", parent=base, fontSize=8.5, leading=11, textColor=MUTED)
cell = ParagraphStyle("c", parent=base, fontSize=8.6, leading=11)
cellb = ParagraphStyle("cb", parent=cell, fontName="Helvetica-Bold")
h1 = ParagraphStyle("h1", parent=base, fontName="Helvetica-Bold", fontSize=18, leading=22, spaceAfter=2)
h2 = ParagraphStyle("h2", parent=base, fontName="Helvetica-Bold", fontSize=11.5, leading=15, spaceBefore=8, spaceAfter=4)


def _p(text, style=cell):
    return Paragraph(escape(str(text)), style)


def _table(rows, widths):
    t = Table([[_p(c, cellb if i == 0 else cell) for c in r] for i, r in enumerate(rows)],
              colWidths=[w * mm for w in widths], repeatRows=1)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, LINE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F3F2EE")),
                           ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]))
    return t


def _img(path: Path, width_mm: float):
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    return Image(str(path), width=width_mm * mm, height=width_mm * mm * h / w)


def build(check: dict, data_dir: Path) -> bytes:
    r = check["result"]
    clear = r["line_clear"]
    s = [Paragraph("Line clearance record", h1),
         _p(f"Check {check['event_id']} · {check['time']} · model {check['model']}{' (MOCK, not a real check)' if check.get('mock') else ''}", small),
         Spacer(1, 6),
         Paragraph(f"<font color='{'#137333' if clear else '#B3261E'}'><b>{'LINE CLEAR' if clear else 'NOT CLEAR'}</b></font>",
                   ParagraphStyle("v", parent=base, fontSize=16, leading=20)),
         _p(r.get("summary", ""), base), Spacer(1, 6)]
    ref, cur = Path(data_dir) / check["ref_marked"], Path(data_dir) / check["cur_marked"]
    if ref.exists() and cur.exists():
        imgs = Table([[_p("Reference (approved)", cellb), _p("Current", cellb)], [_img(ref, 84), _img(cur, 84)]],
                     colWidths=[87 * mm, 87 * mm])
        imgs.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
        s.append(imgs)
    s.append(Paragraph("Changes", h2))
    rows = [["#", "Kind", "Object", "Risk", "Why it matters"]] + [
        [f"C{c['id']}", c["kind"], c.get("label", ""), c.get("risk", ""), c.get("explanation", "")] for c in r["changes"]]
    s.append(_table(rows, [10, 18, 34, 16, 96]) if len(rows) > 1 else _p("No changes.", base))
    if r.get("zones"):
        s.append(Paragraph("Zones", h2))
        zrows = [["Zone", "Rule", "Status", "Details"]] + [
            [z["name"], "must be empty" if z["rule"] == "empty" else "only " + ", ".join(z["allowed"]),
             "OK" if z["ok"] else "VIOLATION",
             "; ".join(z["problems"]) or ", ".join(o["name"] for o in z["objects"]) or "nothing inside"]
            for z in r["zones"]]
        s.append(_table(zrows, [34, 40, 22, 78]))
    if r.get("batch"):        # feature 11
        b = r["batch"]
        s.append(Paragraph(f"Labels (current batch LOT {b.get('lot', '')})", h2))
        if b.get("error"):
            s.append(_p(f"Labels could not be read: {b['error']}. Check them by hand.", base))
        lrows = [["Object", "Lot", "Expiry", "Code", "Print"]] + [
            [str(it["box_id"]), it["lot"], it["expiry"], it.get("code") or "", it.get("text") or ""] for it in b.get("items", [])]
        s.append(_table(lrows, [16, 26, 24, 50, 58]) if len(lrows) > 1 else _p("No labels seen.", base))
        if b.get("problems"):
            prows = [["Object", "Problem", "Risk", "Details"]] + [
                [str(p["box_id"]), p["kind"].replace("_", " "), p["risk"], p["text"]] for p in b["problems"]]
            s.append(_table(prows, [16, 30, 16, 112]))
    if r.get("filtered"):
        s.append(Paragraph("Candidates rejected as false alarms", h2))
        s += [_p(f"C{c['id']} {c['kind']}: {c.get('explanation', '')}", base) for c in r["filtered"]]
    s.append(Paragraph("Sign-off", h2))
    so = check.get("signoff")
    if so:
        s.append(_p(f"{so['operator']} · {so['decision'].upper()} · {so['time']}" + (f" · {so['note']}" if so.get("note") else ""), base))
    else:
        s.append(_p("Not signed off yet. Operator: ____________________   Decision: approve / reject   Time: ________", base))
    s += [Spacer(1, 10), _p("Trova assists the operator; the operator decides. Every request, Grok decision and "
                            f"sign-off is in the audit log (data/events.jsonl), check id {check['event_id']}.", small)]
    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm,
                      title=f"Line clearance record {check['event_id']}", author="Trova").build(s)
    return buf.getvalue()
