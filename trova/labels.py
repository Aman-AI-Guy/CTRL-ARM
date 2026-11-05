"""Feature 11: read labels and codes, check the batch. Line clearance exists to stop a leftover from the PREVIOUS
batch (a carton, a leaflet, a label roll) from ending up in the next one. Trova reads every object's label:

- QR codes and barcodes are decoded locally by OpenCV (exact, offline);
- Grok reads the printed text on close-ups: product, lot / batch number, expiry date ("?" for unreadable);
- the batch check flags: a lot different from the current batch, a code that disagrees with the print,
  an expired date, an unreadable lot or expiry, a label nobody read (Grok offline or silent on that object).
  Any of these makes the line NOT clear. The code's expiry wins over the printed one when it parses.

`label_sheet()` makes a printable sheet of demo labels (current lot, an old lot, an expired one).
"""

from __future__ import annotations

import calendar
import datetime as dt
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .boxes import Box
from .grok import GrokClient, register_mock
from .marks import HIGHLIGHT, _label, contact_sheet, draw_marks

ITEM = {"type": "object",
        "properties": {"box_id": {"type": "integer"}, "has_label": {"type": "boolean"}, "text": {"type": "string"},
                       "product": {"type": "string"}, "lot": {"type": "string"}, "expiry": {"type": "string"}},
        "required": ["box_id", "has_label", "text", "product", "lot", "expiry"], "additionalProperties": False}
SCHEMA = {"type": "object", "properties": {"items": {"type": "array", "items": ITEM}}, "required": ["items"],
          "additionalProperties": False}

PROMPT = """You read labels for Trova during pharmaceutical line clearance.
Image 1: the table from above; every object has a magenta box with a number (1 to {n}).
Image 2: close-ups, one per number.

For EVERY number return one item:
- has_label: true if the object carries a printed label, carton print or code.
- text: the printed text exactly as printed (short; keep line breaks as " / ").
- product: the product name if printed, else "".
- lot: the lot or batch number (after LOT, Lot, Lotto, Batch, Ch.-B.), else "".
- expiry: the expiry date as printed (after EXP, Exp., Scad., Scadenza, Use by), else "".
Copy characters exactly. Write "?" for any character you cannot read; never guess a lot number.
Codes decoded by the scanner (exact, trust them over the print if they disagree, and report both):
{codes}"""

LOT_RE = re.compile(r"\b(?:LOTTO|LOT|BATCH|CH\.?-?B\.?)(?![A-Z])\s*[:#.]?\s*([A-Z0-9?][A-Z0-9?\-/]*)", re.I)
EXP_RE = re.compile(r"\b(?:EXP\.?|SCAD\.?|SCADENZA|USE BY)\s*[:.]?\s*([0-9]{1,4}[-/.][0-9]{1,4}(?:[-/.][0-9]{2,4})?)(?![0-9])",
                    re.I)


def parse_lot(text: str) -> str:
    for m in LOT_RE.finditer(text or ""):
        if re.search(r"[0-9?]", m.group(1)):        # a lot number has digits ("no lot here" is not lot HERE)
            return m.group(1).upper()
    return ""


def parse_expiry_text(text: str) -> str:
    m = EXP_RE.search(text or "")
    return m.group(1) if m else ""


def parse_date(s: str) -> dt.date | None:
    """2027-06 | 2027-06-30 | 06/2027 | 06/27 | 30/06/2027 | 30.06.27  ->  the last valid day."""
    s = (s or "").strip()
    try:
        if m := re.fullmatch(r"(\d{4})[-/.](\d{1,2})(?:[-/.](\d{1,2}))?", s):
            y, mo, d = int(m[1]), int(m[2]), int(m[3]) if m[3] else None
        elif m := re.fullmatch(r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})", s):
            d, mo, y = int(m[1]), int(m[2]), int(m[3])
        elif m := re.fullmatch(r"(\d{1,2})[-/.](\d{2,4})", s):
            d, mo, y = None, int(m[1]), int(m[2])
        else:
            return None
        y += 2000 if y < 100 else 0
        return dt.date(y, mo, d or calendar.monthrange(y, mo)[1])
    except ValueError:
        return None


def norm_lot(s: str) -> str:
    return re.sub(r"[\s\-]", "", (s or "").upper())


def decode_codes(frame: np.ndarray, boxes: list[Box] | None = None) -> list[dict]:
    """Every QR code and barcode in the frame: text, kind, centre (pixels). OpenCV's multi-code detector can miss
    one code when several are in view, so each object box without a code is decoded again on its own close-up."""
    out = _decode(frame)
    for b in boxes or []:
        if any(b.contains(*c["center"]) for c in out):
            continue
        pad = 0.1 * max(b.w, b.h)
        x0, y0 = max(0, int(b.x0 - pad)), max(0, int(b.y0 - pad))
        x1, y1 = min(frame.shape[1], int(b.x1 + pad)), min(frame.shape[0], int(b.y1 + pad))
        if x1 - x0 < 30 or y1 - y0 < 30:
            continue
        for c in _decode(frame[y0:y1, x0:x1]):
            c["center"] = [c["center"][0] + x0, c["center"][1] + y0]
            out.append(c)
    return out


def _decode(frame: np.ndarray) -> list[dict]:
    out = []
    try:
        ok, texts, pts, _ = cv2.QRCodeDetector().detectAndDecodeMulti(frame)
        if ok and pts is not None:
            out += [{"text": t, "kind": "qr", "center": [float(p[:, 0].mean()), float(p[:, 1].mean())]}
                    for t, p in zip(texts, pts) if t]
    except cv2.error:
        pass
    try:
        ok, texts, kinds, pts = cv2.barcode.BarcodeDetector().detectAndDecodeWithType(frame)
        if ok and pts is not None:
            out += [{"text": t, "kind": k.lower() or "barcode", "center": [float(p[:, 0].mean()), float(p[:, 1].mean())]}
                    for t, k, p in zip(texts, kinds, pts) if t]
    except (cv2.error, AttributeError):
        pass
    return out


def _codes_by_box(codes: list[dict], boxes: list[Box]) -> dict[int, list[dict]]:
    by: dict[int, list[dict]] = {}
    for c in codes:
        inside = [b for b in boxes if b.contains(*c["center"])]
        if inside:
            by.setdefault(min(inside, key=lambda b: b.area).id, []).append(c)
    return by


@dataclass
class LabelResult:
    items: list[dict] = field(default_factory=list)
    codes: list[dict] = field(default_factory=list)
    problems: list[dict] = field(default_factory=list)
    lot: str = ""                       # the current batch the check was made against
    latency_s: float = 0.0
    offline: bool = False

    def as_dict(self) -> dict:
        return {"items": self.items, "codes": self.codes, "problems": self.problems, "lot": self.lot,
                "ok": not self.problems, "latency_s": round(self.latency_s, 2), "offline": self.offline}


def read_labels(grok: GrokClient, frame: np.ndarray, boxes: list[Box], lot: str = "",
                today: dt.date | None = None) -> LabelResult:
    codes = decode_codes(frame, boxes)
    res = LabelResult(codes=codes, lot=lot)
    if not boxes:
        return res
    by = _codes_by_box(codes, boxes)
    listing = "\n".join(f"- number {i}: {' | '.join(c['text'] for c in cs)}" for i, cs in sorted(by.items())) or "- none"
    r = grok.json(name="trova_labels", prompt=PROMPT.format(n=len(boxes), codes=listing), schema=SCHEMA,
                  images=[draw_marks(frame, boxes), contact_sheet(frame, boxes, cell=300, cols=4)],
                  mock_context={"boxes": boxes, "codes_by_box": by})
    res.latency_s, res.offline = r.latency_s, r.offline
    seen = {b.id for b in boxes}
    got = {it.get("box_id"): it for it in r.data.get("items", []) if isinstance(it, dict) and it.get("box_id") in seen}
    unread = []          # nobody read the print: offline (codes only), or Grok skipped the box
    for b in boxes:
        it = got.get(b.id)
        if it is None or r.offline:
            unread.append(b)
        it = it or {}
        code_text = " | ".join(c["text"] for c in by.get(b.id, []))
        print_lot, code_lot = (it.get("lot") or "").upper().strip(), parse_lot(code_text)
        print_exp = (it.get("expiry") or "").strip()
        print_exp = parse_expiry_text(print_exp) or print_exp          # "EXP 2025-06" -> "2025-06"
        code_exp = parse_expiry_text(code_text)
        expiry = code_exp if parse_date(code_exp) else (print_exp or code_exp)   # the decoded code is exact
        if not (it.get("has_label") or code_text or print_lot):
            continue
        res.items.append({"box_id": b.id, "box": b.to_list(), "text": it.get("text", ""), "product": it.get("product", ""),
                          "lot": code_lot or print_lot, "print_lot": print_lot, "code": code_text, "code_lot": code_lot,
                          "expiry": expiry, "print_expiry": print_exp})
    why = "Grok offline, only codes were decoded" if r.offline else "Grok gave no answer for it"
    res.problems = check_batch(res.items, lot, today) + [
        {"box_id": b.id, "kind": "not_read", "risk": "medium",
         "text": f"object {b.id}: printed label not read ({why}): check it by hand"} for b in unread]
    return res


def check_batch(items: list[dict], lot: str, today: dt.date | None = None) -> list[dict]:
    today = today or dt.date.today()
    out = []

    def add(it, kind, risk, text):
        out.append({"box_id": it["box_id"], "kind": kind, "risk": risk, "text": text})
    for it in items:
        n = f"object {it['box_id']}"
        if it["code_lot"] and it["print_lot"] and "?" not in it["print_lot"] and norm_lot(it["code_lot"]) != norm_lot(it["print_lot"]):
            add(it, "code_mismatch", "high", f"{n}: the code says lot {it['code_lot']} but the print says {it['print_lot']}")
        if "?" in it["lot"]:
            add(it, "unreadable", "medium", f"{n}: lot number not readable ({it['lot']}): check by hand")
        elif lot and it["lot"] and norm_lot(it["lot"]) != norm_lot(lot):
            add(it, "wrong_batch", "high", f"{n}: lot {it['lot']} is not the current batch {lot}: leftover from another batch?")
        elif lot and not it["lot"]:
            add(it, "no_lot", "medium", f"{n}: has a label but no lot number was found")
        d = parse_date(it["expiry"]) if it["expiry"] else None
        if it["expiry"] and d is None:
            add(it, "unreadable_expiry", "medium", f"{n}: expiry date not readable ({it['expiry']}): check by hand")
        elif d and d < today:
            add(it, "expired", "high", f"{n}: expired on {d.isoformat()}")
    return out


def draw_labels(frame: np.ndarray, boxes: list[Box], res: LabelResult) -> np.ndarray:
    img = draw_marks(frame, boxes)
    bad = {p["box_id"] for p in res.problems}
    s = max(0.5, frame.shape[1] / 2200)
    for it in res.items:
        b = Box.from_list(it["box"])
        col = (40, 40, 220) if it["box_id"] in bad else HIGHLIGHT
        cv2.rectangle(img, (int(b.x0), int(b.y0)), (int(b.x1), int(b.y1)), col, max(3, frame.shape[1] // 400))
        txt = " ".join(x for x in (f"LOT {it['lot']}" if it["lot"] else "", f"EXP {it['expiry']}" if it["expiry"] else "") if x)
        _label(img, txt or "label", int(b.x0), int(b.y1) + int(30 * s), col, s)
    return img


def label_sheet(out_dir: Path, lot: str = "A123", old_lot: str = "B777") -> Path:
    """A4-ish PNG of 6 cut-out labels: 3 current lot, 1 old lot, 1 expired, 1 with a code that disagrees with the print."""
    out_dir.mkdir(parents=True, exist_ok=True)
    enc = cv2.QRCodeEncoder.create()
    year = dt.date.today().year
    specs = [(f"LOT {lot} EXP {year + 2}-06", f"Paracetamol 500 mg / LOT {lot} / EXP {year + 2}-06")] * 3 + [
        (f"LOT {old_lot} EXP {year + 1}-03", f"Paracetamol 500 mg / LOT {old_lot} / EXP {year + 1}-03"),
        (f"LOT {lot} EXP {year - 1}-01", f"Paracetamol 500 mg / LOT {lot} / EXP {year - 1}-01"),
        (f"LOT {old_lot} EXP {year + 2}-06", f"Paracetamol 500 mg / LOT {lot} / EXP {year + 2}-06")]
    W, H, cw, ch = 2480, 3508, 1100, 1000
    sheet = np.full((H, W), 255, np.uint8)
    for i, (code, printed) in enumerate(specs):
        x, y = 140 + (i % 2) * (cw + 100), 160 + (i // 2) * (ch + 120)
        cv2.rectangle(sheet, (x, y), (x + cw, y + ch), 0, 3)
        q = enc.encode(code)
        q = cv2.resize(q, (q.shape[1] * 22, q.shape[0] * 22), interpolation=cv2.INTER_NEAREST)
        q = cv2.copyMakeBorder(q, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)
        sheet[y + 40:y + 40 + q.shape[0], x + (cw - q.shape[1]) // 2:x + (cw - q.shape[1]) // 2 + q.shape[1]] = q
        for j, line in enumerate(printed.split(" / ")):
            cv2.putText(sheet, line, (x + 60, y + 40 + q.shape[0] + 70 + j * 75), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 0, 5)
    cv2.putText(sheet, f"Trova demo labels: current batch LOT {lot}. Cut out, stick on cartons or blisters.",
                (140, H - 80), cv2.FONT_HERSHEY_SIMPLEX, 1.6, 0, 3)
    p = out_dir / "trova_labels.png"
    cv2.imwrite(str(p), sheet)
    return p


def _mock_labels(prompt: str, images: list, ctx: dict) -> dict:
    """Offline: report what the codes say (the print is assumed to match); objects without codes have no label."""
    items = []
    for b in ctx["boxes"]:
        text = " | ".join(c["text"] for c in ctx["codes_by_box"].get(b.id, []))
        items.append({"box_id": b.id, "has_label": bool(text), "text": text, "product": "",
                      "lot": parse_lot(text), "expiry": parse_expiry_text(text)})
    return {"items": items}


register_mock("trova_labels", _mock_labels)
