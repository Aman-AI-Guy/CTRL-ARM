"""'Find this': a request in words and/or a photo -> one numbered box (or an honest "not found").

Set-of-marks: plain code proposes numbered boxes, Grok chooses among them. Grok never has to
produce pixel coordinates for the main path. If Grok says the object is visible but has no box
(the proposals missed it), we fall back to asking Grok for a box directly.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import numpy as np

from .boxes import Box
from .grok import GrokClient, register_mock
from .marks import color_name, contact_sheet, crop, draw_marks

SELECT_SCHEMA = {
    "type": "object",
    "properties": {
        "box_id": {"type": ["integer", "null"]},
        "visible_but_unboxed": {"type": "boolean"},
        "candidate_ids": {"type": "array", "items": {"type": "integer"}},
        "clarifying_question": {"type": ["string", "null"]},
        "label": {"type": "string"},
        "reason": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["box_id", "visible_but_unboxed", "candidate_ids", "clarifying_question", "label", "reason",
                 "confidence"],
    "additionalProperties": False,
}

BOX4 = {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4}
DIRECT_SCHEMA = {
    "type": "object",
    "properties": {
        "found": {"type": "boolean"},
        "box_2d": {"anyOf": [BOX4, {"type": "null"}]},
        "label": {"type": "string"},
        "reason": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["found", "box_2d", "label", "reason", "confidence"],
    "additionalProperties": False,
}

SELECT_PROMPT = """You are the eyes of Trova, a robot arm that finds objects on a work table for a person.

Image 1 is the table seen from above. Every candidate object has a magenta box with a number (1 to {n}).
Image 2 is a sheet of close-ups, one per number.{photo_line}

Request from the person: "{request}"{history}

Rules:
- Choose the number whose object best matches the request. Match on what the object is and what it is used for, not only on colour. "Something to write with" can be a pen or a marker.
- Requests about position ("left of the mug", "the one at the top") use the positions in image 1. Top of the image is the far side of the table.
- If nothing on the table matches, set box_id to null. Never choose a box just to give an answer.
- If the object is clearly visible but has no box, set box_id to null and visible_but_unboxed to true.
- If two or more boxes fit the request equally well, set box_id to null, list them in candidate_ids, and write one short clarifying_question that tells them apart. Otherwise clarifying_question is null and candidate_ids lists only your choice (or is empty).
- label: a short name for the chosen object. reason: one sentence. Write label, reason and clarifying_question in the language of the request.
- confidence: from 0 to 1."""

DIRECT_PROMPT = """Find this object in the image: "{request}".{photo_line}
Coordinates are integers normalised to 0-1000 on both axes: box_2d = [x_min, y_min, x_max, y_max], origin at the top-left.
Return a tight box. If it is not visible, found = false and box_2d = null. Never guess a box for something you cannot see.
Write label and reason in the language of the request."""


@dataclass
class FindResult:
    status: str                       # found | not_found | clarify | error
    box: Box | None = None
    boxes: list[Box] = field(default_factory=list)
    label: str = ""
    reason: str = ""
    confidence: float = 0.0
    clarifying_question: str | None = None
    candidate_ids: list[int] = field(default_factory=list)
    mode: str = "som"                 # som | direct
    grok_latency_s: float = 0.0
    total_s: float = 0.0
    grok_calls: int = 0
    marked: np.ndarray | None = None  # the exact image Grok saw

    def as_dict(self) -> dict:
        return {"status": self.status, "box": self.box.as_dict() if self.box else None,
                "label": self.label, "reason": self.reason, "confidence": round(self.confidence, 2),
                "clarifying_question": self.clarifying_question, "candidate_ids": self.candidate_ids,
                "mode": self.mode, "grok_latency_s": round(self.grok_latency_s, 2),
                "total_s": round(self.total_s, 2), "grok_calls": self.grok_calls,
                "n_candidates": len(self.boxes)}


def _history_text(history: list[tuple[str, str]] | None) -> str:
    if not history:
        return ""
    lines = [f'\nYou asked: "{q}". The person answered: "{a}".' for q, a in history]
    return "".join(lines) + "\nUse the answer to choose."


def find(grok: GrokClient, frame: np.ndarray, boxes: list[Box], request: str = "",
         query_image: np.ndarray | None = None, history: list[tuple[str, str]] | None = None,
         direct_fallback: bool = True) -> FindResult:
    t0 = time.perf_counter()
    request = (request or "").strip() or "the object in the photo"
    photo_line = ("\nImage 3 is a photo of the object the person wants. It may be a similar item, seen from "
                  "another angle or in other light." if query_image is not None else "")
    res = FindResult(status="not_found", boxes=boxes)

    if boxes:
        marked = draw_marks(frame, boxes)
        images = [marked, contact_sheet(frame, boxes)] + ([query_image] if query_image is not None else [])
        prompt = SELECT_PROMPT.format(n=len(boxes), photo_line=photo_line, request=request,
                                      history=_history_text(history))
        ctx = {"boxes": boxes, "colors": {b.id: color_name(crop(frame, b, pad=0)) for b in boxes},
               "request": request, "history": history or [], "has_photo": query_image is not None,
               "query_color": color_name(query_image) if query_image is not None else None}
        r = grok.json(name="trova_select", prompt=prompt, schema=SELECT_SCHEMA, images=images, mock_context=ctx)
        d = r.data
        res.marked, res.grok_calls, res.grok_latency_s = marked, 1, r.latency_s
        res.label, res.reason = d.get("label", ""), d.get("reason", "")
        res.confidence = float(d.get("confidence") or 0)
        ids = {b.id: b for b in boxes}
        res.candidate_ids = [i for i in d.get("candidate_ids", []) if i in ids]
        if d.get("clarifying_question") and len(res.candidate_ids) >= 2:
            res.status, res.clarifying_question = "clarify", d["clarifying_question"]
        elif d.get("box_id") in ids:
            res.status, res.box = "found", ids[d["box_id"]]
        elif d.get("visible_but_unboxed") and direct_fallback:
            _direct(grok, frame, request, query_image, photo_line, res)
    elif direct_fallback:
        _direct(grok, frame, request, query_image, photo_line, res)

    res.total_s = time.perf_counter() - t0
    return res


def _direct(grok, frame, request, query_image, photo_line, res: FindResult) -> None:
    images = [frame] + ([query_image] if query_image is not None else [])
    prompt = DIRECT_PROMPT.format(request=request, photo_line=photo_line.replace("Image 3", "Image 2"))
    r = grok.json(name="trova_direct", prompt=prompt, schema=DIRECT_SCHEMA, images=images,
                  mock_context={"request": request})
    d = r.data
    res.mode = "direct"
    res.grok_calls += 1
    res.grok_latency_s += r.latency_s
    res.label, res.reason = d.get("label", res.label), d.get("reason", res.reason)
    res.confidence = float(d.get("confidence") or 0)
    if d.get("found") and d.get("box_2d"):
        H, W = frame.shape[:2]
        box = Box.from_norm1000(d["box_2d"], W, H, id=0, source="grok", label=res.label)
        if box.area > 16:
            res.status, res.box = "found", box


# ---------------------------------------------------------------- mock Grok (offline mode)

_COLOR_WORDS = {
    "red": ["red", "rosso", "rossa", "rossi", "rosse"], "blue": ["blue", "blu"],
    "green": ["green", "verde", "verdi"], "yellow": ["yellow", "giallo", "gialla", "gialli"],
    "black": ["black", "nero", "nera", "neri"], "orange": ["orange", "arancione", "arancio"],
    "purple": ["purple", "viola", "violet"], "white": ["white", "bianco", "bianca"],
}


def _colors_in(text: str) -> list[str]:
    words = set(re.findall(r"[a-zà-ù]+", text.lower()))
    return [c for c, ws in _COLOR_WORDS.items() if words & set(ws)]


_ITALIAN = {"il", "la", "lo", "le", "gli", "un", "una", "qualcosa", "trova", "per", "quale", "dove", "rosso", "rossa",
            "blu", "verde", "giallo", "nero", "nera", "cosa", "tazza", "penna", "forbici", "numero"}


def is_italian(text: str) -> bool:
    return bool(set(re.findall(r"[a-zà-ù]+", (text or "").lower())) & _ITALIAN)


def _mock_select(prompt: str, images: list, ctx: dict) -> dict:
    """Offline stand-in for Grok: matches colour words (EN/IT) or the photo's colour. Flow only."""
    it = is_italian(ctx["request"])
    wanted = _colors_in(ctx["request"]) or ([ctx["query_color"]] if ctx.get("query_color") else [])
    for _, answer in ctx.get("history", []):
        m = re.search(r"\d+", answer)
        if m and int(m.group()) in ctx["colors"]:
            return {"box_id": int(m.group()), "visible_but_unboxed": False, "candidate_ids": [int(m.group())],
                    "clarifying_question": None, "label": f"{'oggetto' if it else 'object'} {m.group()}",
                    "reason": "Mock: chosen from your answer.", "confidence": 0.9}
    hits = [i for i, c in ctx["colors"].items() if c in wanted]
    if len(hits) == 1:
        return {"box_id": hits[0], "visible_but_unboxed": False, "candidate_ids": hits, "clarifying_question": None,
                "label": f"{ctx['colors'][hits[0]]} object", "reason": "Mock: the only object of that colour.",
                "confidence": 0.8}
    if len(hits) > 1:
        return {"box_id": None, "visible_but_unboxed": False, "candidate_ids": hits[:3],
                "clarifying_question": (f"Ne vedo {len(hits)}: quale, il numero {' o il '.join(map(str, hits[:3]))}?" if it
                                        else f"I see {len(hits)}: which one, number {' or '.join(map(str, hits[:3]))}?"),
                "label": "", "reason": "Mock: several objects match.", "confidence": 0.4}
    return {"box_id": None, "visible_but_unboxed": False, "candidate_ids": [], "clarifying_question": None,
            "label": "", "reason": "Mock mode matches colour words only; add XAI_API_KEY for real understanding.",
            "confidence": 0.0}


def _mock_direct(prompt: str, images: list, ctx: dict) -> dict:
    return {"found": False, "box_2d": None, "label": "", "reason": "Mock: direct grounding needs the real API.",
            "confidence": 0.0}


register_mock("trova_select", _mock_select)
register_mock("trova_direct", _mock_direct)
