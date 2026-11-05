"""Feature 1: table inventory map. "What's on the table?"

One Grok call names every numbered object (name, category, one telling detail) and writes a one-line summary.
Plain code draws the labelled map. The names are also kept with a saved reference, so "what changed" can say
WHICH object went missing ("removed: the scissors"), not just "an object was removed".
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .boxes import Box
from .grok import GrokClient, register_mock
from .marks import HIGHLIGHT, MARK, _label, color_name, contact_sheet, crop, draw_marks

SCHEMA = {
    "type": "object",
    "properties": {
        "objects": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "box_id": {"type": "integer"},
                "name": {"type": "string"},
                "category": {"type": "string", "enum": ["tool", "stationery", "container", "packaging", "electronics",
                                                         "personal item", "food", "other"]},
                "note": {"type": "string"},
            },
            "required": ["box_id", "name", "category", "note"],
            "additionalProperties": False}},
        "summary": {"type": "string"},
    },
    "required": ["objects", "summary"],
    "additionalProperties": False,
}

PROMPT = """Image 1 is a work table seen from above. Every object has a magenta box with a number (1 to {n}).
Image 2 is a sheet of close-ups, one per number.

For EVERY number, return: box_id; name: 2 to 4 specific words ("blue ballpoint pen", not "object"); category;
note: one short visual detail that tells it apart from similar objects ("orange handles", "cap missing").
If a box holds two touching objects, name both ("pen and marker"). If a box is empty or a shadow, name "nothing".
summary: one sentence a person can hear, saying what is on the table.
{lang}"""

LANG = {"en": "Write names, notes and summary in English.", "it": "Write names, notes and summary in Italian.",
        "both": "Write names in English; write the summary in English, then in Italian after ' / '."}


@dataclass
class Inventory:
    items: dict[int, dict] = field(default_factory=dict)   # box id -> {name, category, note}
    summary: str = ""
    latency_s: float = 0.0
    time: str = ""
    offline: bool = False          # names made up locally (colours) because Grok was unreachable

    def names(self) -> dict[int, str]:
        return {i: d["name"] for i, d in self.items.items()}

    def as_dict(self) -> dict:
        return {"summary": self.summary, "time": self.time, "latency_s": round(self.latency_s, 2),
                "objects": [{"box_id": i, **d} for i, d in sorted(self.items.items())]}


def describe(grok: GrokClient, frame: np.ndarray, boxes: list[Box], lang: str = "en") -> Inventory:
    inv = Inventory(time=time.strftime("%H:%M:%S"))
    if not boxes:
        inv.summary = "The table is empty." if lang != "it" else "Il tavolo è vuoto."
        return inv
    ctx = {"boxes": boxes, "colors": {b.id: color_name(crop(frame, b, pad=0)) for b in boxes}}
    r = grok.json(name="trova_inventory", prompt=PROMPT.format(n=len(boxes), lang=LANG.get(lang, LANG["en"])),
                  schema=SCHEMA, images=[draw_marks(frame, boxes), contact_sheet(frame, boxes)], mock_context=ctx)
    ids = {b.id for b in boxes}
    for o in r.data.get("objects", []):
        if o.get("box_id") in ids and o.get("name", "").lower() != "nothing":
            inv.items[o["box_id"]] = {"name": o["name"], "category": o.get("category", "other"), "note": o.get("note", "")}
    inv.summary, inv.latency_s, inv.offline = r.data.get("summary", ""), r.latency_s, r.offline
    return inv


def draw_map(frame: np.ndarray, boxes: list[Box], inv: Inventory) -> np.ndarray:
    """The table with each box labelled '3 · blue pen'."""
    img = frame.copy()
    scale = max(0.5, frame.shape[1] / 1500)
    lw = max(2, int(frame.shape[1] / 600))
    for b in boxes:
        name = inv.items.get(b.id, {}).get("name")
        color = HIGHLIGHT if name else MARK
        cv2.rectangle(img, (int(b.x0), int(b.y0)), (int(b.x1), int(b.y1)), color, lw)
        _label(img, f"{b.id} {name}" if name else str(b.id), b.x0, b.y0, color, scale)
    return img


def _mock(prompt: str, images: list, ctx: dict) -> dict:
    objs = [{"box_id": b.id, "name": f"{ctx['colors'][b.id]} object", "category": "other",
             "note": f"Mock: {ctx['colors'][b.id]}"} for b in ctx["boxes"]]
    return {"objects": objs, "summary": f"Mock: {len(objs)} objects on the table."}


register_mock("trova_inventory", _mock)
