"""'What changed': reference scene vs current scene -> added / removed / moved objects.

1. Plain code finds candidate changes: objects in both scenes are matched by position, size and
   colour histogram; unmatched ones are added or removed; matched ones that travelled are moved.
2. Persistence: a change must show up in at least 2 of 3 frames, so a passing hand or a flicker
   does not count. Frames with a hand or the arm in view are retried.
3. Grok verifies every candidate (is it a real object change, or a shadow, glare, the same
   object?), names it and explains why it matters on a cleared line. Rejected candidates are
   kept in the log as "filtered", which is where false alarms are removed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .boxes import Box
from .grok import GrokClient, register_mock
from .marks import color_name, crop, draw_changes, pair_sheet
from .proposals import match_lighting

KINDS = ("added", "removed", "moved")


@dataclass
class Change:
    id: int
    kind: str
    ref_box: Box | None = None
    cur_box: Box | None = None
    label: str = ""
    explanation: str = ""
    risk: str = ""
    verified: bool | None = None   # None = not checked by Grok yet

    def as_dict(self) -> dict:
        return {"id": self.id, "kind": self.kind, "label": self.label, "explanation": self.explanation,
                "risk": self.risk, "verified": self.verified,
                "ref_box": self.ref_box.to_list() if self.ref_box else None,
                "cur_box": self.cur_box.to_list() if self.cur_box else None}


@dataclass
class ChangeCheck:
    changes: list[Change] = field(default_factory=list)    # confirmed (or unverified) changes
    filtered: list[Change] = field(default_factory=list)   # candidates Grok rejected
    line_clear: bool = True
    summary: str = ""
    occluded: bool = False
    frames_used: int = 0
    grok_latency_s: float = 0.0
    total_s: float = 0.0
    ref_marked: np.ndarray | None = None
    cur_marked: np.ndarray | None = None
    current: np.ndarray | None = None

    def as_dict(self) -> dict:
        return {"line_clear": self.line_clear, "summary": self.summary, "occluded": self.occluded,
                "changes": [c.as_dict() for c in self.changes],
                "filtered": [c.as_dict() for c in self.filtered], "frames_used": self.frames_used,
                "grok_latency_s": round(self.grok_latency_s, 2), "total_s": round(self.total_s, 2)}


def _hist(frame: np.ndarray, box: Box) -> np.ndarray:
    c = crop(frame, box, pad=-0.1)  # inner part only: less table in the histogram
    if not c.size:
        c = crop(frame, box, pad=0)
    hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    # Hue means nothing for black, grey and white pixels (it flips with tiny light changes), so
    # coloured pixels go in a hue x saturation histogram and the rest in a brightness histogram.
    chroma = ((hsv[..., 1] > 50) & (hsv[..., 2] > 50)).astype(np.uint8)
    h1 = cv2.calcHist([hsv], [0, 1], chroma, [18, 4], [0, 180, 50, 256]).flatten()
    h2 = cv2.calcHist([hsv], [2], 1 - chroma, [8], [0, 256]).flatten()
    v = np.concatenate([h1, h2]).astype(np.float32)
    n = float(v.sum())
    return v / n if n else v


def _shape(frame: np.ndarray, box: Box) -> tuple[float, float, float]:
    """Rotation-proof shape of the object in a box: (elongation, fill, object area in px).
    Lets matching tell a black mug from a black phone, which colour alone cannot."""
    c = crop(frame, box, pad=0.12)
    if c.shape[0] < 4 or c.shape[1] < 4:
        return 1.0, 1.0, box.area
    lab = cv2.cvtColor(c, cv2.COLOR_BGR2LAB).astype(np.float32)
    border = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    d = np.abs(lab - np.median(border, 0))
    m = ((0.5 * d[..., 0] + d[..., 1] + d[..., 2]) > 30).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 1.0, 1.0, box.area
    cnt = max(cnts, key=cv2.contourArea)
    w, h = cv2.minAreaRect(cnt)[1]
    if min(w, h) < 1:
        return 1.0, 1.0, box.area
    area = float(cv2.contourArea(cnt))
    return max(w, h) / min(w, h), area / (w * h), area


def diff_objects(ref_frame: np.ndarray, ref_boxes: list[Box], cur_frame: np.ndarray, cur_boxes: list[Box],
                 move_frac: float = 0.015, min_sim: float = 0.5) -> list[Change]:
    """Greedy matching on position + size + colour + shape. Deterministic, no model."""
    cur_frame = match_lighting(cur_frame, ref_frame)   # compare colours under the same lighting
    H, W = cur_frame.shape[:2]
    diag = float(np.hypot(W, H))
    rh = [_hist(ref_frame, b) for b in ref_boxes]
    ch = [_hist(cur_frame, b) for b in cur_boxes]
    rs = [_shape(ref_frame, b) for b in ref_boxes]
    cs = [_shape(cur_frame, b) for b in cur_boxes]
    pairs = []
    for i, rb in enumerate(ref_boxes):
        for j, cb in enumerate(cur_boxes):
            sim = float(cv2.compareHist(rh[i], ch[j], cv2.HISTCMP_CORREL))
            ratio = (cs[j][2] + 1) / (rs[i][2] + 1)          # object area, not box area: rotation-proof
            elong = max(rs[i][0], cs[j][0]) / min(rs[i][0], cs[j][0])
            fill = abs(rs[i][1] - cs[j][1])
            if sim < min_sim:
                continue
            # Same place + same colours = same object: shape noise (thin pens, glare) must not split it.
            # The shape gate is only for telling different objects apart across the table.
            same_place = rb.iou(cb) > 0.5 and sim > 0.7
            fill_tol = 0.3 if max(rs[i][0], cs[j][0]) > 3 else 0.18   # thin objects: fill is noisy
            if not same_place and (not (0.6 < ratio < 1.67) or elong > 1.5 or fill > fill_tol):
                continue
            dist = float(np.hypot(rb.center[0] - cb.center[0], rb.center[1] - cb.center[1])) / diag
            cost = dist * 2.0 + (1 - sim) + abs(np.log(ratio)) * 0.5 + np.log(elong) * 0.5 + fill
            pairs.append((cost, i, j, dist))
    pairs.sort()
    used_r, used_c, changes = set(), set(), []
    for cost, i, j, dist in pairs:
        if i in used_r or j in used_c:
            continue
        used_r.add(i)
        used_c.add(j)
        if dist > move_frac:
            changes.append(Change(0, "moved", ref_boxes[i], cur_boxes[j]))
    changes += [Change(0, "removed", ref_boxes[i], None) for i in range(len(ref_boxes)) if i not in used_r]
    changes += [Change(0, "added", None, cur_boxes[j]) for j in range(len(cur_boxes)) if j not in used_c]
    for k, c in enumerate(changes, start=1):
        c.id = k
    return changes


def _same(a: Change, b: Change) -> bool:
    if a.kind != b.kind:
        return False
    ba, bb = (a.cur_box or a.ref_box), (b.cur_box or b.ref_box)
    return ba is not None and bb is not None and ba.iou(bb) > 0.3


def persistent_changes(per_frame: list[list[Change]], min_votes: int = 2) -> list[Change]:
    """Keep changes seen in at least min_votes of the frames (taken from the last frame)."""
    if not per_frame:
        return []
    if len(per_frame) == 1:
        return per_frame[0]
    kept = [c for c in per_frame[-1] if sum(any(_same(c, o) for o in f) for f in per_frame) >= min_votes]
    for k, c in enumerate(kept, start=1):
        c.id = k
    return kept


def naive_pixel_diff(ref_frame: np.ndarray, cur_frame: np.ndarray, thresh: int = 35,
                     min_area_frac: float = 0.0008) -> list[Box]:
    """The baseline we beat: raw pixel difference + connected components. Shadows, light and
    noise all count as changes here."""
    g1 = cv2.GaussianBlur(cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    g2 = cv2.GaussianBlur(cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    m = (cv2.absdiff(g1, g2) > thresh).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(m)
    H, W = m.shape
    return [Box(float(x), float(y), float(x + w), float(y + h), source="naive")
            for x, y, w, h, a in stats[1:] if a >= min_area_frac * W * H]


VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "changes": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "change_id": {"type": "integer"},
                "is_real": {"type": "boolean"},
                "label": {"type": "string"},
                "explanation": {"type": "string"},
                "risk": {"type": "string", "enum": ["low", "medium", "high"]},
            },
            "required": ["change_id", "is_real", "label", "explanation", "risk"],
            "additionalProperties": False}},
        "line_clear": {"type": "boolean"},
        "summary": {"type": "string"},
    },
    "required": ["changes", "line_clear", "summary"],
    "additionalProperties": False,
}

VERIFY_PROMPT = """You check a work table for Trova, the way an operator checks a packaging line before the next batch (line clearance).

Image 1: the REFERENCE scene (the approved state). Boxes mark where removed or moved objects used to be.
Image 2: the CURRENT scene. Boxes mark added objects and the new place of moved objects.
Image 3: one row per candidate change C1..C{n}: a close-up before and after.

The candidates come from a pixel comparison, so some may be false alarms: a shadow, glare, a light change, a hand, or the same object barely shifted.

Candidates:
{listing}

For every candidate return change_id, is_real, a short label for the object, an explanation and a risk.
- explanation: one sentence on what happened and why it matters on a cleared line (a leftover from the last batch, a foreign object, a missing item), or why it is a false alarm.
- risk: high for foreign or leftover items that could end up in a product, medium for moved or missing items, low otherwise.
line_clear is true only if there are no real changes. summary: one sentence for the operator.
{language}"""


LANG_LINES = {"en": "Write label, explanation and summary in English.",
              "it": "Write label, explanation and summary in Italian.",
              "both": "Write label in English; explanation and summary in English, then the same in Italian after ' / '."}
NO_CHANGE = {"en": "No differences from the reference.", "it": "Nessuna differenza rispetto al riferimento.",
             "both": "No differences from the reference. / Nessuna differenza rispetto al riferimento."}


def verify_changes(grok: GrokClient, ref_frame: np.ndarray, cur_frame: np.ndarray, changes: list[Change],
                   lang: str = "en", ref_names: dict[int, str] | None = None) -> tuple[list, list, bool, str, float]:
    lang = lang if lang in LANG_LINES else "en"
    language = LANG_LINES[lang]
    if not changes:
        return [], [], True, NO_CHANGE[lang], 0.0
    ref_names = ref_names or {}

    def line(c):
        was = ref_names.get(c.ref_box.id) if c.ref_box is not None else None
        return f"- C{c.id}: {c.kind}" + (f" (in the reference this was: {was})" if was else "")
    listing = "\n".join(line(c) for c in changes)
    pairs = [(f"C{c.id}", crop(ref_frame, c.ref_box) if c.ref_box else crop(ref_frame, c.cur_box),
              crop(cur_frame, c.cur_box) if c.cur_box else crop(cur_frame, c.ref_box)) for c in changes]
    images = [draw_changes(ref_frame, changes, "ref"), draw_changes(cur_frame, changes, "cur"), pair_sheet(pairs)]
    ctx = {"lang": lang, "changes": changes, "colors": {c.id: color_name(crop(cur_frame if c.cur_box else ref_frame,
                                                                  c.cur_box or c.ref_box, pad=0)) for c in changes}}
    r = grok.json(name="trova_verify", prompt=VERIFY_PROMPT.format(n=len(changes), listing=listing, language=language),
                  schema=VERIFY_SCHEMA, images=images, mock_context=ctx)
    verdicts = {v["change_id"]: v for v in r.data.get("changes", [])}
    real, filtered = [], []
    for c in changes:
        v = verdicts.get(c.id)
        if v is None:          # Grok skipped it: keep it, unverified, so nothing is silently dropped
            real.append(c)
            continue
        c.verified, c.label, c.explanation, c.risk = v["is_real"], v["label"], v["explanation"], v["risk"]
        (real if v["is_real"] else filtered).append(c)
    return real, filtered, (not real) and bool(r.data.get("line_clear", True)), r.data.get("summary", ""), r.latency_s


def check(grok: GrokClient | None, ref_frame: np.ndarray, ref_boxes: list[Box], frames: list[np.ndarray],
          boxes_per_frame: list[list[Box]], occluded: bool = False, verify: bool = True, lang: str = "en",
          ref_names: dict[int, str] | None = None) -> ChangeCheck:
    t0 = time.perf_counter()
    per_frame = [diff_objects(ref_frame, ref_boxes, f, b) for f, b in zip(frames, boxes_per_frame)]
    cands = persistent_changes(per_frame)
    cur = frames[-1]
    out = ChangeCheck(occluded=occluded, frames_used=len(frames), current=cur)
    if verify and grok is not None:
        out.changes, out.filtered, out.line_clear, out.summary, out.grok_latency_s = verify_changes(
            grok, ref_frame, cur, cands, lang, ref_names)
    else:
        out.changes, out.line_clear = cands, not cands
        out.summary = f"{len(cands)} candidate change(s), not verified by Grok."
    out.ref_marked = draw_changes(ref_frame, out.changes, "ref")
    out.cur_marked = draw_changes(cur, out.changes, "cur")
    out.total_s = time.perf_counter() - t0
    return out


def _mock_verify(prompt: str, images: list, ctx: dict) -> dict:
    italian = ctx.get("lang") == "it"
    it = {"added": "aggiunto", "removed": "rimosso", "moved": "spostato"}
    items = [{"change_id": c.id, "is_real": True,
              "label": (f"oggetto {ctx['colors'].get(c.id, '')}" if italian else f"{ctx['colors'].get(c.id, '')} object").strip(),
              "explanation": (f"Mock: oggetto {it[c.kind]} rispetto al riferimento." if italian
                              else f"Mock: object {c.kind} compared with the reference."),
              "risk": "high" if c.kind == "added" else "medium"} for c in ctx["changes"]]
    summary = (f"Mock: {len(items)} differenze trovate." if italian else f"Mock: {len(items)} differences found.") \
        if items else ("Mock: linea libera." if italian else "Mock: line clear.")
    return {"changes": items, "line_clear": not items, "summary": summary}


register_mock("trova_verify", _mock_verify)
