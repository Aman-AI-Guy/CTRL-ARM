"""Frozen scene snapshot: stable track ids, uncertainty, and selection that does not move the arm.

`Box.id` stays the reading-order number of one picture, which finder and clarify already use.
`track_id` is assigned here and kept across snapshots while the camera view is the same.
Grok never chooses either number. Selecting an object records a target and does not start a motion.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import cv2
import numpy as np

from .boxes import Box
from .finder import _colors_in, is_italian
from .marks import color_name, crop

IOU_MIN = 0.35

ORDINALS = {
    "first": 1, "1st": 1, "prima": 1, "primo": 1,
    "second": 2, "2nd": 2, "seconda": 2, "secondo": 2,
    "third": 3, "3rd": 3, "terza": 3, "terzo": 3,
    "fourth": 4, "4th": 4, "quarta": 4, "quarto": 4,
    "fifth": 5, "5th": 5, "quinta": 5, "quinto": 5,
}
_ORDINAL = re.compile(r"\b(" + "|".join(ORDINALS) + r")\b", re.IGNORECASE)
_LEADING = re.compile(r"^(?:please\s+)?(?:seleziona|select|scegli|choose|pick)\s+", re.IGNORECASE)
_ID = re.compile(r"(?:oggetto|object|numero|number|id|#)\s*(\d+)\s*$", re.IGNORECASE)
_TOKEN = re.compile(r"[a-zà-ù0-9]+", re.IGNORECASE)
# Words that do not identify which instance. Colour and other words stay.
_SKIP = {
    "the", "a", "an", "of", "one", "please", "select", "seleziona", "scegli", "choose", "pick",
    "object", "oggetto", "numero", "number", "pen", "penna", "item", "thing",
    "il", "lo", "la", "i", "gli", "le", "un", "una", "di", "del", "della", "dei",
}


@dataclass
class TrackState:
    """Ids already handed out. `previous` is the last view: box (with track_id) and its colour histogram."""

    next_id: int = 1
    previous: list[tuple[Box, np.ndarray]] = field(default_factory=list)
    frame_size: tuple[int, int] | None = None


@dataclass
class Decision:
    status: str                 # selected | clarify | not_found | defer
    track_ids: list[int] = field(default_factory=list)
    question: str | None = None
    reason: str = ""
    confidence: float = 0.0


def _copy(box: Box) -> Box:
    return Box(box.x0, box.y0, box.x1, box.y1, id=box.id, source=box.source, label=box.label,
               score=box.score, touches_border=box.touches_border, extra=dict(box.extra))


def _hist(frame: np.ndarray, box: Box) -> np.ndarray:
    img = crop(frame, box, pad=0)
    if img.size == 0:
        return np.zeros((8, 8, 8), np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist, alpha=1, norm_type=cv2.NORM_L1)
    return hist


def _similar(a: np.ndarray, b: np.ndarray) -> float:
    score = float(cv2.compareHist(a, b, cv2.HISTCMP_CORREL))
    return max(0.0, score)


def assign_tracks(state: TrackState, boxes: list[Box], frame: np.ndarray,
                  reset_reason: str | None = None) -> tuple[list[Box], str | None]:
    """Copy `boxes` and set `extra['track_id']`. A reset drops the matches and never reuses an id."""
    height, width = frame.shape[:2]
    reason = reset_reason
    if state.frame_size is not None and state.frame_size != (width, height):
        reason = reason or "frame size changed"
    if reason:
        state.previous = []
    state.frame_size = (width, height)

    current_h = [_hist(frame, box) for box in boxes]
    pairs: list[tuple[float, int, int]] = []
    for i, box in enumerate(boxes):
        for j, (prev, prev_h) in enumerate(state.previous):
            overlap = box.iou(prev)
            if overlap < IOU_MIN:
                continue
            pairs.append((overlap + 0.05 * _similar(current_h[i], prev_h), i, j))
    pairs.sort(reverse=True)
    match: dict[int, int] = {}
    used: set[int] = set()
    for _score, i, j in pairs:
        if i in match or j in used:
            continue
        match[i] = j
        used.add(j)

    out: list[Box] = []
    stored: list[tuple[Box, np.ndarray]] = []
    for i, box in enumerate(boxes):
        copied = _copy(box)
        if i in match:
            copied.extra["track_id"] = int(state.previous[match[i]][0].extra["track_id"])
        else:
            copied.extra["track_id"] = state.next_id
            state.next_id += 1
        copied.extra["color"] = color_name(crop(frame, box, pad=0))
        out.append(copied)
        stored.append((copied, current_h[i]))
    state.previous = stored
    return out, reason


def local_confidence(frame: np.ndarray, box: Box) -> float:
    """How distinct the blob looks. This is not a classifier probability."""
    img = crop(frame, box, pad=0)
    if img.size == 0:
        return 0.2
    sat = float(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1].mean()) / 255.0
    score = 0.45 + 0.4 * sat
    if box.touches_border:
        score *= 0.6
    return round(float(np.clip(score, 0.15, 0.9)), 2)


def normal_name(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip().lower())


def mark_ambiguous(objects: list[dict]) -> None:
    """Same normalised name on two instances: the person has to pick one."""
    counts: dict[str, int] = {}
    keys = []
    for obj in objects:
        key = normal_name(obj.get("name", ""))
        keys.append(key)
        if key:
            counts[key] = counts.get(key, 0) + 1
    for obj, key in zip(objects, keys, strict=True):
        obj["ambiguous"] = bool(key) and counts[key] > 1


def build_objects(boxes: list[Box], frame: np.ndarray, names: dict[int, dict]) -> list[dict]:
    objects = []
    for box in boxes:
        named = names.get(box.id, {})
        objects.append({
            "track_id": int(box.extra["track_id"]),
            "box_id": int(box.id),
            "name": named.get("name", ""),
            "category": named.get("category", ""),
            "note": named.get("note", ""),
            "box": box.to_list(),
            "center": [round(c, 1) for c in box.center],
            "color": box.extra.get("color") or color_name(crop(frame, box, pad=0)),
            "touches_border": bool(box.touches_border),
            "confidence": local_confidence(frame, box),
            "ambiguous": False,
            "source": box.source,
        })
    mark_ambiguous(objects)
    return objects


def question_for(candidates: list[dict], text: str) -> str:
    ids = [str(obj["track_id"]) for obj in candidates]
    if is_italian(text):
        return f"Ne vedo {len(candidates)} simili: quale, il numero {' o il '.join(ids)}?"
    return f"I see {len(candidates)} that match: which one, number {' or '.join(ids)}?"


def _body(text: str) -> str:
    return _LEADING.sub("", (text or "").strip()).strip()


def decide_text(text: str, objects: list[dict]) -> Decision:
    """Resolve an id or an ordinal in code. Anything else is left to the finder (`defer`)."""
    body = _body(text)
    if not body:
        return Decision("not_found", reason="Empty request")
    id_match = _ID.fullmatch(body)
    if id_match:
        number = int(id_match.group(1))
        hits = [obj for obj in objects if obj["track_id"] == number]
        if not hits:
            hits = [obj for obj in objects if obj["box_id"] == number]
        if len(hits) == 1:
            return Decision("selected", [hits[0]["track_id"]], reason=f"id {number}",
                            confidence=float(hits[0]["confidence"]))
        if len(hits) > 1:
            return Decision("clarify", [obj["track_id"] for obj in hits], question_for(hits, text),
                            reason="that id matches more than one object")
        return Decision("not_found", reason=f"No object {number} in this snapshot")

    ordinal = _ORDINAL.search(body)
    if ordinal:
        index = ORDINALS[ordinal.group(1).lower()]
        colors = _colors_in(body)
        if colors:
            candidates = [obj for obj in objects
                          if obj.get("color") in colors or any(color in obj.get("name", "").lower() for color in colors)]
        else:
            tokens = [tok for tok in _TOKEN.findall(_ORDINAL.sub(" ", body).lower()) if tok not in _SKIP and tok not in ORDINALS]
            if tokens:
                candidates = [obj for obj in objects if any(
                    tok in f"{obj.get('name', '')} {obj.get('note', '')} {obj.get('color', '')}".lower() for tok in tokens)]
            else:
                candidates = list(objects)
        candidates = sorted(candidates, key=lambda obj: (obj["box_id"], obj["track_id"]))
        if len(candidates) >= index:
            chosen = candidates[index - 1]
            return Decision("selected", [chosen["track_id"]], reason=f"ordinal {index}",
                            confidence=float(chosen["confidence"]))
        return Decision("not_found", reason=f"There is no number {index} among {len(candidates)} matches")
    return Decision("defer")


def from_finder(status: str, box_id: int | None, candidate_ids: list[int], confidence: float,
                reason: str, objects: list[dict], text: str) -> Decision:
    """Turn a finder answer into track ids. A shared name never becomes a silent choice."""
    by_box = {obj["box_id"]: obj for obj in objects}
    if status == "clarify":
        chosen = [by_box[i] for i in candidate_ids if i in by_box]
        if len(chosen) < 2:
            return Decision("not_found", reason=reason or "the choices are not in this snapshot", confidence=confidence)
        return Decision("clarify", [obj["track_id"] for obj in chosen], question_for(chosen, text),
                        reason=reason or "several objects match", confidence=confidence)
    if status == "found" and box_id in by_box:
        obj = by_box[box_id]
        if obj["ambiguous"]:
            key = normal_name(obj["name"])
            group = [item for item in objects if normal_name(item.get("name", "")) == key]
            return Decision("clarify", [item["track_id"] for item in group], question_for(group, text),
                            reason="more than one object has this name", confidence=confidence)
        return Decision("selected", [obj["track_id"]], reason=reason or "matched the description",
                        confidence=confidence or float(obj["confidence"]))
    return Decision("not_found", reason=reason or "not in this snapshot", confidence=confidence)


def pack(decision: Decision, objects: list[dict], authoritative: bool, snapshot_id: str) -> dict:
    """JSON for /api/select. Motion stays off. Mock and offline names are not authoritative."""
    by_id = {obj["track_id"]: obj for obj in objects}
    candidates = [by_id[i] for i in decision.track_ids if i in by_id]
    chosen = candidates[0] if decision.status == "selected" and len(candidates) == 1 else None
    status = decision.status
    question = decision.question
    if status == "selected" and chosen is None:
        status, chosen, candidates, question = "not_found", None, [], None
    if status != "clarify":
        shown = [chosen] if chosen else []
    else:
        shown = candidates
    reason = decision.reason or "no match"
    reason = f"{reason}. selection does not move the arm"
    if not authoritative:
        reason += ". Names are from mock or offline recognition and cannot authorize motion"
    confidence = decision.confidence or (float(chosen["confidence"]) if chosen else 0.0)
    return {
        "status": status,
        "object": chosen,
        "candidates": shown if status == "clarify" else ([] if chosen is None else [chosen]),
        "question": question if status == "clarify" else None,
        "confidence": round(float(confidence or 0), 2),
        "authoritative": bool(authoritative),
        "motion_allowed": False,
        "reason": reason,
        "snapshot_id": snapshot_id,
    }
