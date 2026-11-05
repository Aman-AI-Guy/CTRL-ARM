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
POSE_MOVE_MM = 25.0       # past this, image overlap is not an identity: the camera moved with the arm
APPEAR_MIN = 0.72         # histogram correlation required for a cross-view guess
WORLD_MATCH_MM = 40.0     # plane points closer than this can confirm a cross-view id

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
    """Ids already handed out. `previous` is the last view: box (with track_id) and its colour histogram.

    `pose_mm` is the robot pose of that view. A large change means the camera moved, so an
    image-space overlap is not treated as the same object.
    """

    next_id: int = 1
    previous: list[tuple[Box, np.ndarray]] = field(default_factory=list)
    previous_frame: np.ndarray | None = field(default=None, repr=False)
    frame_size: tuple[int, int] | None = None
    pose_mm: tuple[float, float, float] | None = None


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


def _pose_dist(a, b) -> float:
    return float(sum((float(a[i]) - float(b[i])) ** 2 for i in range(3)) ** 0.5)


def remember_world(state: TrackState, track_id: int, world_mm) -> None:
    """Attach a plane intersection to a track. Later views can confirm the id with it."""
    point = [round(float(v), 2) for v in world_mm[:3]]
    for box, _hist_row in state.previous:
        if int(box.extra.get("track_id", -1)) == int(track_id):
            box.extra["world_mm"] = point


def assign_tracks(state: TrackState, boxes: list[Box], frame: np.ndarray,
                  reset_reason: str | None = None, *, pose_mm=None) -> tuple[list[Box], str | None]:
    """Copy `boxes` and set `extra['track_id']`. A reset drops the matches and never reuses an id.

    When `pose_mm` has moved, image overlap is ignored. A match then needs appearance, and it
    stays `uncertain` until a plane position agrees as well (`geometric`).
    """
    height, width = frame.shape[:2]
    reason = reset_reason
    if state.frame_size is not None and state.frame_size != (width, height):
        reason = reason or "frame size changed"
    camera_moved = False
    if pose_mm is not None and state.pose_mm is not None and _pose_dist(pose_mm, state.pose_mm) > POSE_MOVE_MM:
        camera_moved = True
    if pose_mm is not None and state.pose_mm is not None and len(pose_mm) >= 6 and len(state.pose_mm) >= 6:
        camera_moved = camera_moved or any(abs((float(pose_mm[i])-state.pose_mm[i]+180) % 360-180) > 2 for i in range(3, 6))
    if reason:
        state.previous = []
    elif not camera_moved and state.previous_frame is not None:
        boxes = recover_regions(state.previous, state.previous_frame, frame, boxes)
    state.frame_size = (width, height)

    current_h = [_hist(frame, box) for box in boxes]
    match: dict[int, int] = {}
    kind: dict[int, str] = {}
    if camera_moved and state.previous:
        pairs: list[tuple[float, int, int, str]] = []
        for i, box in enumerate(boxes):
            for j, (prev, prev_h) in enumerate(state.previous):
                appearance = _similar(current_h[i], prev_h)
                previous_world, new_world = prev.extra.get("world_mm"), box.extra.get("world_mm")
                geometric = None
                if previous_world and new_world:
                    geometric = _pose_dist(previous_world, new_world)
                if geometric is not None and geometric <= WORLD_MATCH_MM and appearance >= 0.45:
                    pairs.append((2.0 + appearance, i, j, "geometric"))
                elif appearance >= APPEAR_MIN and geometric is None:
                    pairs.append((appearance, i, j, "uncertain"))
        pairs.sort(reverse=True)
        used: set[int] = set()
        for _score, i, j, how in pairs:
            if i in match or j in used:
                continue
            match[i], kind[i] = j, how
            used.add(j)
        reason = reason or "camera moved; image-space ids are not treated as the same object"
    else:
        pairs_iou: list[tuple[float, int, int]] = []
        for i, box in enumerate(boxes):
            for j, (prev, prev_h) in enumerate(state.previous):
                overlap = box.iou(prev)
                if overlap < IOU_MIN or _similar(current_h[i], prev_h) < APPEAR_MIN:
                    continue
                pairs_iou.append((overlap + 0.05 * _similar(current_h[i], prev_h), i, j))
        pairs_iou.sort(reverse=True)
        used = set()
        for _score, i, j in pairs_iou:
            if i in match or j in used:
                continue
            match[i], kind[i] = j, "same_view"
            used.add(j)

    out: list[Box] = []
    stored: list[tuple[Box, np.ndarray]] = []
    for i, box in enumerate(boxes):
        copied = _copy(box)
        if i in match:
            previous = state.previous[match[i]][0]
            copied.extra["track_id"] = int(previous.extra["track_id"])
            copied.extra["association"] = kind.get(i, "same_view")
            if previous.extra.get("world_mm") and "world_mm" not in copied.extra:
                copied.extra["world_mm"] = list(previous.extra["world_mm"])
        else:
            copied.extra["track_id"] = state.next_id
            state.next_id += 1
            copied.extra["association"] = "new"
        copied.extra["color"] = color_name(crop(frame, box, pad=0))
        out.append(copied)
        stored.append((copied, current_h[i]))
    state.previous = stored
    state.previous_frame = frame.copy()
    if pose_mm is not None:
        state.pose_mm = tuple(float(v) for v in pose_mm)
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
        raw_name = str(named.get("name") or "").strip()
        if not raw_name or raw_name.lower() in {"unknown", "?", "n/a", "none", "nothing"}:
            raw_name = "unknown"
            label_ok = False
        else:
            label_ok = bool(named.get("label_authoritative"))
        objects.append({
            "track_id": int(box.extra["track_id"]),
            "box_id": int(box.id),
            "name": raw_name,
            "category": named.get("category", ""),
            "note": named.get("note", ""),
            "semantic": named.get("note", ""),
            "geometric_detection": True,
            "recognition_state": "recognized" if raw_name != "unknown" else "unknown",
            "image_localization_state": "region_identified",
            "metric_localization_state": "not_localized",
            "localization_provenance": box.extra.get("localization_provenance", "opencv_blob"),
            "region_confidence": box.extra.get("region_confidence"),
            "label_authoritative": label_ok,
            "name_from_model": bool(named.get("name_from_model")),
            "box": box.to_list(),
            "center": [round(c, 1) for c in box.center],
            "color": box.extra.get("color") or color_name(crop(frame, box, pad=0)),
            "touches_border": bool(box.touches_border),
            "confidence": local_confidence(frame, box),
            "ambiguous": False,
            "source": box.source,
            "association": box.extra.get("association", "same_view"),
            "identity_uncertain": box.extra.get("association") == "uncertain",
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
    if chosen and chosen.get('label_authoritative') is False:
        authoritative = False
        reason += '. The selected object has no authoritative live label'
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


def adopt_found(state, frame, boxes, objects, found):
    """Put a grounded finder result into the same snapshot and track namespace as blobs."""
    from .regions import validated_region
    box = found.box
    if found.status != "found" or box is None:
        return boxes, objects
    validated_region([box.x0, box.y0, box.x1, box.y1], frame, source=box.source)
    names = {obj["box_id"]: dict(obj) for obj in objects}
    if found.mode == "direct":
        overlap = next((b for b in boxes if b.iou(box) >= 0.65), None)
        if overlap is not None:
            box.id = overlap.id
            boxes = [box if b.id == overlap.id else b for b in boxes]
        else:
            box.id = max((b.id for b in boxes), default=0) + 1
            boxes = [*boxes, box]
    names[box.id] = {"name": found.label or "unknown", "note": found.reason,
                     "name_from_model": bool(found.label) and box.source != "manual", "label_authoritative": found.recognition_live}
    boxes, _ = assign_tracks(state, boxes, frame)
    found.box = next(b for b in boxes if b.id == box.id)
    return boxes, build_objects(boxes, frame, names)


def recover_regions(previous, old_frame, frame, boxes):
    """Recover non-blob regions only with a unique, strong local pixel match in the same view.

    No box survives merely because its name or old pixel position is still remembered.
    Rotation/pose changes and reset are handled by assign_tracks before this function.
    """
    if old_frame.shape != frame.shape:
        return boxes
    out = list(boxes)
    h, w = frame.shape[:2]
    for prev, _ in previous:
        if prev.source not in {"grok", "manual"} or any(b.iou(prev) >= IOU_MIN for b in out):
            continue
        x0, y0, x1, y1 = prev.to_list()
        template = old_frame[y0:y1, x0:x1]
        if template.size == 0 or float(template.std()) < 3:
            continue
        sx, sy = max(0, x0-30), max(0, y0-30)
        search = frame[sy:min(h, y1+30), sx:min(w, x1+30)]
        scores = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, (dx, dy) = cv2.minMaxLoc(scores)
        if not np.isfinite(score) or score < 0.94:
            continue
        other = scores.copy()
        other[max(0, dy-5):dy+6, max(0, dx-5):dx+6] = -1
        if float(other.max()) > score - 0.01:
            continue
        box = _copy(prev)
        box.x0, box.y0 = sx+dx, sy+dy
        box.x1, box.y1 = box.x0+template.shape[1], box.y0+template.shape[0]
        box.id = max((b.id for b in out), default=0)+1
        box.extra.pop("world_mm", None)
        box.extra["localization_provenance"] = "visual_template_reobservation"
        box.extra["region_confidence"] = float(score)
        out.append(box)
    return out
