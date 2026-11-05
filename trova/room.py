"""Feature 14: look around the room ("Where's Waldo when Waldo hides in a real room", the Track A brief).

The camera on the xArm looks sideways from the flange, so the arm can work as a pan head:
  1. go to the taught LOOK pose (joint angles; camera looking out at the room),
  2. turn one joint (TROVA_PAN_JOINT, default 1 = the base) in steps, one photo per step,
  3. Grok picks the photo that shows the requested object, then draws its box in that photo,
  4. the arm turns so the camera (and its LED ring) faces the object.
Pixels per degree are measured from the overlap of neighbouring photos (ORB features); if that fails, from
TROVA_CAMERA_HFOV_DEG. Every joint target is checked against the safety box (forward kinematics) and STOP aborts.
"""

from __future__ import annotations

import os
import statistics
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .boxes import Box
from .finder import DIRECT_PROMPT, DIRECT_SCHEMA, _colors_in
from .grok import GrokClient, register_mock
from .marks import HIGHLIGHT, _label

PICK_SCHEMA = {"type": "object",
               "properties": {"photo": {"type": ["integer", "null"]}, "label": {"type": "string"},
                              "reason": {"type": "string"}, "confidence": {"type": "number"}},
               "required": ["photo", "label", "reason", "confidence"], "additionalProperties": False}

PICK_PROMPT = """You are the eyes of Trova, a robot arm with a camera, looking around a room for a person.
The image is a sheet of {n} photos taken while the arm turned, numbered 1 to {n} in the order taken.

Request: "{request}"

Return the number of the photo where the requested object is most clearly and completely visible.
If it is in none of them, photo = null. Never pick a photo just to give an answer.
label: a short name of the object. reason: one sentence saying where it is in the photo (left, centre, right).
Write label and reason in the language of the request. confidence: 0 to 1."""


def pan_settings() -> dict:
    lo, hi, step = (float(v) for v in os.getenv("TROVA_ROOM_PAN", "-90,90,30").split(","))
    return {"joint": int(os.getenv("TROVA_PAN_JOINT", "1")) - 1, "angles": [float(a) for a in np.arange(lo, hi + 1e-6, step)],
            "hfov": float(os.getenv("TROVA_CAMERA_HFOV_DEG", "70")), "sign": float(os.getenv("TROVA_PAN_SIGN", "1"))}


@dataclass
class View:
    angle: float              # pan angle relative to the look pose (degrees)
    frame: np.ndarray
    path: str = ""


@dataclass
class RoomFind:
    status: str = "not_found"           # found | not_found | stopped | error
    request: str = ""
    view: int | None = None             # index into views
    angle: float | None = None          # where the arm turned to face it (relative to the look pose)
    box: Box | None = None
    label: str = ""
    reason: str = ""
    confidence: float = 0.0
    px_per_deg: float = 0.0
    px_per_deg_source: str = ""
    grok_latency_s: float = 0.0
    views: list[View] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"status": self.status, "request": self.request, "view": None if self.view is None else self.view + 1,
                "n_views": len(self.views), "angle": None if self.angle is None else round(self.angle, 1),
                "box": self.box.as_dict() if self.box else None, "label": self.label, "reason": self.reason,
                "confidence": round(self.confidence, 2), "px_per_deg": round(self.px_per_deg, 2),
                "px_per_deg_source": self.px_per_deg_source, "grok_latency_s": round(self.grok_latency_s, 2),
                "view_angles": [round(v.angle, 1) for v in self.views], "views": [v.path for v in self.views]}


def scan(arm, grab, look: list[float], joint: int, angles: list[float], settle_s: float = 0.4,
         sleep=time.sleep) -> list[View] | None:
    """Visit each pan angle and take a photo. None if a move was refused or STOP was pressed."""
    views = []
    for a in angles:
        q = list(look)
        q[joint] = look[joint] + a
        if not arm.go_joints(q):
            return None
        sleep(settle_s)
        views.append(View(angle=float(a), frame=grab()))
    return views


def pixel_shift(a: np.ndarray, b: np.ndarray) -> float | None:
    """Horizontal shift of the scene from photo a to photo b (pixels), from matched ORB features."""
    orb = cv2.ORB_create(1500)
    ga, gb = (cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in (a, b))
    ka, da = orb.detectAndCompute(ga, None)
    kb, db = orb.detectAndCompute(gb, None)
    if da is None or db is None or len(ka) < 20 or len(kb) < 20:
        return None
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
    if len(matches) < 15:
        return None
    pa = np.float32([ka[m.queryIdx].pt for m in matches])
    pb = np.float32([kb[m.trainIdx].pt for m in matches])
    M, inl = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=4.0)
    if M is None or inl is None or int(inl.sum()) < 12:
        return None
    return float(M[0, 2])


def measure_px_per_deg(views: list[View], hfov: float, sign: float) -> tuple[float, str]:
    """Signed pixels the scene moves per degree of pan, from neighbouring photos; else from the field of view."""
    rates = []
    for v1, v2 in zip(views, views[1:]):
        d = v2.angle - v1.angle
        s = pixel_shift(v1.frame, v2.frame)
        if s is not None and abs(d) > 1e-6 and abs(s) > 5:
            rates.append(s / d)
    if len(rates) >= 2 and max(rates) * min(rates) > 0:          # consistent sign
        return statistics.median(rates), f"measured from {len(rates)} photo pairs"
    w = views[0].frame.shape[1] if views else 1280
    return sign * w / hfov, f"from TROVA_CAMERA_HFOV_DEG={hfov:g}"


def sheet(views: list[View], cell_w: int = 420) -> np.ndarray:
    """Numbered photos in a grid, in the order taken."""
    cols = min(4, max(1, len(views)))
    h0, w0 = views[0].frame.shape[:2]
    cell_h = int(cell_w * h0 / w0)
    rows = (len(views) + cols - 1) // cols
    out = np.full((rows * (cell_h + 8) + 8, cols * (cell_w + 8) + 8, 3), 30, np.uint8)
    for i, v in enumerate(views):
        r, c = divmod(i, cols)
        y, x = 8 + r * (cell_h + 8), 8 + c * (cell_w + 8)
        out[y:y + cell_h, x:x + cell_w] = cv2.resize(v.frame, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        _label(out, str(i + 1), x + 6, y + 34, (180, 0, 158), 1.0)
    return out


def _color_stats(views: list[View]) -> list[dict]:
    """For the offline stand-in only: how much of each named colour each photo holds, and where."""
    from .synthetic import COLORS
    out = []
    for v in views:
        small = cv2.resize(v.frame, (320, int(320 * v.frame.shape[0] / v.frame.shape[1])), interpolation=cv2.INTER_AREA)
        stats = {}
        for name, bgr in COLORS.items():
            m = (np.linalg.norm(small.astype(np.int16) - np.array(bgr, np.int16), axis=2) < 45).astype(np.uint8)
            n, lab, st, _ = cv2.connectedComponentsWithStats(m)
            if n > 1:
                k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
                x, y, w, h, area = st[k]
                H, W = small.shape[:2]
                stats[name] = (area / (H * W), [int(1000 * x / W), int(1000 * y / H), int(1000 * (x + w) / W),
                                                int(1000 * (y + h) / H)])
        out.append(stats)
    return out


def find(grok: GrokClient, views: list[View], request: str) -> RoomFind:
    res = RoomFind(request=request, views=views)
    if not views:
        return res
    stats = _color_stats(views)
    r = grok.json(name="trova_room_pick", prompt=PICK_PROMPT.format(n=len(views), request=request), schema=PICK_SCHEMA,
                  images=[sheet(views)], mock_context={"request": request, "stats": stats})
    res.grok_latency_s += r.latency_s
    res.label, res.reason, res.confidence = r.data.get("label", ""), r.data.get("reason", ""), float(r.data.get("confidence") or 0)
    k = r.data.get("photo")
    if not isinstance(k, int) or not 1 <= k <= len(views):
        return res
    v = views[k - 1]
    rb = grok.json(name="trova_room_box", prompt=DIRECT_PROMPT.format(request=request, photo_line=""), schema=DIRECT_SCHEMA,
                   images=[v.frame], mock_context={"request": request, "stats": stats[k - 1]})
    res.grok_latency_s += rb.latency_s
    d = rb.data
    if d.get("found") and Box.valid_norm1000(d.get("box_2d")):
        H, W = v.frame.shape[:2]
        res.box = Box.from_norm1000(d["box_2d"], W, H, id=0, source="grok", label=res.label)
        res.view, res.status = k - 1, "found"
        res.reason = d.get("reason") or res.reason
    return res


def aim_angle(res: RoomFind, ppd: float) -> float:
    """Pan angle that puts the object's centre in the middle of the picture, never past what the photo showed."""
    v = res.views[res.view]
    half = v.frame.shape[1] / 2
    offset = (half - res.box.center[0]) / ppd
    return float(v.angle + max(-abs(half / ppd), min(abs(half / ppd), offset)))


def draw_found(res: RoomFind) -> np.ndarray:
    v = res.views[res.view]
    img = v.frame.copy()
    b = res.box
    cv2.rectangle(img, (int(b.x0), int(b.y0)), (int(b.x1), int(b.y1)), HIGHLIGHT, max(4, img.shape[1] // 250))
    _label(img, res.label or "found", int(b.x0), max(30, int(b.y0) - 8), HIGHLIGHT, max(0.6, img.shape[1] / 1600))
    return img


# ---------------------------------------------------------------- offline stand-ins (colour words only)
def _mock_pick(prompt: str, images: list, ctx: dict) -> dict:
    wanted = _colors_in(ctx["request"])
    best = max(((s.get(c, (0, None))[0], i) for i, s in enumerate(ctx["stats"]) for c in wanted), default=(0, None))
    if not wanted or best[0] < 0.002:
        return {"photo": None, "label": "", "reason": "Mock: no photo has that colour.", "confidence": 0.0}
    return {"photo": best[1] + 1, "label": f"{wanted[0]} object", "reason": "Mock: the photo with most of that colour.",
            "confidence": 0.7}


def _mock_box(prompt: str, images: list, ctx: dict) -> dict:
    wanted = _colors_in(ctx["request"])
    hits = [ctx["stats"][c] for c in wanted if c in ctx["stats"]]
    if not hits:
        return {"found": False, "box_2d": None, "label": "", "reason": "Mock: not in this photo.", "confidence": 0.0}
    return {"found": True, "box_2d": max(hits)[1], "label": f"{wanted[0]} object", "reason": "Mock: colour box.",
            "confidence": 0.7}


register_mock("trova_room_pick", _mock_pick)
register_mock("trova_room_box", _mock_box)
