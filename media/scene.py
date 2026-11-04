"""A drawn top-down work table with everyday objects, a 2-link cobot seen from above, and a hand.
Used by make_concept_video.py. Object boxes in the video come from trova.proposals on these frames."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import cv2
import numpy as np

TW, TH = 1280, 800
TABLE_BGR = np.array([196, 208, 220], np.float32)


def _rot(pts, cx, cy, a):
    a = math.radians(a)
    c, s = math.cos(a), math.sin(a)
    return np.array([[cx + x * c - y * s, cy + x * s + y * c] for x, y in pts], np.int32)


def _rect(w, h):
    return [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]


def _poly(img, pts, color):
    cv2.fillPoly(img, [pts], color, cv2.LINE_AA)


@dataclass
class Obj:
    key: str
    kind: str
    x: float
    y: float
    a: float = 0.0
    color: tuple = (40, 40, 200)

    def at(self, x, y, a=None):
        return replace(self, x=x, y=y, a=self.a if a is None else a)


def draw_obj(img, o: Obj, mono=None):
    C = (lambda c: mono) if mono is not None else (lambda c: c)
    x, y, a = o.x, o.y, o.a
    if o.kind == "pen":
        _poly(img, _rot(_rect(220, 18), x, y, a), C((205, 110, 30)))
        _poly(img, _rot([(-110 - 22, 0), (-110, -9), (-110, 9)], x, y, a), C((120, 120, 125)))
        _poly(img, _rot([(r[0] + 85, r[1]) for r in _rect(60, 24)], x, y, a), C((150, 70, 20)))
    elif o.kind == "marker":
        _poly(img, _rot(_rect(180, 36), x, y, a), C((35, 35, 38)))
        _poly(img, _rot([(r[0] + 105, r[1]) for r in _rect(50, 40)], x, y, a), C((95, 95, 100)))
    elif o.kind == "mug":
        cv2.ellipse(img, (int(x + 56), int(y)), (30, 24), 0, -80, 80, C(tuple(int(v * 0.85) for v in o.color)), 13,
                    cv2.LINE_AA)
        cv2.circle(img, (int(x), int(y)), 60, C(o.color), -1, cv2.LINE_AA)
        cv2.circle(img, (int(x), int(y)), 47, C(tuple(int(v * 0.55) for v in o.color)), -1, cv2.LINE_AA)
    elif o.kind == "scissors":
        for s in (-1, 1):
            _poly(img, _rot([(-6, s * 2), (150, s * 1), (146, s * 9), (-6, s * 10)], x, y, a), C((118, 118, 126)))
            _poly(img, _rot([(2, s * 3), (-44, s * 14), (-48, s * 24), (-2, s * 12)], x, y, a), C((30, 130, 245)))
            cx, cy = _rot([(-70, s * 28)], x, y, a)[0]
            cv2.ellipse(img, (int(cx), int(cy)), (30, 19), a + s * 12, 0, 360, C((30, 130, 245)), 11, cv2.LINE_AA)
        cv2.circle(img, (int(x), int(y)), 8, C((60, 60, 66)), -1, cv2.LINE_AA)
    elif o.kind == "blister":
        _poly(img, _rot(_rect(230, 128), x, y, a), C((215, 196, 170)))
        for i in range(5):
            for j in range(2):
                px, py = _rot([(-84 + i * 42, -28 + j * 56)], x, y, a)[0]
                cv2.circle(img, (int(px), int(py)), 17, C((248, 248, 250)), -1, cv2.LINE_AA)
                if mono is None:
                    cv2.circle(img, (int(px), int(py)), 17, (170, 165, 160), 2, cv2.LINE_AA)
    elif o.kind == "tape":
        cv2.circle(img, (int(x), int(y)), 48, C((40, 196, 238)), 28, cv2.LINE_AA)
    elif o.kind == "stapler":
        _poly(img, _rot(_rect(230, 58), x, y, a), C((40, 40, 45)))
        _poly(img, _rot([(r[0] + 10, r[1]) for r in _rect(180, 20)], x, y, a), C((45, 45, 190)))
    elif o.kind == "usb":
        _poly(img, _rot(_rect(72, 28), x, y, a), C((200, 120, 35)))
        _poly(img, _rot([(r[0] + 48, r[1]) for r in _rect(26, 20)], x, y, a), C((190, 190, 195)))


def make_table(seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:TH, 0:TW].astype(np.float32)
    light = 1.0 - 0.10 * (((xx - TW * 0.42) / TW) ** 2 + ((yy - TH * 0.35) / TH) ** 2)
    img = TABLE_BGR[None, None, :] * light[..., None]
    grain = cv2.GaussianBlur(rng.normal(0, 1, (TH, TW)).astype(np.float32), (0, 0), sigmaX=18, sigmaY=2)
    img += grain[..., None] * 6 + rng.normal(0, 1.6, (TH, TW, 1))
    return np.clip(img, 0, 255).astype(np.uint8)


def render(table: np.ndarray, objs: list[Obj], gain=(1.0, 1.0, 1.0)) -> np.ndarray:
    img = table.astype(np.float32)
    m = np.zeros((TH, TW), np.uint8)
    for o in objs:
        draw_obj(m, replace(o, x=o.x + 9, y=o.y + 11), mono=255)
    m = cv2.GaussianBlur(m, (31, 31), 0).astype(np.float32) / 255
    img *= (1 - 0.2 * m)[..., None]
    img = img.astype(np.uint8)
    for o in objs:
        draw_obj(img, o)
    if gain != (1.0, 1.0, 1.0):
        img = np.clip(img.astype(np.float32) * np.array(gain, np.float32), 0, 255).astype(np.uint8)
    return img


# ---------------------------------------------------------------- arm (top view, 2 links)
BASE = (640.0, 905.0)
L1 = L2 = 500.0
PARK = (1180.0, 980.0)


def ik(tx, ty):
    dx, dy = tx - BASE[0], ty - BASE[1]
    d = min(math.hypot(dx, dy), L1 + L2 - 1)
    c2 = (d * d - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    q2 = -math.acos(max(-1.0, min(1.0, c2)))
    q1 = math.atan2(dy, dx) - math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2))
    ex, ey = BASE[0] + L1 * math.cos(q1), BASE[1] + L1 * math.sin(q1)
    return (ex, ey), (ex + L2 * math.cos(q1 + q2), ey + L2 * math.sin(q1 + q2))


def draw_arm(img: np.ndarray, tip) -> None:
    (ex, ey), (px, py) = ik(*tip)
    pts = [BASE, (ex, ey), (px, py)]
    sh = np.zeros(img.shape[:2], np.uint8)
    for (x0, y0), (x1, y1), w in ((pts[0], pts[1], 38), (pts[1], pts[2], 30)):
        cv2.line(sh, (int(x0 + 22), int(y0 + 28)), (int(x1 + 22), int(y1 + 28)), 255, w, cv2.LINE_AA)
    sh = cv2.GaussianBlur(sh, (41, 41), 0).astype(np.float32) / 255
    img[:] = (img.astype(np.float32) * (1 - 0.28 * sh)[..., None]).astype(np.uint8)
    for (x0, y0), (x1, y1), w in ((pts[0], pts[1], 40), (pts[1], pts[2], 32)):
        cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)), (72, 74, 80), w + 6, cv2.LINE_AA)
        cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)), (228, 228, 231), w, cv2.LINE_AA)
    for (x, y), r in ((pts[0], 46), (pts[1], 30), (pts[2], 22)):
        cv2.circle(img, (int(x), int(y)), r + 4, (60, 62, 68), -1, cv2.LINE_AA)
        cv2.circle(img, (int(x), int(y)), r, (215, 160, 20), -1, cv2.LINE_AA)   # UR-style blue caps
    cv2.circle(img, (int(px), int(py)), 9, (245, 245, 245), -1, cv2.LINE_AA)


# ---------------------------------------------------------------- hand from the top edge
SKIN = (128, 160, 212)


def draw_hand(img: np.ndarray, x: float, y: float) -> None:
    sh = np.zeros(img.shape[:2], np.uint8)
    cv2.line(sh, (int(x + 70), -80), (int(x + 16), int(y + 14)), 255, 120, cv2.LINE_AA)
    cv2.ellipse(sh, (int(x + 14), int(y + 22)), (64, 76), 10, 0, 360, 255, -1, cv2.LINE_AA)
    sh = cv2.GaussianBlur(sh, (51, 51), 0).astype(np.float32) / 255
    img[:] = (img.astype(np.float32) * (1 - 0.3 * sh)[..., None]).astype(np.uint8)
    cv2.line(img, (int(x + 60), -80), (int(x + 4), int(y - 6)), (110, 140, 190), 112, cv2.LINE_AA)
    cv2.ellipse(img, (int(x), int(y)), (62, 74), 10, 0, 360, SKIN, -1, cv2.LINE_AA)
    for dx in (-36, -12, 12, 34):
        cv2.ellipse(img, (int(x + dx), int(y + 78 - abs(dx) * 0.4)), (12, 30), 8, 0, 360, SKIN, -1, cv2.LINE_AA)
    cv2.ellipse(img, (int(x + 64), int(y + 10)), (14, 34), -40, 0, 360, SKIN, -1, cv2.LINE_AA)


def initial_objects() -> list[Obj]:
    return [
        Obj("pen", "pen", 300, 250, 18),
        Obj("mug_red", "mug", 575, 215, color=(45, 45, 205)),
        Obj("scissors", "scissors", 860, 245, -25),
        Obj("tape", "tape", 1095, 225),
        Obj("marker", "marker", 205, 560, -35),
        Obj("blister", "blister", 470, 545),
        Obj("stapler", "stapler", 780, 610, 8),
        Obj("mug_black", "mug", 1050, 545, color=(42, 42, 42)),
    ]


USB = Obj("usb", "usb", 640, 420, 30)
SCISSORS_NEW = (925, 400, -8)
