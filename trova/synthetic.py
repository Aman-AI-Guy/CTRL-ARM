"""A synthetic top-down work table: coloured objects on a lit, noisy tabletop, with soft shadows.

It powers the offline demo (TROVA_CAMERA=demo), the unit tests and the synthetic eval cases.
The jury will use real objects; this only proves the pipeline end to end before we get there.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import cv2
import numpy as np

from .boxes import Box

COLORS = {  # name -> BGR
    "red": (45, 45, 210), "blue": (200, 95, 35), "green": (60, 160, 60), "yellow": (40, 205, 235),
    "black": (35, 35, 35), "orange": (30, 135, 245), "purple": (150, 60, 135),
}
KINDS = [("mug", "circle"), ("pen", "bar"), ("box", "rect"), ("tape", "ring"), ("cube", "square"),
         ("phone", "rect"), ("stapler", "bar"), ("lid", "circle")]


@dataclass
class ObjSpec:
    name: str
    color: str
    shape: str
    cx: float
    cy: float
    size: float
    angle: float = 0.0

    @property
    def label(self) -> str:
        return f"{self.color} {self.name}"


def _draw(img, o: ObjSpec, color, offset=(0, 0)):
    cx, cy = int(o.cx + offset[0]), int(o.cy + offset[1])
    s = o.size
    if o.shape == "circle":
        cv2.circle(img, (cx, cy), int(s / 2), color, -1, cv2.LINE_AA)
    elif o.shape == "ring":
        cv2.circle(img, (cx, cy), int(s / 2), color, max(6, int(s / 6)), cv2.LINE_AA)
    else:
        w, h = {"rect": (s, s * 0.6), "bar": (s * 1.4, s * 0.2), "square": (s * 0.7, s * 0.7)}[o.shape]
        pts = cv2.boxPoints(((cx, cy), (w, h), o.angle)).astype(np.int32)
        cv2.fillPoly(img, [pts], color, cv2.LINE_AA)


def object_box(o: ObjSpec) -> Box:
    mask = np.zeros((int(o.cy + o.size * 2), int(o.cx + o.size * 2)), np.uint8)
    _draw(mask, o, 255)
    ys, xs = np.nonzero(mask)
    return Box(float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()), label=o.label, source="truth")


class SyntheticTable:
    def __init__(self, width: int = 1280, height: int = 800, n_objects: int = 8, seed: int = 7):
        self.w, self.h = width, height
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.objects: list[ObjSpec] = []
        self._table = self._make_table()
        for _ in range(n_objects):
            self.add_random()

    def _make_table(self) -> np.ndarray:
        yy, xx = np.mgrid[0:self.h, 0:self.w].astype(np.float32)
        light = 1.0 - 0.12 * (((xx - self.w * 0.4) / self.w) ** 2 + ((yy - self.h * 0.3) / self.h) ** 2)
        base = np.array([196, 208, 218], np.float32)  # light beige, BGR
        return np.clip(base[None, None, :] * light[..., None], 0, 255)

    def _free_spot(self, size: float) -> tuple[float, float] | None:
        margin = size * 0.9 + 20
        for _ in range(200):
            cx = self.rng.uniform(margin, self.w - margin)
            cy = self.rng.uniform(margin, self.h - margin)
            if all(np.hypot(cx - o.cx, cy - o.cy) > (size + o.size) * 0.85 + 30 for o in self.objects):
                return cx, cy
        return None

    def add_random(self, color: str | None = None, kind: tuple[str, str] | None = None) -> ObjSpec | None:
        name, shape = kind or self.rng.choice(KINDS)
        color = color or self.rng.choice(list(COLORS))
        size = self.rng.uniform(70, 130)
        spot = self._free_spot(size)
        if spot is None:
            return None
        o = ObjSpec(name, color, shape, spot[0], spot[1], size, self.rng.uniform(0, 180))
        self.objects.append(o)
        return o

    def render(self, shadows: bool = True, noise: float = 3.0, hand: bool = False) -> np.ndarray:
        img = self._table.copy()
        if shadows:
            for o in self.objects:
                m = np.zeros((self.h, self.w), np.uint8)
                _draw(m, o, 255, offset=(o.size * 0.08, o.size * 0.08))
                m = cv2.GaussianBlur(m, (21, 21), 0).astype(np.float32) / 255.0
                img *= (1.0 - 0.15 * m)[..., None]
        img = img.astype(np.uint8)
        for o in self.objects:
            _draw(img, o, COLORS[o.color])
        if hand:  # a skin-coloured arm entering from the bottom edge
            cv2.ellipse(img, (int(self.w * 0.55), self.h), (90, 260), 10, 0, 360, (120, 160, 215), -1, cv2.LINE_AA)
        if noise:
            img = np.clip(img.astype(np.float32) + self.np_rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
        return img

    def empty(self) -> np.ndarray:
        objects, self.objects = self.objects, []
        try:
            return self.render(shadows=False)
        finally:
            self.objects = objects

    def truth(self) -> list[Box]:
        return [object_box(o) for o in self.objects]

    def tamper(self, kind: str | None = None) -> dict:
        """Add, remove or move one object, like a judge would. Returns what was done."""
        kind = kind or self.rng.choice(["add", "remove", "move"])
        if kind == "add" or not self.objects:
            o = self.add_random()
            return {"kind": "added", "label": o.label if o else None}
        o = self.rng.choice(self.objects)
        if kind == "remove":
            self.objects.remove(o)
            return {"kind": "removed", "label": o.label}
        self.objects.remove(o)
        spot = self._free_spot(o.size)
        if spot:
            o.cx, o.cy = spot
        self.objects.append(o)
        return {"kind": "moved", "label": o.label}
