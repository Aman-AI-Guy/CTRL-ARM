"""Feature 2: line-clearance zones, the way a packaging line is really checked.

A zone is a polygon on the camera image with a rule:
  empty    nothing may be inside (the line area after changeover)
  allowed  only objects whose name matches the allowed list (the reject bin may hold only blister packs)
Every "what changed" check also evaluates the zones; any violation makes the line NOT clear, with the reason.
Names for "allowed" zones come from the inventory call (Grok); "empty" zones need no Grok at all.
Zones are saved in data/zones.json (image coordinates, so redraw them if the camera moves).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .boxes import Box

RULES = ("empty", "allowed")
EMPTY_OVERLAP = 0.05     # an object with more than this fraction of its box inside an 'empty' zone violates it
SEVERAL = re.compile(r"(\band\b|\bwith\b|\be\b|\bcon\b|[&+,/])")   # "leaflet and blister pack" is two objects


@dataclass
class Zone:
    name: str
    polygon: list[list[float]]
    rule: str = "empty"
    allowed: list[str] = field(default_factory=list)

    def contains(self, box: Box) -> bool:
        poly = np.array(self.polygon, np.float32).reshape(-1, 1, 2)
        return cv2.pointPolygonTest(poly, tuple(float(v) for v in box.center), False) >= 0

    def overlap(self, box: Box) -> float:
        """Fraction of the box's area that lies inside the zone."""
        x0, y0 = int(np.floor(box.x0)), int(np.floor(box.y0))
        w, h = max(1, int(np.ceil(box.x1)) - x0), max(1, int(np.ceil(box.y1)) - y0)
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [np.round(np.array(self.polygon, np.float64) - (x0, y0)).astype(np.int32).reshape(-1, 1, 2)], 1)
        return float(m.mean())

    def touches(self, box: Box) -> bool:
        """'empty' zones: any real part of the object inside counts, not only its centre."""
        return self.contains(box) or self.overlap(box) > EMPTY_OVERLAP


def load_zones(path: Path) -> list[Zone]:
    if not Path(path).exists():
        return []
    return [Zone(**z) for z in json.loads(Path(path).read_text())]


def save_zones(path: Path, zones: list[Zone]) -> None:
    Path(path).write_text(json.dumps([asdict(z) for z in zones], indent=2))


def _words(s: str) -> list[str]:
    return [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in re.findall(r"[a-zà-ù0-9]+", s.lower())]


def _allowed(name: str, allowed: list[str]) -> bool:
    """Whole words: every word of an allowed item, in order, in the object's name ("blue blister pack" is a
    "blister pack"; "pencil sharpener" is not a "pen"). A name that joins several objects is never allowed."""
    n = name.lower().strip()
    if not n or SEVERAL.search(n):
        return False
    nw = _words(n)
    for a in allowed:
        aw = _words(a)
        if aw and any(nw[i:i + len(aw)] == aw for i in range(len(nw) - len(aw) + 1)):
            return True
    return False


def evaluate(zones: list[Zone], boxes: list[Box], names: dict[int, str] | None = None) -> list[dict]:
    names = names or {}
    out = []
    for z in zones:
        inside = [b for b in boxes if (z.touches(b) if z.rule == "empty" else z.contains(b))]
        objs = [{"id": b.id, "name": names.get(b.id, f"object {b.id}"), "box": b.to_list()} for b in inside]
        problems = []
        if z.rule == "empty":
            problems = [f"{o['name']} must not be in {z.name}" for o in objs]
        elif z.rule == "allowed":
            problems = [f"{o['name']} is not allowed in {z.name}" for o in objs
                        if o["id"] not in names or not _allowed(o["name"], z.allowed)]
        out.append({"name": z.name, "rule": z.rule, "allowed": z.allowed, "objects": objs, "ok": not problems,
                    "problems": problems})
    return out


def draw_zones(img: np.ndarray, zones: list[Zone], results: list[dict] | None = None,
               draft: list[list[float]] | None = None) -> np.ndarray:
    ok = {r["name"]: r["ok"] for r in (results or [])}
    overlay = img.copy()
    lw = max(2, img.shape[1] // 500)
    for z in zones:
        color = (60, 60, 230) if ok.get(z.name) is False else (230, 160, 0)
        pts = np.array(z.polygon, np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(img, [pts], True, color, lw, cv2.LINE_AA)
        x, y = pts[:, 0, 0].min(), pts[:, 0, 1].min()
        tag = f"{z.name}: {'empty' if z.rule == 'empty' else 'only ' + ', '.join(z.allowed)}"
        cv2.putText(img, tag, (int(x) + 6, int(y) + 26), cv2.FONT_HERSHEY_SIMPLEX, max(0.5, img.shape[1] / 1800),
                    color, 2, cv2.LINE_AA)
    img = cv2.addWeighted(overlay, 0.15, img, 0.85, 0)
    if draft:
        pts = np.array(draft, np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], False, (255, 0, 200), lw, cv2.LINE_AA)
        for p in draft:
            cv2.circle(img, (int(p[0]), int(p[1])), lw * 3, (255, 0, 200), -1)
    return img
