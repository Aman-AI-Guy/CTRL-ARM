"""Drawing for Grok and for people: numbered boxes ("set of marks"), crops, contact sheets, colours."""

from __future__ import annotations

import cv2
import numpy as np

from .boxes import Box

MARK = (255, 0, 200)       # magenta: rare on real tables, so numbers stay readable
HIGHLIGHT = (60, 220, 60)  # green
CHANGE = {"added": (0, 140, 255), "removed": (60, 60, 230), "moved": (230, 160, 0)}


def _label(img, text, x, y, bg, scale):
    font = cv2.FONT_HERSHEY_SIMPLEX
    th = max(1, int(scale * 2))
    (tw, tht), base = cv2.getTextSize(text, font, scale, th)
    x, y = int(max(0, x)), int(max(tht + 4, y))
    cv2.rectangle(img, (x, y - tht - 6), (x + tw + 6, y + base - 2), bg, -1)
    cv2.putText(img, text, (x + 3, y - 3), font, scale, (255, 255, 255), th, cv2.LINE_AA)


def draw_marks(frame: np.ndarray, boxes: list[Box], highlight: int | None = None,
               dim_others: bool = False) -> np.ndarray:
    """Numbered boxes on a copy of the frame. This exact image is what Grok sees."""
    img = frame.copy()
    scale = max(0.5, frame.shape[1] / 1400)
    lw = max(2, int(frame.shape[1] / 600))
    for b in boxes:
        hit = highlight is not None and b.id == highlight
        if dim_others and not hit:
            continue
        color = HIGHLIGHT if hit else MARK
        cv2.rectangle(img, (int(b.x0), int(b.y0)), (int(b.x1), int(b.y1)), color, lw * (2 if hit else 1))
        _label(img, str(b.id), b.x0, b.y0, color, scale)
    return img


def draw_changes(frame: np.ndarray, changes: list, which: str) -> np.ndarray:
    """which='ref' draws removed/moved at their old place; 'cur' draws added/moved at the new place."""
    img = frame.copy()
    scale = max(0.5, frame.shape[1] / 1400)
    lw = max(2, int(frame.shape[1] / 500))
    for ch in changes:
        b = ch.ref_box if which == "ref" else ch.cur_box
        if b is None:
            continue
        color = CHANGE.get(ch.kind, MARK)
        cv2.rectangle(img, (int(b.x0), int(b.y0)), (int(b.x1), int(b.y1)), color, lw)
        _label(img, f"C{ch.id} {ch.kind}", b.x0, b.y0, color, scale)
    return img


def crop(frame: np.ndarray, box: Box, pad: float = 0.15, min_side: int = 24) -> np.ndarray:
    H, W = frame.shape[:2]
    px, py = box.w * pad + 4, box.h * pad + 4
    x0, y0 = int(max(0, box.x0 - px)), int(max(0, box.y0 - py))
    x1, y1 = int(min(W, box.x1 + px)), int(min(H, box.y1 + py))
    if x1 - x0 < min_side or y1 - y0 < min_side:
        cx, cy = box.center
        x0, x1 = int(max(0, cx - min_side)), int(min(W, cx + min_side))
        y0, y1 = int(max(0, cy - min_side)), int(min(H, cy + min_side))
    return frame[y0:y1, x0:x1].copy()


def _fit(img: np.ndarray, cell: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = (cell - 8) / max(h, w)
    img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    out = np.full((cell, cell, 3), 255, np.uint8)
    y, x = (cell - img.shape[0]) // 2, (cell - img.shape[1]) // 2
    out[y:y + img.shape[0], x:x + img.shape[1]] = img
    return out


def contact_sheet(frame: np.ndarray, boxes: list[Box], cell: int = 180, cols: int = 6) -> np.ndarray:
    """A grid of close-ups, one per numbered box, so Grok can see small objects clearly."""
    tiles = []
    for b in boxes:
        t = _fit(crop(frame, b), cell)
        _label(t, str(b.id), 2, 26, MARK, 0.7)
        tiles.append(t)
    if not tiles:
        return np.full((cell, cell, 3), 255, np.uint8)
    rows = []
    for i in range(0, len(tiles), cols):
        row = tiles[i:i + cols]
        row += [np.full((cell, cell, 3), 255, np.uint8)] * (cols - len(row))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def pair_sheet(pairs: list[tuple[str, np.ndarray | None, np.ndarray | None]], cell: int = 200) -> np.ndarray:
    """One row per candidate change: [before crop | after crop], labelled C<id>."""
    blank = np.full((cell, cell, 3), 235, np.uint8)
    rows = []
    for name, before, after in pairs:
        b = _fit(before, cell) if before is not None and before.size else blank.copy()
        a = _fit(after, cell) if after is not None and after.size else blank.copy()
        _label(b, f"{name} before", 2, 26, (90, 90, 90), 0.6)
        _label(a, f"{name} after", 2, 26, (90, 90, 90), 0.6)
        rows.append(np.hstack([b, a]))
    return np.vstack(rows) if rows else blank


HUE_NAMES = [(10, "red"), (22, "orange"), (34, "yellow"), (85, "green"), (100, "teal"),
             (130, "blue"), (160, "purple"), (180, "red")]


def color_name(img: np.ndarray) -> str:
    """Dominant colour of a crop, ignoring the beige/grey table. Used by mock mode and logs."""
    if img is None or not img.size:
        return "unknown"
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    v, s = hsv[:, 2], hsv[:, 1]
    dark = v < 70
    vivid = (s > 90) & ~dark
    if dark.mean() > 0.25 and dark.sum() >= vivid.sum():
        return "black"
    if vivid.sum() < 0.05 * len(hsv):
        return "white" if v.mean() > 180 else "grey"
    hue = float(np.median(hsv[vivid, 0]))
    for limit, name in HUE_NAMES:
        if hue <= limit:
            return name
    return "red"


def jpeg(img: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""
