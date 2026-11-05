"""Object proposals: every object on the table as a box, with no model and no training.

Main method: compare the frame with an "empty table" image (captured once per arm slot with the
arm at home). Colour differences are measured in LAB with lightness down-weighted, so shadows
and slow light changes are not objects. Without an empty-table image we estimate one with a
large median filter (works on a uniform table when objects are small relative to the frame).

Boxes that touch the frame edge and are large are flagged: that is usually a hand or the arm.
"""

from __future__ import annotations

import cv2
import numpy as np

from .boxes import Box, number_boxes


def estimate_background(frame: np.ndarray, object_scale: float = 0.35) -> np.ndarray:
    h, w = frame.shape[:2]
    small_w = 320
    scale = small_w / w
    small = cv2.resize(frame, (small_w, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    k = max(31, int(min(small.shape[:2]) * object_scale)) | 1
    bg = cv2.medianBlur(small, k)
    return cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)


def match_lighting(frame: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Per-channel gain so the frame's overall colour matches the reference (exposure and white
    balance drift between captures). Objects cover a minority of the table, so medians are
    dominated by the tabletop."""
    if reference.shape != frame.shape:
        reference = cv2.resize(reference, (frame.shape[1], frame.shape[0]))
    a = cv2.resize(frame, (160, 100), interpolation=cv2.INTER_AREA).reshape(-1, 3).astype(np.float32)
    b = cv2.resize(reference, (160, 100), interpolation=cv2.INTER_AREA).reshape(-1, 3).astype(np.float32)
    gain = np.clip(np.median(b, 0) / np.maximum(np.median(a, 0), 1.0), 0.6, 1.6)
    if np.all(np.abs(gain - 1) < 0.02):
        return frame
    return np.clip(frame.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def foreground_mask(frame: np.ndarray, background: np.ndarray, thresh: float = 30.0,
                    roi: np.ndarray | None = None) -> np.ndarray:
    if background.shape != frame.shape:
        background = cv2.resize(background, (frame.shape[1], frame.shape[0]))
    frame = match_lighting(frame, background)
    lab_f = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_b = cv2.cvtColor(background, cv2.COLOR_BGR2LAB).astype(np.float32)
    d = np.abs(lab_f - lab_b)
    diff = 0.5 * d[..., 0] + d[..., 1] + d[..., 2]   # lightness counts half: shadows stay background
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    mask = (diff > thresh).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    if roi is not None:
        mask = cv2.bitwise_and(mask, roi)
    return mask


def merge_boxes(boxes: list[Box], iou_thresh: float = 0.25, contain_thresh: float = 0.8) -> list[Box]:
    """Merge boxes that overlap a lot or sit inside one another (one object split in parts)."""
    boxes = sorted(boxes, key=lambda b: -b.area)
    merged = True
    while merged:
        merged = False
        out: list[Box] = []
        for b in boxes:
            for i, o in enumerate(out):
                if b.iou(o) > iou_thresh or b.coverage_by(o) > contain_thresh or o.coverage_by(b) > contain_thresh:
                    out[i] = o.union(b)
                    merged = True
                    break
            else:
                out.append(b)
        boxes = out
    return boxes


def find_blobs(frame: np.ndarray, background: np.ndarray | None = None, thresh: float = 30.0,
               min_area_frac: float = 0.0008, max_area_frac: float = 0.25,
               roi: np.ndarray | None = None) -> list[Box]:
    if background is None:
        background = estimate_background(frame)
    mask = foreground_mask(frame, background, thresh, roi)
    H, W = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[Box] = []
    for c in contours:
        a = cv2.contourArea(c)
        if a < min_area_frac * W * H:
            continue
        x, y, w, h = cv2.boundingRect(c)
        touches = x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1
        # Larger than max_area_frac: a tray, a binder, or the light changed. Never dropped: it is kept and flagged,
        # so a check sees it as a change and Grok cannot call it a false alarm.
        extra = {"oversize": True} if a > max_area_frac * W * H and not touches else {}
        boxes.append(Box(float(x), float(y), float(x + w), float(y + h), source="blob", score=float(a),
                         touches_border=touches, extra=extra))
    return number_boxes(merge_boxes(boxes))


def is_occluded(boxes: list[Box], frame_shape, area_frac: float = 0.03) -> bool:
    """A large blob touching the frame edge is almost always a hand, a sleeve or the arm."""
    H, W = frame_shape[:2]
    return any(b.touches_border and b.area > area_frac * W * H for b in boxes)


def drop_occluders(boxes: list[Box], frame_shape, area_frac: float = 0.03) -> list[Box]:
    H, W = frame_shape[:2]
    return number_boxes([b for b in boxes if not (b.touches_border and b.area > area_frac * W * H)])
