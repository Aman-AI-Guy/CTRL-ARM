"""Validate image regions without claiming object semantics or metric robot coordinates."""

import numpy as np

from .boxes import Box


def validated_region(values, frame, *, source="manual", label=""):
    coords = np.asarray(values, dtype=float)
    h, w = frame.shape[:2]
    if coords.shape != (4,) or not np.isfinite(coords).all():
        raise ValueError("A region requires four finite pixel coordinates")
    x0, y0, x1, y1 = coords
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError("Region must be ordered and inside the snapshot")
    if x1 - x0 < 4 or y1 - y0 < 4 or (x1 - x0) * (y1 - y0) > 0.8 * w * h:
        raise ValueError("Region is too small or covers almost the entire image")
    return Box(
        *coords,
        source=source,
        label=label,
        touches_border=x0 == 0 or y0 == 0 or x1 == w or y1 == h,
    )
