"""Camera pixels <-> table millimetres, via a homography of the table plane.

Two ways to calibrate, both a few seconds per arm slot (the mount may move between slots):
- ArUco: tape 5 printed markers (ids 0-4) on the table, measure where their centres are in mm,
  put those in data/markers.json, press "Calibrate (ArUco)".
- Clicks: use 5+ points for robot coordinates, or 4+ for table coordinates, in the live view.

Until calibrated, a default maps the image rectangle onto a table_w x table_h mm rectangle, so the
simulated arm works out of the box.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

from .config import atomic_json_write

ARUCO_DICT = cv2.aruco.DICT_4X4_50
MAX_ERROR_MM = 15.0      # a fit worse than this has a wrong point in it


def _flat(pts: np.ndarray) -> bool:
    """True if the points are (nearly) on one line: no usable homography."""
    diag2 = max(float(np.sum((a - b) ** 2)) for a, b in combinations(pts, 2))
    if len(pts) == 4:     # an exact fit: every triple must span a real triangle
        return any(abs((b - a)[0] * (c - a)[1] - (b - a)[1] * (c - a)[0]) / 2 < 0.01 * diag2
                   for a, b, c in combinations(pts, 3))
    return cv2.contourArea(cv2.convexHull(pts.astype(np.float32))) < 0.05 * diag2


@dataclass
class Calibration:
    H: np.ndarray | None = None                       # 3x3, pixel -> mm
    points: list[list[float]] = field(default_factory=list)   # [u, v, x_mm, y_mm]
    source: str = "none"                              # none | default | aruco | aruco_robot | clicks | arm
    size: list[int] | None = None                     # [w, h] of the camera image it was made on
    pose: list[float] | None = None                   # camera on the arm: the survey pose it was made at

    camera: str | None = None                         # source specification used to fit robot-frame points

    @property
    def ok(self) -> bool:
        return self.H is not None

    @classmethod
    def default(cls, width: int, height: int, table_w_mm: float, table_h_mm: float) -> "Calibration":
        pts = [[0, 0, 0, 0], [width, 0, table_w_mm, 0], [width, height, table_w_mm, table_h_mm],
               [0, height, 0, table_h_mm]]
        c = cls(points=[list(map(float, p)) for p in pts], source="default")
        c.fit()
        return c

    def fit(self) -> float:
        if self.source in ("arm", "aruco_robot") and len(self.points) < 5:
            raise ValueError("Need at least 5 points for arm calibration; four points cannot measure fit error")
        if any(len(p) != 4 or not all(math.isfinite(v) for v in p) for p in self.points):
            raise ValueError("Calibration points must contain four finite coordinates")
        if len(self.points) < 4:
            raise ValueError("Need at least 4 points to calibrate")
        src = np.array([[p[0], p[1]] for p in self.points], np.float64)
        dst = np.array([[p[2], p[3]] for p in self.points], np.float64)
        for a, b in combinations(range(len(src)), 2):
            if np.hypot(*(src[a] - src[b])) < 10 or np.hypot(*(dst[a] - dst[b])) < 3:
                raise ValueError("Two calibration points are on the same spot: reset, then use spread-out points")
        if _flat(src) or _flat(dst):
            raise ValueError("Calibration points are nearly in a line: reset, then use points near the table corners")
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC if len(self.points) > 4 else 0, 5.0)
        if H is None:
            raise ValueError("Points are degenerate (collinear?)")
        self.H = H
        err = self.error_mm()
        if err > MAX_ERROR_MM:
            self.H = None
            raise ValueError(f"Calibration error {err:.0f} mm: one of the points is wrong. Reset and redo")
        return err

    def pixel_to_table(self, u: float, v: float) -> tuple[float, float]:
        p = cv2.perspectiveTransform(np.array([[[u, v]]], np.float64), self.H)[0, 0]
        return float(p[0]), float(p[1])

    def table_to_pixel(self, x: float, y: float) -> tuple[float, float]:
        p = cv2.perspectiveTransform(np.array([[[x, y]]], np.float64), np.linalg.inv(self.H))[0, 0]
        return float(p[0]), float(p[1])

    def error_mm(self) -> float:
        if not self.ok or not self.points:
            return float("nan")
        errs = [np.hypot(*(np.array(self.pixel_to_table(p[0], p[1])) - np.array(p[2:4]))) for p in self.points]
        return float(np.max(errs))

    def as_dict(self) -> dict:
        return {"source": self.source, "points": self.points, "ok": self.ok, "size": self.size, "pose": self.pose,
                "camera": self.camera, "error_mm": None if not self.ok else round(self.error_mm(), 2)}

    def save(self, path: Path) -> None:
        atomic_json_write(path, {"source": self.source, "points": self.points, "size": self.size, "pose": self.pose,
                                 "camera": self.camera})

    @classmethod
    def load(cls, path: Path) -> "Calibration | None":
        try:
            if not path.exists():
                return None
            d = json.loads(path.read_text())
            c = cls(points=d["points"], source=d.get("source", "clicks"), size=d.get("size"), pose=d.get("pose"), camera=d.get("camera"))
            if c.size is not None and (not isinstance(c.size, list) or len(c.size) != 2
                    or not all(type(v) is int and v > 0 for v in c.size)):
                raise ValueError("invalid calibration image size")
            if c.pose is not None and (not isinstance(c.pose, list) or not c.pose
                    or not all(math.isfinite(v) for v in c.pose)):
                raise ValueError("invalid calibration survey pose")
            c.fit()
        except (ValueError, KeyError, TypeError, IndexError, OSError, cv2.error) as e:
            logging.getLogger(__name__).warning("Ignoring invalid calibration %s: %s", path, e)
            return None
        return c


def detect_aruco(frame: np.ndarray) -> dict[int, tuple[float, float]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(ARUCO_DICT), cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(gray)
    if ids is None:
        return {}
    return {int(i): tuple(map(float, c.reshape(-1, 2).mean(0))) for i, c in zip(ids.ravel(), corners)}


def load_marker_positions(path: Path, table_w_mm: float, table_h_mm: float) -> dict[int, tuple[float, float]]:
    """Marker centre positions in mm. Default: ids 0-3 at the table corners, clockwise from top-left."""
    if path.exists():
        return {int(k): tuple(v) for k, v in json.loads(path.read_text()).items() if not k.startswith("_")}
    return {0: (0.0, 0.0), 1: (table_w_mm, 0.0), 2: (table_w_mm, table_h_mm), 3: (0.0, table_h_mm)}


def calibrate_aruco(frame: np.ndarray, positions_mm: dict[int, tuple[float, float]]) -> Calibration:
    found = detect_aruco(frame)
    pts = [[*found[i], *positions_mm[i]] for i in sorted(found) if i in positions_mm]
    if len(pts) < 4:
        raise ValueError(f"Saw markers {sorted(found)}; need 4 of {sorted(positions_mm)}")
    c = Calibration(points=[list(map(float, p)) for p in pts], source="aruco", size=[frame.shape[1], frame.shape[0]])
    c.fit()
    return c


def make_marker_sheet(out_dir: Path, ids=(0, 1, 2, 3, 4), size_px: int = 600) -> list[Path]:
    """PNG per marker. Print at a fixed size (e.g. 5 cm) with a white border."""
    out_dir.mkdir(parents=True, exist_ok=True)
    d = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    paths = []
    for i in ids:
        m = cv2.aruco.generateImageMarker(d, i, size_px)
        m = cv2.copyMakeBorder(m, 60, 100, 60, 60, cv2.BORDER_CONSTANT, value=255)
        cv2.putText(m, f"Trova marker {i}", (60, m.shape[0] - 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 2)
        p = out_dir / f"aruco_{i}.png"
        cv2.imwrite(str(p), m)
        paths.append(p)
    return paths
