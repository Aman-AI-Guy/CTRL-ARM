"""Camera pixels <-> table millimetres, via a homography of the table plane.

Two ways to calibrate, both a few seconds per arm slot (the mount may move between slots):
- ArUco: tape 4 printed markers (ids 0-3) on the table, measure where their centres are in mm,
  put those in data/markers.json, press "Calibrate (ArUco)".
- Clicks: click 4+ points in the live view and type their table coordinates in mm.

Until calibrated, a default maps the image rectangle onto a table_w x table_h mm rectangle, so the
simulated arm works out of the box.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

ARUCO_DICT = cv2.aruco.DICT_4X4_50


@dataclass
class Calibration:
    H: np.ndarray | None = None                       # 3x3, pixel -> mm
    points: list[list[float]] = field(default_factory=list)   # [u, v, x_mm, y_mm]
    source: str = "none"                              # none | default | aruco | clicks

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
        if len(self.points) < 4:
            raise ValueError("Need at least 4 points to calibrate")
        src = np.array([[p[0], p[1]] for p in self.points], np.float64)
        dst = np.array([[p[2], p[3]] for p in self.points], np.float64)
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC if len(self.points) > 4 else 0, 5.0)
        if H is None:
            raise ValueError("Points are degenerate (collinear?)")
        self.H = H
        return self.error_mm()

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
        return float(np.mean(errs))

    def as_dict(self) -> dict:
        return {"source": self.source, "points": self.points, "ok": self.ok,
                "error_mm": None if not self.ok else round(self.error_mm(), 2)}

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"source": self.source, "points": self.points}, indent=2))

    @classmethod
    def load(cls, path: Path) -> "Calibration | None":
        if not path.exists():
            return None
        d = json.loads(path.read_text())
        c = cls(points=d["points"], source=d.get("source", "clicks"))
        c.fit()
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
    c = Calibration(points=[list(map(float, p)) for p in pts], source="aruco")
    c.fit()
    return c


def make_marker_sheet(out_dir: Path, ids=(0, 1, 2, 3), size_px: int = 600) -> list[Path]:
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
