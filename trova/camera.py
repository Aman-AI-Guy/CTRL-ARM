"""Frame sources. Pick one with TROVA_CAMERA, or switch live from the UI.

  demo                 synthetic table (offline demo; supports "tamper")
  0, 1, 2 ...          webcam index. An iPhone via Continuity Camera shows up as one of these.
                       Turn OFF Center Stage / Portrait / Studio Light (macOS Control Centre ->
                       Video Effects): they reframe the image and break calibration.
  path/to/video.mp4    a recording of the station, looped (work without the arm)
  path/to/image.jpg    one still
  path/to/folder/      every jpg/png in the folder; step with next/prev
An uploaded photo (e.g. a phone shot of the table) overrides the source until cleared.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .synthetic import SyntheticTable


class FrameSource:
    kind = "base"

    def read(self) -> np.ndarray:
        raise NotImplementedError

    def info(self) -> dict:
        return {"kind": self.kind}

    def close(self) -> None:
        pass


class DemoSource(FrameSource):
    kind = "demo"

    def __init__(self, seed: int = 7):
        self.table = SyntheticTable(seed=seed)
        self.hand = False

    def read(self) -> np.ndarray:
        return self.table.render(hand=self.hand)

    def empty(self) -> np.ndarray:
        return self.table.empty()

    def info(self) -> dict:
        return {"kind": self.kind, "objects": [o.label for o in self.table.objects]}


class WebcamSource(FrameSource):
    kind = "webcam"

    def __init__(self, index: int, width: int = 1920, height: int = 1080):
        self.index = index
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {index}")
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._run = True
        threading.Thread(target=self._loop, daemon=True).start()
        t0 = time.time()
        while self._frame is None and time.time() - t0 < 5:
            time.sleep(0.05)
        if self._frame is None:
            raise RuntimeError(f"Camera {index} opened but sent no frames")

    def _loop(self):
        while self._run:   # always keep only the newest frame, so finds never use a stale one
            ok, f = self.cap.read()
            if ok:
                with self._lock:
                    self._frame = f
            else:
                time.sleep(0.02)

    def read(self) -> np.ndarray:
        with self._lock:
            return self._frame.copy()

    def info(self) -> dict:
        f = self._frame
        return {"kind": self.kind, "index": self.index, "size": None if f is None else [f.shape[1], f.shape[0]]}

    def close(self) -> None:
        self._run = False
        self.cap.release()


class VideoSource(FrameSource):
    kind = "video"

    def __init__(self, path: str):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 15
        self.t0 = time.time()
        self.last: np.ndarray | None = None

    def read(self) -> np.ndarray:
        # play in real time and loop
        n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        idx = int((time.time() - self.t0) * self.fps) % n
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = self.cap.read()
        if ok:
            self.last = f
        if self.last is None:
            raise RuntimeError("Video gave no frames")
        return self.last.copy()

    def info(self) -> dict:
        return {"kind": self.kind, "path": self.path}


class ImageSource(FrameSource):
    kind = "images"

    def __init__(self, path: str):
        p = Path(path)
        self.files = sorted([f for f in p.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")]) \
            if p.is_dir() else [p]
        if not self.files:
            raise RuntimeError(f"No images in {path}")
        self.i = 0
        self._cache: dict[int, np.ndarray] = {}

    def step(self, delta: int) -> None:
        self.i = (self.i + delta) % len(self.files)

    def read(self) -> np.ndarray:
        if self.i not in self._cache:
            img = cv2.imread(str(self.files[self.i]))
            if img is None:
                raise RuntimeError(f"Cannot read {self.files[self.i]}")
            self._cache[self.i] = img
        return self._cache[self.i].copy()

    def info(self) -> dict:
        return {"kind": self.kind, "file": str(self.files[self.i]), "index": self.i, "count": len(self.files)}


def open_source(spec: str) -> FrameSource:
    spec = str(spec).strip()
    if spec == "demo":
        return DemoSource()
    if spec.isdigit():
        return WebcamSource(int(spec))
    p = Path(spec)
    if p.is_dir() or p.suffix.lower() in (".jpg", ".jpeg", ".png"):
        return ImageSource(spec)
    return VideoSource(spec)
