"""Frame sources. Pick one with TROVA_CAMERA, or switch live from the UI.

  demo                 synthetic table (offline demo; supports "tamper")
  0, 1, 2 ...          webcam index. An iPhone via Continuity Camera shows up as one of these.
  http://.../stream.mjpg  a network camera (the arm's Pi: http://trova-cam.local:8080/stream.mjpg), read like a
                       webcam: newest frame only, refused when older than 2 s
                       Turn OFF Center Stage / Portrait / Studio Light (macOS Control Centre ->
                       Video Effects): they reframe the image and break calibration.
  path/to/video.mp4    a recording of the station, looped (work without the arm)
  path/to/image.jpg    one still
  path/to/folder/      every jpg/png in the folder; step with next/prev
An uploaded photo (e.g. a phone shot of the table) overrides the source until cleared.
"""

from __future__ import annotations

import os
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


STALE_S = 2.0   # a webcam frame older than this is refused
STREAM_TIMEOUT_MS = 1500   # network streams: open/read give up before close() stops waiting for the reader


class WebcamSource(FrameSource):
    """Only the reader thread touches the capture after start-up, and it releases it on its way out: releasing it
    under a blocked cap.read() crashes on macOS."""
    kind = "webcam"
    JOIN_S = 2.0            # close(): how long to wait for the reader
    FIRST_FRAME_S = 5.0     # start-up: how long to wait for the first frame

    def __init__(self, index: int | str, width: int = 1920, height: int = 1080):
        self.index, self.width, self.height = index, width, height
        self.cap = self._open()
        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError(f"Cannot open camera {index}")
        self._frame: np.ndarray | None = None
        self._ts = 0.0                    # when the newest frame arrived
        self._size: list[int] | None = None
        self.size_changed: str | None = None
        self._lock = threading.Lock()
        self._run = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        t0 = time.time()
        while self._frame is None and time.time() - t0 < self.FIRST_FRAME_S:
            time.sleep(0.05)
        if self._frame is None:
            self.close()                  # never leave a reader reconnecting forever
            raise RuntimeError(f"Camera {index} opened but sent no frames")

    def _open(self):
        if isinstance(self.index, str) and "://" in self.index:
            cap = cv2.VideoCapture(self.index, cv2.CAP_FFMPEG, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, STREAM_TIMEOUT_MS,
                                                                cv2.CAP_PROP_READ_TIMEOUT_MSEC, STREAM_TIMEOUT_MS])
        else:
            cap = cv2.VideoCapture(self.index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)      # also after every reopen: calibration needs the same size
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return cap

    def _reopen(self) -> None:
        self.cap.release()
        time.sleep(0.5)
        self.cap = self._open()

    def _loop(self):
        try:
            opened = time.time()
            while self._run:   # always keep only the newest frame, so finds never use a stale one
                ok, f = self.cap.read()
                if ok:
                    size = [f.shape[1], f.shape[0]]
                    if self._size and size != self._size:
                        self.size_changed = f"{self._size[0]}x{self._size[1]} -> {size[0]}x{size[1]} at {time.strftime('%H:%M:%S')}"
                        print(f"Camera {self.index}: frame size changed {self.size_changed}", flush=True)
                    self._size = size
                    with self._lock:
                        self._frame, self._ts = f, time.time()
                    continue
                time.sleep(0.02)
                if self._run and time.time() - max(self._ts, opened) >= STALE_S:   # dropped (Continuity, Wi-Fi)
                    self._reopen()
                    opened = time.time()
        finally:
            self.cap.release()

    def read(self) -> np.ndarray:
        with self._lock:
            age = time.time() - self._ts
            if age > STALE_S:      # never run a find or a line-clearance check on an old picture
                raise RuntimeError(f"Camera {self.index} sent no new frame for {age:.0f} s: check the cable or phone")
            return self._frame.copy()

    def info(self) -> dict:
        f = self._frame
        return {"kind": self.kind, "index": self.index, "size": None if f is None else [f.shape[1], f.shape[0]],
                "requested": [self.width, self.height], "size_changed": self.size_changed,
                "frame_age_s": round(time.time() - self._ts, 1)}

    def close(self) -> None:
        self._run = False
        self._thread.join(timeout=self.JOIN_S)   # the reader releases the capture itself once it leaves cap.read()


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
        self._lock = threading.Lock()     # several request threads read at once; seek + read must not interleave

    def read(self) -> np.ndarray:
        # play in real time and loop
        with self._lock:
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


class ArmCameraSim(DemoSource):
    """The camera on the arm, simulated (TROVA_CAMERA=armsim, with TROVA_ARM=cartesian:fake): what the arm would see.
    At the survey pose: the table. At the look pose (any pan angle): a slice of a room panorama. Elsewhere: a blurred,
    shifted 'moving' view, so code that forgets to go back to the survey pose fails in rehearsal, not in the demo."""
    kind = "armsim"
    HFOV = 70.0

    def __init__(self, seed: int = 7):
        super().__init__(seed)
        self.arm, self.data_dir = None, None
        self.w, self.h = 1280, 800
        self.px_per_deg = self.w / self.HFOV
        self.room = SyntheticTable(width=int(self.w + 2 * 160 * self.px_per_deg), height=self.h, n_objects=26, seed=seed + 100)
        self.room_img = self.room.render(noise=0)

    def attach(self, arm, data_dir=None) -> None:
        self.arm, self.data_dir = arm, data_dir

    def _pose(self) -> str | float:
        from .arm import CartesianArm, load_poses
        if not isinstance(self.arm, CartesianArm):
            return "survey"
        if self.arm.at_home():
            return "survey"
        look = load_poses(self.data_dir).get("look_joints") if self.data_dir else None
        try:
            q = self.arm.backend.joints()
        except Exception:  # noqa: BLE001
            return "moving"
        if look:
            pan = int(os.getenv("TROVA_PAN_JOINT", "1")) - 1
            if all(abs(a - b) < 1.0 for i, (a, b) in enumerate(zip(q, look)) if i != pan):
                return q[pan] - look[pan]
        return "moving"

    def read(self) -> np.ndarray:
        where = self._pose()
        if where == "survey":
            return super().read()
        if where == "moving":
            return cv2.GaussianBlur(np.roll(super().read(), 200, axis=1), (31, 31), 0)
        cx = self.room_img.shape[1] / 2 - where * self.px_per_deg     # pan +: the view turns left, the scene moves right
        x0 = int(round(cx - self.w / 2))
        x0 = max(0, min(self.room_img.shape[1] - self.w, x0))
        view = self.room_img[:, x0:x0 + self.w].astype(np.float32)
        return np.clip(view + np.random.normal(0, 2.5, view.shape), 0, 255).astype(np.uint8)


def open_source(spec: str) -> FrameSource:
    spec = str(spec).strip()
    if spec == "demo":
        return DemoSource()
    if spec == "armsim":
        return ArmCameraSim()
    if spec.isdigit():
        return WebcamSource(int(spec))
    if "://" in spec:          # a live stream must never fall back to the video-file reader, which repeats old frames
        return WebcamSource(spec)
    p = Path(spec)
    if p.is_dir() or p.suffix.lower() in (".jpg", ".jpeg", ".png"):
        return ImageSource(spec)
    return VideoSource(spec)


def list_cameras(out_dir: Path = Path("data"), max_index: int = 6) -> list[dict]:
    """Try camera indices 0..max_index-1: resolution, mean brightness and a snapshot (data/camera_<i>.jpg), so the
    team can tell which index is the camera on the arm, the Mac's own camera, or an iPhone (Continuity Camera)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        f = None
        for _ in range(15):                     # the first frames are often black while exposure settles
            ok, g = cap.read()
            if ok:
                f = g
            time.sleep(0.05)
        cap.release()
        if f is None:
            continue
        p = out_dir / f"camera_{i}.jpg"
        cv2.imwrite(str(p), f)
        found.append({"index": i, "size": [f.shape[1], f.shape[0]], "brightness": round(float(f.mean()), 1), "snapshot": str(p)})
    return found


def main() -> None:
    import subprocess
    import sys
    if "--list" not in sys.argv:
        sys.exit("usage: python -m trova.camera --list      (then TROVA_CAMERA=<index> in .env)")
    if sys.platform == "darwin":
        try:
            names = subprocess.run(["system_profiler", "SPCameraDataType"], capture_output=True, text=True, timeout=15).stdout
            print("macOS cameras (usually in index order):")
            print("\n".join("  " + n.strip().rstrip(":") for n in names.splitlines() if n.startswith("    ") and n.strip().endswith(":")))
        except (OSError, subprocess.SubprocessError):
            pass
    for c in list_cameras():
        print(f"index {c['index']}: {c['size'][0]}x{c['size'][1]}, brightness {c['brightness']}, snapshot {c['snapshot']}")
    print("Open the snapshots: the arm camera shows the table (or the room) from the sphere. Put its index in TROVA_CAMERA.")


if __name__ == "__main__":
    main()
