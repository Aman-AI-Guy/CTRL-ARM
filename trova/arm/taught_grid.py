"""Point at any table position with ANY arm, without inverse kinematics.

Teach once per slot: put the arm in compliant mode (torque off, staff present), move the tip to
just above a known table point (e.g. a marker or a grid crossing, about 2 cm above the surface),
and press "Record pose". 9-16 poses spread over the table are enough. To point at (x, y) we
blend the joint angles of the nearest taught poses (inverse-distance weighting).

The real arm only has to implement JointBackend (4 methods). See backend_template.py.
"""

from __future__ import annotations

import json
import math
import threading
from abc import ABC, abstractmethod
from pathlib import Path

from .base import ArmDriver
from .cartesian import Limits


class JointBackend(ABC):
    @abstractmethod
    def read_joints(self) -> list[float]: ...

    @abstractmethod
    def move_joints(self, q: list[float], duration_s: float, stop_event: threading.Event) -> bool:
        """Move to q over about duration_s; return False early if stop_event gets set."""

    def fk_mm(self, q: list[float]) -> tuple[float, float, float]:
        raise NotImplementedError("grid backend needs forward kinematics to validate joint targets")

    @abstractmethod
    def hold(self) -> None:
        """Stop and hold the current position (torque stays on)."""

    @abstractmethod
    def set_compliant(self, compliant: bool) -> None:
        """True = torque off so a person can move the arm by hand for teaching."""


class TaughtGridArm(ArmDriver):
    name = "taught_grid"

    def __init__(self, backend: JointBackend, path: Path, k: int = 4, duration_s: float = 1.5,
                 limits: Limits | None = None, z_mm: float = 150.0):
        self.backend = backend
        self.limits, self.z, self.last_error = limits, z_mm, None
        self.path = Path(path)
        self.k = k
        self.duration_s = duration_s
        self.samples: list[dict] = []   # {"x":, "y":, "q": [...]}
        self.home_q: list[float] | None = None
        self.state = "idle"
        self.tip: list[float] | None = None
        self._stop = threading.Event()
        self.load()

    # ----- teaching
    def record(self, x_mm: float, y_mm: float) -> dict:
        s = {"x": float(x_mm), "y": float(y_mm), "q": [float(v) for v in self.backend.read_joints()]}
        self.samples.append(s)
        self.save()
        return s

    def record_home(self) -> list[float]:
        self.home_q = [float(v) for v in self.backend.read_joints()]
        self.save()
        return self.home_q

    def save(self) -> None:
        self.path.write_text(json.dumps({"home_q": self.home_q, "samples": self.samples}, indent=2))

    def load(self) -> None:
        if self.path.exists():
            d = json.loads(self.path.read_text())
            self.home_q, self.samples = d.get("home_q"), d.get("samples", [])

    # ----- motion
    def joints_for(self, x: float, y: float) -> list[float]:
        if not all(math.isfinite(v) for v in (x, y)):
            raise ValueError("Grid coordinates must be finite")
        if not self.samples:
            raise RuntimeError("No taught poses yet: record at least 4 across the table")
        near = sorted(self.samples, key=lambda s: math.hypot(s["x"] - x, s["y"] - y))[: self.k]
        d0 = math.hypot(near[0]["x"] - x, near[0]["y"] - y)
        if d0 < 1.0:
            return list(near[0]["q"])
        w = [1.0 / math.hypot(s["x"] - x, s["y"] - y) ** 2 for s in near]
        n = len(near[0]["q"])
        return [sum(wi * s["q"][j] for wi, s in zip(w, near)) / sum(w) for j in range(n)]

    def _go(self, q: list[float], tip) -> bool:
        if self._stop.is_set():          # STOP latched until the server calls ready()
            self.state = "stopped"
            return False
        try:
            if not q or not all(math.isfinite(v) for v in q):
                raise ValueError("Grid joints must be finite")
            if self.limits is None:
                raise ValueError("No safety box configured for taught grid")
            if type(self.backend) is not FakeJointBackend:
                err = self.limits.check(*self.backend.fk_mm(q))
                if err:
                    raise ValueError(err)
        except Exception as e:  # FK or controller errors must refuse dispatch
            self.state, self.last_error = "refused", str(e)
            return False
        self.state = "moving"
        ok = self.backend.move_joints(q, self.duration_s, self._stop)
        self.state = "idle" if ok else "stopped"
        if ok:
            self.tip = tip
        return ok

    def home(self) -> None:
        if self.home_q is not None:
            self._go(self.home_q, None)

    def point_at(self, x_mm: float, y_mm: float) -> bool:
        err = self.limits.check(x_mm, y_mm, self.z) if self.limits else "No safety box configured for taught grid"
        if err:
            self.state, self.last_error = "refused", err
            return False
        return self._go(self.joints_for(x_mm, y_mm), [x_mm, y_mm])

    def stop(self) -> None:
        self._stop.set()
        self.backend.hold()
        self.state = "stopped"

    def status(self) -> dict:
        return {"name": self.name, "state": self.state, "tip_mm": self.tip, "taught_poses": len(self.samples),
                "home_taught": self.home_q is not None, "simulated": False, "error": self.last_error}


class FakeJointBackend(JointBackend):
    """For tests and for rehearsing the teaching flow without hardware: joints = [x, y]."""

    def __init__(self):
        self.q = [0.0, 0.0]

    def read_joints(self):
        return list(self.q)

    def move_joints(self, q, duration_s, stop_event):
        if stop_event.is_set():
            return False
        self.q = list(q)
        return True

    def hold(self):
        pass

    def set_compliant(self, compliant):
        pass
