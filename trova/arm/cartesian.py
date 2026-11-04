"""Cartesian cobots (UR3e/UR5e, xArm5, Elite CS63/CS66): move the tool tip straight to (x, y, z).

All five event arms accept a Cartesian target in their base frame, so no taught grid is needed:
calibration pairs camera pixels directly with the arm's own TCP readings (move the tip onto a
point by hand in freedrive, click it in the live view), giving pixel -> robot millimetres.

Safety, in this order:
- every target is clamped against a workspace box (TROVA_ARM_LIMITS) and refused outside it;
- the tool hovers at TROVA_ARM_Z_MM above the table plane, never touching it;
- speed is capped (TROVA_ARM_SPEED_MM_S, default 100 mm/s);
- stop() commands a controlled stop and holds; the staff e-stop stays the primary stop.
Each robot family only implements CartesianBackend (connect, move, stop, tcp, freedrive).
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .base import ArmDriver


class CartesianBackend(ABC):
    name = "cartesian"

    @abstractmethod
    def move_linear(self, x: float, y: float, z: float, speed_mm_s: float) -> None:
        """Start a straight-line move of the TCP to (x, y, z) mm, tool pointing down. Must not block long."""

    @abstractmethod
    def is_moving(self) -> bool: ...

    @abstractmethod
    def stop(self) -> None:
        """Controlled stop, then hold position (not a power-off e-stop)."""

    @abstractmethod
    def tcp_mm(self) -> tuple[float, float, float]: ...

    def freedrive(self, on: bool) -> None:
        raise NotImplementedError(f"{self.name}: freedrive not supported, jog from the pendant instead")

    def clear_errors(self) -> None:
        pass

    def close(self) -> None:
        pass


@dataclass
class Limits:
    x: tuple[float, float]
    y: tuple[float, float]
    z: tuple[float, float]

    @classmethod
    def parse(cls, s: str | None) -> "Limits | None":
        if not s:
            return None
        v = [float(p) for p in s.split(",")]
        if len(v) != 6:
            raise ValueError("TROVA_ARM_LIMITS must be xmin,xmax,ymin,ymax,zmin,zmax (mm, robot base frame)")
        return cls((v[0], v[1]), (v[2], v[3]), (v[4], v[5]))

    def check(self, x: float, y: float, z: float) -> str | None:
        for name, val, (lo, hi) in (("x", x, self.x), ("y", y, self.y), ("z", z, self.z)):
            if not lo <= val <= hi:
                return f"{name}={val:.0f} mm is outside the safe box [{lo:.0f}, {hi:.0f}]"
        return None


class CartesianArm(ArmDriver):
    name = "cartesian"

    def __init__(self, backend: CartesianBackend, z_hover_mm: float, home_mm: tuple[float, float, float] | None,
                 limits: Limits | None, speed_mm_s: float = 100.0, timeout_s: float = 20.0):
        self.backend = backend
        self.name = backend.name
        self.z = z_hover_mm
        self.home_mm = home_mm
        self.limits = limits
        self.speed = speed_mm_s
        self.timeout = timeout_s
        self.state = "idle"
        self.last_error: str | None = None
        self.step: str | None = None
        self._stop = threading.Event()

    def _go(self, x: float, y: float, z: float) -> bool:
        if self.limits:
            err = self.limits.check(x, y, z)
            if err:
                self.state, self.last_error = "refused", err
                return False
        self._stop.clear()
        self.state, self.last_error = "moving", None
        try:
            self.backend.move_linear(x, y, z, self.speed)
            t0 = time.time()
            time.sleep(0.05)
            while self.backend.is_moving():
                if self._stop.is_set():
                    self.state = "stopped"
                    return False
                if time.time() - t0 > self.timeout:
                    self.backend.stop()
                    self.state, self.last_error = "error", "move timed out"
                    return False
                time.sleep(0.02)
        except Exception as e:  # noqa: BLE001 - surface any SDK error in the UI, never crash the server
            self.state, self.last_error = "error", f"{type(e).__name__}: {e}"
            return False
        if self._stop.is_set():
            self.state = "stopped"
            return False
        try:   # a robot in local mode or in a protective stop ignores the move silently: say so
            tx, ty, tz = self.backend.tcp_mm()
            off = ((tx - x) ** 2 + (ty - y) ** 2 + (tz - z) ** 2) ** 0.5
        except Exception:  # noqa: BLE001
            off = 0.0
        if off > 10.0:
            self.state, self.last_error = "error", (f"stopped {off:.0f} mm from the target: is the robot in remote mode, "
                                                    "or in a protective stop?")
            return False
        self.state = "idle"
        return True

    def point_at(self, x_mm: float, y_mm: float) -> bool:
        return self._go(x_mm, y_mm, self.z)

    def run_waypoints(self, wps: list[tuple], speed_factor: float = 0.5) -> bool:
        """Feature 7: a fixed sequence of straight moves (label, x, y, z). Stops at the first refusal or stop.
        All waypoints are checked against the safety box BEFORE the first move."""
        if self.limits:
            for label, x, y, z in wps:
                err = self.limits.check(x, y, z)
                if err:
                    self.state, self.last_error = "refused", f"{label}: {err}"
                    return False
        speed, self.speed = self.speed, self.speed * speed_factor
        try:
            for label, x, y, z in wps:
                self.step = label
                if not self._go(x, y, z):
                    return False
            return True
        finally:
            self.speed, self.step = speed, None

    def home(self) -> None:
        if self.home_mm:
            self._go(*self.home_mm)

    def stop(self) -> None:
        self._stop.set()
        try:
            self.backend.stop()
        finally:
            self.state = "stopped"

    def tcp_mm(self) -> tuple[float, float, float]:
        return self.backend.tcp_mm()

    def freedrive(self, on: bool) -> None:
        self.backend.freedrive(on)

    def status(self) -> dict:
        try:
            tcp = [round(v, 1) for v in self.backend.tcp_mm()]
        except Exception:  # noqa: BLE001
            tcp = None
        return {"name": self.name, "state": self.state, "tip_mm": tcp[:2] if tcp else None, "tcp_mm": tcp,
                "z_hover_mm": self.z, "error": self.last_error, "step": self.step, "simulated": isinstance(self.backend, FakeCartesian)}


class FakeCartesian(CartesianBackend):
    """Simulated Cartesian robot for tests and rehearsal: moves at the commanded speed, can be stopped."""
    name = "fake-cartesian"

    def __init__(self, start=(300.0, 0.0, 200.0)):
        self.pos = list(start)
        self.target = list(start)
        self.speed = 100.0
        self.free = False
        self.history: list[tuple] = []
        self._t = time.time()
        self._lock = threading.Lock()

    def _advance(self):
        now = time.time()
        dt, self._t = now - self._t, now
        d = [t - p for t, p in zip(self.target, self.pos)]
        dist = sum(v * v for v in d) ** 0.5
        step = self.speed * dt
        if dist <= step:
            self.pos = list(self.target)
        else:
            self.pos = [p + v / dist * step for p, v in zip(self.pos, d)]

    def move_linear(self, x, y, z, speed_mm_s):
        with self._lock:
            self._advance()
            self.target, self.speed = [x, y, z], speed_mm_s
            self.history.append((x, y, z))

    def is_moving(self):
        with self._lock:
            self._advance()
            return self.pos != self.target

    def stop(self):
        with self._lock:
            self._advance()
            self.target = list(self.pos)

    def tcp_mm(self):
        with self._lock:
            self._advance()
            return tuple(self.pos)

    def freedrive(self, on):
        self.free = on
