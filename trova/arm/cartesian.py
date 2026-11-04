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

START_GRACE_S = 0.5   # a controller may report 'not moving' for a moment after accepting a move


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

    def move_joints(self, q: list[float], speed_deg_s: float) -> None:
        """Start a joint-space move to the taught angles (degrees). Optional: needed for taught poses."""
        raise NotImplementedError(f"{self.name}: joint moves not supported, use TROVA_ARM_HOME_MM")

    def joints(self) -> list[float]:
        raise NotImplementedError(f"{self.name}: joint angles not available")

    def fk_mm(self, q: list[float]) -> tuple[float, float, float]:
        """Where the tool would be at these joint angles: lets the safety box check joint moves too."""
        raise NotImplementedError

    def is_level(self) -> bool:
        """Can straight-line moves start from here? Only the 5-axis xArm needs a level flange (manual p. 156)."""
        return True

    def level_joints(self, q: list[float]) -> list[float]:
        """The same pose with the flange levelled in place (first supervised run, stage 0)."""
        raise NotImplementedError(f"{self.name}: no flange levelling")

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
                 limits: Limits | None, speed_mm_s: float = 100.0, timeout_s: float = 20.0,
                 home_joints: list[float] | None = None, joint_speed_deg_s: float = 20.0):
        self.backend = backend
        self.name = backend.name
        self.z = z_hover_mm
        self.home_mm = home_mm
        self.home_joints = home_joints      # a taught joint pose wins over home_mm (camera on the arm: the survey pose)
        self.joint_speed = joint_speed_deg_s
        self.limits = limits
        self.speed = speed_mm_s
        self.timeout = timeout_s
        self.state = "idle"
        self.last_error: str | None = None
        self.step: str | None = None
        self._stop = threading.Event()

    def _halted(self) -> bool:
        """A STOP arrived: stop again (it may have raced a move still being sent, which would replace it)."""
        try:
            self.backend.stop()
        except Exception as e:  # a raced move needs a successful resend; never hide its failure
            self.state, self.last_error = "error", f"STOP resend failed: {e}"
            return False
        self.state = "stopped"
        return False

    def _go(self, x: float, y: float, z: float) -> bool:
        if self._stop.is_set():          # STOP latched: refuse until the server calls ready()
            self.state = "stopped"
            return False
        if self.limits:
            err = self.limits.check(x, y, z)
            if err:
                self.state, self.last_error = "refused", err
                return False
        self.state, self.last_error = "moving", None
        try:
            self.backend.move_linear(x, y, z, self.speed)
            if self._stop.is_set():
                return self._halted()
            t0 = time.time()
            seen = False
            while True:
                if self._stop.is_set():
                    return self._halted()
                moving = self.backend.is_moving()
                seen = seen or moving
                if not moving and (seen or time.time() - t0 > START_GRACE_S):   # some arms start late
                    break
                if time.time() - t0 > self.timeout:
                    self.backend.stop()
                    self.state, self.last_error = "error", "move timed out"
                    return False
                time.sleep(0.02)
        except Exception as e:  # noqa: BLE001 - surface any SDK error in the UI, never crash the server
            self.state, self.last_error = "error", f"{type(e).__name__}: {e}"
            return False
        if self._stop.is_set():
            return self._halted()
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

    def approach(self, x_mm: float, y_mm: float, z_mm: float, speed_mm_s: float) -> bool:
        """First supervised run: one straight move to (x, y, z) at most at speed_mm_s; same box, STOP and reached check."""
        speed, self.speed = self.speed, min(self.speed, speed_mm_s)
        try:
            return self._go(x_mm, y_mm, z_mm)
        finally:
            self.speed = speed

    def run_waypoints(self, wps: list[tuple], speed_factor: float = 0.5) -> bool:
        """Feature 7: a fixed sequence of straight moves (label, x, y, z). Stops at the first refusal or stop.
        All waypoints are checked against the safety box BEFORE the first move."""
        if self.limits:
            for label, x, y, z in wps:
                err = self.limits.check(x, y, z)
                if err:
                    self.state, self.last_error = "refused", f"{label}: {err}"
                    return False
        speed, self.speed = self.speed, self.speed * speed_factor   # STOP stays latched across waypoints
        try:
            for label, x, y, z in wps:
                self.step = label
                if not self._go(x, y, z):
                    return False
            return True
        finally:
            self.speed, self.step = speed, None

    def home(self) -> None:
        if self.home_joints:
            self.go_joints(self.home_joints)
        elif self.home_mm:
            self._go(*self.home_mm)

    def go_joints(self, q: list[float], tol_deg: float = 1.0) -> bool:
        """Joint-space move to taught angles (degrees), with the same STOP latch, timeout and reached check as _go."""
        q = [float(v) for v in q]
        if self._stop.is_set():
            self.state = "stopped"
            return False
        if self.limits is None:
            self.state, self.last_error = "refused", "No safety box: cannot validate the joint target"
            return False
        if self.limits:
            try:
                err = self.limits.check(*self.backend.fk_mm(q))
            except NotImplementedError:
                err = "forward kinematics unavailable: cannot validate the joint target against the safety box"
            except Exception as e:  # noqa: BLE001
                err = f"forward kinematics failed: {e}"
            if err:
                self.state, self.last_error = "refused", f"joint pose: {err}"
                return False
        self.state, self.last_error = "moving", None
        try:
            self.backend.move_joints(q, self.joint_speed)
            if self._stop.is_set():
                return self._halted()
            t0, seen = time.time(), False
            while True:
                if self._stop.is_set():
                    return self._halted()
                moving = self.backend.is_moving()
                seen = seen or moving
                if not moving and (seen or time.time() - t0 > START_GRACE_S):
                    break
                if time.time() - t0 > self.timeout:
                    self.backend.stop()
                    self.state, self.last_error = "error", "joint move timed out"
                    return False
                time.sleep(0.02)
            got = self.backend.joints()
        except Exception as e:  # noqa: BLE001
            self.state, self.last_error = "error", f"{type(e).__name__}: {e}"
            return False
        if self._stop.is_set():
            return self._halted()
        off = max(abs(a - b) for a, b in zip(got, q))
        if off > tol_deg:
            self.state, self.last_error = "error", f"stopped {off:.1f} deg from the taught pose"
            return False
        self.state = "idle"
        return True

    def at_home(self, tol_mm: float = 3.0, tol_deg: float = 1.0) -> bool:
        """Is the arm parked at home right now (for a camera on the arm: is this the calibrated view)?"""
        if self.state == "moving":
            return False
        try:
            if self.home_joints:
                return bool(max(abs(a - b) for a, b in zip(self.backend.joints(), self.home_joints)) <= tol_deg)
            if self.home_mm:
                return bool(sum((a - b) ** 2 for a, b in zip(self.backend.tcp_mm(), self.home_mm)) ** 0.5 <= tol_mm)
        except Exception:  # noqa: BLE001
            return False
        return False

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
                "z_hover_mm": self.z, "error": self.last_error, "step": self.step, "simulated": isinstance(self.backend, FakeCartesian),
                "home": "joints" if self.home_joints else ("mm" if self.home_mm else None)}


class FakeCartesian(CartesianBackend):
    """Simulated Cartesian robot for tests and rehearsal: moves at the commanded speed, can be stopped."""
    name = "fake-cartesian"

    def __init__(self, start=(300.0, 0.0, 200.0)):
        self.pos = list(start)
        self.target = list(start)
        self.speed = 100.0
        self.free = False
        self.history: list[tuple] = []
        self.q = self.ik(self.pos)          # joint angles (toy model: see fk)
        self.q_target: list[float] | None = None
        self._t = time.time()
        self._lock = threading.Lock()

    @staticmethod
    def fk(q):            # toy forward model, only for rehearsal and tests: 1 deg of joint = 1 mm
        return [300.0 + q[1], q[0], 200.0 + q[2]]

    @staticmethod
    def ik(p):
        return [p[1], p[0] - 300.0, p[2] - 200.0, 0.0, 0.0]

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
        if self.q_target is not None and self.pos == self.target:
            self.q, self.q_target = list(self.q_target), None
        elif self.q_target is None:
            self.q = self.ik(self.pos)[:3] + list(self.q[3:])

    def move_linear(self, x, y, z, speed_mm_s):
        with self._lock:
            self._advance()
            self.target, self.speed, self.q_target = [x, y, z], speed_mm_s, None
            self.history.append((x, y, z))

    def move_joints(self, q, speed_deg_s):
        with self._lock:
            self._advance()
            self.target, self.speed, self.q_target = self.fk(q), speed_deg_s, list(q)
            self.history.append(("joints", *q))

    def joints(self):
        with self._lock:
            self._advance()
            return list(self.q)

    def fk_mm(self, q):
        return tuple(self.fk(q))

    def is_level(self):     # toy model: level when J4 = -(J2 + J3), as on the xArm 5
        q = self.joints()
        return abs(q[3] + q[1] + q[2]) < 0.5

    def level_joints(self, q):
        return [q[0], q[1], q[2], -(q[1] + q[2]), *q[4:]]

    def is_moving(self):
        with self._lock:
            self._advance()
            return self.pos != self.target

    def stop(self):
        with self._lock:
            self._advance()
            self.target, self.q_target = list(self.pos), None

    def tcp_mm(self):
        with self._lock:
            self._advance()
            return tuple(self.pos)

    def freedrive(self, on):
        self.free = on
