"""Simulated xArm 5 for rehearsal: the real xArm 5 forward kinematics (scan3d.flange), joint-space motion only.

TROVA_ARM=xarm5sim. It subclasses FakeCartesian, so the motion gateway treats it as simulated: hardware mode is
refused and no command ever reaches a controller. Joint moves take real time at the commanded speed (scaled by
TROVA_SIM_FAST), so STOP, timeouts and the UI see a moving arm. Cartesian line moves are not simulated: the table
scan, the camera-over-object approach and a taught home_joints pose all move in joint space.

Start pose: TROVA_SIM_START_JOINTS (degrees J1..J5), default all zero (the xArm's zero pose, TCP 207, 0, 112 mm).
"""

from __future__ import annotations

import math
import os
import threading
import time

import numpy as np

from ..scan3d.flange import flange_pose
from .cartesian import FakeCartesian


def _rpy_deg(r: np.ndarray) -> tuple[float, float, float]:
    """Roll, pitch, yaw (degrees, xArm convention R = Rz(yaw) Ry(pitch) Rx(roll))."""
    pitch = math.asin(max(-1.0, min(1.0, -float(r[2, 0]))))
    roll = math.atan2(float(r[2, 1]), float(r[2, 2]))
    yaw = math.atan2(float(r[1, 0]), float(r[0, 0]))
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


class Xarm5Sim(FakeCartesian):
    name = "xarm5-sim"

    def __init__(self, address: str = ""):
        raw = os.getenv("TROVA_SIM_START_JOINTS", "").strip()
        start = [float(v) for v in raw.split(",")] if raw else [0.0] * 5
        if len(start) != 5:
            raise ValueError("TROVA_SIM_START_JOINTS needs five joint angles J1..J5 in degrees")
        self._lock = threading.Lock()
        self.fast = max(1.0, float(os.getenv("TROVA_SIM_FAST", "1")))
        self.q = list(start)
        self.q_from, self.q_goal = list(start), list(start)
        self.t0, self.duration = time.time(), 0.0
        self.free = False
        self.history: list[tuple] = []

    # ------------------------------------------------------------ motion
    def _current(self) -> list[float]:
        if self.duration <= 0:
            return list(self.q_goal)
        s = min(1.0, (time.time() - self.t0) * self.fast / self.duration)
        return [a + (b - a) * s for a, b in zip(self.q_from, self.q_goal)]

    def move_joints(self, q, speed_deg_s):
        q = [float(v) for v in q]
        if len(q) != 5:
            raise ValueError("the xArm 5 has five joints")
        with self._lock:
            now = self._current()
            travel = max(abs(a - b) for a, b in zip(now, q))
            self.q_from, self.q_goal = now, q
            self.t0, self.duration = time.time(), travel / max(0.1, float(speed_deg_s))
            self.history.append(("joints", *q))

    def move_linear(self, x, y, z, speed_mm_s):
        raise NotImplementedError("xarm5sim moves in joint space only (table scan, camera-over-object, home_joints)")

    def is_moving(self):
        with self._lock:
            return self.duration > 0 and (time.time() - self.t0) * self.fast < self.duration

    def stop(self):
        with self._lock:
            now = self._current()
            self.q_from = self.q_goal = now
            self.duration = 0.0

    def joints(self):
        with self._lock:
            return self._current()

    # ------------------------------------------------------------ kinematics (same as the controller's)
    def fk_mm(self, q):
        return tuple(float(v) for v in flange_pose(q)[:3, 3])

    def tcp_mm(self):
        return self.fk_mm(self.joints())

    def tcp_pose(self):
        t = flange_pose(self.joints())
        return (*[float(v) for v in t[:3, 3]], *_rpy_deg(t[:3, :3]))

    def is_level(self):
        _x, _y, _z, roll, pitch, _yaw = self.tcp_pose()
        return abs(abs(roll) - 180) < 3 and abs(pitch) < 3

    def level_joints(self, q):
        raise NotImplementedError("xarm5sim: no flange levelling (first-run stage 0 uses the real controller FK)")

    def freedrive(self, on):
        self.free = on
