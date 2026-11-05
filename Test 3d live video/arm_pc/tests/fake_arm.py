"""Enough of xarm.wrapper.XArmAPI for rehearsals and tests: no robot, moves are instant, kinematics are ours.

    from fake_arm import FakeArm
    arm = FakeArm(q=[0, -20, -60, 80, 0])
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import orbit_here as oh  # noqa: E402


def matrix_rpy(r: np.ndarray) -> tuple[float, float, float]:
    """Inverse of orbit_here.rpy_matrix (xArm convention R = Rz(yaw) Ry(pitch) Rx(roll)), degrees."""
    pitch = math.asin(max(-1.0, min(1.0, -r[2, 0])))
    return (math.degrees(math.atan2(r[2, 1], r[2, 2])), math.degrees(pitch), math.degrees(math.atan2(r[1, 0], r[0, 0])))


class FakeArm:
    def __init__(self, q=(0, -20, -60, 80, 0)):
        self.q = np.array(q, float)
        self.mode = self.state = 0
        self.error_code = self.warn_code = 0
        self.tcp_offset = [0, 0, 0, 0, 0, 0]
        self.version = "fake"
        self.moves: list[list[float]] = []

    def _pose(self, q):
        f = oh.fk(q)
        return [*f[:3, 3], *matrix_rpy(f[:3, :3])]

    def get_servo_angle(self, *a, **k):
        return 0, list(self.q) + [0.0, 0.0]

    def get_position(self, *a, **k):
        return 0, self._pose(self.q)

    def get_forward_kinematics(self, angles, *a, **k):
        return 0, self._pose(np.array(angles[:5], float))

    def set_servo_angle(self, angle=None, speed=None, mvacc=None, wait=False, timeout=None, **k):
        self.q = oh.clip(angle[:5])
        self.moves.append([float(v) for v in self.q])
        return 0

    def get_err_warn_code(self, *a, **k):
        return 0, [0, 0]

    def get_is_moving(self):
        return False

    def set_mode(self, m):
        self.mode = m
        return 0

    def set_state(self, s):
        self.state = s
        return 0

    def motion_enable(self, *a, **k):
        return 0

    def clean_error(self):
        return 0

    def clean_warn(self):
        return 0

    def disconnect(self):
        pass
