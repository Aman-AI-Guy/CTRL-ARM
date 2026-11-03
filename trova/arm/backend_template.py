"""TEMPLATE for the real arm. Owner: Hardware (P1). Fill this in on Wednesday morning.

We don't know the arm yet. The briefs mention LeRobot demos and a 3D-printed tool, so it may be
a LeRobot-style arm (e.g. SO-100/SO-101 with Feetech servos), but confirm with staff at 09:00.
Whatever it is, you only need these four methods; everything else (pointing anywhere on the
table, stop, UI) already works through TaughtGridArm.

Run with:  TROVA_ARM=grid:trova.arm.backend_template:MyArmBackend  python -m trova
"""

from __future__ import annotations

import threading
import time

from .taught_grid import JointBackend


class MyArmBackend(JointBackend):
    def __init__(self):
        # TODO: open the connection (serial port / SDK client). Example port on a Mac:
        #       /dev/tty.usbmodemXXXX. Ask staff for the calibration file the arm already uses.
        raise NotImplementedError("Fill in MyArmBackend for the event's arm (see docstring)")

    def read_joints(self) -> list[float]:
        # TODO: return current joint positions (any unit, as long as move_joints takes the same)
        raise NotImplementedError

    def move_joints(self, q: list[float], duration_s: float, stop_event: threading.Event) -> bool:
        # Interpolate in small steps so stop_event is checked often (every 20 ms).
        start = self.read_joints()
        steps = max(1, int(duration_s / 0.02))
        for i in range(1, steps + 1):
            if stop_event.is_set():
                self.hold()
                return False
            t = i / steps
            t = t * t * (3 - 2 * t)  # smoothstep: gentle start and stop
            target = [a + (b - a) * t for a, b in zip(start, q)]
            self._send(target)
            time.sleep(0.02)
        return True

    def hold(self) -> None:
        # TODO: command the CURRENT position as the goal, torque ON. Never torque off here.
        self._send(self.read_joints())

    def set_compliant(self, compliant: bool) -> None:
        # TODO: torque off (True) / on (False) for teaching by hand. Only with staff present.
        raise NotImplementedError

    def _send(self, q: list[float]) -> None:
        # TODO: write goal positions to the servos
        raise NotImplementedError
