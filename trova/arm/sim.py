"""A simulated arm: the tip moves across the table at a realistic speed and can be stopped.
The live view draws it as a crosshair, so the whole flow works with no hardware."""

from __future__ import annotations

import math
import threading
import time

from .base import ArmDriver


class SimArm(ArmDriver):
    name = "sim"

    def __init__(self, table_w_mm: float = 600, table_h_mm: float = 400, speed_mm_s: float = 250):
        self.home_xy = (table_w_mm / 2, table_h_mm + 60)   # parked beyond the near edge
        self.tip = list(self.home_xy)
        self.speed = speed_mm_s
        self.state = "idle"
        self.target: tuple[float, float] | None = None
        self.step: str | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

    def _move(self, x: float, y: float) -> bool:
        if self._stop.is_set():          # STOP latched: refuse until the server calls ready()
            with self._lock:
                self.state = "stopped"
            return False
        with self._lock:
            self.state, self.target = "moving", (x, y)
        while True:
            if self._stop.is_set():
                with self._lock:
                    self.state = "stopped"
                return False
            dx, dy = x - self.tip[0], y - self.tip[1]
            d = math.hypot(dx, dy)
            step = self.speed * 0.02
            with self._lock:
                if d <= step:
                    self.tip = [x, y]
                    self.state = "idle"
                    return True
                self.tip = [self.tip[0] + dx / d * step, self.tip[1] + dy / d * step]
            time.sleep(0.02)

    def home(self) -> None:
        self._move(*self.home_xy)

    def point_at(self, x_mm: float, y_mm: float) -> bool:
        return self._move(x_mm, y_mm)

    def run_waypoints(self, wps: list[tuple], speed_factor: float = 0.5) -> bool:
        """Feature 7 in the simulator: the tip follows the XY path (height is not drawn)."""
        for label, x, y, _z in wps:
            self.step = label
            if not self._move(x, y):
                self.step = None
                return False
        self.step = None
        return True

    def stop(self) -> None:
        self._stop.set()

    def status(self) -> dict:
        with self._lock:
            return {"name": self.name, "state": self.state, "tip_mm": [round(v, 1) for v in self.tip],
                    "target_mm": list(self.target) if self.target else None, "simulated": True, "step": self.step}
