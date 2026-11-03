"""The one interface every arm implements. The server only ever talks to this.

Drivers may block inside point_at()/home(): the server runs them in a worker thread, and
stop() is called from another thread, so stop() must interrupt a motion in progress.
Stop means HOLD the current position, never torque off (a torque-off arm drops).
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class ArmDriver(ABC):
    name = "base"

    @abstractmethod
    def home(self) -> None:
        """Move to the parked pose, out of the camera's view of the table."""

    @abstractmethod
    def point_at(self, x_mm: float, y_mm: float) -> bool:
        """Bring the tip just above (x, y) on the table. Returns True if it got there."""

    @abstractmethod
    def stop(self) -> None:
        """Halt now and hold position. Safe to call at any time, from any thread."""

    @abstractmethod
    def status(self) -> dict:
        """At least: {"name", "state": idle|moving|stopped|error, "tip_mm": [x, y] | None}."""
