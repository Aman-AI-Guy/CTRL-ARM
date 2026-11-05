"""Feature 7 (EXPERIMENTAL): lift an object's slotted tray with the printed fork and carry it.

The arms have no gripper: the printed tool has a fork, a pusher and a tip, and only slotted trays can be lifted.
A lift is a fixed sequence of straight moves (robot base frame, mm), each one checked by the safety box:

    P  pre-approach  = centre - a * (insert + standoff)   at z_hover, then down to z_slot
    S  slot entry    = centre - a * insert                 at z_slot
    C  under centre  = centre                              at z_slot      (fork fully in)
       lift                                                 to z_slot + lift
       carry to the place centre                            at z_carry
       set down                                             to z_slot
       retract along -a, then up                            to z_hover

`a` is the direction the fork travels into the slot (TROVA_TRAY_APPROACH, robot XY). OFF by default
(TROVA_ALLOW_LIFT=0): measure z_slot and the approach on the real tray with staff before enabling it.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass


@dataclass
class TrayParams:
    z_slot: float          # tool height when the fork enters the slot
    insert: float          # how far the fork slides in
    lift: float            # how high the tray is lifted
    standoff: float        # clearance before the slot
    approach: tuple[float, float]   # unit vector: direction of insertion in robot XY

    @classmethod
    def from_env(cls) -> "TrayParams":
        ax, ay = (float(v) for v in os.getenv("TROVA_TRAY_APPROACH", "0,1").split(","))
        n = math.hypot(ax, ay)
        if not math.isfinite(n) or n == 0:
            raise ValueError("TROVA_TRAY_APPROACH must be a finite non-zero vector")
        return cls(z_slot=float(os.getenv("TROVA_TRAY_Z_SLOT_MM", "30")),
                   insert=float(os.getenv("TROVA_TRAY_INSERT_MM", "40")),
                   lift=float(os.getenv("TROVA_TRAY_LIFT_MM", "40")),
                   standoff=float(os.getenv("TROVA_TRAY_STANDOFF_MM", "60")),
                   approach=(ax / n, ay / n))


def lift_allowed() -> bool:
    return os.getenv("TROVA_ALLOW_LIFT", "0").lower() in ("1", "true", "yes")


def waypoints(pick: tuple[float, float], place: tuple[float, float], z_hover: float, p: TrayParams) -> list[tuple]:
    """(label, x, y, z) for the whole lift-and-place sequence."""
    ax, ay = p.approach

    def back(c, d):
        return c[0] - ax * d, c[1] - ay * d

    z_up = p.z_slot + p.lift
    z_carry = max(z_hover, z_up + 30)
    pre, entry = back(pick, p.insert + p.standoff), back(pick, p.insert)
    pre2, entry2 = back(place, p.insert + p.standoff), back(place, p.insert)
    return [("above the approach", *pre, z_hover), ("down to slot height", *pre, p.z_slot),
            ("to the slot", *entry, p.z_slot), ("fork in", *pick, p.z_slot), ("lift", *pick, z_up),
            ("up to carry height", *pick, z_carry), ("carry", *place, z_carry), ("lower", *place, z_up),
            ("set down", *place, p.z_slot), ("fork out", *entry2, p.z_slot), ("clear the slot", *pre2, p.z_slot),
            ("up", *pre2, z_hover)]
