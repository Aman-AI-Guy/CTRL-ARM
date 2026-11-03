"""Arm drivers. make_arm() picks one from TROVA_ARM.

  sim                      simulated arm drawn on the live view (default)
  ur:<ip>                  UR3e / UR5e            (trova/arm/ur.py)
  xarm:<ip>                UFactory xArm5         (trova/arm/xarm.py)
  elite:<ip>               Elite CS63 / CS66      (trova/arm/elite.py)
  cartesian:fake           simulated Cartesian robot, to rehearse calibration and safety
  grid:<module>:<Class>    any other arm, via taught poses (taught_grid.py)
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path

from .base import ArmDriver
from .cartesian import CartesianArm, CartesianBackend, FakeCartesian, Limits
from .sim import SimArm
from .taught_grid import FakeJointBackend, JointBackend, TaughtGridArm

CARTESIAN = {"ur": ("trova.arm.ur", "URBackend"), "xarm": ("trova.arm.xarm", "XArmBackend"),
             "elite": ("trova.arm.elite", "EliteBackend")}


def _xyz(s: str | None):
    return tuple(float(v) for v in s.split(",")) if s else None


def make_arm(spec: str, data_dir: Path, table_w_mm: float, table_h_mm: float) -> ArmDriver:
    if spec == "sim":
        return SimArm(table_w_mm, table_h_mm)
    if spec.startswith("grid:"):
        target = spec[5:]
        if target == "fake":
            backend: JointBackend = FakeJointBackend()
        else:
            module, cls = target.rsplit(":", 1)
            backend = getattr(importlib.import_module(module), cls)()
        return TaughtGridArm(backend, data_dir / "taught_grid.json")
    kind, _, address = spec.partition(":")
    if kind == "cartesian" or kind in CARTESIAN:
        if kind == "cartesian":
            cb: CartesianBackend = FakeCartesian()
        else:
            module, cls = CARTESIAN[kind]
            cb = getattr(importlib.import_module(module), cls)(address)
        return CartesianArm(
            cb,
            z_hover_mm=float(os.getenv("TROVA_ARM_Z_MM", "150")),
            home_mm=_xyz(os.getenv("TROVA_ARM_HOME_MM")),
            limits=Limits.parse(os.getenv("TROVA_ARM_LIMITS")),
            speed_mm_s=float(os.getenv("TROVA_ARM_SPEED_MM_S", "100")),
        )
    raise ValueError(f"Unknown TROVA_ARM '{spec}'")


__all__ = ["ArmDriver", "SimArm", "TaughtGridArm", "JointBackend", "FakeJointBackend", "CartesianArm",
           "CartesianBackend", "FakeCartesian", "Limits", "make_arm"]
