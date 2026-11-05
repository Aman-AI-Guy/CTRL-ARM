"""Real collision shapes of the xArm 5 and of the camera head, as sets of spheres. numpy only.

Part of the arm kit (this folder). robot_model.json is built by model/build_robot_model.py from UFactory's link
meshes and the head's CAD ("Camera Gatto.step" = model/camera_head.step).

    geo = RobotGeometry.load()
    turn = geo.head_turn(camera_axis_in_flange_frame)    # how the head is turned on the flange (4 bolt positions)
    offset, off_deg = geo.camera(camera_axis_in_flange_frame)
    hits = geo.collisions(fk_frames(q), turn, table_z=0, table_margin=15, obj_xy=(400, 0), obj_radius=60,
                          obj_top=120, obj_margin=20)

Frames: fk_frames(q) of the xArm 5 URDF (5 frames after joints 1..5; the last one is the flange). Units: mm.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

LINKS = ("link_base", "link1", "link2", "link3", "link4", "link5")
# shapes that can reach the table or the object (the base and link 1 stand on the table)
MOVING = ("link2", "link3", "link4", "link5", "head")
# pairs that can touch each other; neighbours (joined by one joint) are left out
SELF_PAIRS = (("head", "link_base"), ("head", "link1"), ("head", "link2"), ("head", "link3"),
              ("link5", "link_base"), ("link5", "link1"), ("link5", "link2"),
              ("link4", "link_base"), ("link4", "link1"))


def _rz3(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


class RobotGeometry:
    def __init__(self, data: dict):
        self.data = data
        self.shapes = {n: (np.array(d["c"], float), np.array(d["r"], float)) for n, d in data["links"].items()}
        h = data["head"]
        self.shapes["head"] = (np.array(h["c"], float), np.array(h["r"], float))
        self.cam_center = np.array(h["camera_center_mm"], float)
        self.cam_axis = np.array(h["camera_axis"], float)
        # bounding sphere per shape (local frame) for a quick reject
        self.bound = {}
        for n, (c, r) in self.shapes.items():
            m = c.mean(0)
            self.bound[n] = (m, float((np.linalg.norm(c - m, axis=1) + r).max()))

    @classmethod
    def load(cls, path: str | Path | None = None) -> "RobotGeometry":
        path = Path(path) if path else Path(__file__).with_name("robot_model.json")
        return cls(json.loads(path.read_text(encoding="utf-8")))

    # ---------------------------------------------------------------- the head on the flange

    def turn_angle(self, axis_flange, snap_deg: float = 0.0) -> float:
        """Angle (deg) the head is turned about the flange axis so that the CAD camera direction points where the
        measured axis points (its component across the flange axis). The 4 bolts allow only 4 positions 90 deg apart
        (at 0 or 45 deg from the flange's x axis, depending on the hole pattern): snap_deg=45 rounds a rough
        measurement (a start pose aimed by eye) to the nearest possible position."""
        a = np.asarray(axis_flange, float)
        if math.hypot(a[0], a[1]) < 1e-9:
            raise ValueError("camera axis along the flange axis: the CAD says it looks sideways")
        ang = math.degrees(math.atan2(a[1], a[0]) - math.atan2(self.cam_axis[1], self.cam_axis[0]))
        if snap_deg:
            ang = round(ang / snap_deg) * snap_deg
        return (ang + 180.0) % 360.0 - 180.0

    def head_turn(self, axis_flange, snap_deg: float = 0.0) -> np.ndarray:
        """Rotation (flange <- head frame) for turn_angle()."""
        return _rz3(math.radians(self.turn_angle(axis_flange, snap_deg)))

    def camera(self, axis_flange) -> tuple[np.ndarray, float]:
        """Optical centre in the flange frame for this camera axis, and the angle (deg) between the axis and the CAD
        viewing direction (0 = the camera looks exactly where the CAD says: perpendicular to the flange axis)."""
        turn = self.head_turn(axis_flange)
        a = np.asarray(axis_flange, float) / np.linalg.norm(axis_flange)
        dev = math.degrees(math.acos(max(-1.0, min(1.0, float(a @ (turn @ self.cam_axis))))))
        return turn @ self.cam_center, dev

    # ---------------------------------------------------------------- collisions

    def world(self, frames, turn) -> dict:
        """Sphere centres in the robot base frame and radii, per shape."""
        out = {"link_base": self.shapes["link_base"]}
        for i, name in enumerate(LINKS[1:]):
            f = frames[i]
            c, r = self.shapes[name]
            out[name] = (c @ f[:3, :3].T + f[:3, 3], r)
        f = frames[4]
        c, r = self.shapes["head"]
        out["head"] = (c @ (f[:3, :3] @ turn).T + f[:3, 3], r)
        return out

    def _bound_world(self, name, frames, turn):
        m, rad = self.bound[name]
        if name == "link_base":
            return m, rad
        if name == "head":
            f = frames[4]
            return f[:3, :3] @ (turn @ m) + f[:3, 3], rad
        f = frames[LINKS.index(name) - 1]
        return f[:3, :3] @ m + f[:3, 3], rad

    def collisions(self, frames, turn, table_z: float, table_margin: float, obj_xy, obj_radius: float,
                   obj_top: float, obj_margin: float, self_margin: float = 5.0) -> list[str]:
        """Which of 'table', 'object', 'self <a>/<b>' the arm with the head would touch (with these margins)."""
        w = self.world(frames, turn)
        hits = set()
        ox, oy = float(obj_xy[0]), float(obj_xy[1])
        for name in MOVING:
            c, r = w[name]
            if (c[:, 2] - r < table_z + table_margin).any():
                hits.add("table")
            near = np.hypot(c[:, 0] - ox, c[:, 1] - oy) < obj_radius + r + obj_margin
            if (near & (c[:, 2] - r < obj_top + obj_margin)).any():
                hits.add("object")
        for a, b in SELF_PAIRS:
            ma, ra = self._bound_world(a, frames, turn)
            mb, rb = self._bound_world(b, frames, turn)
            if np.linalg.norm(ma - mb) > ra + rb + self_margin:
                continue
            ca, rra = w[a]
            cb, rrb = w[b]
            d = np.linalg.norm(ca[:, None, :] - cb[None, :, :], axis=2) - rra[:, None] - rrb[None, :]
            if d.min() < self_margin:
                hits.add(f"self {a}/{b}")
        return sorted(hits)

    def clearance(self, frames, turn, a: str, b: str) -> float:
        """Smallest gap (mm, negative = overlap) between two shapes' spheres."""
        w = self.world(frames, turn)
        ca, ra = w[a]
        cb, rb = w[b]
        return float((np.linalg.norm(ca[:, None, :] - cb[None, :, :], axis=2) - ra[:, None] - rb[None, :]).min())
