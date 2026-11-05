"""Pixel of the arm camera <-> xArm 5 robot frame (mm). numpy + OpenCV only; copy this file with camera_robot.json.

The Pi camera sits in the head on the xArm 5 flange. camera_robot.json (made by scan3d/handeye.py from the arm's
own joints and ArUco markers of known size) holds the camera's intrinsics, where the camera sits on the flange,
and the table height in the robot frame. With the joints the arm reports when the picture was taken:

    from pix2robot import Calibration
    cal = Calibration.load("camera_robot.json")
    q = arm.get_servo_angle()[1][:5]                        # degrees, J1..J5, read while the arm stands still
    top = cal.pixel_to_object_top(u, v, q, height_mm=35)   # robot mm of the object's top under pixel (u, v)
    foot = cal.pixel_to_table(u, v, q)                     # robot mm on the table under pixel (u, v)
    u, v = cal.robot_to_pixel([x, y, z], q)                # and back
    z_min = cal.min_head_centre_z(top[2])                  # keep the head sphere's centre at least this high

Robot frame: xArm base, z up, mm. The table is cal.table_z (-34 by the fit; the arm stands on a 40 mm pedestal), or
a slightly tilted plane cal.table_z_at(x, y) when the calibration measured one ("table_plane").
Picture size: the calibration's (the camera's default stream, 1600x900); cal.for_size(w, h) rescales it for another
size with the same aspect ratio (e.g. 640x360). Joints must be the ones at the moment of the picture: take the
picture with the arm standing still.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

D2R = math.pi / 180.0


# ---------------------------------------------------------------- xArm 5 forward kinematics (= the controller's)
def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1.0]])


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])


def _tr(x, y, z):
    t = np.eye(4)
    t[:3, 3] = (x, y, z)
    return t


_FIXED = [_tr(0, 0, 267.0), _rx(-math.pi / 2), _tr(53.5, -284.5, 0), _tr(77.5, 342.5, 0),
          _tr(76.0, 97.0, 0) @ _rx(-math.pi / 2)]


def flange_pose(q) -> np.ndarray:
    """4x4 flange pose in the robot base frame for joints J1..J5 in degrees (all 0 -> (207, 0, 112), pointing down)."""
    t = np.eye(4)
    for fixed, qi in zip(_FIXED, list(q)[:5]):
        t = t @ fixed @ _rz(float(qi) * D2R)
    return t


def rodrigues(rvec) -> np.ndarray:
    return cv2.Rodrigues(np.asarray(rvec, float).reshape(3, 1))[0]


# ---------------------------------------------------------------- calibration
class Calibration:
    def __init__(self, d: dict):
        self.data = d
        self.width, self.height = int(d["image_size"][0]), int(d["image_size"][1])
        self.K = np.array(d["K"], float)
        self.dist = np.array(d["dist"], float)
        self.T_fc = np.array(d["T_flange_camera"], float)       # camera frame (OpenCV: x right, y down, z forward)
        self.table_z = float(d["table_z_mm"])                   # at the work area; table_z_at(x, y) if tilted
        tp = d.get("table_plane")                               # z = a x + b y + c (robot mm), when measured
        self.plane = (float(tp["a"]), float(tp["b"]), float(tp["c"])) if tp else (0.0, 0.0, self.table_z)
        hs = d.get("head_sphere") or {}
        self.head_c = np.array(hs.get("center_flange_mm", [0, 0, 45]), float)
        self.head_r = float(hs.get("radius_mm", 60))

    @classmethod
    def load(cls, path: str | Path = Path(__file__).with_name("camera_robot.json")) -> "Calibration":
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    def for_size(self, width: int, height: int) -> "Calibration":
        """Same camera at another picture size with the SAME aspect ratio (the stream is scaled, not cropped)."""
        if abs(width / height - self.width / self.height) > 0.01:
            raise ValueError(f"{width}x{height} is not the aspect ratio of {self.width}x{self.height}: recalibrate")
        s = width / self.width
        d = dict(self.data, image_size=[width, height])
        K = self.K.copy()
        K[0, :] *= s
        K[1, :] *= s
        d["K"] = K.tolist()
        return Calibration(d)

    # ------------------------------------------------------------ geometry
    def camera_pose(self, q) -> np.ndarray:
        """4x4 camera pose in the robot frame (columns: image x, image y, optical axis, optical centre)."""
        return flange_pose(q) @ self.T_fc

    def ray(self, u: float, v: float, q) -> tuple[np.ndarray, np.ndarray]:
        """Optical centre and unit direction (robot frame) of the ray through pixel (u, v)."""
        n = cv2.undistortPoints(np.array([[[u, v]]], float), self.K, self.dist).reshape(2)
        T = self.camera_pose(q)
        d = T[:3, :3] @ np.array([n[0], n[1], 1.0])
        return T[:3, 3].copy(), d / np.linalg.norm(d)

    def pixel_to_plane(self, u: float, v: float, q, z: float, slope=(0.0, 0.0)) -> np.ndarray:
        """Robot-frame point (mm) where pixel (u, v)'s ray meets the plane z' = z + a x + b y (slope = (a, b);
        default horizontal at height z)."""
        c, d = self.ray(u, v, q)
        a, b = slope
        den = d[2] - a * d[0] - b * d[1]
        s = (z + a * c[0] + b * c[1] - c[2]) / den if abs(den) > 1e-6 else -1.0
        if s <= 0:
            raise ValueError("this pixel's ray does not reach that plane (looking up or parallel)")
        return c + d * s

    def table_z_at(self, x: float, y: float) -> float:
        a, b, c = self.plane
        return a * x + b * y + c

    def pixel_to_table(self, u: float, v: float, q) -> np.ndarray:
        a, b, c = self.plane
        return self.pixel_to_plane(u, v, q, c, (a, b))

    def pixel_to_object_top(self, u: float, v: float, q, height_mm: float) -> np.ndarray:
        """Point under pixel (u, v) on the plane of an object's top face (table + height). Use the pixel of the
        object's centre SEEN FROM ABOVE; for a tilted view the centre of its outline is not exactly its centre."""
        a, b, c = self.plane
        return self.pixel_to_plane(u, v, q, c + float(height_mm), (a, b))

    def robot_to_pixel(self, xyz, q) -> tuple[float, float]:
        T = np.linalg.inv(self.camera_pose(q))
        p = T[:3, :3] @ np.asarray(xyz, float) + T[:3, 3]
        if p[2] <= 0:
            raise ValueError("the point is behind the camera")
        uv, _ = cv2.projectPoints(p.reshape(1, 1, 3), np.zeros(3), np.zeros(3), self.K, self.dist)
        return float(uv[0, 0, 0]), float(uv[0, 0, 1])

    def mm_per_pixel(self, q, z: float) -> float:
        """Size of one pixel on the horizontal plane z near the picture centre (mm)."""
        a = self.pixel_to_plane(self.width / 2, self.height / 2, q, z)
        b = self.pixel_to_plane(self.width / 2 + 10, self.height / 2, q, z)
        return float(np.linalg.norm(b - a) / 10)

    # ------------------------------------------------------------ safety
    def head_sphere_world(self, q) -> tuple[np.ndarray, float]:
        """Centre (robot mm) and radius of a sphere holding the whole camera head."""
        T = flange_pose(q)
        return T[:3, :3] @ self.head_c + T[:3, 3], self.head_r

    def min_head_centre_z(self, top_z: float, clearance: float = 30.0) -> float:
        """Lowest allowed z for the head sphere's centre above an object whose top is at top_z (robot mm)."""
        return float(top_z) + self.head_r + clearance


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="pixel -> robot mm with camera_robot.json")
    ap.add_argument("u", type=float)
    ap.add_argument("v", type=float)
    ap.add_argument("--q", required=True, help="joints J1..J5 in degrees, comma-separated (--q=-93.5,... if J1 < 0)")
    ap.add_argument("--height", type=float, default=0.0, help="object height above the table, mm")
    ap.add_argument("--cal", default=str(Path(__file__).with_name("camera_robot.json")))
    a = ap.parse_args()
    cal = Calibration.load(a.cal)
    q = [float(x) for x in a.q.split(",")]
    p = cal.pixel_to_object_top(a.u, a.v, q, a.height)
    print(f"pixel ({a.u:.0f}, {a.v:.0f}) at height {a.height:.0f} mm -> robot x {p[0]:.1f}  y {p[1]:.1f}  z {p[2]:.1f} mm"
          f"   ({cal.mm_per_pixel(q, p[2]):.3f} mm/px there); head centre must stay above z "
          f"{cal.min_head_centre_z(p[2]):.0f}")
