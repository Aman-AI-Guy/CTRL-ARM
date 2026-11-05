"""Orbit the xArm 5 camera around an object: teach, fit, plan, check, run.

The Pi camera sits in the printed sphere on the flange and looks sideways. The xArm 5 has 5 joints: enough to put
the camera at any reachable point (3 numbers) AND aim it at the object (2 numbers); only the roll of the picture is
left to the arm. A Cartesian program with a fixed roll/pitch/yaw (G-code, set_position) cannot do that on a 5-axis
arm, so we plan in JOINT space with our own forward kinematics (the xArm 5 URDF) and send joint moves.

    python -m scan3d.orbit teach --ip 192.168.1.xxx     # live view + crosshair: record poses by hand
    python -m scan3d.orbit fit --height 120 --radius 50 # camera axis + object centre from those poses
    python -m scan3d.orbit plan                         # orbit -> data/orbit/plan.json + plan.png (no motion)
    python -m scan3d.orbit check --ip ...               # our kinematics vs the controller, where the camera looks now
    python -m scan3d.orbit run --ip ... --step --only 0,6,12,18   # test some views, Enter before each move
    python -m scan3d.orbit run --ip ... --snapshots     # whole orbit, a still at every view -> data/scans/<id>/images
    python -m scan3d.orbit simulate --object 350,0      # synthetic teach.json: rehearse everything with --ip fake

The hard part is where the object is, so we measure it, check it and correct it:
  1. teach: object on the table, M = manual mode, move the arm by hand.
       T     = flange straight above the object's centre (optional, makes the fit much stiffer)
       SPACE = the object is on the crosshair. 3-6 poses from different sides, near/far, high/low.
  2. fit: camera axis, object centre and the miss of every pose in mm (< 10 mm is good).
  3. plan: rings around the object; views out of reach are moved closer/higher or dropped (all printed).
  4. run --step --only ...: look at some views. Object off-centre? teach again AT that view, centre it by hand,
     SPACE, fit + plan again. Every correction is one more pose for the fit.
  5. run: the whole orbit (--snapshots, or scan3d.capture recording in another terminal).

Units: mm and degrees. Robot base frame: z up, table at orbit.table_z_mm (0 when the arm stands on the table).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import ROOT, add_config_args, config_from_args

# Real shapes of the arm and of the camera head (arm_pc/robot_geometry.py + robot_model.json, numpy only, shared with
# arm_pc/orbit_here.py). Built by arm_pc/model/build_robot_model.py from UFactory's meshes and "Camera Gatto.step".
if str(ROOT / "arm_pc") not in sys.path:
    sys.path.insert(0, str(ROOT / "arm_pc"))
from robot_geometry import RobotGeometry  # noqa: E402

_GEO: dict = {}


def cad_geometry(o: dict):
    """The CAD model when orbit.use_cad_geometry is on (default), else None (capsules + camera_offset_mm)."""
    if not o.get("use_cad_geometry", True):
        return None
    if "geo" not in _GEO:
        _GEO["geo"] = RobotGeometry.load()
    return _GEO["geo"]

D2R = math.pi / 180.0
ORBIT_DIR = ROOT / "data" / "orbit"

# ---------------------------------------------------------------- xArm 5 kinematics (xarm_description URDF)


def _rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1.0]])


def _rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])


def _tr(x: float, y: float, z: float) -> np.ndarray:
    t = np.eye(4)
    t[:3, 3] = (x, y, z)
    return t


# Fixed transform of each joint origin in its parent frame; every joint turns about its local z.
# All joints at 0 -> flange at (207, 0, 112) pointing down (roll 180, pitch 0, yaw 0), as the controller reports.
_FIXED = [_tr(0, 0, 267.0), _rx(-math.pi / 2), _tr(53.5, -284.5, 0), _tr(77.5, 342.5, 0),
          _tr(76.0, 97.0, 0) @ _rx(-math.pi / 2)]
XARM5_LIMITS = np.array([(-360, 360), (-118, 120), (-225, 11), (-97, 180), (-360, 360)], float)
JOINT_LIMITS = XARM5_LIMITS.copy()     # what the planner may use (orbit.joint_limits_deg can narrow it)


def fk_frames(q) -> list[np.ndarray]:
    """4x4 frame after each joint (shoulder, shoulder, elbow, wrist, flange) for joints q in degrees."""
    t, out = np.eye(4), []
    for fixed, qi in zip(_FIXED, q):
        t = t @ fixed @ _rz(float(qi) * D2R)
        out.append(t)
    return out


def fk(q) -> np.ndarray:
    return fk_frames(q)[-1]


def clip_joints(q) -> np.ndarray:
    return np.clip(np.asarray(q, float), JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])


def rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """xArm convention (degrees): R = Rz(yaw) Ry(pitch) Rx(roll)."""
    r, p, y = roll * D2R, pitch * D2R, yaw * D2R
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def matrix_rpy(r: np.ndarray) -> tuple[float, float, float]:
    pitch = math.asin(max(-1.0, min(1.0, -r[2, 0])))
    return (math.atan2(r[2, 1], r[2, 2]) / D2R, pitch / D2R, math.atan2(r[1, 0], r[0, 0]) / D2R)


def pose_matrix(pose) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = rpy_matrix(*pose[3:6])
    t[:3, 3] = pose[:3]
    return t


# ---------------------------------------------------------------- camera on the flange


def axis_from_angles(az: float, el: float) -> np.ndarray:
    """Unit vector in the flange frame: az around the flange axis from flange +x, el towards flange +z (degrees)."""
    a, e = az * D2R, el * D2R
    return np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])


def angles_from_axis(v) -> tuple[float, float]:
    v = np.asarray(v, float) / np.linalg.norm(v)
    return math.atan2(v[1], v[0]) / D2R, math.asin(max(-1.0, min(1.0, v[2]))) / D2R


@dataclass
class Camera:
    offset: np.ndarray      # optical centre in the flange frame, mm
    axis: np.ndarray        # optical axis (unit) in the flange frame

    def ray(self, q) -> tuple[np.ndarray, np.ndarray]:
        f = fk(q)
        return f[:3, :3] @ self.offset + f[:3, 3], f[:3, :3] @ self.axis


def view_target(target, az: float, el: float, dist: float) -> tuple[np.ndarray, np.ndarray]:
    """Camera position on a sphere around the target (az around +z, el above the horizontal) and its view direction."""
    a, e = az * D2R, el * D2R
    u = np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
    return np.asarray(target, float) + dist * u, -u


def solve_view(cam: Camera, pos, direction, seed, scale: float = 250.0, iters: int = 60,
               tol_mm: float = 1.0, tol_deg: float = 0.5):
    """Joints that put the camera at pos looking along direction (Levenberg-Marquardt from seed), or None."""
    pos, direction = np.asarray(pos, float), np.asarray(direction, float)
    tol_dir = 2 * math.sin(tol_deg * D2R / 2)

    def res(q):
        p, v = cam.ray(q)
        return np.concatenate([p - pos, (v - direction) * scale])

    q = clip_joints(seed)
    r = res(q)
    cost, lam = float(r @ r), 1e-3
    for _ in range(iters):
        if np.linalg.norm(r[:3]) < tol_mm and np.linalg.norm(r[3:]) / scale < tol_dir:
            return q
        jac = np.empty((6, 5))
        for j in range(5):
            dq = np.zeros(5)
            dq[j] = 1e-3
            jac[:, j] = (res(q + dq) - r) / 1e-3
        jtj, g = jac.T @ jac, jac.T @ r
        while True:
            step = -np.linalg.solve(jtj + lam * (np.diag(np.diag(jtj)) + 1e-6 * np.eye(5)), g)
            big = np.abs(step).max()
            if big > 15:
                step *= 15 / big
            qn = clip_joints(q + step)
            rn = res(qn)
            cn = float(rn @ rn)
            if cn < cost:
                q, r, cost, lam = qn, rn, cn, max(lam * 0.3, 1e-9)
                break
            lam *= 10
            if lam > 1e9:
                return None
    ok = np.linalg.norm(r[:3]) < tol_mm and np.linalg.norm(r[3:]) / scale < tol_dir
    return q if ok else None


# ---------------------------------------------------------------- scene and collisions (coarse capsules)


@dataclass
class Scene:
    target: np.ndarray          # look-at point (object centre), mm
    object_xy: np.ndarray       # object axis
    object_radius: float
    object_top: float
    table_z: float
    table_margin: float
    object_margin: float
    link_radius: float
    camera_radius: float
    base_radius: float = 80.0       # the arm's own column (base + J1 link) as a vertical cylinder at x = y = 0
    base_height: float = 330.0
    geometry: object = None         # RobotGeometry: real shapes (then the capsule fields above are not used)
    head_turn: np.ndarray = None    # how the head sits on the flange (from the camera axis)
    self_margin: float = 5.0


def collisions(q, cam: Camera, sc: Scene) -> list[str]:
    """What the arm would touch at q: 'table', 'object', 'self <a>/<b>' with the real shapes (CAD), or
    'table' / 'object' / 'robot base' with the capsule model (links as capsules, camera as a ball)."""
    if sc.geometry is not None:
        return sc.geometry.collisions(fk_frames(q), sc.head_turn, sc.table_z, sc.table_margin, sc.object_xy,
                                      sc.object_radius, sc.object_top, sc.object_margin, sc.self_margin)
    f = fk_frames(q)
    shoulder, elbow, wrist, flange = (f[i][:3, 3] for i in (1, 2, 3, 4))
    cam_p = f[4][:3, :3] @ cam.offset + flange
    hits = set()
    for k, (a, b, rad) in enumerate(((shoulder, elbow, sc.link_radius), (elbow, wrist, sc.link_radius),
                                     (wrist, flange, sc.link_radius), (flange, cam_p, sc.camera_radius))):
        for t in np.linspace(0, 1, 8):
            p = a + (b - a) * t
            if p[2] - rad < sc.table_z + sc.table_margin:
                hits.add("table")
            if k > 0 and math.hypot(p[0], p[1]) < sc.base_radius + rad and p[2] - rad < sc.base_height:
                hits.add("robot base")
            if (math.hypot(p[0] - sc.object_xy[0], p[1] - sc.object_xy[1]) < sc.object_radius + rad + sc.object_margin
                    and p[2] - rad < sc.object_top + sc.object_margin):
                hits.add("object")
    return sorted(hits)


def path_collisions(q0, q1, cam: Camera, sc: Scene, steps: int = 8) -> list[str]:
    """Collisions along the straight joint-space move q0 -> q1 (what set_servo_angle does)."""
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    hits = set()
    for t in np.linspace(0, 1, steps + 1):
        hits.update(collisions(q0 + (q1 - q0) * t, cam, sc))
    return sorted(hits)


# ---------------------------------------------------------------- config / files


def orbit_cfg(cfg: dict) -> dict:
    """The orbit section; orbit.joint_limits_deg (5 x [min, max]) narrows the xArm 5 limits for every plan."""
    o = {k: v for k, v in cfg.get("orbit", {}).items() if not k.startswith("_")}
    lim = o.get("joint_limits_deg")
    JOINT_LIMITS[:] = XARM5_LIMITS
    if lim:
        lim = np.array(lim, float)
        JOINT_LIMITS[:, 0] = np.maximum(XARM5_LIMITS[:, 0], lim[:, 0])
        JOINT_LIMITS[:, 1] = np.minimum(XARM5_LIMITS[:, 1], lim[:, 1])
    return o


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)


def camera_from_model(model: dict) -> Camera:
    return Camera(np.array(model["camera_offset_mm"], float), np.array(model["camera_axis_flange"], float))


def scene_from_model(model: dict, o: dict) -> Scene:
    obj = model["object"]
    geo = cad_geometry(o) if model.get("camera_from") == "cad" else None
    if geo is not None:
        tm, om = float(o.get("cad_table_margin_mm", 15)), float(o.get("cad_object_margin_mm", 20))
    else:
        tm, om = float(o["table_margin_mm"]), float(o["object_margin_mm"])
    return Scene(np.array(model["target_mm"], float), np.array(obj["xy_mm"], float), float(obj["radius_mm"]),
                 float(obj["table_z_mm"]) + float(obj["height_mm"]), float(obj["table_z_mm"]),
                 tm, om, float(o["link_radius_mm"]),
                 float(o["camera_radius_mm"]), float(o.get("base_radius_mm", 80)), float(o.get("base_height_mm", 330)),
                 geo, geo.head_turn(model["camera_axis_flange"]) if geo is not None else None,
                 float(o.get("self_margin_mm", 5)))


def r3(v) -> list[float]:
    return [round(float(x), 2) for x in v]


# ---------------------------------------------------------------- fit: camera axis + object centre from taught poses


def _target_for(rays, fixed_xy, fixed_z):
    """Point closest (least squares) to all rays, with x,y and/or z held fixed when known."""
    projs = [(np.eye(3) - np.outer(v, v), p) for p, v in rays]
    if fixed_xy is not None and fixed_z is not None:
        return np.array([fixed_xy[0], fixed_xy[1], fixed_z], float)
    if fixed_xy is not None:
        xy0 = np.array([fixed_xy[0], fixed_xy[1], 0.0])
        num = den = 0.0
        for m, p in projs:
            a = m[:, 2]
            b = m @ p - m @ xy0
            num += a @ b
            den += a @ a
        return np.array([xy0[0], xy0[1], num / max(den, 1e-9)])
    a = sum(m for m, _ in projs)
    b = sum(m @ p for m, p in projs)
    return np.linalg.lstsq(a, b, rcond=None)[0]


def fit_model(teach: dict, o: dict, height: float | None = None, radius: float | None = None,
              fit_offset: bool | None = None, say=print) -> dict:
    """Least-squares fit of the camera axis in the flange frame (and the object centre when no T pose was taken,
    and the camera offset with fit_offset) so that every SPACE pose's optical axis passes through the centre."""
    centers = [np.array(p["q"], float) for p in teach["poses"] if p["kind"] == "center"]
    touches = [np.array(p["q"], float) for p in teach["poses"] if p["kind"] == "touch"]
    if not centers:
        raise ValueError("no SPACE (object on the crosshair) poses in the teach file")
    height = float(height if height is not None else o["object_height_mm"])
    radius = float(radius if radius is not None else o["object_radius_mm"])
    table_z = float(o["table_z_mm"])
    geo = cad_geometry(o)
    fit_offset = bool(o.get("fit_offset", False) if fit_offset is None else fit_offset) and geo is None
    offset0 = np.array(o["camera_offset_mm"], float)
    az0, el0 = angles_from_axis(o["camera_axis_flange"])
    flanges = [fk(q) for q in centers]
    fixed_xy = np.mean([fk(q)[:2, 3] for q in touches], axis=0) if touches else None
    fixed_z = table_z + height / 2 if touches else None
    if len(centers) + (2 if touches else 0) < 3:
        say("  ! few poses: take at least 3 SPACE poses (or 1 T + 2 SPACE) from different sides")

    def unpack(x):
        axis = axis_from_angles(x[0], x[1])
        if geo is not None:                  # the CAD gives the optical centre; it turns with the head on the flange
            return axis, geo.camera(axis)[0]
        off = offset0 + x[2:5] if len(x) > 2 else offset0
        return axis, off

    def residual(x):
        axis, off = unpack(x)
        rays = [(f[:3, :3] @ off + f[:3, 3], f[:3, :3] @ axis) for f in flanges]
        tgt = _target_for(rays, fixed_xy, fixed_z)
        out = []
        for p, v in rays:
            d = tgt - p
            out.extend(d - v * (v @ d))
            out.append(max(0.0, 30.0 - v @ d) * 10)          # the object must be in FRONT of the camera
        if len(x) > 2:
            out.extend(x[2:5] * 0.05)                         # keep the offset near the measured one
        return np.array(out), tgt

    best = None
    for az in np.arange(-180, 180, 10.0):                     # coarse grid, then Levenberg-Marquardt
        for el in np.arange(-80, 81, 10.0):
            x = np.array([az, el] + ([0, 0, 0] if fit_offset else []), float)
            c = float(np.sum(residual(x)[0] ** 2))
            if best is None or c < best[0]:
                best = (c, x)
    # the measured guess too, in case the grid's best is a mirror solution with a similar cost
    xg = np.array([az0, el0] + ([0, 0, 0] if fit_offset else []), float)
    cg = float(np.sum(residual(xg)[0] ** 2))
    x = best[1] if best[0] <= cg else xg
    lam = 1e-3
    r = residual(x)[0]
    cost = float(r @ r)
    for _ in range(200):
        jac = np.empty((len(r), len(x)))
        for j in range(len(x)):
            dx = np.zeros(len(x))
            dx[j] = 1e-4
            jac[:, j] = (residual(x + dx)[0] - r) / 1e-4
        jtj, g = jac.T @ jac, jac.T @ r
        improved = False
        for _ in range(20):
            xn = x - np.linalg.solve(jtj + lam * (np.diag(np.diag(jtj)) + 1e-9 * np.eye(len(x))), g)
            rn = residual(xn)[0]
            if float(rn @ rn) < cost:
                x, r, cost, lam, improved = xn, rn, float(rn @ rn), max(lam * 0.3, 1e-12), True
                break
            lam *= 10
        if not improved or np.linalg.norm(g) < 1e-9:
            break

    axis, off = unpack(x)
    _, tgt = residual(x)
    cam = Camera(off, axis)
    per_pose, dists, elevs, azims = [], [], [], []
    for q in centers:
        p, v = cam.ray(q)
        d = tgt - p
        along = float(v @ d)
        miss = float(np.linalg.norm(d - v * along))
        dist = float(np.linalg.norm(d))
        per_pose.append({"q": r3(q), "miss_mm": round(miss, 1), "off_axis_deg": round(math.atan2(miss, along) / D2R, 2),
                         "distance_mm": round(dist, 1)})
        dists.append(dist)
        elevs.append(math.asin((p[2] - tgt[2]) / dist) / D2R)
        azims.append(math.atan2(p[1] - tgt[1], p[0] - tgt[0]) / D2R)
    rays = [cam.ray(q) for q in centers]
    eig = np.linalg.eigvalsh(sum(np.eye(3) - np.outer(v, v) for _, v in rays))
    rms = math.sqrt(np.mean([pp["miss_mm"] ** 2 for pp in per_pose]))
    obj_xy = fixed_xy if fixed_xy is not None else tgt[:2]
    model = {
        "camera_offset_mm": r3(off),
        "camera_from": "cad" if geo is not None else "config",
        "camera_vs_cad_deg": round(geo.camera(axis)[1], 2) if geo is not None else None,
        "camera_axis_flange": [round(float(v), 5) for v in axis],
        "camera_axis_angles_deg": {"az": round(angles_from_axis(axis)[0], 2), "el": round(angles_from_axis(axis)[1], 2)},
        "target_mm": r3(tgt),
        "object": {"xy_mm": r3(obj_xy), "radius_mm": radius, "height_mm": height, "table_z_mm": table_z},
        "fit": {"rms_miss_mm": round(rms, 2), "poses": per_pose, "touch_poses": len(touches),
                "target_from": "T pose + height" if touches else "rays", "weakest_direction": round(float(eig[0]), 3),
                "fit_offset": fit_offset},
        "distance_mm": round(float(np.mean(dists)), 1),
        "elevation_deg": round(float(np.mean(elevs)), 1),
        "start_azimuth_deg": round(float(azims[-1]), 1),
        "seeds": [r3(q) for q in centers + touches],
        "teach_file": teach.get("_path", ""),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if teach.get("truth"):
        t = teach["truth"]
        model["fit"]["vs_truth"] = {
            "target_err_mm": round(float(np.linalg.norm(tgt - np.array(t["target_mm"]))), 2),
            "axis_err_deg": round(math.acos(max(-1.0, min(1.0, float(axis @ np.array(t["camera_axis_flange"]))))) / D2R, 3)}
    return model


def print_fit(model: dict, say=print) -> None:
    f = model["fit"]
    say(f"camera axis in the flange frame: az {model['camera_axis_angles_deg']['az']:.1f} deg, "
        f"el {model['camera_axis_angles_deg']['el']:.1f} deg  -> {model['camera_axis_flange']}")
    if model.get("camera_from") == "cad":
        dev = model["camera_vs_cad_deg"]
        say(f"camera offset (flange frame): {model['camera_offset_mm']} mm from the CAD head; the fitted axis is "
            f"{dev:.1f} deg from the CAD direction (perpendicular to the flange)")
        if dev > 8:
            say("  ! more than 8 deg: a pose is wrong, or the camera sits tilted in the sphere (check it)")
    else:
        say(f"camera offset (flange frame): {model['camera_offset_mm']} mm{'  (fitted)' if f['fit_offset'] else ''}")
    say(f"object centre (look-at point): {model['target_mm']} mm   from {f['target_from']}")
    for i, p in enumerate(f["poses"]):
        say(f"  pose {i}: miss {p['miss_mm']:6.1f} mm   off axis {p['off_axis_deg']:5.2f} deg   at {p['distance_mm']:.0f} mm")
    say(f"rms miss {f['rms_miss_mm']:.1f} mm  (< 10 good, 10-25 usable, > 25: a pose is wrong or the camera moved)")
    if f["target_from"] == "rays" and f["weakest_direction"] < 0.15:
        say("  ! rays nearly parallel: the object position is poorly defined. Add poses from other sides or a T pose")
    say(f"mean distance {model['distance_mm']:.0f} mm, elevation {model['elevation_deg']:.0f} deg, "
        f"last pose at azimuth {model['start_azimuth_deg']:.0f} deg")
    tilt = model["camera_axis_angles_deg"]["el"]
    say(f"with the flange pointing down the camera looks {'down' if tilt >= 0 else 'up'} {abs(tilt):.0f} deg: an orbit at "
        f"elevation {tilt:.0f} deg needs no flange tilt (base + head only, the most reliable): plan --elevations auto")
    if tilt < 10:
        say("  tip: tilt the camera down 20-30 deg in its mount (or put the object on a stand) so the 'auto' orbit "
            "sees the object from above the table")
    if "vs_truth" in f:
        say(f"vs synthetic truth: target {f['vs_truth']['target_err_mm']} mm, axis {f['vs_truth']['axis_err_deg']} deg")


# ---------------------------------------------------------------- plan

RELAX = [(1.0, 0), (1.0, 10), (0.9, 0), (0.9, 10), (1.0, -10), (0.8, 0), (0.8, 10), (1.1, 0), (1.0, 20),
         (0.8, 20), (0.7, 10), (0.7, 25), (1.2, 0)]        # (distance factor, extra elevation): first that works


def _generic_seeds(pos) -> list[np.ndarray]:
    """Flange pointing down (q2 + q3 + q4 = 0: the three pitch joints are parallel) with the camera turned by J5,
    i.e. 'move the base and turn the head', plus a few tilted ones."""
    a = math.atan2(pos[1], pos[0]) / D2R
    out = []
    for q2, q3 in ((0, -60), (25, -85), (45, -115), (-15, -35)):
        for tilt in (0, 25, -25):
            for q5 in (-180, -90, 0, 90, 180):
                out.append(np.array([a, q2, q3, -(q2 + q3) + tilt, q5], float))
    return out


def _distinct(q, sols, tol=3.0) -> bool:
    return all(np.abs(q - s).max() > tol for s in sols)


def _view_candidates(cam, sc, tgt, az, el, dist, seeds, max_cand=10):
    """Distinct collision-free IK solutions for one view; relaxed (closer / higher) only if the nominal one has none.
    Returns (list of joints, distance, elevation, relax rank)."""
    for rank, (dscale, de) in enumerate(RELAX):
        d, e = dist * dscale, min(el + de, 80.0)
        pos, view = view_target(tgt, az, e, d)
        sols: list[np.ndarray] = []
        for s in seeds + _generic_seeds(pos):
            q = solve_view(cam, pos, view, s, scale=d)
            if q is None or collisions(q, cam, sc):
                continue
            for k in (0, -360, 360):          # J5 +-360 is the same picture, but not the same cable twist
                qq = q.copy()
                qq[4] += k
                if JOINT_LIMITS[4, 0] <= qq[4] <= JOINT_LIMITS[4, 1] and _distinct(qq, sols):
                    sols.append(qq)
            if len(sols) >= max_cand:
                break
        if sols:
            return sols, d, e, rank
    return [], dist, el, -1


def plan_orbit(model: dict, o: dict, distance=None, elevations=None, step=None, span=None, direction=None,
               say=print) -> dict:
    """Views on rings around the object, several IK solutions per view, then the cheapest collision-free way
    through them (dynamic programming over views): joint travel + penalties for relaxed, skipped and big moves."""
    cam, sc = camera_from_model(model), scene_from_model(model, o)
    tgt = sc.target
    dist = float(distance or o.get("distance_mm") or model["distance_mm"])
    elevs = elevations or o.get("elevations_deg") or [model["elevation_deg"]]
    step = float(step or o["step_deg"])
    span = float(span or o["span_deg"])
    direction = int(direction or o.get("direction", 1)) or 1
    max_jump = float(o["max_joint_step_deg"])
    taught = [np.array(s, float) for s in model.get("seeds", [])]
    az0 = float(model["start_azimuth_deg"])
    n = int(round(span / step)) if span >= 360 else int(round(span / step)) + 1

    targets = []                                          # serpentine: ring 2 comes back the way ring 1 went
    for ri, el in enumerate(elevs):
        sgn = direction if ri % 2 == 0 else -direction
        start = az0 if ri % 2 == 0 else az0 + direction * step * (n - 1)
        targets += [(ri, float(el), start + sgn * k * step) for k in range(n)]

    views = []                                            # candidates, seeded by the previous views' (branches go on)
    for i, (ri, el, az) in enumerate(targets):
        prev = [q for v in views[-2:] for q in v["cands"]]
        cands, d, e, rank = _view_candidates(cam, sc, tgt, az, el, dist, prev + taught)
        views.append({"ring": ri, "az": az, "el": e, "d": d, "rank": rank, "cands": cands})

    skip_pen, relax_pen, jump_pen, max_skip = 400.0, 15.0, 60.0, 3
    path_cache: dict = {}

    def path_ok(i, a, j, b) -> bool:
        key = (i, a, j, b)
        if key not in path_cache:
            qa, qb = views[i]["cands"][a], views[j]["cands"][b]
            steps = max(4, int(math.ceil(np.abs(qb - qa).max() / 5.0)))
            path_cache[key] = not path_collisions(qa, qb, cam, sc, steps)
        return path_cache[key]

    def node_cost(j):
        return relax_pen * (views[j]["rank"] > 0) + 3.0 * max(views[j]["rank"], 0)

    best: dict = {}                                       # (view, cand) -> (cost, previous node)
    start_q = taught[-1] if taught else None
    for j, v in enumerate(views):
        for b, qb in enumerate(v["cands"]):
            c0 = skip_pen * j + node_cost(j)
            if start_q is not None:
                c0 += 0.2 * float(np.abs(qb - start_q).max())
            entry = (c0, None)
            for i in range(max(0, j - 1 - max_skip), j):
                for a, qa in enumerate(views[i]["cands"]):
                    if (i, a) not in best:
                        continue
                    jump = float(np.abs(qb - qa).max())
                    c = best[(i, a)][0] + jump + jump_pen * (jump > max_jump) + skip_pen * (j - i - 1) + node_cost(j)
                    if c < entry[0] and path_ok(i, a, j, b):
                        entry = (c, (i, a))
            best[(j, b)] = entry
    chosen: dict = {}
    if best:
        last = len(views) - 1
        end = min(best, key=lambda k: best[k][0] + skip_pen * (last - k[0]))
        node = end
        while node is not None:
            chosen[node[0]] = node[1]
            node = best[node][1]

    poses, prev_q = [], None
    for i, v in enumerate(views):
        if i not in chosen:
            poses.append({"i": i, "ring": v["ring"], "az_deg": round(v["az"] % 360, 1), "el_deg": round(v["el"], 1),
                          "q": None, "note": "out of reach or in collision" if not v["cands"]
                          else "reachable, but no collision-free way to it from its neighbours"})
            continue
        q = v["cands"][chosen[i]]
        p, w = cam.ray(q)
        jump = float(np.abs(q - prev_q).max()) if prev_q is not None else 0.0
        poses.append({"i": i, "ring": v["ring"], "az_deg": round(v["az"] % 360, 1), "el_deg": round(v["el"], 1),
                      "distance_mm": round(v["d"], 1), "relaxed": v["rank"] > 0, "q": r3(q),
                      "camera_mm": r3(p), "camera_dir": [round(float(x), 4) for x in w],
                      "max_joint_jump_deg": round(jump, 1)})
        prev_q = q

    ok = [p for p in poses if p["q"] is not None]
    qs = np.array([p["q"] for p in ok]) if ok else np.zeros((0, 5))
    travel = float(np.sum(np.abs(np.diff(qs, axis=0)).max(axis=1))) if len(qs) > 1 else 0.0
    speed, dwell = float(o["joint_speed_deg_s"]), float(o["dwell_s"])
    summary = {
        "views": len(poses), "reachable": len(ok), "skipped": [p["i"] for p in poses if p["q"] is None],
        "relaxed": [p["i"] for p in ok if p["relaxed"]],
        "j5_span_deg": round(float(qs[:, 4].max() - qs[:, 4].min()), 1) if len(qs) else 0.0,
        "big_jumps": [p["i"] for p in ok if p["max_joint_jump_deg"] > max_jump],
        "est_time_s": round(travel / speed * 1.3 + len(ok) * (dwell + 0.5), 0),
    }
    plan = {"model": model, "settings": {"distance_mm": dist, "elevations_deg": list(elevs), "step_deg": step,
                                         "span_deg": span, "direction": direction},
            "summary": summary, "poses": poses, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    return plan


def print_plan(plan: dict, say=print) -> None:
    s, st = plan["summary"], plan["settings"]
    say(f"orbit: {s['views']} views, rings at elevation {st['elevations_deg']} deg, every {st['step_deg']:.0f} deg, "
        f"distance {st['distance_mm']:.0f} mm, span {st['span_deg']:.0f} deg")
    for p in plan["poses"]:
        if p["q"] is None:
            say(f"  {p['i']:3d}  az {p['az_deg']:6.1f}  el {p['el_deg']:5.1f}   SKIPPED: {p['note']}")
        else:
            flag = "  relaxed" if p["relaxed"] else ""
            say(f"  {p['i']:3d}  az {p['az_deg']:6.1f}  el {p['el_deg']:5.1f}  d {p['distance_mm']:5.0f}  "
                f"q [{', '.join(f'{v:7.1f}' for v in p['q'])}]{flag}")
    say(f"reachable {s['reachable']}/{s['views']}; skipped {s['skipped'] or 'none'}; relaxed {s['relaxed'] or 'none'}")
    if s["skipped"] or s["big_jumps"]:
        say("  gaps or big moves come from views whose line of sight is tangent to the base: the xArm 5 can tilt the "
            "flange only towards/away from its base. Try --elevations auto, a smaller --distance, or the object "
            "farther from the base (400-450 mm)")
    say(f"J5 (camera roll) turns {s['j5_span_deg']:.0f} deg in total: check the camera cable has the slack for it")
    if s["big_jumps"]:
        say(f"  ! big joint jumps before views {s['big_jumps']} (path checked for collisions, but watch them)")
    say(f"estimated time {s['est_time_s']:.0f} s")


def plot_plan(plan: dict, o: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = plan["model"]
    cam, sc = camera_from_model(model), scene_from_model(model, o)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6.2))
    ax1.add_patch(plt.Circle((0, 0), 60, color="0.6"))
    ax1.text(0, -95, "xArm base", ha="center", fontsize=8)
    ax1.add_patch(plt.Circle(sc.object_xy, sc.object_radius, color="tab:orange", alpha=0.6))
    ax1.plot(*sc.target[:2], "k+", ms=12)
    colors = plt.cm.viridis(np.linspace(0, 0.85, max(1, len(plan["settings"]["elevations_deg"]))))
    for p in plan["poses"]:
        if p["q"] is None:
            pos, _ = view_target(sc.target, p["az_deg"], p["el_deg"], plan["settings"]["distance_mm"])
            ax1.plot(pos[0], pos[1], "rx", ms=8)
            continue
        c = np.array(p["camera_mm"])
        col = colors[p["ring"]]
        ax1.plot([c[0], sc.target[0]], [c[1], sc.target[1]], color=col, lw=0.4, alpha=0.5)
        ax1.plot(c[0], c[1], "o", color=col, ms=5)
        ax1.annotate(str(p["i"]), c[:2], fontsize=6, xytext=(3, 3), textcoords="offset points")
        f = fk_frames(p["q"])
        chain = np.array([f[1][:3, 3], f[2][:3, 3], f[3][:3, 3], f[4][:3, 3], c])
        ax1.plot(chain[:, 0], chain[:, 1], color="0.5", lw=0.5, alpha=0.4)
        rad = np.hypot(chain[:, 0], chain[:, 1])
        ax2.plot(rad, chain[:, 2], color=col, lw=0.6, alpha=0.5)
    ax1.set_aspect("equal")
    ax1.set_xlabel("x (mm)")
    ax1.set_ylabel("y (mm)")
    ax1.set_title("top view: camera positions, view lines, arm (grey); red x = skipped")
    ax1.grid(alpha=0.3)
    r_obj = math.hypot(*sc.object_xy)
    ax2.add_patch(plt.Rectangle((r_obj - sc.object_radius, sc.table_z), 2 * sc.object_radius,
                                sc.object_top - sc.table_z, color="tab:orange", alpha=0.6))
    ax2.axhline(sc.table_z, color="k", lw=1)
    ax2.axhline(sc.table_z + sc.table_margin, color="r", lw=0.5, ls="--")
    ax2.set_aspect("equal")
    ax2.set_xlabel("distance from the base axis (mm)")
    ax2.set_ylabel("z (mm)")
    ax2.set_title("side view (every pose unrolled around the base)")
    ax2.grid(alpha=0.3)
    s = plan["summary"]
    fig.suptitle(f"orbit plan: {s['reachable']}/{s['views']} views reachable, J5 span {s['j5_span_deg']:.0f} deg, "
                 f"~{s['est_time_s']:.0f} s")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------- arm connection (real or fake)


class FakeArm:
    """Enough of xarm.wrapper.XArmAPI for rehearsals and tests: moves are instant, kinematics are ours."""

    def __init__(self, q=(0, -20, -60, 80, 0)):
        self.q = np.array(q, float)
        self.mode = self.state = 0
        self.error_code = self.warn_code = 0
        self.tcp_offset = [0, 0, 0, 0, 0, 0]
        self.version = "fake"
        self.moves: list[list[float]] = []

    def _pose(self, q):
        f = fk(q)
        return [*f[:3, 3], *matrix_rpy(f[:3, :3])]

    def get_servo_angle(self, *a, **k):
        return 0, list(self.q) + [0.0, 0.0]

    def get_position(self, *a, **k):
        return 0, self._pose(self.q)

    def get_forward_kinematics(self, angles, *a, **k):
        return 0, self._pose(np.array(angles[:5], float))

    def set_servo_angle(self, angle=None, speed=None, mvacc=None, wait=False, timeout=None, **k):
        self.q = clip_joints(angle[:5])
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


def connect_arm(ip: str, say=print):
    if not ip:
        raise SystemExit("no arm IP: --ip 192.168.1.xxx, or set orbit.arm_ip in scan3d/config.json (--ip fake to rehearse)")
    if ip == "fake":
        say("arm: FAKE (no motion, our kinematics)")
        return FakeArm()
    try:
        from xarm.wrapper import XArmAPI
    except ImportError:
        raise SystemExit("pip install xarm-python-sdk  (into .venv)") from None
    arm = XArmAPI(ip, is_radian=False)
    code, ew = arm.get_err_warn_code()
    say(f"arm {ip}: version {getattr(arm, 'version', '?')}, mode {arm.mode}, state {arm.state}, "
        f"error {ew[0] if code == 0 else '?'}, warning {ew[1] if code == 0 else '?'}")
    return arm


def enable(arm) -> None:
    arm.clean_error()
    arm.clean_warn()
    arm.motion_enable(True)
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(0.3)


def halt(arm) -> None:
    """Stop and hold (same sequence as trova/arm/xarm.py): not a power cut."""
    arm.set_state(4)
    time.sleep(0.2)
    arm.motion_enable(True)
    arm.set_mode(0)
    arm.set_state(0)


def read_joints(arm) -> np.ndarray:
    code, q = arm.get_servo_angle()
    if code != 0 or not q:
        raise RuntimeError(f"get_servo_angle code {code}")
    return np.array([float(v) for v in q[:5]])


# ---------------------------------------------------------------- commands


def cmd_simulate(args, cfg) -> int:
    """Synthetic teach file from a known camera and object: rehearse fit/plan/run without the arm."""
    o = orbit_cfg(cfg)
    rng = np.random.default_rng(args.seed)
    ox, oy = (float(v) for v in args.object.split(","))
    h = float(args.height if args.height is not None else o["object_height_mm"])
    table_z = float(o["table_z_mm"])
    tgt = np.array([ox, oy, table_z + h / 2])
    true_axis = axis_from_angles(args.axis_az, args.axis_el)
    geo = cad_geometry(o)
    cam = Camera(geo.camera(true_axis)[0] if geo is not None else np.array(o["camera_offset_mm"], float), true_axis)
    poses = []
    probe = Camera(np.zeros(3), np.array([0, 0, 1.0]))
    q = solve_view(probe, [ox, oy, table_z + h + 120], [0, 0, -1], [math.atan2(oy, ox) / D2R, 0, -60, 60, 0])
    if q is not None and not args.no_touch:
        poses.append({"kind": "touch", "q": r3(q + rng.normal(0, 0.3, 5)), "tcp": None})
    to_base = math.atan2(-oy, -ox) / D2R
    for k, daz in enumerate((-70, -25, 20, 65)):
        pos, view = view_target(tgt, to_base + daz, 25 + 8 * (k % 2), 240 + 30 * (k % 2))
        for s in [*_generic_seeds(pos), *(np.array(p["q"]) for p in poses)]:
            q = solve_view(cam, pos, view, s, scale=250)
            if q is not None:
                break
        if q is None:
            continue
        poses.append({"kind": "center", "q": r3(q + rng.normal(0, args.noise_deg, 5)), "tcp": None})
    teach = {"arm_ip": "fake", "poses": poses, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
             "truth": {"target_mm": r3(tgt), "camera_axis_flange": [float(v) for v in true_axis],
                       "camera_offset_mm": r3(cam.offset)}}
    out = Path(args.teach)
    save_json(out, teach)
    print(f"synthetic teach file with {len(poses)} poses (object at {r3(tgt)}, true axis az {args.axis_az} "
          f"el {args.axis_el}) -> {out}")
    return 0


def cmd_fit(args, cfg) -> int:
    o = orbit_cfg(cfg)
    teach = load_json(Path(args.teach))
    teach["_path"] = str(args.teach)
    model = fit_model(teach, o, args.height, args.radius, args.fit_offset or None)
    print_fit(model)
    save_json(Path(args.model), model)
    print(f"-> {args.model}")
    return 0


def _elevations(text, model):
    """'15,40' -> [15, 40]; 'auto' -> the camera's own down-tilt (orbit with the flange pointing down)."""
    if not text:
        return None
    return [float(v) if v.strip() != "auto" else model["camera_axis_angles_deg"]["el"] for v in text.split(",")]


def cmd_plan(args, cfg) -> int:
    o = orbit_cfg(cfg)
    model_path = Path(args.model)
    if args.refit or not model_path.exists():
        teach = load_json(Path(args.teach))
        teach["_path"] = str(args.teach)
        model = fit_model(teach, o, args.height, args.radius)
        print_fit(model)
        save_json(model_path, model)
        print()
    else:
        model = load_json(model_path)
    t0 = time.time()
    plan = plan_orbit(model, o, args.distance, _elevations(args.elevations, model), args.step, args.span,
                      args.direction)
    print_plan(plan)
    out = Path(args.plan)
    save_json(out, plan)
    png = out.with_suffix(".png")
    plot_plan(plan, o, png)
    print(f"-> {out}  and  {png}   ({time.time() - t0:.1f} s)")
    return 0


def cmd_check(args, cfg) -> int:
    """No motion: compare our kinematics with the controller's, and say where the camera looks now."""
    arm = connect_arm(args.ip or orbit_cfg(cfg).get("arm_ip", ""))
    try:
        q = read_joints(arm)
        print(f"joints now: {r3(q)}")
        tcp = np.array(getattr(arm, "tcp_offset", None) or [0] * 6, float)
        t_tcp = pose_matrix(tcp)
        worst = 0.0
        tests = [q, q + np.array([10, -10, 15, -20, 30]), np.zeros(5), np.array([30, 10, -80, 40, 45])]
        for qt in tests:
            code, pose = arm.get_forward_kinematics(list(qt), input_is_radian=False, return_is_radian=False)
            if code != 0:
                print(f"  controller FK code {code} for {r3(qt)}")
                continue
            rc = rpy_matrix(*pose[3:6])
            best = None
            for label, ours in (("with TCP offset", fk(qt) @ t_tcp), ("flange", fk(qt))):
                dp = float(np.linalg.norm(ours[:3, 3] - np.array(pose[:3])))
                da = math.acos(max(-1.0, min(1.0, (np.trace(rc.T @ ours[:3, :3]) - 1) / 2))) / D2R
                if best is None or dp < best[0]:
                    best = (dp, da, label)
            dp, da, label = best
            worst = max(worst, dp)
            print(f"  q {r3(qt)}: position diff {dp:6.2f} mm, rotation diff {da:5.2f} deg ({label})")
        print("kinematics OK" if worst < 1.0 else "! our kinematics disagree with the controller: do NOT run plans")
        code, pose = arm.get_position()
        if code == 0:
            print(f"tool pose (controller, TCP offset {r3(tcp)}): {r3(pose)}")
        model_path = Path(args.model)
        if model_path.exists():
            model = load_json(model_path)
            cam = camera_from_model(model)
            p, v = cam.ray(q)
            tgt = np.array(model["target_mm"])
            d = tgt - p
            along = float(v @ d)
            miss = float(np.linalg.norm(d - v * along))
            print(f"camera at {r3(p)} looking {[round(float(x), 3) for x in v]}")
            print(f"object centre {r3(tgt)}: {along:.0f} mm in front, {miss:.1f} mm off the axis "
                  f"({math.atan2(miss, along) / D2R:.1f} deg). Object on the crosshair now? Then the model is right "
                  f"here; if not, centre it by hand and add the pose with teach (SPACE).")
    finally:
        arm.disconnect()
    return 0


class _Grabber(threading.Thread):
    def __init__(self, host: str):
        super().__init__(daemon=True)
        sys.path.insert(0, str(ROOT))
        from trova_cam import TrovaCam
        self.cam = TrovaCam(host)
        self.frame, self.error, self.running = None, "", True

    def run(self):
        while self.running:
            try:
                self.frame, self.error = self.cam.get_frame(), ""
            except Exception as e:  # noqa: BLE001  (CameraError, socket errors)
                self.error = str(e)
                time.sleep(0.5)


def cmd_teach(args, cfg) -> int:
    import cv2

    o = orbit_cfg(cfg)
    path = Path(args.teach)
    teach = load_json(path) if path.exists() and not args.new else {"poses": []}
    teach["arm_ip"] = args.ip or o.get("arm_ip", "")
    teach["camera_host"] = cfg["pi"]["host"]
    teach.pop("truth", None)
    arm = connect_arm(teach["arm_ip"])
    grab = _Grabber(cfg["pi"]["host"])
    grab.start()
    manual, msg = False, "M manual mode | SPACE object on crosshair | T flange above object | U undo | F fit | Q save+quit"
    print(msg)
    win = "orbit teach"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    try:
        while True:
            img = grab.frame.copy() if grab.frame is not None else np.zeros((616, 820, 3), np.uint8)
            h, w = img.shape[:2]
            cv2.line(img, (w // 2 - 60, h // 2), (w // 2 + 60, h // 2), (0, 255, 0), 2)
            cv2.line(img, (w // 2, h // 2 - 60), (w // 2, h // 2 + 60), (0, 255, 0), 2)
            cv2.circle(img, (w // 2, h // 2), int(0.08 * w), (0, 255, 0), 1)
            n_c = sum(p["kind"] == "center" for p in teach["poses"])
            n_t = sum(p["kind"] == "touch" for p in teach["poses"])
            lines = [f"SPACE poses {n_c}  T poses {n_t}  manual {'ON' if manual else 'off'}",
                     grab.error and f"camera: {grab.error[:70]}", msg]
            for k, line in enumerate(x for x in lines if x):
                cv2.putText(img, line, (12, 30 + 28 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                cv2.putText(img, line, (12, 30 + 28 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow(win, img)
            key = cv2.waitKey(40) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("m"):
                manual = not manual
                arm.set_mode(2 if manual else 0)
                arm.set_state(0)
                print("manual mode ON: move the arm by hand" if manual else "manual mode off: the arm holds")
            elif key in (ord(" "), ord("t")):
                q = read_joints(arm)
                code, pose = arm.get_position()
                kind = "center" if key == ord(" ") else "touch"
                teach["poses"].append({"kind": kind, "q": r3(q), "tcp": r3(pose) if code == 0 else None,
                                       "t": time.strftime("%H:%M:%S")})
                save_json(path, teach)
                print(f"{kind} pose {len(teach['poses']) - 1}: q {r3(q)}")
            elif key == ord("u") and teach["poses"]:
                print(f"removed {teach['poses'].pop()}")
                save_json(path, teach)
            elif key == ord("f"):
                try:
                    print_fit(fit_model(dict(teach, _path=str(path)), o))
                except ValueError as e:
                    print(f"fit: {e}")
    finally:
        grab.running = False
        if manual:
            arm.set_mode(0)
            arm.set_state(0)
        arm.disconnect()
        cv2.destroyAllWindows()
        save_json(path, teach)
        print(f"{len(teach['poses'])} poses -> {path}")
    return 0


def cmd_run(args, cfg) -> int:
    o = orbit_cfg(cfg)
    plan = load_json(Path(args.plan))
    model = plan["model"]
    cam, sc = camera_from_model(model), scene_from_model(model, o)
    poses = [p for p in plan["poses"] if p["q"] is not None]
    if args.reverse:
        poses = poses[::-1]
    stops = {int(v) for v in args.only.split(",")} if args.only else {p["i"] for p in poses}
    if args.only:              # travel through the views in between (checked path), stop only at the chosen ones
        idx = [k for k, p in enumerate(poses) if p["i"] in stops]
        if not idx:
            raise SystemExit(f"none of {sorted(stops)} is a reachable view in the plan")
        poses = poses[idx[0]: idx[-1] + 1]
    speed = float(args.speed or o["joint_speed_deg_s"])
    acc = float(o["joint_acc_deg_s2"])
    dwell = float(args.dwell if args.dwell is not None else o["dwell_s"])

    arm = connect_arm(args.ip or o.get("arm_ip", ""))
    shots = None
    try:
        q_now = read_joints(arm)
        first = np.array(poses[0]["q"])
        hits = path_collisions(q_now, first, cam, sc)
        print(f"now {r3(q_now)} -> first view {poses[0]['i']} {r3(first)}: max joint move "
              f"{np.abs(first - q_now).max():.0f} deg")
        if hits:
            print(f"! the straight joint move to the first view would touch: {hits}. Move the arm by hand near view "
                  f"{poses[0]['i']} (or near a taught pose) and run again.")
            return 2
        print(f"{len(poses)} moves, stop at {len([p for p in poses if p['i'] in stops])} views, {speed:.0f} deg/s, "
              f"dwell {dwell} s{', snapshots' if args.snapshots else ''}")
        if not args.yes and not input("Move the arm? Someone at the e-stop. [y/N] ").strip().lower().startswith("y"):
            return 1
        if args.snapshots:
            sys.path.insert(0, str(ROOT))
            from trova_cam import TrovaCam
            sid = time.strftime("%Y%m%d-%H%M%S") + "-orbit"
            shots = {"dir": ROOT / o.get("scans_dir", "data/scans") / sid, "cam": TrovaCam(cfg["pi"]["host"]),
                     "rows": [], "sid": sid}
            (shots["dir"] / "images").mkdir(parents=True, exist_ok=True)
        enable(arm)
        t0 = time.time()
        for k, p in enumerate(poses):
            stop_here = p["i"] in stops
            if args.step and stop_here:
                if input(f"Enter = move to view {p['i']} (az {p['az_deg']}, el {p['el_deg']}), q = quit: ").strip() == "q":
                    break
            code = arm.set_servo_angle(angle=list(p["q"]), speed=speed, mvacc=acc, wait=True, timeout=120)
            if code != 0 or getattr(arm, "error_code", 0):
                print(f"! move to view {p['i']} failed: code {code}, error {getattr(arm, 'error_code', '?')}")
                halt(arm)
                return 3
            if not stop_here:
                continue
            time.sleep(dwell)
            line = f"view {p['i']:3d}  az {p['az_deg']:6.1f}  el {p['el_deg']:5.1f}  ({k + 1}/{len(poses)})"
            if shots:
                jpg = shots["cam"].get_jpeg()
                name = f"{p['i']:03d}.jpg"
                (shots["dir"] / "images" / name).write_bytes(jpg)
                shots["rows"].append({"image": name, **{x: p[x] for x in ("i", "az_deg", "el_deg", "q", "camera_mm",
                                                                         "camera_dir")}})
                line += f"  {name} {len(jpg) // 1024} kB"
            print(line)
        print(f"done in {time.time() - t0:.0f} s")
    except KeyboardInterrupt:
        print("\nCtrl+C: stopping the arm (it holds position)")
        halt(arm)
        return 130
    finally:
        if shots and shots["rows"]:
            save_json(shots["dir"] / "orbit.json", {"scan_id": shots["sid"], "plan": str(args.plan),
                                                    "model": model, "images": shots["rows"]})
            print(f"{len(shots['rows'])} stills -> {shots['dir'] / 'images'}\n"
                  f"next: .venv/Scripts/python -m scan3d.reconstruct {shots['dir'] / 'images'} --mode preview")
        arm.disconnect()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m scan3d.orbit", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--teach", default=str(ORBIT_DIR / "teach.json"))
    ap.add_argument("--model", default=str(ORBIT_DIR / "model.json"))
    ap.add_argument("--plan", default=str(ORBIT_DIR / "plan.json"))
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("teach", help="live view + crosshair, record poses moved by hand")
    s.add_argument("--ip", default="")
    s.add_argument("--new", action="store_true", help="start a new teach file instead of adding to it")

    s = sub.add_parser("fit", help="camera axis + object centre from the taught poses")
    s.add_argument("--height", type=float, help="object height in mm (default orbit.object_height_mm)")
    s.add_argument("--radius", type=float, help="object radius in mm, for collisions (default orbit.object_radius_mm)")
    s.add_argument("--fit-offset", action="store_true", help="also fit the camera offset (needs >= 5 varied poses)")

    s = sub.add_parser("plan", help="orbit plan + picture, no motion")
    s.add_argument("--refit", action="store_true", help="fit the teach file again first")
    s.add_argument("--height", type=float)
    s.add_argument("--radius", type=float)
    s.add_argument("--distance", type=float, help="camera to object centre, mm (default: taught)")
    s.add_argument("--elevations", help="rings, degrees above the object centre, e.g. 15,40 or auto,45 "
                                        "(auto = the camera's own tilt: flange pointing down; default: taught)")
    s.add_argument("--step", type=float, help="degrees between views (default orbit.step_deg)")
    s.add_argument("--span", type=float, help="degrees around the object (default orbit.span_deg = 360)")
    s.add_argument("--direction", type=int, choices=(1, -1), help="1 = counter-clockwise seen from above")

    s = sub.add_parser("check", help="no motion: our kinematics vs the controller, where the camera looks now")
    s.add_argument("--ip", default="")

    s = sub.add_parser("run", help="move along the plan")
    s.add_argument("--ip", default="")
    s.add_argument("--step", action="store_true", help="Enter before every stop")
    s.add_argument("--only", help="stop only at these views, e.g. 0,6,12 (passes through the ones in between)")
    s.add_argument("--snapshots", action="store_true", help="a still from the Pi at every stop -> data/scans/<id>/")
    s.add_argument("--speed", type=float, help="joint speed deg/s (default orbit.joint_speed_deg_s)")
    s.add_argument("--dwell", type=float, help="seconds still at every stop (default orbit.dwell_s)")
    s.add_argument("--reverse", action="store_true", help="run the plan backwards (e.g. to come back)")
    s.add_argument("--yes", action="store_true", help="no confirmation prompt")

    s = sub.add_parser("simulate", help="synthetic teach file for a rehearsal (--ip fake)")
    s.add_argument("--object", default="350,0", help="object centre x,y in mm (robot base frame)")
    s.add_argument("--height", type=float)
    s.add_argument("--axis-az", type=float, default=0.0, help="true camera axis az in the flange frame, deg")
    s.add_argument("--axis-el", type=float, default=-10.0, help="true camera axis el in the flange frame, deg")
    s.add_argument("--noise-deg", type=float, default=0.3, help="hand-centering noise on every joint, deg")
    s.add_argument("--no-touch", action="store_true", help="no T pose (object from rays only)")
    s.add_argument("--seed", type=int, default=1)

    args = ap.parse_args(argv)
    cfg = config_from_args(args)
    return {"teach": cmd_teach, "fit": cmd_fit, "plan": cmd_plan, "check": cmd_check, "run": cmd_run,
            "simulate": cmd_simulate}[args.cmd](args, cfg)


if __name__ == "__main__":
    sys.exit(main())
