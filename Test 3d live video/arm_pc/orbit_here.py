"""Camera orbit around the object, planned from the arm's CURRENT pose. Runs on the PC that drives the xArm 5.

Start pose (set by hand / Studio before running): the Pi camera on the arm looks STRAIGHT DOWN at the object,
object in the middle of the picture. From that pose and the measured distance the script works out where the
object is, then plans views on a cone around it (camera always aimed at the object centre) with joint-space IK
(xArm 5 URDF kinematics, same as scan3d/orbit.py), checks joint limits and collisions, and moves joint by joint.

Needs only numpy + xarm-python-sdk (both already in the Trova repo's dependencies). Copy the whole arm_pc folder:
robot_geometry.py + robot_model.json give the REAL shapes for the collision checks (UFactory's xArm 5 meshes and the
camera head's CAD, "Camera Gatto.step"): table, object, and the head against the arm itself. They also give the
camera's optical centre (42 mm in front of the flange axis, 40 mm out from the flange face) and tell that it looks
perpendicular to the flange axis, so the real view direction at the start pose comes from the arm angles.
Without them (or with --no-cad) it falls back to capsules and a guessed camera offset.

    python orbit_here.py --ip 192.168.1.185                       # DRY RUN: read joints, plan, print. No motion
    python orbit_here.py --ip 192.168.1.185 --go --max-views 2 --pause   # test: 2 views, Enter at each, back to start
    python orbit_here.py --ip 192.168.1.185 --go                  # whole orbit, then back to the start pose
    python orbit_here.py --q0=0,-10,-50,60,0                      # offline plan from typed joints (no arm)

All lengths mm, angles degrees, robot base frame (z up). Ctrl+C stops the arm (it holds position).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

try:
    from robot_geometry import MOVING, RobotGeometry
except ImportError:          # robot_geometry.py not copied: capsule model
    RobotGeometry, MOVING = None, ()

D2R = math.pi / 180.0

# ---------------------------------------------------------------- xArm 5 kinematics (xarm_description URDF)


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


# All joints at 0 -> flange at (207, 0, 112) pointing down, as the controller reports.
_FIXED = [_tr(0, 0, 267.0), _rx(-math.pi / 2), _tr(53.5, -284.5, 0), _tr(77.5, 342.5, 0),
          _tr(76.0, 97.0, 0) @ _rx(-math.pi / 2)]
XARM5_LIMITS = np.array([(-360, 360), (-118, 120), (-225, 11), (-97, 180), (-360, 360)], float)
LIMITS = XARM5_LIMITS.copy()
LIMITS[1, 0] = -60          # J2 >= -60: never lean back behind the base (same as scan3d config)


def fk_frames(q):
    t, out = np.eye(4), []
    for fixed, qi in zip(_FIXED, q):
        t = t @ fixed @ _rz(float(qi) * D2R)
        out.append(t)
    return out


def fk(q):
    return fk_frames(q)[-1]


def clip(q):
    return np.clip(np.asarray(q, float), LIMITS[:, 0], LIMITS[:, 1])


def rpy_matrix(roll, pitch, yaw):
    r, p, y = roll * D2R, pitch * D2R, yaw * D2R
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def pose_matrix(pose):
    t = np.eye(4)
    t[:3, :3] = rpy_matrix(*pose[3:6])
    t[:3, 3] = pose[:3]
    return t


class Camera:
    """Optical centre (offset) and optical axis in the FLANGE frame."""

    def __init__(self, offset, axis):
        self.offset = np.asarray(offset, float)
        self.axis = np.asarray(axis, float) / np.linalg.norm(axis)

    def ray(self, q):
        f = fk(q)
        return f[:3, :3] @ self.offset + f[:3, 3], f[:3, :3] @ self.axis


def solve_view(cam, pos, direction, seed, scale=250.0, iters=60, tol_mm=1.0, tol_deg=0.5):
    """Joints putting the camera at pos looking along direction (Levenberg-Marquardt from seed), or None."""
    pos, direction = np.asarray(pos, float), np.asarray(direction, float)
    tol_dir = 2 * math.sin(tol_deg * D2R / 2)

    def res(q):
        p, v = cam.ray(q)
        return np.concatenate([p - pos, (v - direction) * scale])

    q = clip(seed)
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
            qn = clip(q + step)
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


# ---------------------------------------------------------------- collisions
# Real shapes (robot_geometry.py: xArm meshes + camera head CAD, as spheres) when available, else capsules.


class Scene:
    def __init__(self, obj_xy, obj_radius, obj_top, table_z, table_margin=40.0, obj_margin=30.0,
                 link_radius=45.0, camera_radius=45.0, base_radius=80.0, base_height=330.0,
                 geometry=None, head_turn=None, self_margin=5.0):
        self.obj_xy, self.obj_radius, self.obj_top, self.table_z = np.asarray(obj_xy, float), obj_radius, obj_top, table_z
        self.table_margin, self.obj_margin = table_margin, obj_margin
        self.link_radius, self.camera_radius = link_radius, camera_radius
        self.base_radius, self.base_height = base_radius, base_height
        self.geometry, self.head_turn, self.self_margin = geometry, head_turn, self_margin
        self.boxes = []          # --box keep-outs: (lo xyz, hi xyz), kept obj_margin away like the object
        self.notes = {}


def box_gap(c, r, lo, hi):
    """Gaps (mm) between spheres (centres c, radii r) and an axis-aligned box."""
    c = np.atleast_2d(c)
    return np.linalg.norm(np.clip(c, lo, hi) - c, axis=1) - r


def collisions(q, cam, sc):
    if sc.geometry is not None:
        frames = fk_frames(q)
        hits = sc.geometry.collisions(frames, sc.head_turn, sc.table_z, sc.table_margin, sc.obj_xy,
                                      sc.obj_radius, sc.obj_top, sc.obj_margin, sc.self_margin)
        if sc.boxes:
            w = sc.geometry.world(frames, sc.head_turn)
            for k, (lo, hi) in enumerate(sc.boxes):
                if any((box_gap(*w[n], lo, hi) < sc.obj_margin).any() for n in MOVING):
                    hits.append(f"box {k}")
        return sorted(hits)
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
            if (math.hypot(p[0] - sc.obj_xy[0], p[1] - sc.obj_xy[1]) < sc.obj_radius + rad + sc.obj_margin
                    and p[2] - rad < sc.obj_top + sc.obj_margin):
                hits.add("object")
            for kb, (lo, hi) in enumerate(sc.boxes):
                if box_gap(p, rad, lo, hi)[0] < sc.obj_margin:
                    hits.add(f"box {kb}")
    return sorted(hits)


def path_collisions(q0, q1, cam, sc):
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    steps = max(4, int(math.ceil(np.abs(q1 - q0).max() / 5.0)))
    hits = set()
    for t in np.linspace(0, 1, steps + 1):
        hits.update(collisions(q0 + (q1 - q0) * t, cam, sc))
    return sorted(hits)


# ---------------------------------------------------------------- setup from the start pose


_GEOMETRY: dict = {}


def geometry(a):
    """The CAD collision model next to this script, or None (--no-cad, or the files were not copied)."""
    if a.no_cad or RobotGeometry is None:
        return None
    if "geo" not in _GEOMETRY:
        try:
            _GEOMETRY["geo"] = RobotGeometry.load()
        except (OSError, ValueError) as e:
            print(f"  ! robot_model.json not loaded ({e}): capsule collision model")
            _GEOMETRY["geo"] = None
    return _GEOMETRY["geo"]


def setup_from_start(q0, a):
    """Camera model + object position from: at q0 the camera is aimed (by eye) straight down at the object, a.dist
    away. With the CAD head the camera axis is perpendicular to the flange axis, turned like the measured 'down'
    direction: the real view direction at q0 then follows from the arm angles, not from the aiming by eye."""
    cal = None
    if getattr(a, "calib", None):                   # camera_robot.json (scan3d/handeye.py): the measured camera
        cal = json.loads(Path(a.calib).read_text(encoding="utf-8"))
        T_fc = np.array(cal["T_flange_camera"], float)
        if not a.axis_flange:
            a.axis_flange = [float(v) for v in T_fc[:3, 2]]
        if a.cam_offset is None:
            a.cam_offset = [float(v) for v in T_fc[:3, 3]]
    f0 = fk(q0)
    down = np.array([0.0, 0.0, -1.0])
    geo = geometry(a)
    down_f = f0[:3, :3].T @ down
    if geo is not None and not a.axis_flange and a.axis_from == "cad" and math.hypot(down_f[0], down_f[1]) < 0.5:
        raise SystemExit(f"the flange points {math.degrees(math.acos(min(1.0, abs(down_f[2])))):.0f} deg from "
                         "vertical at the start pose, but the camera looks sideways from the flange (CAD): it cannot "
                         "be looking straight down. Aim the camera at the object from above first.")
    if a.axis_flange:
        axis_f, axis_note = np.array(a.axis_flange, float), "--axis-flange"
    elif geo is not None and a.axis_from == "cad":
        raw = geo.turn_angle(down_f)
        turn = geo.turn_angle(down_f, a.head_snap) if a.head_turn is None else a.head_turn
        axis_f = geo.head_turn([math.cos(math.radians(turn)), math.sin(math.radians(turn)), 0]) @ geo.cam_axis
        how = ("--head-turn" if a.head_turn is not None else
               f"start pose says {raw:.1f}, bolts allow multiples of {a.head_snap:g}" if a.head_snap else "start pose")
        axis_note = f"CAD (perpendicular to the flange), head turned {turn:.0f} deg on the flange ({how})"
        if a.head_turn is None and a.head_snap and abs(((raw - turn) + 180) % 360 - 180) > 15:
            axis_note += "  ! start pose more than 15 deg off: check the aim, or give --head-turn"
    else:
        axis_f, axis_note = down_f, "start pose = straight down"
    if a.cam_offset is not None:
        offset, off_note = a.cam_offset, "--cam-offset"
    elif geo is not None:
        offset, dev = geo.camera(axis_f)
        off_note = "CAD head" + (f" ({dev:.1f} deg between this axis and the CAD one)" if dev > 0.5 else "")
    else:
        offset, off_note = [0.0, 0.0, 60.0], "guess (no CAD)"
    cam = Camera(offset, axis_f)
    p0, v0 = cam.ray(q0)
    if getattr(a, "target", None):                  # look-at point given in the robot frame
        target = np.array(a.target, float) + np.array(a.shift, float)
    else:
        target = p0 + a.dist * v0 + np.array(a.shift, float)    # object centre at mid height
    if getattr(a, "table_z", None) is not None:
        table_z = float(a.table_z)
    elif cal is not None and getattr(a, "target", None):
        table_z = float(cal["table_z_mm"])
    else:
        table_z = target[2] - a.obj_height / 2
    tm = a.table_margin if a.table_margin is not None else (15.0 if geo is not None else 40.0)
    om = a.obj_margin if a.obj_margin is not None else (20.0 if geo is not None else 30.0)
    sc = Scene(target[:2], a.obj_radius, table_z + a.obj_height, table_z, table_margin=tm, obj_margin=om,
               geometry=geo, head_turn=geo.head_turn(axis_f) if geo is not None else None, self_margin=a.self_margin)
    for x0, y0, x1, y1, ztop in getattr(a, "box", None) or []:
        sc.boxes.append((np.array([min(x0, x1), min(y0, y1), table_z]), np.array([max(x0, x1), max(y0, y1), ztop])))
    if cal is not None:
        axis_note, off_note = f"calibration {Path(a.calib).name}", f"calibration {Path(a.calib).name}"
    sc.notes = {"axis": axis_note, "offset": off_note,
                "start_tilt_deg": math.degrees(math.acos(max(-1.0, min(1.0, float(-v0[2])))))}
    return cam, sc, target, p0, v0


def generic_seeds(pos):
    a = math.atan2(pos[1], pos[0]) / D2R
    out = []
    for q2, q3 in ((0, -60), (25, -85), (45, -115), (-15, -35)):
        for tilt in (0, 45, -45, 90, -90):
            for q5 in (-180, -90, 0, 90, 180):
                out.append(np.array([a, q2, q3, -(q2 + q3) + tilt, q5], float))
    return out


def view_pose(target, az, el, dist):
    u = np.array([math.cos(el * D2R) * math.cos(az * D2R), math.cos(el * D2R) * math.sin(az * D2R), math.sin(el * D2R)])
    return target + dist * u, -u


def solve_one(cam, sc, target, az, el, dist, prev, q0):
    """Collision-free solutions for one view, nearest to prev first (J5 +-360 variants included)."""
    pos, view = view_pose(target, az, el, dist)
    sols = []
    seeds = [prev, q0]
    for stage in (seeds, generic_seeds(pos)):
        for s in stage:
            q = solve_view(cam, pos, view, s, scale=dist)
            if q is None or collisions(q, cam, sc):
                continue
            for k in (0, -360, 360):
                qq = q.copy()
                qq[4] += k
                if LIMITS[4, 0] <= qq[4] <= LIMITS[4, 1] and all(np.abs(qq - x).max() > 3 for x in sols):
                    sols.append(qq)
        if sols:
            break
    sols.sort(key=lambda x: float(np.abs(x - prev).max()) + 0.2 * abs(x[4] - q0[4]) / 10)
    return sols


def per(values, k):
    """--orbit-dist / --dwell take one value or one per ring / view: the k-th, or the last one."""
    v = values if isinstance(values, (list, tuple)) else [values]
    return float(v[min(k, len(v) - 1)])


def plan(q0, a, start_az, direction):
    cam, sc, target, p0, v0 = setup_from_start(q0, a)
    n = int(round(a.span / a.step)) if a.span >= 360 else int(round(a.span / a.step)) + 1
    views = []
    for ri, tilt in enumerate(a.tilts):
        sgn = direction if ri % 2 == 0 else -direction
        start = start_az if ri % 2 == 0 else start_az + direction * a.step * (n - 1)
        views += [(ri, 90.0 - tilt, start + sgn * k * a.step) for k in range(n)]
    poses, prev = [], np.asarray(q0, float)
    pref0 = None                       # --branch-from-view: pick every view's arm configuration like THAT view's
    bv = getattr(a, "branch_from_view", None)
    if bv is not None and 0 <= bv < len(views):
        rb, eb, zb = views[bv]
        cands = solve_one(cam, sc, target, zb, eb, per(a.orbit_dist, rb), prev, q0)
        pref0 = cands[0] if cands else None
    for i, (ri, el, az) in enumerate(views):
        chosen, note = None, "out of reach / collision"
        pref = pref0 if (i == 0 and pref0 is not None) else prev
        for q in solve_one(cam, sc, target, az, el, per(a.orbit_dist, ri), pref, q0):
            hits = path_collisions(prev, q, cam, sc)
            if not hits:
                chosen = q
                break
            note = f"no collision-free move to it ({', '.join(hits)})"
        if chosen is None:
            poses.append({"i": i, "ring": ri, "az": round(az % 360, 1), "el": round(el, 1), "q": None, "note": note})
            continue
        p, w = cam.ray(chosen)
        poses.append({"i": i, "ring": ri, "az": round(az % 360, 1), "el": round(el, 1),
                      "q": [round(float(x), 2) for x in chosen], "camera_mm": [round(float(x), 1) for x in p],
                      "jump_deg": round(float(np.abs(chosen - prev).max()), 1)})
        prev = chosen
    ok = [p for p in poses if p["q"]]
    back = path_collisions(prev, q0, cam, sc) if ok else []
    qs = np.array([q0] + [p["q"] for p in ok] + [q0])
    travel = float(np.sum(np.abs(np.diff(qs, axis=0)).max(axis=1)))
    j5 = qs[:, 4]
    return {"cam": cam, "scene": sc, "target": target, "p0": p0, "v0": v0, "poses": poses,
            "reachable": len(ok), "travel_deg": travel, "j5_span": float(j5.max() - j5.min()),
            "return_hits": back, "start_az": start_az, "direction": direction,
            "est_s": travel / a.speed * 1.4 + sum(per(a.dwell, k) + 0.4 for k in range(len(ok)))}


def best_plan(q0, a):
    if a.start_az is not None:
        starts = [a.start_az]
    else:
        base_dir = math.atan2(-fk(q0)[1, 3], -fk(q0)[0, 3]) / D2R      # from the object towards the robot base
        starts = [base_dir + k for k in (0, 90, 180, 270)]
    dirs = [a.direction] if a.direction else [1, -1]
    best = None
    for s in starts:
        for d in dirs:
            p = plan(q0, a, s, d)
            key = (-p["reachable"], bool(p["return_hits"]), p["j5_span"] > a.j5_max, p["travel_deg"])
            if best is None or key < best[0]:
                best = (key, p)
    return best[1]


def describe(q0, pl, a):
    cam, sc, tgt, p0 = pl["cam"], pl["scene"], pl["target"], pl["p0"]
    ang = math.acos(abs(float(cam.axis @ np.array([0, 0, 1.0])))) / D2R
    print(f"start joints      {[round(float(x), 2) for x in q0]}")
    print(f"camera at start   {[round(float(x), 1) for x in p0]}   axis in flange frame {np.round(cam.axis, 3).tolist()} "
          f"({ang:.0f} deg from the flange axis): {sc.notes['axis']}")
    print(f"camera offset     {np.round(cam.offset, 1).tolist()} mm in the flange frame: {sc.notes['offset']}")
    tilt = sc.notes["start_tilt_deg"]
    print(f"start view        {tilt:.1f} deg away from straight down"
          + ("" if sc.notes["axis"].startswith("start") else " (from the arm angles and the CAD)"))
    if tilt > 10:
        print("  ! the start pose is far from straight down: the object is placed along the REAL view line, which is "
              "right if the object is on the crosshair; if not, fix the start pose (scan3d.aim_check)")
    at_start = collisions(q0, cam, sc)
    if at_start:
        print(f"  ! the START pose already counts as touching {at_start}: every move out of it is refused. Move the "
              "arm a little away (object, table, or wrist less bent) and run again"
              + ("; or lower --self-margin" if any(h.startswith("self") for h in at_start) else ""))
    if sc.geometry is not None:
        print(f"collisions        real shapes: xArm 5 meshes + camera head CAD; margins table {sc.table_margin:.0f}, "
              f"object {sc.obj_margin:.0f}, arm/head {sc.self_margin:.0f} mm")
    else:
        print(f"collisions        capsules (no CAD model); margins table {sc.table_margin:.0f}, object {sc.obj_margin:.0f} mm")
    for k, (lo, hi) in enumerate(sc.boxes):
        print(f"keep-out box {k}    x {lo[0]:.0f}..{hi[0]:.0f}  y {lo[1]:.0f}..{hi[1]:.0f}  up to z {hi[2]:.0f} "
              f"(+{sc.obj_margin:.0f} mm margin)")
    if ang < 25:
        print("  ! the camera looks almost ALONG the flange axis: a 5-axis arm cannot aim it around an object; "
              "expect skipped views")
    print(f"object centre     {[round(float(x), 1) for x in tgt]}  ({math.hypot(tgt[0], tgt[1]):.0f} mm from the base axis)")
    print(f"table z           {sc.table_z:.0f} mm in the robot frame "
          f"(should be about 0 if the arm stands on the same table: else check --dist)")
    print(f"orbit             rings at {a.tilts} deg from vertical, camera {'/'.join(f'{per(a.orbit_dist, k):.0f}' for k in range(len(a.tilts)))} mm from the object, "
          f"every {a.step:.0f} deg over {a.span:.0f} deg, start az {pl['start_az'] % 360:.0f}, "
          f"{'counter-clockwise' if pl['direction'] > 0 else 'clockwise'} seen from above")
    for p in pl["poses"]:
        if p["q"] is None:
            print(f"  {p['i']:3d}  az {p['az']:6.1f}  el {p['el']:5.1f}   SKIPPED: {p['note']}")
        else:
            print(f"  {p['i']:3d}  az {p['az']:6.1f}  el {p['el']:5.1f}  q [{', '.join(f'{v:7.1f}' for v in p['q'])}]"
                  f"  cam z {p['camera_mm'][2]:5.0f}  jump {p['jump_deg']:5.1f}")
    print(f"reachable {pl['reachable']}/{len(pl['poses'])}, J5 turns {pl['j5_span']:.0f} deg in total "
          f"(check the camera cable), ~{pl['est_s']:.0f} s")
    if pl["j5_span"] > a.j5_max:
        print(f"  ! J5 span above --j5-max {a.j5_max:.0f}: use a smaller --span (e.g. 270)")
    if pl["return_hits"]:
        print(f"  ! the move back to the start would touch {pl['return_hits']}: use --no-return and bring it back by hand")


# ---------------------------------------------------------------- arm


def connect(ip):
    from xarm.wrapper import XArmAPI
    arm = XArmAPI(ip, is_radian=False)
    code, ew = arm.get_err_warn_code()
    print(f"arm {ip}: version {getattr(arm, 'version', '?')}, mode {arm.mode}, state {arm.state}, "
          f"error {ew[0] if code == 0 else '?'}, warning {ew[1] if code == 0 else '?'}")
    return arm


def read_joints(arm):
    code, q = arm.get_servo_angle()
    if code != 0 or not q:
        raise RuntimeError(f"get_servo_angle code {code}")
    return np.array([float(v) for v in q[:5]])


def check_kinematics(arm, q0):
    """Our FK vs the controller's, at the start pose and a few others. Returns the worst position error (mm)."""
    tcp = np.array(getattr(arm, "tcp_offset", None) or [0] * 6, float)
    worst = 0.0
    for qt in (q0, q0 + np.array([10, -10, 15, -20, 30]), np.zeros(5)):
        code, pose = arm.get_forward_kinematics(list(qt) + [0, 0], input_is_radian=False, return_is_radian=False)
        if code != 0:
            print(f"  controller FK code {code}")
            return 1e9
        rc = rpy_matrix(*pose[3:6])
        res = []
        for ours in (fk(qt) @ pose_matrix(tcp), fk(qt)):
            dp = float(np.linalg.norm(ours[:3, 3] - np.array(pose[:3])))
            da = math.acos(max(-1.0, min(1.0, (np.trace(rc.T @ ours[:3, :3]) - 1) / 2))) / D2R
            res.append((dp, da))
        dp, da = min(res)
        worst = max(worst, dp)
        print(f"  FK check q {[round(float(x), 1) for x in qt]}: {dp:.2f} mm, {da:.2f} deg")
    return worst


def enable(arm):
    arm.clean_error()
    arm.clean_warn()
    arm.motion_enable(True)
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(0.3)


def halt(arm):
    arm.set_state(4)
    time.sleep(0.2)
    arm.motion_enable(True)
    arm.set_mode(0)
    arm.set_state(0)


def move(arm, q, a, label):
    code = arm.set_servo_angle(angle=[float(x) for x in q], speed=a.speed, mvacc=a.acc, wait=True, timeout=120)
    if code != 0 or getattr(arm, "error_code", 0):
        print(f"! move to {label} failed: code {code}, error {getattr(arm, 'error_code', '?')}")
        halt(arm)
        return False
    return True


class Snapshots:
    """--snapshots: a still from the camera + the joints READ from the arm at the start pose and at every view
    (after the dwell, the arm standing still). For the camera <-> robot calibration (scan3d/handeye.py)."""

    def __init__(self, url, folder):
        self.url, self.folder, self.rows = url, Path(folder), []
        self.folder.mkdir(parents=True, exist_ok=True)

    def take(self, arm, view, fetch=None):
        import urllib.request
        q = read_joints(arm)
        try:
            data = fetch() if fetch else urllib.request.urlopen(self.url, timeout=10).read()
        except OSError as e:
            print(f"  ! snapshot at view {view}: {e}")
            return
        name = f"{view}.jpg" if isinstance(view, str) else f"{int(view):03d}.jpg"
        (self.folder / name).write_bytes(data)
        self.rows.append({"view": view, "file": name, "q": [round(float(x), 4) for x in q],
                          "t": time.strftime("%H:%M:%S")})


def run(arm, q0, pl, a):
    poses = [p for p in pl["poses"] if p["q"]]
    if a.max_views:
        poses = poses[:a.max_views]
    print(f"\n{len(poses)} views at {a.speed:.0f} deg/s, dwell {a.dwell} s, "
          f"{'back to the start pose at the end' if a.ret else 'NO return'}"
          f"{', snapshot at every view -> ' + a.snapshots if a.snapshots else ''}")
    if input("Move the arm? Someone at the e-stop. [y/N] ").strip().lower()[:1] != "y":
        return 1
    snaps = Snapshots(a.snapshot_url, a.snapshots) if a.snapshots else None
    if snaps:
        snaps.take(arm, "start")
    enable(arm)
    t0 = time.time()
    log = []
    try:
        print(f"START {time.strftime('%H:%M:%S')}  (press Enter in scan3d.capture now)")
        for k, p in enumerate(poses):
            if a.pause and input(f"Enter = view {p['i']} (az {p['az']}, el {p['el']}), q = stop: ").strip() == "q":
                break
            if not move(arm, p["q"], a, f"view {p['i']}"):
                return 3
            t_arrive = time.time() - t0
            time.sleep(per(a.dwell, k))
            if snaps:
                snaps.take(arm, p["i"])
            row = {"i": p["i"], "t_arrive": round(t_arrive, 3), "q": p["q"]}
            try:
                row["q_read"] = [round(float(x), 4) for x in read_joints(arm)]
            except RuntimeError:
                pass
            row["t"] = round(time.time() - t0, 2)       # [t_arrive, t] = standing still: scan3d/armpose.py
            log.append(row)
            print(f"  view {p['i']:3d}  az {p['az']:6.1f}  el {p['el']:5.1f}  ({k + 1}/{len(poses)})  {time.time() - t0:5.1f} s")
        print(f"ORBIT END {time.strftime('%H:%M:%S')}  {time.time() - t0:.0f} s  (press Enter in scan3d.capture now)")
        if a.ret and not pl["return_hits"]:
            move(arm, q0, a, "start pose")
            print("back at the start pose")
    except KeyboardInterrupt:
        print("\nCtrl+C: stopping (the arm holds position)")
        halt(arm)
        return 130
    finally:
        out = {"q0": [float(x) for x in q0], "start_pc": round(t0, 4), "dwell_s": list(a.dwell), "views": log}
        if snaps:
            out["snapshots"] = {"folder": str(snaps.folder), "url": snaps.url, "rows": snaps.rows}
            (snaps.folder / "snapshots.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
            print(f"{len(snaps.rows)} snapshots -> {snaps.folder}")
        Path(a.log).write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


# ---------------------------------------------------------------- main


def floats(text, n=None):
    v = [float(x) for x in text.split(",")]
    if n and len(v) != n:
        raise argparse.ArgumentTypeError(f"need {n} comma-separated numbers")
    return v


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ip", default=os.environ.get("XARM_IP"),
                    help="xArm IP: reads the start pose from the arm (default: env XARM_IP)")
    ap.add_argument("--q0", type=lambda s: floats(s, 5), help="start joints J1..J5 (offline plan, no arm)")
    ap.add_argument("--go", action="store_true", help="MOVE the arm (default: dry run, print the plan)")
    ap.add_argument("--dist", type=float, default=432,
                    help="camera to the object's mid height at the start pose (measured from the markers: 432)")
    ap.add_argument("--orbit-dist", type=lambda t: floats(t), default=[350.0],
                    help="camera to object centre during the orbit; one per ring allowed (e.g. 250,100,250)")
    ap.add_argument("--tilts", type=lambda s: floats(s), default=[40.0],
                    help="rings, degrees from vertical (e.g. 35,60 for two rings)")
    ap.add_argument("--step", type=float, default=20, help="degrees between views")
    ap.add_argument("--span", type=float, default=360, help="degrees around the object")
    ap.add_argument("--start-az", type=float, help="first view azimuth (default: best of 4)")
    ap.add_argument("--direction", type=int, choices=(1, -1), help="1 = counter-clockwise from above (default: best)")
    ap.add_argument("--shift", type=lambda s: floats(s, 3), default=[0.0, 0.0, 0.0],
                    help="move the look-at point, robot frame dx,dy,dz mm (correct an off-centre object)")
    ap.add_argument("--cam-offset", type=lambda s: floats(s, 3),
                    help="camera optical centre in the flange frame, mm (z = out of the flange); default from the CAD")
    ap.add_argument("--axis-flange", type=lambda s: floats(s, 3), help="camera axis in the flange frame (default: "
                    "see --axis-from)")
    ap.add_argument("--axis-from", choices=("cad", "start"), default="cad",
                    help="cad: perpendicular to the flange as in the CAD, turned like the start pose (default); "
                         "start: exactly the start pose's straight-down direction")
    ap.add_argument("--head-snap", type=float, default=45,
                    help="the head can sit on the flange only every 90 deg (bolts), at 0 or 45 deg from the flange x "
                         "axis: round the turn read from the start pose to this step (0 = no rounding)")
    ap.add_argument("--head-turn", type=float, help="the head's turn on the flange in deg, if known (skips the guess)")
    ap.add_argument("--no-cad", action="store_true", help="ignore robot_model.json: capsules + guessed camera offset")
    ap.add_argument("--obj-height", type=float, default=35, help="object height (mouse 35)")
    ap.add_argument("--obj-radius", type=float, default=140, help="keep-out radius around the object (mouse + marker cubes)")
    ap.add_argument("--table-margin", type=float, help="nothing goes lower than table + this (default 15 with "
                    "the CAD shapes, 40 with capsules)")
    ap.add_argument("--obj-margin", type=float, help="gap kept from the object (default 20 CAD / 30 capsules)")
    ap.add_argument("--self-margin", type=float, default=5, help="gap kept between the head and the arm, mm")
    ap.add_argument("--speed", type=float, default=15, help="joint speed deg/s")
    ap.add_argument("--acc", type=float, default=60, help="joint acceleration deg/s^2")
    ap.add_argument("--dwell", type=lambda t: floats(t), default=[0.6],
                    help="seconds still at every view (sharp frames); one per view allowed (e.g. 2,30,2)")
    ap.add_argument("--j5-max", type=float, default=400, help="warn when J5 turns more than this in total")
    ap.add_argument("--max-views", type=int, default=0, help="only the first N views (tests)")
    ap.add_argument("--pause", action="store_true", help="Enter before every view")
    ap.add_argument("--no-return", dest="ret", action="store_false", help="stay at the last view")
    ap.add_argument("--force", action="store_true", help="move even if the FK check disagrees (do not)")
    ap.add_argument("--out", help="plan file (default runs/<date-time>_plan.json next to this script)")
    ap.add_argument("--log", help="log of the views reached (default runs/<date-time>_log.json)")
    ap.add_argument("--calib", help="camera_robot.json from scan3d/handeye.py: use the MEASURED camera (axis + "
                    "optical centre in the flange frame) and its table height instead of the CAD guess")
    ap.add_argument("--target", type=lambda t: floats(t, 3), help="look-at point x,y,z in the robot frame (mm), e.g. "
                    "from calib/pix2robot.py; replaces --dist. With --tilts 0 --span 0 --orbit-dist H: one view "
                    "straight down from H mm above it")
    ap.add_argument("--branch-from-view", type=int, help="solve this view first (e.g. the lowest) and keep its arm "
                    "configuration for all views: a wrist flip then happens on the way to view 0, not next to the object")
    ap.add_argument("--table-z", type=float, help="table height in the robot frame (default: --calib's, else from "
                    "--dist and --obj-height)")
    ap.add_argument("--box", action="append", type=lambda t: floats(t, 5), metavar="X0,Y0,X1,Y1,ZTOP",
                    help="keep-out box in the robot frame (mm), from the table up to ZTOP, kept --obj-margin away; "
                    "repeatable. For things taller than the object: the marker cubes (53 mm) next to a 35 mm mouse "
                    "stick out of the --obj-radius cylinder, which is only as tall as the object. Write --box=-150,... "
                    "when the first number is negative")
    ap.add_argument("--snapshots", nargs="?", const="auto", help="save a camera still + the joints read at the start "
                    "pose and at every view (default folder runs/<date-time>_snaps); use --dwell 1.5 or more")
    ap.add_argument("--snapshot-url", default=os.environ.get("CAMERA_SNAPSHOT_URL",
                                                             "http://192.168.5.160:8080/snapshot.jpg"))
    a = ap.parse_args(argv)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    runs = Path(__file__).resolve().parent / "runs"
    if a.out is None or a.log is None:
        runs.mkdir(exist_ok=True)
    a.out = a.out or str(runs / f"{stamp}_plan.json")
    a.log = a.log or str(runs / f"{stamp}_log.json")
    if a.snapshots == "auto":
        runs.mkdir(exist_ok=True)
        a.snapshots = str(runs / f"{stamp}_snaps")

    if not a.ip and a.q0 is None:
        ap.error("--ip (read the arm) or --q0 (offline)")
    arm = connect(a.ip) if a.ip else None
    try:
        q0 = np.array(a.q0, float) if a.q0 is not None else read_joints(arm)
        if arm is not None:
            worst = check_kinematics(arm, q0)
            print("kinematics OK" if worst < 2.0 else f"! kinematics differ by {worst:.1f} mm from the controller")
            if worst >= 2.0 and a.go and not a.force:
                print("not moving: our kinematics disagree with the controller")
                return 2
        t = time.time()
        pl = best_plan(q0, a)
        describe(q0, pl, a)
        print(f"(planned in {time.time() - t:.1f} s)")
        Path(a.out).write_text(json.dumps({
            "q0": [float(x) for x in q0], "args": {k: v for k, v in vars(a).items() if k not in ("ip",)},
            "target_mm": [float(x) for x in pl["target"]], "table_z_mm": pl["scene"].table_z,
            "camera_axis_flange": [float(x) for x in pl["cam"].axis],
            "camera_offset_flange": [float(x) for x in pl["cam"].offset],
            "collision_model": "cad" if pl["scene"].geometry is not None else "capsules", "poses": pl["poses"],
            "reachable": pl["reachable"], "j5_span_deg": pl["j5_span"], "start_az": pl["start_az"],
            "direction": pl["direction"], "time": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1), encoding="utf-8")
        print(f"-> {a.out}")
        if not a.go:
            print("dry run: nothing moved. Add --go to move.")
            return 0
        if arm is None:
            print("--go needs --ip")
            return 2
        if not pl["reachable"]:
            return 2
        now = read_joints(arm)
        if np.abs(now - q0).max() > 0.5:
            print("the arm moved since the plan was made: run again")
            return 2
        return run(arm, q0, pl, a)
    finally:
        if arm is not None:
            arm.disconnect()


if __name__ == "__main__":
    sys.exit(main())
