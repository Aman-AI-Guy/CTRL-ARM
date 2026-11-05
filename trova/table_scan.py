"""Voice table scan with the V2 camera model: look around at a fixed height, map every object in robot millimetres,
find one by words, ask when several match, then place the camera above the chosen one.

Geometry (all from the V2 calibration file, config/calibration_candidate.json, SHA-256 pinned):
- the xArm 5 forward kinematics and the camera's pose on the flange (scan3d.flange.FlangeCalibration);
- the table plane z = a x + b y + c in the robot base frame.

Look pose: TCP at TROVA_TABLE_SCAN_Z_MM (default 600), image centre aimed at the table, image horizon level. The
tilt is the shallowest whose top image edge (corners included) still lands on the table within
TROVA_TABLE_SCAN_FAR_MM of the robot base, so the room around the table stays out of the picture. The scan turns
only J1 between views: every view has the same height and tilt.

Object position: the bottom-centre of each box is where the object meets the table (the camera looks down and
forward); its ray meets the V2 table plane. The footprint centre is that point pushed away from the camera by half
the measured width. This is an estimate from one calibrated image, not a measurement of the object.

Camera over object: joints with the object at the image centre, the camera TROVA_TABLE_SCAN_HOVER_MM away and
looking as straight down as the 5-axis arm allows, the head sphere above the tallest assumed object.

V2 is not physically qualified (docs/M62_QUALIFICATION.md). Simulation and dry-run always work. Hardware motion
additionally needs TROVA_TABLE_SCAN_V2_HARDWARE=accept-unqualified-v2, set by the operator, on top of the existing
TROVA_MOTION_MODE=hardware + TROVA_HARDWARE_UNLOCK=MOVE, workspace limits and per-move confirmation. Every move still
goes through CartesianArm.go_joints and so through the MotionGateway.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .grok import GrokClient, register_mock
from .motion import XARM5_JOINT_LIMITS_DEG
from .scan3d.flange import FlangeCalibration, flange_pose

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "config" / "calibration_candidate.json"
HARDWARE_ACK = "accept-unqualified-v2"
MARGIN_DEG = 2.0
LO = np.array([lo for lo, _hi in XARM5_JOINT_LIMITS_DEG]) + MARGIN_DEG
HI = np.array([hi for _lo, hi in XARM5_JOINT_LIMITS_DEG]) - MARGIN_DEG
UNKNOWN = {"", "unknown", "nothing", "none", "n/a", "?"}
# Real arm, 2026-11-05: the xArm controller stopped the folded-wrist scan pose (J4 -> 174.5) with C22 self-collision
# at J4 = 164 deg (camera head against the arm). Every solver here rejects J4 above this and prefers the
# straight-wrist family (J4 ~ -17, J5 ~ +44), which gives the same view with the flange pointing forward.
WRIST_FOLD_MAX_DEG = 120.0
STRAIGHT_WRIST = (-55.0, -27.0, -17.0, 44.0)          # J2..J5 seed of the straight-wrist look pose


# ---------------------------------------------------------------- settings
@dataclass
class Settings:
    z_mm: float = 600.0                 # TCP height while looking around
    center_deg: float | None = None     # base bearing of the middle of the table (0 = +X, -90 = -Y);
                                        # None = the bearing of the start (rest) pose: the table is under it
    pan_deg: list[float] = field(default_factory=lambda: [-60.0, -30.0, 0.0, 30.0, 60.0])   # J1 offsets
    far_mm: float = 800.0               # the top image edge must land on the table within this radius
    hover_mm: float = 300.0             # camera-to-object distance when the camera is placed above it
    object_max_mm: float = 120.0        # tallest object assumed under the camera head
    joint_speed_deg_s: float = 10.0     # cap for every table-scan move
    settle_s: float = 0.8               # wait after arriving before the picture
    merge_mm: float = 60.0              # two sightings closer than this are the same object
    clearance_mm: float = 30.0          # camera head surface above the table along every path


def _floats(raw: str) -> list[float]:
    return [float(v) for v in raw.replace(";", ",").split(",") if v.strip()]


def settings(env: dict | None = None) -> Settings:
    env = os.environ if env is None else env
    s = Settings()
    num = {"z_mm": "TROVA_TABLE_SCAN_Z_MM", "center_deg": "TROVA_TABLE_SCAN_CENTER_DEG",
           "far_mm": "TROVA_TABLE_SCAN_FAR_MM", "hover_mm": "TROVA_TABLE_SCAN_HOVER_MM",
           "object_max_mm": "TROVA_TABLE_SCAN_OBJECT_MAX_MM", "joint_speed_deg_s": "TROVA_TABLE_SCAN_SPEED_DEG_S",
           "settle_s": "TROVA_TABLE_SCAN_SETTLE_S", "merge_mm": "TROVA_TABLE_SCAN_MERGE_MM"}
    for attr, key in num.items():
        raw = str(env.get(key, "") or "").strip()
        if raw:
            setattr(s, attr, float(raw))
    raw = str(env.get("TROVA_TABLE_SCAN_PAN_DEG", "") or "").strip()     # "from,to,step" relative to the centre
    if raw:
        a, b, step = _floats(raw)
        if step <= 0 or b < a:
            raise ValueError("TROVA_TABLE_SCAN_PAN_DEG must be from,to,step with from <= to and step > 0")
        s.pan_deg = [round(a + i * step, 6) for i in range(int(math.floor((b - a) / step + 1e-9)) + 1)]
    s.joint_speed_deg_s = min(max(1.0, s.joint_speed_deg_s), 20.0)       # never faster than 20 deg/s
    if not 150.0 <= s.hover_mm <= 600.0:
        raise ValueError("TROVA_TABLE_SCAN_HOVER_MM must be between 150 and 600")
    return s


def hardware_acknowledged(env: dict | None = None) -> bool:
    env = os.environ if env is None else env
    return str(env.get("TROVA_TABLE_SCAN_V2_HARDWARE", "")).strip() == HARDWARE_ACK


# ---------------------------------------------------------------- V2 calibration
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_v2(manifest: Path = MANIFEST) -> tuple[FlangeCalibration, dict]:
    """The explicitly selected V2 artifact, refused if its bytes differ from the pinned SHA-256."""
    sel = json.loads(Path(manifest).read_text(encoding="utf-8"))
    path = Path(manifest).resolve().parents[1] / sel["candidate"]
    digest = sha256(path)
    if digest != sel["sha256"]:
        raise ValueError(f"V2 calibration {path.name} differs from the pinned SHA-256: refusing to use it")
    return FlangeCalibration.load(path), {"path": sel["candidate"], "sha256": digest,
                                          "physical_qualification": bool(sel.get("physical_qualification"))}


def for_frame(cal: FlangeCalibration, width: int, height: int) -> FlangeCalibration:
    return cal if (width, height) == (cal.width, cal.height) else cal.for_size(width, height)


# ---------------------------------------------------------------- solvers
def _solve_all(fun, seeds) -> list[np.ndarray]:
    """One local solution per seed. Several arm configurations can satisfy the same aim; callers filter them."""
    from scipy.optimize import least_squares   # reconstruction extra; imported only when a pose is planned
    return [least_squares(fun, np.clip(np.asarray(s0, float), LO, HI), bounds=(LO, HI)).x for s0 in seeds]


def _wrap(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def _nearest(solutions: list[np.ndarray], ref) -> np.ndarray | None:
    ref = np.asarray(ref, float)
    return min(solutions, key=lambda q: float(np.max(np.abs(q - ref)))) if solutions else None


def _centre_dir_cam(cal: FlangeCalibration) -> np.ndarray:
    """Direction (camera frame) of the ray through the image centre, lens distortion removed."""
    n = cv2.undistortPoints(np.array([[[cal.width / 2, cal.height / 2]]], float), cal.K, cal.dist).reshape(2)
    d = np.array([n[0], n[1], 1.0])
    return d / np.linalg.norm(d)


def _aim(cal: FlangeCalibration, dcam: np.ndarray, target: np.ndarray, q) -> tuple[np.ndarray, float, np.ndarray]:
    """Miss vector (mm, perpendicular to the image-centre ray), distance along that ray, camera pose.

    The image-centre RAY is used, not a projection: OpenCV's distortion polynomial folds points far outside the
    field of view back toward the image centre, so 'target projects to the centre' can be satisfied by a pose
    that looks somewhere else entirely."""
    cam = cal.camera_pose(q)
    d = cam[:3, :3] @ dcam
    v = target - cam[:3, 3]
    along = float(v @ d)
    return v - along * d, along, cam


def aim_error_deg(cal: FlangeCalibration, target, q) -> float:
    """Angle between the image-centre ray and the direction to target (degrees)."""
    miss, along, _cam = _aim(cal, _centre_dir_cam(cal), np.asarray(target, float), q)
    return float(math.degrees(math.atan2(np.linalg.norm(miss), along)))


def look_pose(cal: FlangeCalibration, bearing_deg: float, z_mm: float, aim_mm: float, seed=None) -> np.ndarray:
    b = math.radians(bearing_deg)
    x, y = aim_mm * math.cos(b), aim_mm * math.sin(b)
    target = np.array([x, y, cal.table_z_at(x, y)])
    ref = np.asarray(seed if seed is not None else [bearing_deg, *STRAIGHT_WRIST], float)
    dcam = _centre_dir_cam(cal)

    def residual(q):
        miss, along, cam = _aim(cal, dcam, target, q)
        return [*(miss / 5), min(0.0, along) / 5, (flange_pose(q)[2, 3] - z_mm) / 5,
                cam[2, 0] * 20, *(0.002 * (np.asarray(q) - ref))]

    j2, j3, j4, j5 = STRAIGHT_WRIST
    seeds = ([seed] if seed is not None else []) + [[bearing_deg, j2, j3, j4, j5],
                                                    [bearing_deg + 15, j2, j3, j4, j5],
                                                    [bearing_deg - 15, j2, j3, j4, j5],
                                                    [bearing_deg, -47, -32, -17, j5],
                                                    [bearing_deg, -70, -21, -13, j5]]
    u = np.array([math.cos(b), math.sin(b)])

    def sensible(q) -> bool:
        cam = cal.camera_pose(q)
        return (q[3] <= WRIST_FOLD_MAX_DEG                         # folded wrist: controller C22 self-collision
                and aim_error_deg(cal, target, q) <= 0.3 and abs(flange_pose(q)[2, 3] - z_mm) <= 2
                and abs(cam[2, 0]) < 0.02                         # image horizon level
                and abs(_wrap(q[0] - bearing_deg)) < 45           # the arm faces the bearing it looks at
                and float(cam[:2, 2] @ u) > 0.1                   # looking outward, away from the robot
                and -50 < float(cam[:2, 3] @ u) < aim_mm)         # camera between the base and the aim point
    good = [q for q in _solve_all(residual, seeds) if sensible(q)]
    q = _nearest(good, ref)
    if q is None:
        raise ValueError(f"no look pose at bearing {bearing_deg:.0f} deg, height {z_mm:.0f} mm, aim {aim_mm:.0f} mm")
    return q


def image_flipped(cal: FlangeCalibration, q) -> bool:
    """True when the picture comes out upside down at joints q (image 'down' points up in the world). This is the
    case in the straight-wrist family: the flange points forward and the sideways camera is rolled 180 deg.
    Callers rotate the frame 180 deg so OpenCV, Grok and people see it upright; pixel_of() maps back to V2."""
    return float(cal.camera_pose(q)[2, 1]) > 0.0


def pixel_of(cal: FlangeCalibration, u: float, v: float, flipped: bool) -> tuple[float, float]:
    """Pixel in the upright (possibly rotated) picture -> pixel in the camera's own frame, as V2 expects."""
    return (cal.width - 1 - u, cal.height - 1 - v) if flipped else (u, v)


def view_on_table(cal: FlangeCalibration, q) -> dict:
    """Where the upright image lands on the table: nearest and farthest edge distances from the robot base."""
    flipped = image_flipped(cal, q)

    def dist(u, v):
        p = cal.pixel_to_table(*pixel_of(cal, u, v, flipped), q)
        return float(math.hypot(p[0], p[1]))
    try:
        far = max(dist(0, 0), dist(cal.width / 2, 0), dist(cal.width - 1, 0))
    except ValueError:
        far = float("inf")                       # the top edge looks above the horizon: the room is in view
    near = dist(cal.width / 2, cal.height - 1)
    return {"near_mm": round(near), "far_mm": round(far) if math.isfinite(far) else None,
            "table_only": math.isfinite(far) and near < far}     # near < far: looking outward, not back


def plan_look(cal: FlangeCalibration, s: Settings, bearing_deg: float, seed=None) -> dict:
    """Shallowest tilt (largest aim distance) whose whole top edge stays on the table within s.far_mm."""
    best, q_prev = None, seed
    for aim in np.arange(250.0, s.far_mm, 20.0):
        try:
            q = look_pose(cal, bearing_deg, s.z_mm, float(aim), q_prev)
        except ValueError:
            continue
        view = view_on_table(cal, q)
        if not view["table_only"] or view["far_mm"] > s.far_mm:
            break
        best, q_prev = {"q": q, "aim_mm": float(aim), "view": view}, q
    if best is None:
        raise ValueError(f"no look pose at Z={s.z_mm:.0f} mm keeps the whole picture on the table within "
                         f"{s.far_mm:.0f} mm (raise TROVA_TABLE_SCAN_FAR_MM or lower TROVA_TABLE_SCAN_Z_MM)")
    return best


def camera_over_pose(cal: FlangeCalibration, target, hover_mm: float, seed) -> np.ndarray:
    target = np.asarray(target, float)
    ref = np.asarray(seed, float)
    dcam = _centre_dir_cam(cal)

    def residual(q):
        miss, along, cam = _aim(cal, dcam, target, q)
        return [*(miss / 5), (along - hover_mm) / 5, cam[2, 0] * 10, (cam[2, 2] + 1) * 2,
                *(0.002 * (np.asarray(q) - ref))]

    bearing = math.degrees(math.atan2(target[1], target[0]))
    j2, j3, j4, j5 = STRAIGHT_WRIST
    seeds = [seed, np.asarray(seed) + [0, 10, 10, -10, 0], np.asarray(seed) + [0, -10, -10, 10, 0],
             [bearing, j2, j3, j4, j5], [bearing, -20, -40, -30, j5], [bearing, 10, -60, -40, j5],
             [bearing, 30, -90, -30, j5]]

    def sensible(q) -> bool:
        cam = cal.camera_pose(q)
        return (q[3] <= WRIST_FOLD_MAX_DEG                          # folded wrist: controller C22 self-collision
                and aim_error_deg(cal, target, q) <= 0.5
                and abs(np.linalg.norm(cam[:3, 3] - target) - hover_mm) <= 10
                and -float(cam[2, 2]) >= math.sin(math.radians(45)))    # looking down at it, not across
    q = _nearest([q for q in _solve_all(residual, seeds) if sensible(q)], ref)   # the smallest move from here
    if q is None:
        raise ValueError("the arm cannot place the camera above that point at the requested distance")
    return q


def level_over_pose(cal: FlangeCalibration, target, z_mm: float, seed, limits=None) -> np.ndarray:
    """Camera over (or, if it cannot get there, pointing at) target with the TCP held at height z_mm: the
    image-centre ray goes through the object, the camera looks down as steeply as the arm allows, the TCP stays
    inside the workspace box. No descent: the height of the look pose is kept."""
    target = np.asarray(target, float)
    ref = np.asarray(seed, float)
    dcam = _centre_dir_cam(cal)
    box = getattr(limits, "x", None) and (limits.x, limits.y)

    def outside(p):
        if not box:
            return [0.0, 0.0]
        return [max(0.0, lo + 40 - v) + max(0.0, v - hi + 40) for v, (lo, hi) in zip(p[:2], box)]  # joint paths bulge

    def residual(q):
        miss, along, cam = _aim(cal, dcam, target, q)
        tcp = flange_pose(q)[:3, 3]
        return [*(miss / 2), min(0.0, along) / 2, (tcp[2] - z_mm) / 2, cam[2, 0] * 20,
                (cam[2, 2] + 1) * 3, *(np.asarray(outside(tcp)) / 2), *(0.002 * (np.asarray(q) - ref))]

    bearing = math.degrees(math.atan2(target[1], target[0]))
    j2, j3, j4, j5 = STRAIGHT_WRIST
    seeds = [ref, np.r_[bearing, ref[1:]], np.r_[bearing, ref[1], ref[2] + 15, ref[3] - 15, ref[4]],
             [bearing, j2, j3, j4, j5], [bearing, -20, -40, -30, j5], [bearing, 10, -60, -40, j5]]

    def sensible(q) -> bool:
        cam = cal.camera_pose(q)
        tcp = flange_pose(q)[:3, 3]
        return (q[3] <= WRIST_FOLD_MAX_DEG and aim_error_deg(cal, target, q) <= 0.5
                and abs(tcp[2] - z_mm) <= 3 and abs(cam[2, 0]) < 0.05 and -float(cam[2, 2]) > 0.3
                and max(outside(tcp)) <= 15)
    good = [np.r_[q[:4], _wrap(q[4])] for q in _solve_all(residual, seeds) if sensible(q)]  # J5 +-180: less cable twist
    if not good:
        raise ValueError(f"the arm cannot point the camera at that spot from height Z={z_mm:.0f} mm inside the box")
    steep = max(-float(cal.camera_pose(q)[2, 2]) for q in good)       # steepest view first, then the smallest move
    return _nearest([q for q in good if -float(cal.camera_pose(q)[2, 2]) >= steep - 0.02], ref)


def plan_level_over(cal: FlangeCalibration, s: Settings, start_q, target_mm, limits=None) -> dict:
    """'Mettiti sopra': turn J1 toward the object at constant height, then lean over it at the same TCP height."""
    start_q = np.asarray(start_q, float)
    target = np.asarray(target_mm, float)
    z = float(flange_pose(start_q)[2, 3])
    final = level_over_pose(cal, target, z, start_q, limits)
    turn = start_q.copy()
    turn[0] = final[0]
    steps = ([{"label": "turn toward the object", "q": turn}] if abs(final[0] - start_q[0]) > 0.5 else []) + \
            [{"label": "over the object, same height", "q": final}]
    problems = path_problems(cal, s, steps, start_q, limits)
    prev, low = start_q, z
    for st in steps:                                   # how much a joint-space move sags below the kept height
        for t in np.linspace(0, 1, 30):
            low = min(low, float(flange_pose(prev + (np.asarray(st["q"]) - prev) * t)[2, 3]))
        prev = np.asarray(st["q"])
    if low < z - 60:
        problems.append(f"the move would dip {z - low:.0f} mm below the kept height")
    cam = cal.camera_pose(final)
    down = math.degrees(math.asin(min(1.0, max(-1.0, -float(cam[2, 2])))))
    return {"kind": "level_over", "target_mm": [round(float(v), 1) for v in target], "z_mm": round(z, 1),
            "steps": steps, "problems": problems, "start_q": start_q.tolist(), "dip_mm": round(z - low, 1),
            "camera_mm": [round(float(v), 1) for v in cam[:3, 3]], "look_down_deg": round(down, 1),
            "above": down >= 75.0, "distance_mm": round(float(np.linalg.norm(cam[:3, 3] - target)), 1),
            "tcp_mm": [round(float(v), 1) for v in flange_pose(final)[:3, 3]]}


# ---------------------------------------------------------------- path checks
def clearance(cal: FlangeCalibration, qa, qb, n: int = 40) -> dict:
    """Lowest TCP and lowest camera-head surface above the table along the straight joint path qa -> qb."""
    qa, qb = np.asarray(qa, float), np.asarray(qb, float)
    tcp_low = head_low = float("inf")
    for t in np.linspace(0.0, 1.0, n):
        q = qa + (qb - qa) * t
        tcp = flange_pose(q)[:3, 3]
        centre, radius = cal.head_sphere_world(q)
        tcp_low = min(tcp_low, float(tcp[2] - cal.table_z_at(tcp[0], tcp[1])))
        head_low = min(head_low, float(centre[2] - radius - cal.table_z_at(centre[0], centre[1])))
    return {"tcp_mm": round(tcp_low, 1), "head_mm": round(head_low, 1)}


def path_problems(cal: FlangeCalibration, s: Settings, steps: list[dict], start_q, limits=None) -> list[str]:
    """Joint limits, head clearance (never lower than where it starts, at least s.clearance_mm once above it) and the
    workspace box on sampled TCP points. The gateway repeats its own checks before each move."""
    problems, prev = [], np.asarray(start_q, float)
    start_head = clearance(cal, prev, prev, 1)["head_mm"]
    floor = min(s.clearance_mm, start_head - 1.0)
    for step in steps:
        q = np.asarray(step["q"], float)
        if np.any(q < LO) or np.any(q > HI):
            problems.append(f"{step['label']}: a joint is within {MARGIN_DEG:.0f} deg of its xArm 5 limit")
        c = clearance(cal, prev, q)
        step["clearance"] = c
        if c["head_mm"] < floor:
            problems.append(f"{step['label']}: camera head {c['head_mm']:.0f} mm above the table on the way")
        if limits:
            for t in np.linspace(0.0, 1.0, 20):
                p = flange_pose(prev + (q - prev) * t)[:3, 3]
                err = limits.check(*p)
                if err:
                    problems.append(f"{step['label']}: {err}")
                    break
        prev = q
    return problems


# ---------------------------------------------------------------- plans
def start_bearing(start_q) -> float:
    """Bearing (deg) of the TCP at the start pose. The table lies under the rest pose, so this is its middle."""
    tcp = flange_pose(start_q)[:3, 3]
    return round(math.degrees(math.atan2(tcp[1], tcp[0])), 1) if math.hypot(tcp[0], tcp[1]) > 20 else float(start_q[0])


def plan_scan(cal: FlangeCalibration, s: Settings, start_q, limits=None, look_q=None) -> dict:
    """look_q: the operator's taught look pose (joints). It replaces the solver: J1 offsets turn from it, and the
    scan ends back at it."""
    start_q = np.asarray(start_q, float)
    if look_q is not None:
        q_look = np.asarray(look_q, float)
        center = float(q_look[0])
        try:
            view = view_on_table(cal, q_look)
        except ValueError:
            view = {"near_mm": None, "far_mm": None, "table_only": False}
        look = {"q": q_look, "aim_mm": None, "view": view}
    else:
        center = s.center_deg if s.center_deg is not None else start_bearing(start_q)
        look = plan_look(cal, s, center)
        q_look = look["q"]
    views = []
    for off in s.pan_deg:
        q = q_look.copy()
        q[0] = q_look[0] + off
        views.append({"label": f"view {off:+.0f} deg", "offset_deg": off, "bearing_deg": center + off,
                      "q": q, "capture": True})
    if abs(views[-1]["q"][0] - start_q[0]) < abs(views[0]["q"][0] - start_q[0]):
        views.reverse()                          # start from the side nearer to where J1 is now
    lift = q_look.copy()
    lift[0] = start_q[0]                         # rise and tilt in place first; the turn comes after
    steps = ([{"label": "rise and tilt", "q": lift, "capture": False}]
             if float(np.max(np.abs(lift - start_q))) > 0.5 else []) + views   # already there: no empty move
    if look_q is not None and float(np.max(np.abs(np.asarray(steps[-1]["q"]) - q_look))) > 0.5:
        steps.append({"label": "back to the look pose", "q": q_look.copy(), "capture": False})
    problems = path_problems(cal, s, steps, start_q, limits)
    return {"kind": "table_scan", "z_mm": round(float(flange_pose(q_look)[2, 3]), 1) if look_q is not None else s.z_mm,
            "aim_mm": look["aim_mm"], "view": look["view"], "center_deg": center,
            "center_from": "taught pose" if look_q is not None else ("setting" if s.center_deg is not None
                                                                     else "start pose"),
            "steps": steps, "problems": problems, "start_q": start_q.tolist()}


def plan_camera_over(cal: FlangeCalibration, s: Settings, start_q, target_mm, limits=None) -> dict:
    start_q = np.asarray(start_q, float)
    target = np.asarray(target_mm, float)
    final = camera_over_pose(cal, target, s.hover_mm, start_q)
    pre = camera_over_pose(cal, target, s.hover_mm + 150.0, final)    # same line of sight, 150 mm higher first
    steps = [{"label": "above, 150 mm higher", "q": pre}, {"label": "camera above the object", "q": final}]
    problems = path_problems(cal, s, steps, start_q, limits)
    centre, radius = cal.head_sphere_world(final)
    head_floor = cal.min_head_centre_z(target[2] + s.object_max_mm)
    if centre[2] < head_floor:
        problems.append(f"the camera head would be {head_floor - centre[2]:.0f} mm too low for a "
                        f"{s.object_max_mm:.0f} mm object")
    cam = cal.camera_pose(final)
    return {"kind": "camera_over", "target_mm": [round(float(v), 1) for v in target], "hover_mm": s.hover_mm,
            "steps": steps, "problems": problems, "start_q": start_q.tolist(),
            "camera_mm": [round(float(v), 1) for v in cam[:3, 3]],
            "look_down_deg": round(math.degrees(math.asin(min(1.0, max(-1.0, -float(cam[2, 2]))))), 1),
            "tcp_mm": [round(float(v), 1) for v in flange_pose(final)[:3, 3]]}


# ---------------------------------------------------------------- objects
def locate_box(cal: FlangeCalibration, q, box, flipped: bool = False) -> dict:
    """Footprint estimate of one box seen from joints q (robot mm on the V2 table plane). The box is in the upright
    picture; flipped=True means that picture was rotated 180 deg from the camera's own frame."""
    x0, y0, x1, y1 = (float(v) for v in box)
    u, v = (x0 + x1) / 2, min(y1, cal.height - 1.0)
    foot = cal.pixel_to_table(*pixel_of(cal, u, v, flipped), q)
    left = cal.pixel_to_table(*pixel_of(cal, x0, v, flipped), q)
    right = cal.pixel_to_table(*pixel_of(cal, x1, v, flipped), q)
    width = float(np.linalg.norm(right - left))
    cam = cal.camera_pose(q)[:3, 3]
    away = foot[:2] - cam[:2]
    away = away / (np.linalg.norm(away) or 1.0)
    centre = foot.copy()
    centre[:2] = foot[:2] + away * width / 2
    centre[2] = cal.table_z_at(centre[0], centre[1])
    partial = x0 <= 2 or y0 <= 2 or x1 >= cal.width - 3 or y1 >= cal.height - 3
    return {"foot_mm": [round(float(c), 1) for c in foot], "centre_mm": [round(float(c), 1) for c in centre],
            "width_mm": round(width, 1), "support_pixel": [round(u, 1), round(v, 1)], "partial": bool(partial),
            "off_centre_px": round(float(math.hypot((x0 + x1) / 2 - cal.width / 2, (y0 + y1) / 2 - cal.height / 2)), 1)}


def merge(sightings: list[dict], radius_mm: float) -> list[dict]:
    """One map entry per physical object: sightings closer than radius_mm (XY) are merged; the best view (whole box,
    nearest the image centre) gives the name and the close-up."""
    ordered = sorted(sightings, key=lambda o: (o["partial"], o["name"].lower() in UNKNOWN, o["off_centre_px"]))
    objects: list[dict] = []
    for o in ordered:
        xy = np.asarray(o["centre_mm"][:2])
        hit = next((m for m in objects if np.linalg.norm(np.asarray(m["centre_mm"][:2]) - xy) < radius_mm), None)
        if hit is None:
            objects.append({"id": len(objects) + 1, "name": o["name"], "category": o.get("category", ""),
                            "note": o.get("note", ""), "aliases": [], "authoritative": o.get("authoritative", False),
                            "centre_mm": list(o["centre_mm"]), "width_mm": o["width_mm"], "partial": o["partial"],
                            "views": [o["view"]], "best": {k: o[k] for k in ("view", "box", "image", "support_pixel")},
                            "_sightings": [o]})
            continue
        hit["_sightings"].append(o)
        if o["view"] not in hit["views"]:
            hit["views"].append(o["view"])
        if o["name"].lower() not in UNKNOWN and o["name"] != hit["name"] and o["name"] not in hit["aliases"]:
            hit["aliases"].append(o["name"])
        full = [x for x in hit["_sightings"] if not x["partial"]] or hit["_sightings"]
        hit["centre_mm"] = [round(float(np.mean([x["centre_mm"][i] for x in full])), 1) for i in range(3)]
        hit["width_mm"] = round(float(np.median([x["width_mm"] for x in full])), 1)
        hit["partial"] = all(x["partial"] for x in hit["_sightings"])
    for m in objects:
        m["sightings"] = len(m.pop("_sightings"))
        x, y = m["centre_mm"][:2]
        m["distance_mm"] = round(math.hypot(x, y))
        m["bearing_deg"] = round(math.degrees(math.atan2(y, x)), 1)
    return objects


def place_words(obj: dict, center_deg: float, lang: str = "it") -> str:
    """Where an object is, as seen from the robot looking at the middle of the table."""
    rel = (obj["bearing_deg"] - center_deg + 180) % 360 - 180      # + = counter-clockwise = the robot's left
    near = obj["distance_mm"] < 420
    if lang == "it":
        side = "alla sinistra del robot" if rel > 15 else "alla destra del robot" if rel < -15 else "al centro"
        return f"{side}, {'vicino al robot' if near else 'più lontano'}"
    side = "on the robot's left" if rel > 15 else "on the robot's right" if rel < -15 else "in the middle"
    return f"{side}, {'near the robot' if near else 'further away'}"


def annotate_view(frame: np.ndarray, entries: list[dict]) -> np.ndarray:
    """One photo with its OpenCV boxes and labels. entries: {box, text, kind} with kind
    'mapped' (named + on the table, green), 'off_table' (named, foot not on the plane, orange), 'unknown' (grey)."""
    out = frame.copy()
    colours = {"mapped": (60, 200, 60), "off_table": (0, 150, 255), "unknown": (150, 150, 150)}
    scale = max(0.5, out.shape[1] / 1600 * 0.9)
    thick = max(1, round(out.shape[1] / 800))
    for e in sorted(entries, key=lambda e: e["kind"] == "mapped"):      # mapped boxes drawn last, on top
        x0, y0, x1, y1 = (int(round(v)) for v in e["box"])
        c = colours.get(e["kind"], colours["unknown"])
        cv2.rectangle(out, (x0, y0), (x1, y1), c, thick + (1 if e["kind"] == "mapped" else 0), cv2.LINE_AA)
        text = e["text"]
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        ty = y0 - 6 if y0 - th - base - 6 > 0 else min(out.shape[0] - base - 2, y1 + th + 6)
        cv2.rectangle(out, (x0, ty - th - 4), (x0 + tw + 6, ty + base), c, -1)
        cv2.putText(out, text, (x0 + 3, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick, cv2.LINE_AA)
    return out


def contact_sheet(objects: list[dict], frames: dict[str, np.ndarray], cell: int = 220) -> np.ndarray:
    """One close-up per map object, numbered with its map id (for the Grok find)."""
    cols = min(4, max(1, len(objects)))
    rows = max(1, math.ceil(len(objects) / cols))
    sheet = np.full((rows * cell, cols * cell, 3), 255, np.uint8)
    for i, obj in enumerate(objects):
        frame = frames.get(obj["best"]["view"])
        if frame is None:
            continue
        x0, y0, x1, y1 = (int(v) for v in obj["best"]["box"])
        pad = int(0.15 * max(x1 - x0, y1 - y0)) + 4
        crop = frame[max(0, y0 - pad):min(frame.shape[0], y1 + pad), max(0, x0 - pad):min(frame.shape[1], x1 + pad)]
        if crop.size == 0:
            continue
        k = (cell - 34) / max(crop.shape[:2])
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * k)), max(1, int(crop.shape[0] * k))))
        r, c = divmod(i, cols)
        oy, ox = r * cell + 30, c * cell + (cell - crop.shape[1]) // 2
        sheet[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
        cv2.putText(sheet, str(obj["id"]), (c * cell + 6, r * cell + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (180, 0, 180), 2, cv2.LINE_AA)
    return sheet


# ---------------------------------------------------------------- find by words
FIND_SCHEMA = {"type": "object", "properties": {
    "matches": {"type": "array", "items": {"type": "integer"}},
    "reason": {"type": "string"}}, "required": ["matches", "reason"], "additionalProperties": False}

FIND_PROMPT = """A robot arm scanned a work table. The image is a sheet of close-ups: each cell is one object on the
table, with its number in magenta.
Objects (number: name; detail; where it is seen from the robot):
{listing}

Request: "{request}"
Return in "matches" the number of EVERY object that fits the request, including all identical-looking ones. If the
request names a position ("a sinistra", "the one near the robot"), keep only those. Return an empty list if nothing
fits; never guess. "reason": one short sentence in {lang}."""


def find(grok: GrokClient, request: str, objects: list[dict], sheet: np.ndarray | None,
         center_deg: float, lang: str = "it") -> dict:
    named = [o for o in objects if o["name"].lower() not in UNKNOWN]
    if not named:
        return {"status": "not_found", "matches": [], "reason": "the scan has no named objects", "grok_mode": "none"}
    listing = "\n".join(f"{o['id']}: {o['name']}" + (f" (also seen as {', '.join(o['aliases'])})" if o["aliases"] else "")
                        + f"; {o.get('note') or '-'}; {place_words(o, center_deg, 'en')}" for o in named)
    r = grok.json(name="trova_table_find", prompt=FIND_PROMPT.format(listing=listing, request=request,
                                                                     lang="Italian" if lang == "it" else "English"),
                  schema=FIND_SCHEMA, images=[sheet] if sheet is not None else [],
                  mock_context={"objects": named, "request": request})
    ids = {o["id"] for o in named}
    matches = [i for i in dict.fromkeys(r.data.get("matches") or []) if i in ids]
    mode = "offline" if r.offline else "mock" if r.mock else "live"
    out = {"matches": matches, "reason": str(r.data.get("reason") or ""), "grok_mode": mode,
           "latency_s": round(r.latency_s, 2)}
    if not matches:
        return out | {"status": "not_found"}
    if len(matches) == 1:
        return out | {"status": "found", "object_id": matches[0]}
    by_id = {o["id"]: o for o in objects}
    parts = [f"{'il numero' if lang == 'it' else 'number'} {i} {place_words(by_id[i], center_deg, lang)}"
             for i in matches]
    name = by_id[matches[0]]["name"]
    question = (f"Ho trovato {len(matches)} oggetti che corrispondono a \"{name}\": " + "; ".join(parts)
                + ". Quale scegli? Dimmi il numero, oppure toccalo nell'interfaccia."
                if lang == "it" else
                f"I found {len(matches)} objects matching \"{name}\": " + "; ".join(parts)
                + ". Which one? Say its number, or tap it in the interface.")
    return out | {"status": "clarify", "question": question, "candidates": matches}


def resolve_choice(answer: str, candidates: list[int], objects: list[dict], center_deg: float) -> int | None:
    """The operator's answer to the clarify question: a number, or a side/distance word naming one candidate."""
    nums = [int(n) for n in re.findall(r"\d+", answer)]
    words = {"uno": 1, "one": 1, "primo": 1, "first": 1, "due": 2, "two": 2, "secondo": 2, "second": 2,
             "tre": 3, "three": 3, "terzo": 3, "third": 3, "quattro": 4, "four": 4, "cinque": 5, "five": 5}
    nums += [n for w, n in words.items() if re.search(rf"\b{w}\b", answer.lower())]
    for n in nums:
        if n in candidates:
            return n
    t = answer.lower()
    by_id = {o["id"]: o for o in objects}
    keys = [("sinistra", "left"), ("destra", "right"), ("centro", "middle"), ("vicino", "near"), ("lontano", "further")]
    hits = []
    for i in candidates:
        place = place_words(by_id[i], center_deg, "it") + " " + place_words(by_id[i], center_deg, "en")
        if any((it in t or en in t) and (it in place or en in place) for it, en in keys):
            hits.append(i)
    return hits[0] if len(hits) == 1 else None


def _mock_find(prompt: str, images: list, ctx: dict) -> dict:
    words = {w for w in re.findall(r"[a-zà-ù]+", ctx["request"].lower()) if len(w) > 2}
    hits = [o["id"] for o in ctx["objects"]
            if words & set(re.findall(r"[a-zà-ù]+", " ".join([o["name"], *o["aliases"], o.get("category", "")]).lower()))]
    return {"matches": hits, "reason": f"Mock: matched {len(hits)} by words."}


register_mock("trova_table_find", _mock_find)


# ---------------------------------------------------------------- Grok boxes when OpenCV finds too few
DETECT_PROMPT = """Photo from a camera on a robot arm looking at a work table (it may be seen at an angle).
List every distinct physical object on the table: tools (pliers, screwdrivers...), cups, bottles, cans, boxes,
cases, devices, phones, papers, bags. At most {n}. For each give box_2d = [x0, y0, x1, y1] in 0-1000 image
coordinates, tight around the whole object, and a short specific name in {lang}. One box per object; skip the
table surface, loose cables, walls, floor, chairs, people and the robot itself."""
DETECT_SCHEMA = {"type": "object", "required": ["objects"], "properties": {"objects": {"type": "array", "items": {
    "type": "object", "required": ["box_2d", "name"],
    "properties": {"box_2d": {"type": "array", "items": {"type": "number"}}, "name": {"type": "string"}}}}}}
MIN_OPENCV_BOXES = 3


def grok_boxes(grok: GrokClient, frame: np.ndarray, lang: str = "it", max_n: int = 15):
    """Boxes drawn by Grok itself (source='grok', label = its name). Used when OpenCV blobs fail, e.g. a cluttered
    table where cables join everything into one border-touching blob."""
    from .boxes import Box
    r = grok.json(name="trova_table_detect", prompt=DETECT_PROMPT.format(n=max_n, lang="Italian" if lang == "it"
                                                                          else "English"),
                  schema=DETECT_SCHEMA, images=[frame], mock_context={})
    H, W = frame.shape[:2]
    boxes = []
    for o in (r.data.get("objects") or [])[:max_n]:
        try:
            b = Box.from_norm1000(list(o["box_2d"])[:4], W, H, id=len(boxes) + 1, source="grok",
                                  label=str(o.get("name") or "").strip())
        except (KeyError, TypeError, ValueError):
            continue
        if b.x1 - b.x0 >= 6 and b.y1 - b.y0 >= 6 and b.label:
            boxes.append(b)
    return boxes, r


register_mock("trova_table_detect", lambda prompt, images, ctx: {"objects": []})
