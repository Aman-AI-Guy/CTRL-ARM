"""Camera <-> robot calibration for the arm camera: intrinsics + where the camera sits on the flange + table height,
from stills taken at several arm poses (orbit_here.py --snapshots) of the ArUco cubes (markers of known size).

    python -m scan3d.handeye solve arm_pc/runs/<stamp>_snaps      # -> calib/camera_robot.json + calib/report.json
    python -m scan3d.handeye selftest                             # synthetic poses and markers, known answer

Model: robot FK (= the controller's) gives the flange pose; camera = flange x T_flange_camera; each marker is a
rigid square of marker_mm in the robot frame; pinhole + 2 radial terms. One least-squares fit of all marker corners
in all pictures (Levenberg-Marquardt, Huber-weighted, weak priors from the CAD and nominal optics), started from
OpenCV's hand-eye solution. Validation: fit on half the pictures, then project the other half's marker corners
from pixels onto the marker's plane (exactly what pixel -> mm does) and compare in mm.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .config import ROOT, add_config_args, config_from_args

sys.path.insert(0, str(ROOT / "calib"))
from pix2robot import Calibration, flange_pose  # noqa: E402

CALIB_DIR = ROOT / "calib"
CAD_CAMERA_FLANGE = np.array([-29.7, 29.7, 40.1])     # optical centre from the head CAD, head turned 135 deg
HEAD_SPHERE = {"center_flange_mm": [0.0, 0.0, 45.0], "radius_mm": 60.0,
               "note": "Giacomo: the head fits in a ~120 mm sphere; CAD bbox 100 x 100 x 90 mm from the flange face"}


def corners_obj(s: float) -> np.ndarray:
    """ArUco corner order (top-left, top-right, bottom-right, bottom-left) in the marker frame, z out of the face."""
    return np.array([[-s / 2, s / 2, 0], [s / 2, s / 2, 0], [s / 2, -s / 2, 0], [-s / 2, -s / 2, 0]], float)


def T_from(rvec, tvec) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.asarray(rvec, float).reshape(3, 1))[0]
    T[:3, 3] = np.asarray(tvec, float).ravel()
    return T


def rt_of(T) -> np.ndarray:
    return np.concatenate([cv2.Rodrigues(T[:3, :3])[0].ravel(), T[:3, 3]])


# ---------------------------------------------------------------- data
def detect(img: np.ndarray, dictionary: str, ids_wanted) -> dict[int, np.ndarray]:
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary)), params)
    corners, ids, _ = det.detectMarkers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    out = {}
    for c, i in zip(corners, [] if ids is None else ids.ravel()):
        if not ids_wanted or int(i) in ids_wanted:
            out[int(i)] = c.reshape(4, 2).astype(float)
    return out


def load_snaps(folder: Path, cfg: dict):
    meta = json.loads((folder / "snapshots.json").read_text(encoding="utf-8"))
    sc = cfg["aruco_scan"]
    ids = set(sc.get("ids") or [])
    images, obs, size = [], [], None
    for row in meta["snapshots"]["rows"]:
        img = cv2.imread(str(folder / row["file"]))
        if img is None:
            continue
        size = (img.shape[1], img.shape[0])
        found = detect(img, sc["dictionary"], ids)
        k = len(images)
        images.append({"file": row["file"], "view": row["view"], "q": row["q"], "markers": sorted(found)})
        for mid, c in found.items():
            obs.append((k, mid, c))
    return images, obs, size, float(sc["marker_mm"])


def load_legs(legs: list[list[Path]], cfg: dict):
    """Several snapshot folders. Folders of one leg saw the cubes in the same place; between legs the cubes were
    moved, so each leg's markers are separate unknowns: (leg, id), or the plain id when there is one leg. The 'start'
    still is the same picture in every folder of a leg: only the first folder's is kept."""
    images, obs, size, s = [], [], None, None
    many = sum(len(f) for f in legs) > 1
    for li, folders in enumerate(legs):
        for fi, folder in enumerate(folders):
            im, ob, sz, s = load_snaps(folder, cfg)
            if size and sz != size:
                raise SystemExit(f"{folder}: pictures {sz}, the others {size}")
            size = sz or size
            remap = {}
            for k, row in enumerate(im):
                if fi > 0 and row["view"] == "start":
                    continue
                remap[k] = len(images)
                images.append(dict(row, file=f"{folder.name}/{row['file']}" if many else row["file"], leg=li,
                                   folder=str(folder)))
            obs += [(remap[k], m if len(legs) == 1 else (li, m), c) for k, m, c in ob if k in remap]
    return images, obs, size, s


def marker_name(m) -> str:
    return f"{m[1]}@leg{m[0]}" if isinstance(m, tuple) else str(m)


def head_bottom(q, axis_flange) -> np.ndarray:
    """Lowest point (robot mm) of the camera head's CAD at joints q; the head's turn on the flange from the camera
    axis, rounded to the bolt positions (45 deg). What a ruler measures from."""
    sys.path[:0] = [str(ROOT / "arm_pc"), str(ROOT / "arm_pc" / "model")]
    import orbit_here as oh
    from build_robot_model import HEAD_STL, cad_to_head, load_stl
    from robot_geometry import RobotGeometry

    if "verts" not in _HEAD:
        _HEAD["geo"], _HEAD["verts"] = RobotGeometry.load(), cad_to_head(load_stl(HEAD_STL)).reshape(-1, 3)
    turn = _HEAD["geo"].head_turn(axis_flange, 45)
    F = oh.fk_frames(q)[4]
    v = _HEAD["verts"] @ (F[:3, :3] @ turn).T + F[:3, 3]
    return v[int(np.argmin(v[:, 2]))]


_HEAD: dict = {}


def fit_plane(points) -> dict | None:
    """z = a x + b y + c through >= 3 spread points (least squares); a level plane through 1-2 points."""
    P = np.asarray(points, float)
    if len(P) == 0:
        return None
    if len(P) >= 3 and np.linalg.svd(P[:, :2] - P[:, :2].mean(0), compute_uv=False)[-1] > 20 * math.sqrt(len(P)):
        A = np.column_stack([P[:, 0], P[:, 1], np.ones(len(P))])
        a, b, c = np.linalg.lstsq(A, P[:, 2], rcond=None)[0]
    else:
        a, b, c = 0.0, 0.0, float(P[:, 2].mean())
    res = P[:, 2] - (a * P[:, 0] + b * P[:, 1] + c)
    return {"a": float(a), "b": float(b), "c": float(c), "tilt_deg": round(math.degrees(math.atan(math.hypot(a, b))), 3),
            "rms_mm": round(float(np.sqrt(np.mean(res ** 2))), 2), "points": np.round(P, 1).tolist()}


# ---------------------------------------------------------------- model
class Problem:
    def __init__(self, Tbf: list[np.ndarray], obs, marker_mm: float, size, f0: float):
        self.Tbf, self.obs, self.size = Tbf, obs, size
        self.obj = corners_obj(marker_mm)
        self.mids = sorted({m for _, m, _ in obs})
        self.f0 = f0
        self.prior_sigma = {"f": 0.08 * f0, "c": 0.05 * size[0], "k": 0.3, "t_fc": 20.0}
        # scene facts (measured with a ruler): {marker id: ("top", z_mm) | ("side", None)}
        self.scene: dict = {}
        self.scene_sigma = {"z": 1.0, "deg": 0.5}
        # ruler answers: [(marker, z_mm of its face, sigma_mm)] from the head's CAD bottom minus the measured gap
        self.ruler: list = []

    def unpack(self, x):
        f, cx, cy, k1, k2 = x[:5]
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
        dist = np.array([k1, k2, 0, 0, 0.0])
        T_fc = T_from(x[5:8], x[8:11])
        T_bm = {m: T_from(x[11 + 6 * j: 14 + 6 * j], x[14 + 6 * j: 17 + 6 * j]) for j, m in enumerate(self.mids)}
        return K, dist, T_fc, T_bm

    def project(self, x, sel=None):
        K, dist, T_fc, T_bm = self.unpack(x)
        out = []
        for n, (i, m, c) in enumerate(self.obs):
            if sel is not None and n not in sel:
                continue
            T_cm = np.linalg.inv(self.Tbf[i] @ T_fc) @ T_bm[m]
            uv, _ = cv2.projectPoints(self.obj, cv2.Rodrigues(T_cm[:3, :3])[0], T_cm[:3, 3], K, dist)
            out.append(uv.reshape(4, 2) - c)
        return out

    def residuals(self, x, weights=None):
        r = np.concatenate([d.ravel() for d in self.project(x)])
        if weights is not None:
            r = r * weights
        ps = self.prior_sigma
        pri = [(x[0] - self.f0) / ps["f"], (x[1] - self.size[0] / 2) / ps["c"], (x[2] - self.size[1] / 2) / ps["c"],
               x[3] / ps["k"], x[4] / ps["k"]]
        pri += list((x[8:11] - CAD_CAMERA_FLANGE) / ps["t_fc"])
        if self.scene:                                    # cube tops: horizontal at a known z; side faces vertical
            _, _, _, T_bm = self.unpack(x)
            ss = self.scene_sigma
            for m, (kind, z) in self.scene.items():
                if m not in T_bm:
                    continue
                n = T_bm[m][:3, 2]
                if kind == "top":
                    pri += [(T_bm[m][2, 3] - z) / ss["z"], n[0] / math.radians(ss["deg"]), n[1] / math.radians(ss["deg"])]
                else:
                    pri += [n[2] / math.radians(2 * ss["deg"])]
        if self.ruler:
            _, _, _, T_bm = self.unpack(x)
            pri += [(T_bm[m][2, 3] - z) / sig for m, z, sig in self.ruler if m in T_bm]
        return np.concatenate([r, np.array(pri)])


def lm(prob: Problem, x0: np.ndarray, iters: int = 60, huber_px: float = 2.0, say=lambda *_: None):
    x = x0.copy()
    steps = np.array([0.5, 0.5, 0.5, 1e-4, 1e-4] + [1e-5] * 3 + [0.01] * 3 + [1e-5, 1e-5, 1e-5, 0.01, 0.01, 0.01] *
                     len(prob.mids))
    lam = 1e-3

    def weights(x):
        r = np.concatenate([d.ravel() for d in prob.project(x)])
        a = np.abs(r)
        return np.sqrt(np.where(a <= huber_px, 1.0, huber_px / np.maximum(a, 1e-9)))

    w = weights(x)
    r = prob.residuals(x, w)
    cost = float(r @ r)
    for it in range(iters):
        J = np.empty((len(r), len(x)))
        for j in range(len(x)):
            dx = np.zeros(len(x))
            dx[j] = steps[j]
            J[:, j] = (prob.residuals(x + dx, w) - r) / steps[j]
        A, g = J.T @ J, J.T @ r
        improved = False
        for _ in range(12):
            step = -np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), g)
            xn = x + step
            rn = prob.residuals(xn, w)
            cn = float(rn @ rn)
            if cn < cost:
                x, r, cost, lam, improved = xn, rn, cn, max(lam / 3, 1e-9), True
                break
            lam *= 8
        if not improved or np.abs(step).max() < 1e-7:
            break
        if it % 5 == 4:                       # refresh the Huber weights
            w = weights(x)
            r = prob.residuals(x, w)
            cost = float(r @ r)
        say(f"  iter {it:2d}  cost {cost:10.2f}")
    return x


def hand_eye_park(Rg, tg, Rt, tt):
    """AX = XB (Park & Martin 1994): X = camera in the gripper (flange) frame, from gripper->base poses (Rg, tg)
    and target->camera poses (Rt, tt). Same inputs/outputs as cv2.calibrateHandEye (missing in this OpenCV build)."""
    G = [np.block([[R, np.asarray(t, float).reshape(3, 1)], [np.zeros((1, 3)), np.ones((1, 1))]]) for R, t in zip(Rg, tg)]
    C = [np.block([[R, np.asarray(t, float).reshape(3, 1)], [np.zeros((1, 3)), np.ones((1, 1))]]) for R, t in zip(Rt, tt)]
    M, pairs = np.zeros((3, 3)), []
    for i in range(len(G)):
        for j in range(i + 1, len(G)):
            A = np.linalg.inv(G[j]) @ G[i]
            B = C[j] @ np.linalg.inv(C[i])
            a, b = cv2.Rodrigues(A[:3, :3])[0].ravel(), cv2.Rodrigues(B[:3, :3])[0].ravel()
            if np.linalg.norm(a) < 0.05 or np.linalg.norm(b) < 0.05:      # < 3 deg: no information
                continue
            M += np.outer(b, a)
            pairs.append((A, B))
    if len(pairs) < 2:
        raise SystemExit("hand-eye: the poses do not rotate enough (need turns about 2+ different axes)")
    w, V = np.linalg.eigh(M.T @ M)
    R = V @ np.diag(1 / np.sqrt(np.maximum(w, 1e-12))) @ V.T @ M.T
    U, _, Wt = np.linalg.svd(R)
    R = U @ Wt
    if np.linalg.det(R) < 0:
        R = U @ np.diag([1, 1, -1]) @ Wt
    lhs = np.vstack([A[:3, :3] - np.eye(3) for A, _ in pairs])
    rhs = np.concatenate([R @ B[:3, 3] - A[:3, 3] for A, B in pairs])
    t = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    return R, t.reshape(3, 1)


def initial(prob: Problem, images, obs, size, marker_mm):
    """Nominal intrinsics; T_flange_camera from OpenCV's hand-eye on the marker seen most; markers from PnP."""
    f0 = prob.f0
    K = np.array([[f0, 0, size[0] / 2], [0, f0, size[1] / 2], [0, 0, 1.0]])
    pnp = {}
    for n, (i, m, c) in enumerate(obs):
        ok, rv, tv = cv2.solvePnP(corners_obj(marker_mm), c, K, np.zeros(5), flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if ok:
            pnp[n] = T_from(rv, tv)
    counts = {}
    for n, (i, m, c) in enumerate(obs):
        counts[m] = counts.get(m, 0) + (n in pnp)
    ref = max(counts, key=counts.get)
    Rg, tg, Rt, tt = [], [], [], []
    for n, (i, m, c) in enumerate(obs):
        if m == ref and n in pnp:
            Rg.append(prob.Tbf[i][:3, :3])
            tg.append(prob.Tbf[i][:3, 3])
            Rt.append(pnp[n][:3, :3])
            tt.append(pnp[n][:3, 3])
    if len(Rg) < 3:
        raise SystemExit(f"marker {ref} seen in only {len(Rg)} pictures: need 3+ poses")
    R, t = hand_eye_park(Rg, tg, Rt, tt)
    T_fc = np.eye(4)
    T_fc[:3, :3], T_fc[:3, 3] = R, t.ravel()
    x = [f0, size[0] / 2, size[1] / 2, 0.0, 0.0, *rt_of(T_fc)]
    for m in prob.mids:
        Ts = [prob.Tbf[obs[n][0]] @ T_fc @ pnp[n] for n in pnp if obs[n][1] == m]
        T = Ts[len(Ts) // 2].copy()
        T[:3, 3] = np.median([q[:3, 3] for q in Ts], axis=0)
        x += list(rt_of(T))
    return np.array(x, float), ref, len(Rg)


def solve_all(images, obs, size, marker_mm, say=print, x0=None, scene=None, ruler=None):
    f0 = (size[0] / 2) / math.tan(math.radians(62.2 / 2))
    Tbf = [flange_pose(im["q"]) for im in images]
    prob = Problem(Tbf, obs, marker_mm, size, f0)
    prob.scene, prob.ruler = scene or {}, ruler or []
    info = {}
    if x0 is None:
        x0, ref, n_he = initial(prob, images, obs, size, marker_mm)
        info.update(handeye_marker=ref, handeye_pictures=n_he)
    x = lm(prob, x0, say=say)
    # drop observations far off (bad detections), refit
    err = [float(np.sqrt(np.mean(np.sum(d ** 2, axis=1)))) for d in prob.project(x)]
    keep = [n for n, e in enumerate(err) if e < max(3.0, 4 * np.median(err))]
    if len(keep) < len(obs):
        info["dropped_observations"] = [(images[obs[n][0]]["file"], obs[n][1], round(err[n], 1))
                                        for n in range(len(obs)) if n not in keep]
        prob = Problem(Tbf, [obs[n] for n in keep], marker_mm, size, f0)
        prob.scene, prob.ruler = scene or {}, ruler or []
        x = lm(prob, x, say=say)
    return prob, x, info


def backproject_errors(prob_fit: Problem, x_fit, Tbf_test, obs_test, marker_planes: dict) -> list[float]:
    """mm error of detected corners projected (pixel -> plane, as pix2robot does) onto each marker's plane."""
    K, dist, T_fc, T_bm = prob_fit.unpack(x_fit)
    errs = []
    for i, m, c in obs_test:
        if m not in marker_planes:
            continue
        P_true, n_vec = marker_planes[m]
        T_bc = Tbf_test[i] @ T_fc
        nrm = cv2.undistortPoints(c.reshape(-1, 1, 2), K, dist).reshape(-1, 2)
        for k in range(4):
            d = T_bc[:3, :3] @ np.array([nrm[k, 0], nrm[k, 1], 1.0])
            o = T_bc[:3, 3]
            s = ((P_true[k] - o) @ n_vec) / (d @ n_vec)
            errs.append(float(np.linalg.norm(o + s * d - P_true[k])))
    return errs


def marker_geometry(prob: Problem, x) -> dict:
    _, _, _, T_bm = prob.unpack(x)
    out = {}
    for m, T in T_bm.items():
        P = (T[:3, :3] @ prob.obj.T).T + T[:3, 3]
        out[m] = (P, T[:3, 2] / np.linalg.norm(T[:3, 2]))
    return out


def report_and_save(images, obs, size, marker_mm, prob, x, info, cube_mm, out_dir: Path, folder, say=print,
                    table_pts=(), ruler_rows=()):
    K, dist, T_fc, T_bm = prob.unpack(x)
    res = prob.project(x)
    per_obs = [float(np.sqrt(np.mean(np.sum(d ** 2, axis=1)))) for d in res]
    rms = float(np.sqrt(np.mean(np.concatenate([np.sum(d ** 2, axis=1) for d in res]))))
    geo = marker_geometry(prob, x)
    tops = {m: float(P[:, 2].mean()) for m, (P, nv) in geo.items() if nv[2] > 0.95}
    seen = {m: sum(1 for _, k, _ in prob.obs if k == m) for m in geo}
    # under each cube-top marker; a marker seen in < 3 pictures is too loose for the table plane
    feet = [P.mean(0) - nv * cube_mm for m, (P, nv) in geo.items() if nv[2] > 0.95 and seen[m] >= 3]
    plane = fit_plane(feet + [np.asarray(p, float) for p in table_pts])
    if plane:
        cx, cy = np.mean([f[:2] for f in feet], axis=0) if feet else np.mean(np.asarray(table_pts)[:, :2], axis=0)
        table_z = plane["a"] * cx + plane["b"] * cy + plane["c"]
    else:
        table_z = float("nan")
    # hold-out: fit on even pictures, test on odd, and the other way round
    ho = []
    for part in (0, 1):
        fit_imgs = {k for k in range(len(images)) if k % 2 == part}
        o_fit = [o for o in prob.obs if o[0] in fit_imgs]
        o_test = [o for o in prob.obs if o[0] not in fit_imgs]
        if len({o[0] for o in o_fit}) < 4 or not o_test:
            continue
        pf = Problem(prob.Tbf, o_fit, marker_mm, size, prob.f0)
        pf.mids, pf.scene, pf.ruler = prob.mids, prob.scene, prob.ruler
        xf = lm(pf, x, iters=30)
        ho += backproject_errors(pf, xf, prob.Tbf, o_test, {m: g for m, g in geo.items()})
    cam_c = T_fc[:3, 3]
    cam_axis = T_fc[:3, 2]
    say(f"\npictures {len(images)}, marker observations {len(prob.obs)}, markers {[marker_name(m) for m in prob.mids]}")
    say(f"reprojection RMS {rms:.2f} px (per marker view: median {np.median(per_obs):.2f}, max {max(per_obs):.2f})")
    say(f"intrinsics @ {size[0]}x{size[1]}: f {K[0, 0]:.1f} px (nominal {prob.f0:.1f}), c ({K[0, 2]:.1f}, {K[1, 2]:.1f}), "
        f"k1 {dist[0]:+.4f}, k2 {dist[1]:+.4f}")
    say(f"camera in the flange frame: centre {np.round(cam_c, 1).tolist()} mm (CAD {CAD_CAMERA_FLANGE.tolist()}), "
        f"axis {np.round(cam_axis, 3).tolist()} (CAD [-0.707, 0.707, 0])")
    for m, (P, nv) in geo.items():
        say(f"marker {marker_name(m)}: centre {np.round(P.mean(0), 1).tolist()} mm, face normal {np.round(nv, 3).tolist()}")
    say(f"table z {table_z:.1f} mm at the work area (cube-top markers "
        f"{', '.join(f'{marker_name(m)}: {z:.1f}' for m, z in tops.items())} - {cube_mm:g})")
    if plane and (plane["a"] or plane["b"]):
        say(f"table PLANE through {len(plane['points'])} points: tilt {plane['tilt_deg']:.2f} deg, z {plane['c']:.1f} under "
            f"the base, rms {plane['rms_mm']:.1f} mm")
    rr = []
    for row in ruler_rows:                                 # measured gap vs the gap this fit implies
        hb = np.asarray(row["head_bottom_mm"], float)
        if row["to"] == "table":
            z = plane["a"] * hb[0] + plane["b"] * hb[1] + plane["c"] if plane else table_z
        else:
            key = row["key"]
            z = float(geo[key][0][:, 2].mean()) if key in geo else float("nan")
        rr.append(dict(row, fitted_gap_mm=round(float(hb[2] - z), 1)))
        say(f"ruler {row['to']:6s} {row.get('label', '')}: measured {row['gap_mm']:.1f} mm, this fit {hb[2] - z:.1f} mm")
    if ho:
        say(f"HOLD-OUT pixel -> mm on the marker planes: median {np.median(ho):.2f} mm, p95 {np.percentile(ho, 95):.2f}, "
            f"max {max(ho):.2f} ({len(ho)} corners)")

    def mjson(P, nv):
        return {"corners_mm": np.round(P, 2).tolist(), "normal": np.round(nv, 4).tolist()}
    extra = {}
    if any(isinstance(m, tuple) for m in geo):              # several legs: "markers" = where the cubes are now
        last = max(m[0] for m in geo)
        markers = {str(m[1]): mjson(*g) for m, g in geo.items() if m[0] == last}
        extra["markers_by_leg"] = {marker_name(m): mjson(*g) for m, g in geo.items()}
    else:
        markers = {str(m): mjson(*g) for m, g in geo.items()}
    if plane and (plane["a"] or plane["b"]):
        extra["table_plane"] = plane
    if rr:
        extra["ruler"] = rr
    cal = {
        "_note": "scan3d/handeye.py: camera <-> xArm 5 calibration for calib/pix2robot.py. mm, robot base frame, "
                 "OpenCV camera frame (x right, y down, z forward).",
        "image_size": list(size), "K": K.tolist(), "dist": dist.tolist(), "T_flange_camera": T_fc.tolist(),
        "table_z_mm": round(table_z, 2), "table_z_fitted_mm": round(table_z, 2), "cube_height_mm": cube_mm,
        "head_sphere": HEAD_SPHERE, "markers": markers, **extra,
        "quality": {"reprojection_rms_px": round(rms, 3), "holdout_median_mm": round(float(np.median(ho)), 2) if ho else None,
                    "holdout_p95_mm": round(float(np.percentile(ho, 95)), 2) if ho else None,
                    "pictures": len(images), "observations": len(prob.obs)},
        "source": {"snapshots": folder, "time": time.strftime("%Y-%m-%d %H:%M:%S"), **info},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "camera_robot.json").write_text(json.dumps(cal, indent=1), encoding="utf-8")
    (out_dir / "report.json").write_text(json.dumps({
        "per_picture": [{"file": images[i]["file"], "marker": marker_name(m), "rms_px": round(e, 2)}
                        for (i, m, _), e in zip(prob.obs, per_obs)], **cal["quality"]}, indent=1), encoding="utf-8")
    say(f"-> {out_dir / 'camera_robot.json'}")
    return cal


# ---------------------------------------------------------------- self test
def solve_view_cam(T_fc, tgt, tilt, az, d, seed):
    """Joints putting this camera d mm from tgt, tilt deg from vertical at azimuth az, aimed at tgt (or None)."""
    sys.path.insert(0, str(ROOT / "arm_pc"))
    import orbit_here as oh
    cam = oh.Camera(T_fc[:3, 3], T_fc[:3, 2])
    pos, view = oh.view_pose(np.asarray(tgt, float), az, 90.0 - tilt, d)
    for s in [np.asarray(seed, float)] + oh.generic_seeds(pos):
        q = oh.solve_view(cam, pos, view, s, scale=d)
        if q is not None:
            return q
    return None


def synthetic(seed=0, noise=0.3, markers=None, centre=(0.0, -460.0, -30.0)):
    """Truth + pictures from arm poses that look at a point 460 mm in front of the base, like the real setup.
    markers: {id: centre mm} to place the cubes elsewhere (id 3 = a side face, vertical)."""
    rng = np.random.default_rng(seed)
    size = (1600, 900)
    K = np.array([[1290.0, 0, 812], [0, 1290.0, 441], [0, 0, 1]])
    dist = np.array([0.08, -0.12, 0, 0, 0])
    T_fc = np.eye(4)
    # camera axis perpendicular to the flange axis (head turned 135 deg), image x along the flange axis
    z = np.array([-0.7071, 0.7071, 0.0])
    xa = np.array([0, 0, 1.0])
    T_fc[:3, :3] = np.column_stack([xa, np.cross(z, xa), z])
    T_fc[:3, 3] = CAD_CAMERA_FLANGE + np.array([2.0, -3.0, 1.5])
    s = 39.0
    markers = {m: np.asarray(c, float) for m, c in (markers or {2: (-110.0, -460, 3), 4: (110.0, -455, 3),
                                                                  3: (-80.0, -460, -25)}).items()}
    T_bm = {}
    for m, c in markers.items():
        T = np.eye(4)
        if m == 3:                                     # side face, vertical
            T[:3, :3] = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0.0]]).T
        T[:3, 3] = c
        T_bm[m] = T
    # poses: aim the camera at points around (0, -460, -30) from a cone, solved with a small IK on this model
    images, obs = [], []
    seed_q = np.array([-93.5, -0.2, -43.0, -43.5, 48.6])
    for k, (tilt, az, d) in enumerate([(0, 0, 420), (20, 180, 380), (20, 240, 380), (20, 300, 380), (35, 200, 350),
                                       (35, 260, 340), (35, 320, 360), (35, 20, 360), (50, 180, 330), (50, 240, 330),
                                       (50, 300, 330), (50, 0, 330), (10, 90, 450), (25, 140, 400)]):
        tgt = np.asarray(centre, float) + rng.normal(0, 15, 3) * [1, 1, 0]
        q = solve_view_cam(T_fc, tgt, tilt, az, d, seed_q)
        if q is None:
            continue
        T_bc = flange_pose(q) @ T_fc
        found = 0
        for m, T in T_bm.items():
            T_cm = np.linalg.inv(T_bc) @ T
            if (T_cm[:3, :3] @ np.array([0, 0, 1.0]))[2] > -0.2:   # facing away
                continue
            uv, _ = cv2.projectPoints(corners_obj(s), cv2.Rodrigues(T_cm[:3, :3])[0], T_cm[:3, 3], K, dist)
            uv = uv.reshape(4, 2) + rng.normal(0, noise, (4, 2))
            if (uv[:, 0] > 5).all() and (uv[:, 0] < size[0] - 5).all() and (uv[:, 1] > 5).all() and (uv[:, 1] < size[1] - 5).all():
                obs.append((len(images), m, uv))
                found += 1
        if found:
            images.append({"file": f"syn{k}", "view": k, "q": list(q), "markers": []})
        seed_q = q
    truth = {"K": K, "dist": dist, "T_fc": T_fc, "markers": markers}
    return images, obs, size, s, truth


def cmd_selftest(a, cfg) -> int:
    images, obs, size, s, truth = synthetic()
    print(f"synthetic: {len(images)} pictures, {len(obs)} marker views")
    prob, x, info = solve_all(images, obs, size, s, say=lambda *_: None)
    K, dist, T_fc, T_bm = prob.unpack(x)
    dc = float(np.linalg.norm(T_fc[:3, 3] - truth["T_fc"][:3, 3]))
    da = math.degrees(math.acos(min(1.0, float(T_fc[:3, 2] @ truth["T_fc"][:3, 2]))))
    print(f"f {K[0, 0]:.1f} (true {truth['K'][0, 0]}), c ({K[0, 2]:.1f}, {K[1, 2]:.1f}) (true 812, 441), "
          f"k1 {dist[0]:+.3f} (true +0.080)")
    print(f"camera centre error {dc:.2f} mm, axis error {da:.3f} deg")
    geo = marker_geometry(prob, x)
    errs = {m: float(np.linalg.norm(P.mean(0) - truth["markers"][m])) for m, (P, _) in geo.items()}
    print("marker centre errors mm", {m: round(e, 2) for m, e in errs.items()})
    ok = dc < 3 and da < 0.5 and abs(K[0, 0] - truth["K"][0, 0]) < 15 and max(errs.values()) < 2
    print("SELFTEST", "OK" if ok else "FAILED")
    return 0 if ok else 1


def cmd_solve(a, cfg) -> int:
    legs = ([[Path(f).resolve() for f in a.snaps]] if a.snaps else []) + [[Path(f).resolve() for f in g] for g in a.leg or []]
    if not legs:
        raise SystemExit("give a snapshots folder, or --leg FOLDER... per cube placement")
    images, obs, size, s = load_legs(legs, cfg)
    print(f"{len(images)} pictures {size[0]}x{size[1]} in {sum(map(len, legs))} folder(s), {len(legs)} leg(s), "
          f"{len(obs)} marker views: " + ", ".join(f"{im['file']} {im['markers']}" for im in images))
    keys = sorted({m for _, m, _ in obs})
    idof = (lambda k: k[1]) if len(legs) > 1 else (lambda k: k)
    scene = {}
    if a.table_z is not None:
        scene = {k: ("top", a.table_z + a.cube_height) for k in keys if idof(k) in a.top_markers}
        scene.update({k: ("side", None) for k in keys if idof(k) in a.side_markers})
        print(f"scene facts: table z {a.table_z} mm, cube tops (markers {a.top_markers}) at z {a.table_z + a.cube_height} mm, "
              f"side markers {a.side_markers} vertical")
    rows = []
    for row in json.loads(Path(a.ruler).read_text(encoding="utf-8")) if a.ruler else []:
        folder = Path(row["folder"]).resolve()
        li = next((i for i, g in enumerate(legs) if folder in g), None)
        snaps = folder / "snapshots.json"
        if li is None or not snaps.exists():
            print(f"ruler answer for {folder.name}: folder not in this solve, skipped")
            continue
        r = next((r for r in json.loads(snaps.read_text(encoding="utf-8"))["snapshots"]["rows"] if r["view"] == row["view"]), None)
        if r is None:
            print(f"ruler answer for {folder.name} view {row['view']}: no such still, skipped")
            continue
        mk = row.get("marker")
        rows.append(dict(row, q=r["q"], leg=li, key=None if mk is None else (mk if len(legs) == 1 else (li, mk))))
    say = print if a.verbose else (lambda *_: None)
    prob, x, info = solve_all(images, obs, size, s, say=say, scene=scene)
    table_pts, ruler = [], []
    if rows:                                       # head bottom from the first fit's camera axis (head turn)
        axis = prob.unpack(x)[2][:3, 2]
        for row in rows:
            hb = head_bottom(row["q"], axis)
            row["head_bottom_mm"] = [round(float(v), 2) for v in hb]
            if row["to"] == "table":
                table_pts.append([float(hb[0]), float(hb[1]), float(hb[2] - row["gap_mm"])])
            else:
                ruler.append((row["key"], float(hb[2] - row["gap_mm"]), a.ruler_sigma))
        print(f"ruler facts: {len(ruler)} marker heights, {len(table_pts)} table points (sigma {a.ruler_sigma:g} mm)")
        if ruler:
            prob, x, info2 = solve_all(images, obs, size, s, say=say, x0=x, scene=scene, ruler=ruler)
            info.update({k: v for k, v in info2.items() if k == "dropped_observations"})
    info["scene_facts"] = {marker_name(k): v for k, v in scene.items()}
    folders = [str(f) for g in legs for f in g]
    cal = report_and_save(images, obs, size, s, prob, x, info, a.cube_height, Path(a.out),
                          folders[0] if len(folders) == 1 else folders, table_pts=table_pts, ruler_rows=rows)
    if a.table_z is not None:                      # the ruler is the reference for the table height
        cal["table_z_mm"] = float(a.table_z)
        cal["table_z_fitted_mm"] = cal.get("table_z_fitted_mm")
        (Path(a.out) / "camera_robot.json").write_text(json.dumps(cal, indent=1), encoding="utf-8")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m scan3d.handeye", description=__doc__.split("\n\n")[0])
    add_config_args(ap)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("solve")
    s.add_argument("snaps", nargs="*", help="folder(s) from orbit_here.py --snapshots (snapshots.json + jpgs), "
                   "all with the cubes in the same place")
    s.add_argument("--leg", action="append", nargs="+", metavar="FOLDER",
                   help="folders of one cube placement; repeat --leg after moving the cubes (each leg's markers are "
                   "fitted separately; their feet give the table plane)")
    s.add_argument("--ruler", help="JSON list of ruler answers (scan3d/calib_session.py answer): {folder, view, to: "
                   "marker|table, marker, gap_mm}: gap from the head's lowest point down to a cube top / the table")
    s.add_argument("--ruler-sigma", type=float, default=1.5, help="ruler precision, mm")
    s.add_argument("--out", default=str(CALIB_DIR))
    s.add_argument("--cube-height", type=float, default=53.0, help="cube-top markers above the table, mm")
    s.add_argument("--table-z", type=float, help="table height in the robot frame from a ruler (arm on a 40 mm "
                   "pedestal: -40); adds the facts below to the fit")
    s.add_argument("--top-markers", type=lambda t: [int(v) for v in t.split(",")], default=[2, 4],
                   help="markers on the cube TOPS (horizontal, at table + cube height)")
    s.add_argument("--side-markers", type=lambda t: [int(v) for v in t.split(",")], default=[3],
                   help="markers on cube SIDES (vertical)")
    s.add_argument("--verbose", action="store_true")
    sub.add_parser("selftest")
    a = ap.parse_args(argv)
    cfg = config_from_args(a)
    return {"solve": cmd_solve, "selftest": cmd_selftest}[a.cmd](a, cfg)


if __name__ == "__main__":
    sys.exit(main())
