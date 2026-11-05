"""Scan scale and robot frame from the calibrated arm: no markers needed.

    python -m scan3d.armpose attach --scan <id> --log arm_pc/runs/<stamp>_log.json   # a past scan (orbit_scan does it)
    python -m scan3d.armpose frames --scan <id>                                      # the rest frames it would use

orbit_here.py logs, for every view, when the arm arrived and the joints read while it stood still (the dwell). The
frames the Pi captured inside those rest windows (received.csv: Pi capture time on this PC's clock) have a known
camera pose: forward kinematics + T_flange_camera from calib/camera_robot.json. COLMAP gives the same cameras'
centres in its own units; a similarity fit (Umeyama) between the two sets gives the scale (mm per COLMAP unit) and
the transform into the robot frame. scan3d.reconstruct uses it when the scan folder has arm.json.

arm.json = the orbit log + the PC time of START + a copy of the calibration used (a later recalibration must not
change old scans).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from .config import ROOT, add_config_args, config_from_args

sys.path.insert(0, str(ROOT / "calib"))
from pix2robot import Calibration  # noqa: E402

ARM_JSON = "arm.json"
CALIB = ROOT / "calib" / "camera_robot.json"


def write(scan_dir: str | Path, log_path: str | Path, calib_path: str | Path = CALIB, dwell_s=None) -> Path:
    """arm.json in the scan folder. Older logs have no start_pc / t_arrive / q_read: START is then the capture's
    motion start (the Enter pressed on orbit_here's START line, a few ms late) and the rest window the last dwell_s
    seconds before each view's time."""
    scan_dir, log_path = Path(scan_dir), Path(log_path)
    log = json.loads(log_path.read_text(encoding="utf-8"))
    start, src = log.get("start_pc"), "orbit_here"
    if start is None:
        cap = json.loads((scan_dir / "capture.json").read_text(encoding="utf-8"))
        start, src = cap["timings"]["motion_start_pc"], "capture.json motion_start_pc"
    dwell = log.get("dwell_s", dwell_s)
    out = {"log": str(log_path), "start_pc": start, "start_source": src,
           "dwell_s": dwell if isinstance(dwell, list) or dwell is None else [float(dwell)],
           "q0": log.get("q0"), "views": log["views"], "calibration_path": str(calib_path),
           "calibration": json.loads(Path(calib_path).read_text(encoding="utf-8"))}
    p = scan_dir / ARM_JSON
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return p


def load(scan_dir: str | Path) -> dict | None:
    p = Path(scan_dir) / ARM_JSON
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def rest_windows(arm: dict) -> list[tuple[int, float, float, list]]:
    """(view, start, end, joints) on this PC's clock while the arm stood still at each view."""
    dw = arm.get("dwell_s") or [0.6]
    out = []
    for k, v in enumerate(arm["views"]):
        end = arm["start_pc"] + float(v["t"])
        start = arm["start_pc"] + float(v["t_arrive"]) if "t_arrive" in v else end - float(dw[min(k, len(dw) - 1)])
        out.append((int(v["i"]), start, end, v.get("q_read") or v["q"]))
    return out


def rest_frames(scan_dir: str | Path, arm: dict, margin_s: float = 0.1, per_view: int = 2) -> list[dict]:
    """Received frames captured while the arm stood still: up to per_view per view, nearest the window's middle.
    margin_s is dropped at the start of each window (the arm settling)."""
    rows = []
    with open(Path(scan_dir) / "received.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("capture_pc"):
                rows.append((float(r["capture_pc"]), f"{int(r['index']):06d}.jpg"))
    frames = []
    for view, a, b, q in rest_windows(arm):
        a += margin_s
        hits = sorted((abs(t - (a + b) / 2), name, t) for t, name in rows if a <= t <= b)
        frames += [{"name": name, "view": view, "q": [float(x) for x in q], "t_pc": t} for _, name, t in hits[:per_view]]
    return frames


def umeyama(A: np.ndarray, B: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """s, R, t minimising sum |s R a + t - b|^2 over the rows of A and B (N x 3)."""
    ma, mb = A.mean(0), B.mean(0)
    a, b = A - ma, B - mb
    U, S, Vt = np.linalg.svd(b.T @ a / len(A))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / (a ** 2).sum(1).mean())
    return s, R, mb - s * R @ ma


def align(images: dict, frames: list[dict], cal: Calibration, min_frames: int = 5, min_views: int = 4,
          max_residual_mm: float = 5.0) -> dict:
    """images: COLMAP model["images"] (name -> R, t, center). Scale (mm per COLMAP unit) and T (4x4, COLMAP units ->
    robot mm) from the rest frames COLMAP registered; frames that disagree (> 3x the median, at least 3 mm) are
    dropped. ok is False, with a reason, when too few frames are left or they disagree by more than
    max_residual_mm."""
    use = [f for f in frames if f["name"] in images]
    rep = {"ok": False, "rest_frames": len(frames), "registered": len(use)}
    views = {f["view"] for f in use}
    if len(use) < min_frames or len(views) < min_views:
        return rep | {"reason": f"{len(use)} rest frames from {len(views)} views registered "
                                f"(need {min_frames} from {min_views})"}
    A = np.array([images[f["name"]]["center"] for f in use])
    poses = [cal.camera_pose(f["q"]) for f in use]
    B = np.array([T[:3, 3] for T in poses])
    sv = np.linalg.svd(B - B.mean(0), compute_uv=False)
    if sv[1] / np.sqrt(len(B)) < 20.0:
        return rep | {"reason": "the rest poses are almost on a line: the scale and frame are not defined"}
    keep = np.ones(len(use), bool)
    for _ in range(5):
        s, R, t = umeyama(A[keep], B[keep])
        res = np.linalg.norm((s * (R @ A.T)).T + t - B, axis=1)
        new = res <= max(3.0 * float(np.median(res[keep])), 3.0)
        if (new == keep).all():
            break
        keep = new
        if keep.sum() < min_frames:
            return rep | {"reason": f"only {int(keep.sum())} rest frames agree with each other"}
    ang = []
    for f, T, k in zip(use, poses, keep):
        if k:
            Rc = R @ images[f["name"]]["R"].T                     # COLMAP camera axes in the robot frame
            ang.append(np.degrees(np.arccos(np.clip((np.trace(Rc.T @ T[:3, :3]) - 1) / 2, -1, 1))))
    med = float(np.median(res[keep]))
    T4 = np.eye(4)
    T4[:3, :3], T4[:3, 3] = s * R, t
    rep.update(scale=s, T_colmap_to_robot=T4.round(6).tolist(), used=int(keep.sum()),
               views=len({f["view"] for f, k in zip(use, keep) if k}),
               dropped=[f["name"] for f, k in zip(use, keep) if not k],
               residual_mm={"median": round(med, 2), "max": round(float(res[keep].max()), 2)},
               rotation_deg={"median": round(float(np.median(ang)), 2), "max": round(float(max(ang)), 2)})
    if med > max_residual_mm:
        return rep | {"reason": f"camera centres disagree by {med:.1f} mm (median) > {max_residual_mm} mm"}
    return rep | {"ok": True}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m scan3d.armpose", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    sub = ap.add_subparsers(dest="cmd", required=True)
    at = sub.add_parser("attach", help="write arm.json into a scan folder from an orbit_here log")
    at.add_argument("--scan", required=True, help="scan id (data/scans/<id>)")
    at.add_argument("--log", required=True, help="arm_pc/runs/<stamp>_log.json of the orbit that made the scan")
    at.add_argument("--calib", default=str(CALIB))
    at.add_argument("--dwell", type=float, default=0.6, help="dwell per view, for logs without t_arrive (s)")
    fr = sub.add_parser("frames", help="list the rest frames of a scan with arm.json")
    fr.add_argument("--scan", required=True)
    a = ap.parse_args(argv)
    cfg = config_from_args(a)
    scan = ROOT / cfg["paths"]["data_dir"] / "scans" / a.scan
    if a.cmd == "attach":
        print(f"-> {write(scan, a.log, a.calib, a.dwell)}")
    arm = load(scan)
    if arm is None:
        print(f"no {ARM_JSON} in {scan}")
        return 2
    s = cfg["reconstruct"]["common"]
    frames = rest_frames(scan, arm, s["arm_rest_margin_s"], s["arm_rest_per_view"])
    print(f"{len(frames)} rest frames from {len({f['view'] for f in frames})}/{len(arm['views'])} views "
          f"(START from {arm['start_source']}): {' '.join(f['name'] for f in frames)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
