"""Physical check of the camera <-> robot calibration: the camera goes straight above an object it has seen.

    python -m scan3d.hover_check --ip 192.168.1.111            # DRY RUN: find the object, plan, picture; no motion
    python -m scan3d.hover_check --ip 192.168.1.111 --go       # moves (orbit_here.py asks y/N; e-stop manned)
    python -m scan3d.hover_check --analyse arm_pc/runs/<stamp>_hover    # the hover still of a past run, again

1. Reads the joints twice around a Pi still (the arm must stand still, camera looking down at the table).
2. The object = the largest red blob (the red mouse), or --pixel u,v from another detector; with
   calib/camera_robot.json, its top (table + --height) in robot mm.
3. The cube-top ArUco markers (aruco_scan in config.json) in the same still -> one keep-out box per cube: the cubes
   (53 mm) are taller than the mouse (35 mm), so orbit_here's object cylinder does not cover them.
4. arm_pc/orbit_here.py --calib --target: one view with the lens --above mm straight above the top, --dwell s there
   (time for a ruler), a still + the joints at the hover, back to the start pose. The dry run prints what a ruler
   should read (lens and head bottom above the object's top) and the smallest gaps on the way.
5. Analysis of the hover still: the object's centre vs the pixel the calibration puts it at, in mm.

Everything goes to arm_pc/runs/<stamp>_hover/: before.jpg, hover_target.json, plan.json/.png, start.jpg, 000.jpg,
snapshots.json, log.json, hover_check.jpg/.json.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .config import add_config_args, config_from_args
from .locate import CALIB, Calibration, find_object, locate, red_blob  # noqa: F401
from .orbit_scan import CHILD_ENV, KIT, ORBIT_PY, Streamer, py


def predict(q0, q, cal: Calibration, target, boxes, radius: float) -> dict:
    """What a ruler should read at the hover pose q, and the smallest gaps (collision spheres, which stick out
    10-25 mm beyond the real surfaces) on the straight joint move q0 -> q."""
    sys.path[:0] = [str(KIT), str(KIT / "model")]
    import orbit_here as oh
    from build_robot_model import HEAD_STL, cad_to_head, load_stl
    from robot_geometry import MOVING, RobotGeometry

    geo = RobotGeometry.load()
    turn = geo.head_turn(cal.T_fc[:3, 2])
    F = oh.fk_frames(q)[4]
    verts = cad_to_head(load_stl(HEAD_STL)).reshape(-1, 3) @ (F[:3, :3] @ turn).T + F[:3, 3]
    lens = (F @ cal.T_fc)[:3, 3]
    hc, hr = cal.head_sphere_world(q)
    top = float(target[2])
    obst = {f"cube {k}": (np.array([b[0], b[1], cal.table_z]), np.array([b[2], b[3], b[4]])) for k, b in enumerate(boxes)}
    obst["object"] = (np.array([target[0] - radius, target[1] - radius, cal.table_z]),
                      np.array([target[0] + radius, target[1] + radius, top]))
    gaps = {k: (1e9, "") for k in obst}
    q0, q = np.asarray(q0, float), np.asarray(q, float)
    for t in np.linspace(0, 1, 61):
        w = geo.world(oh.fk_frames(q0 + (q - q0) * t), turn)
        for k, (lo, hi) in obst.items():
            for n in MOVING:
                g = float(oh.box_gap(*w[n], lo, hi).min())
                if g < gaps[k][0]:
                    gaps[k] = (g, n)
    return {"lens_above_top_mm": round(float(lens[2] - top), 1),
            "lens_off_axis_mm": round(float(np.hypot(*(lens[:2] - np.asarray(target[:2])))), 1),
            "head_cad_bottom_above_top_mm": round(float(verts[:, 2].min() - top), 1),
            "head_sphere_bottom_above_top_mm": round(float(hc[2] - hr - top), 1),
            "min_gap_on_the_way_mm": {k: [round(g, 1), n] for k, (g, n) in gaps.items()}}


def orbit_cmd(a, cal_path, target, boxes, run_dir: Path, url: str, go: bool) -> list[str]:
    cmd = py(ORBIT_PY, "--ip", a.ip, "--calib", cal_path, "--target=" + ",".join(f"{v:.1f}" for v in target),
             "--tilts", "0", "--span", "0", "--orbit-dist", f"{a.above:g}", "--obj-height", f"{a.height:g}",
             "--obj-radius", f"{a.radius:g}", "--self-margin", "10", "--speed", f"{a.speed:g}", "--dwell", f"{a.dwell:g}",
             "--out", run_dir / "plan.json", "--log", run_dir / "log.json", "--snapshots", run_dir, "--snapshot-url", url)
    cmd += ["--box=" + ",".join(f"{v:g}" for v in b) for b in boxes]
    return cmd + (["--go"] if go else [])


def run_child(cmd, stdin=subprocess.DEVNULL) -> int:
    print("$", " ".join(str(c) for c in cmd[1:]), flush=True)
    p = subprocess.Popen(cmd, cwd=KIT, stdin=stdin, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=CHILD_ENV)
    s = Streamer(p, "  [arm] ")
    s.start()
    try:
        code = p.wait()
    except KeyboardInterrupt:
        print("\nCtrl+C: orbit_here.py stops the arm (it got the Ctrl+C too)")
        code = p.wait()
    s.join()
    return code


def analyse(run_dir: Path, cal: Calibration) -> int:
    info = json.loads((run_dir / "hover_target.json").read_text(encoding="utf-8"))
    if info.get("calib") and Path(info["calib"]).exists():          # the calibration the run moved with
        cal = Calibration.load(info["calib"])
    snaps = run_dir / "snapshots.json"
    rows = json.loads(snaps.read_text(encoding="utf-8"))["snapshots"]["rows"] if snaps.exists() else []
    row = next((r for r in rows if r["view"] != "start"), None)
    if row is None:
        print(f"no still at the hover in {run_dir}")
        return 2
    img = cv2.imread(str(run_dir / row["file"]))
    if img.shape[1::-1] != (cal.width, cal.height):
        cal = cal.for_size(img.shape[1], img.shape[0])
    q, target = row["q"], np.array(info["target_mm"], float)
    pred = cal.robot_to_pixel(target, q)
    out = {"still": row["file"], "q": q, "target_mm": target.tolist(), "predicted_pixel": [round(v, 1) for v in pred],
           "mm_per_px": round(cal.mm_per_pixel(q, target[2]), 4)}
    print(f"hover still {row['file']}: the calibration puts the object's top centre at pixel ({pred[0]:.0f}, {pred[1]:.0f}) "
          f"({out['mm_per_px']:.3f} mm/px there)")
    vis = img.copy()
    cv2.drawMarker(vis, (int(round(pred[0])), int(round(pred[1]))), (0, 255, 0), cv2.MARKER_CROSS, 60, 3)
    blob = find_object(img, q, cal, info.get("find", "red"), info.get("height_mm", 0.0), info.get("diameter_mm"))
    if blob is None:
        print("  object not found in the hover still: look at hover_check.jpg (green cross = where its centre should be)")
    else:
        meas = blob["pixel"]
        d = cal.pixel_to_plane(*meas, q, target[2]) - cal.pixel_to_plane(*pred, q, target[2])
        out.update(seen_pixel=[round(v, 1) for v in meas], offset_mm=[round(float(d[0]), 1), round(float(d[1]), 1)],
                   offset_norm_mm=round(float(np.hypot(d[0], d[1])), 1), cut=blob["cut"])
        print(f"  object's centre seen at ({meas[0]:.0f}, {meas[1]:.0f}): {out['offset_norm_mm']:.1f} mm from where the "
              f"calibration put it (robot dx {d[0]:+.1f}, dy {d[1]:+.1f} mm)")
        if blob["cut"]:
            print(f"  ! the object runs off the picture ({', '.join(blob['cut'])}): its centre is biased along that "
                  "direction; trust only the other one")
        cv2.drawMarker(vis, (int(round(meas[0])), int(round(meas[1]))), (255, 0, 0), cv2.MARKER_TILTED_CROSS, 50, 3)
        x, y, w, h = blob["box_px"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imwrite(str(run_dir / "hover_check.jpg"), vis)
    (run_dir / "hover_check.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    pr = info.get("predicted", {})
    if pr:
        print(f"  ruler check: lens {pr['lens_above_top_mm']:.0f} mm, head bottom (CAD) {pr['head_cad_bottom_above_top_mm']:.0f}"
              f" mm above the object's top (table z {cal.table_z:.1f} in this calibration)")
    print(f"-> {run_dir / 'hover_check.jpg'}  (green + = predicted, blue x = seen)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m scan3d.hover_check", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--ip", default=os.environ.get("XARM_IP"), help="xArm IP (or env XARM_IP)")
    ap.add_argument("--go", action="store_true", help="MOVE (orbit_here.py still asks y/N): someone at the e-stop")
    ap.add_argument("--above", type=float, default=120,
                    help="lens above the object's top, mm (120: the head's bottom ~110 mm above it; lower needs a "
                    "wrist flip or brings the elbow down next to the cubes)")
    ap.add_argument("--height", type=float, default=35, help="object height above the table, mm (red mouse 35)")
    ap.add_argument("--radius", type=float, default=60, help="object keep-out radius, mm (mouse 98 x 58: 57)")
    ap.add_argument("--pixel", type=lambda t: [float(v) for v in t.split(",")],
                    help="object centre u,v in the still (default: --find)")
    ap.add_argument("--find", choices=("red", "disk"), default="red",
                    help="red: the largest red blob (mouse); disk: a white disk of --diameter mm on top (white cup "
                    "upside down: 44); also used to find it again in the hover still")
    ap.add_argument("--diameter", type=float, help="--find disk: diameter of the disk, mm")
    ap.add_argument("--cube-half", type=float, default=32, help="half size of each cube's keep-out box, mm")
    ap.add_argument("--dwell", type=float, default=30, help="seconds at the hover (time for the ruler)")
    ap.add_argument("--speed", type=float, default=10, help="joint speed deg/s")
    ap.add_argument("--calib", default=str(CALIB))
    ap.add_argument("--analyse", metavar="DIR", help="only analyse the hover still of a past run")
    a = ap.parse_args(argv)
    a.calib = str(Path(a.calib).resolve())                  # orbit_here.py runs in arm_pc/
    cfg = config_from_args(a)
    try:
        sys.stdout.reconfigure(errors="replace")
    except AttributeError:
        pass
    cal = Calibration.load(a.calib)
    if a.analyse:
        return analyse(Path(a.analyse), cal)
    if not a.ip:
        ap.error("--ip <arm ip> (or set XARM_IP)")

    try:
        loc = locate(a.ip, a.height, a.pixel, cfg, a.calib, a.cube_half, find=a.find, diameter_mm=a.diameter)
    except RuntimeError as e:
        print(e)
        return 2
    q0, target, boxes, url = np.array(loc["q"]), np.array(loc["top_mm"]), loc["boxes"], loc["snapshot_url"]
    run_dir = KIT / "runs" / (time.strftime("%Y%m%d-%H%M%S") + "_hover")
    run_dir.mkdir(parents=True)
    (run_dir / "before.jpg").write_bytes(loc["jpeg"])
    print(f"joints {np.round(q0, 2).tolist()}, still -> {run_dir / 'before.jpg'}")
    info = {"q0": q0.tolist(), "pixel": loc["pixel"], "height_mm": a.height, "find": a.find, "diameter_mm": a.diameter,
            "target_mm": target.tolist(), "boxes": boxes,
            "above_mm": a.above, "calib": str(a.calib), "table_z_mm": cal.table_z}

    code = run_child(orbit_cmd(a, a.calib, target, boxes, run_dir, url, go=False))
    plan_path = run_dir / "plan.json"
    if code != 0 or not plan_path.exists():
        print(f"dry run failed (exit {code})")
        return 2
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if np.abs(np.array(plan["q0"]) - q0).max() > 0.1:
        print("the arm moved between the still and the plan: run again")
        return 2
    pose = plan["poses"][0]
    if not pose["q"]:
        print(f"no safe hover {a.above:g} mm above it: {pose['note']}. Try a larger --above")
        return 2
    subprocess.run(py(KIT / "tools" / "render_plan.py", plan_path), cwd=KIT, env=CHILD_ENV, capture_output=True)
    info["predicted"] = pr = predict(q0, pose["q"], cal, target, boxes, a.radius)
    (run_dir / "hover_target.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(f"\nat the hover: lens {pr['lens_above_top_mm']:.0f} mm above the object's top ({pr['lens_off_axis_mm']:.1f} mm "
          f"off its axis); head bottom {pr['head_cad_bottom_above_top_mm']:.0f} mm (CAD), "
          f"{pr['head_sphere_bottom_above_top_mm']:.0f} mm (120 mm sphere)")
    print("smallest gaps on the way (collision spheres, real surfaces are 10-25 mm further): "
          + ", ".join(f"{k} {g:.0f} mm ({n})" for k, (g, n) in pr["min_gap_on_the_way_mm"].items()))
    print(f"plan picture: {plan_path.with_suffix('.png')}")
    if not a.go:
        print("\ndry run only: nothing moved. Add --go (someone at the e-stop) to hover and take the still.")
        return 0
    code = run_child(orbit_cmd(a, a.calib, target, boxes, run_dir, url, go=True), stdin=None)
    if code != 0:
        print(f"orbit_here.py exit {code}")
        return code
    return analyse(run_dir, cal)


if __name__ == "__main__":
    sys.exit(main())
