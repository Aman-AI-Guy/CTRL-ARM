"""Point the arm camera at a named object (a Red Bull can) and move it closer: a first test of how the arm moves.

    python -m scan3d.point_at "red bull can" --ip 192.168.1.111         # DRY RUN: still, Grok, target, plan. No motion
    python -m scan3d.point_at "red bull can" --ip 192.168.1.111 --go    # orbit_here asks y/N, then Enter per stage
    python -m scan3d.point_at "red bull can" --still can.jpg --q0=-93.5,-0.2,-43.0,-43.5,48.6   # rehearsal, no arm

Before: the arm stands in a start pose with the Pi camera looking down at the table (as for scan3d.hover_check, e.g.
joints -93.5, -0.2, -43.0, -43.5, 48.6), the can stands in the picture (near x 0, y -480 mm), someone at the e-stop.

1. A still from the Pi camera, with the joints read before and after it (the arm must stand still).
2. Grok boxes the object in the still (--click: click it yourself; --pixel u,v: give it). The box centre is the can's
   middle: its ray (calib/camera_robot.json) meets the plane at table + height/2 on the can's axis; the target is the
   axis point at the can's top (table + --height).
3. arm_pc/orbit_here.py plans two views straight above the target from the CURRENT joints, with the xArm 5 meshes and
   the camera head CAD for collisions (the marker cubes are keep-out boxes):
     stage 1 AIM:      lens --far mm above the can's top, looking straight down at it;
     stage 2 APPROACH: lens --near mm above it (the arm moves the camera straight towards the can);
   then back to the start joints. The plan is explained joint by joint; nothing moves without --go.
4. --go: orbit_here.py asks "Move the arm? [y/N]", then Enter before each stage; a still at each stage. Ctrl+C stops
   the arm (it holds position).
5. Grok finds the can again in each stage still: how far (mm) it is from where the calibration says it should be.

Everything goes to arm_pc/runs/<stamp>_point/: before.jpg, before_annotated.jpg, point_target.json, plan.json/.png,
000.jpg (stage 1), 001.jpg (stage 2), snapshots.json, log.json, stage1_check.jpg, stage2_check.jpg, stage_check.json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from . import hover_check
from .config import add_config_args, config_from_args
from .locate import CALIB, Calibration, cube_boxes, read_still
from .orbit_scan import CHILD_ENV, KIT, py

JOINTS = ("J1 base turn", "J2 shoulder", "J3 elbow", "J4 wrist pitch", "J5 wrist roll")
STAGES = ("AIM", "APPROACH")


def can_target(cal: Calibration, pixel, q, height_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """(top, middle) of an upright object in robot mm from the centre pixel of its whole outline: the ray through it
    meets the object's axis about half way up, also when the camera looks a little sideways."""
    mid = cal.pixel_to_object_top(pixel[0], pixel[1], q, height_mm / 2)
    top = np.array([mid[0], mid[1], cal.table_z_at(mid[0], mid[1]) + height_mm])
    return top, mid


def click_pixel(img: np.ndarray, title="click the can, then Enter (Esc cancels)") -> list[float] | None:
    scale = min(1.0, 1280 / img.shape[1])
    small = cv2.resize(img, None, fx=scale, fy=scale)
    picked: list[float] = []

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN:
            picked[:] = [x / scale, y / scale]
            show = small.copy()
            cv2.drawMarker(show, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 40, 2)
            cv2.imshow(title, show)

    cv2.imshow(title, small)
    cv2.setMouseCallback(title, on_mouse)
    print("click the middle of the can's whole outline (half way between its lid and its bottom), then Enter")
    while True:
        k = cv2.waitKey(50) & 0xFF
        if k in (10, 13, 32) and picked:
            break
        if k == 27:
            picked.clear()
            break
    cv2.destroyAllWindows()
    return picked or None


def detect(img: np.ndarray, jpeg: bytes, a, say=print) -> dict | None:
    """The object's centre pixel: --pixel, --click, or Grok's box."""
    if a.pixel:
        return {"how": "pixel", "pixel": list(a.pixel), "box_px": None}
    if a.click:
        p = click_pixel(img)
        return {"how": "click", "pixel": p, "box_px": None} if p else None
    from .grok_find import GrokFindError, find_box

    say(f'asking Grok for "{a.what}" ...')
    try:
        r = find_box(jpeg, a.what, (img.shape[1], img.shape[0]))
    except GrokFindError as e:
        say(str(e))
        return None
    say(f"Grok ({r['model']}, {r['latency_s']:.1f} s): {r['label']}, confidence {r['confidence']:.2f}, box "
        f"{[round(v) for v in r['box_px']]}, centre pixel ({r['pixel'][0]:.0f}, {r['pixel'][1]:.0f})")
    return {"how": "grok", **r}


def annotate(img, det: dict, cal: Calibration, q, top, mid) -> np.ndarray:
    vis = img.copy()
    if det.get("box_px"):
        x0, y0, x1, y1 = (int(round(v)) for v in det["box_px"])
        cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 0, 0), 3)
    u, v = (int(round(c)) for c in det["pixel"])
    cv2.drawMarker(vis, (u, v), (255, 0, 0), cv2.MARKER_TILTED_CROSS, 40, 3)
    try:
        pt = cal.robot_to_pixel(top, q)
        cv2.drawMarker(vis, (int(round(pt[0])), int(round(pt[1]))), (0, 255, 0), cv2.MARKER_CROSS, 60, 3)
    except ValueError:
        pass
    text = f"target top x {top[0]:.0f} y {top[1]:.0f} z {top[2]:.0f} mm (middle z {mid[2]:.0f})"
    cv2.putText(vis, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(vis, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    return vis


def joint_moves(q_from, q_to) -> str:
    d = np.asarray(q_to, float) - np.asarray(q_from, float)
    parts = [f"{JOINTS[i]} {d[i]:+.1f} deg" for i in range(5) if abs(d[i]) >= 0.5]
    return ", ".join(parts) if parts else "no joint turns more than 0.5 deg"


def explain(q0, plan: dict, cal: Calibration, top, boxes, a, say=print) -> list[dict]:
    """The plan in words: for each stage where the camera goes, which joints turn and by how much, how close the arm
    comes to the can and the cubes. Returns the per-stage numbers."""
    q0 = np.asarray(q0, float)
    cam0 = cal.camera_pose(q0)
    tilt0 = math.degrees(math.acos(max(-1.0, min(1.0, float(-cam0[2, 2])))))
    say("\nWHAT THE ARM WILL DO (J1 + turns counter-clockwise seen from above, J2 + leans the arm forward)")
    say(f"  start     joints [{', '.join(f'{v:.1f}' for v in q0)}], camera at ({cam0[0, 3]:.0f}, {cam0[1, 3]:.0f}, "
        f"{cam0[2, 3]:.0f}) mm, looking {tilt0:.0f} deg away from straight down")
    stages, prev, total = [], q0, 0.0
    for k, pose in enumerate(plan["poses"]):
        name = STAGES[k] if k < len(STAGES) else f"VIEW {k}"
        dist = (a.far, a.near)[k] if k < 2 else None
        if not pose.get("q"):
            say(f"  stage {k + 1}  {name}: NOT POSSIBLE ({pose.get('note', 'no solution')})")
            stages.append({"stage": k + 1, "name": name, "q": None, "note": pose.get("note")})
            continue
        q = np.asarray(pose["q"], float)
        pr = hover_check.predict(prev, q, cal, top, boxes, a.radius)
        secs = float(np.abs(q - prev).max()) / a.speed
        total += secs + a.dwell
        cam = pose.get("camera_mm") or cal.camera_pose(q)[:3, 3].tolist()
        what = ("turn so that the camera is straight above the can and looks down at it" if k == 0 else
                f"move the camera {a.far - a.near:g} mm straight down towards the can" if k == 1 else "next view")
        say(f"  stage {k + 1}  {name}: {what}")
        asked = f" (asked {dist:g})" if dist else ""
        say(f"            camera to ({cam[0]:.0f}, {cam[1]:.0f}, {cam[2]:.0f}) mm: lens {pr['lens_above_top_mm']:.0f} mm "
            f"above the can's top{asked}, {pr['lens_off_axis_mm']:.0f} mm off its axis; head bottom "
            f"{hover_check.head_bottom(pr)} mm above it")
        say(f"            {joint_moves(prev, q)}  (~{secs:.1f} s at {a.speed:g} deg/s, then {a.dwell:g} s still + picture)")
        gaps = ", ".join(f"{n} {g:.0f} mm ({link})" for n, (g, link) in pr["min_gap_on_the_way_mm"].items())
        say(f"            closest on the way (collision spheres; real surfaces are 10-25 mm further): {gaps}")
        stages.append({"stage": k + 1, "name": name, "q": q.tolist(), "joint_change_deg": (q - prev).round(2).tolist(),
                       "seconds": round(secs, 1), **pr})
        prev = q
    back = float(np.abs(q0 - prev).max()) / a.speed
    say(f"  back      to the start joints: {joint_moves(prev, q0)}  (~{back:.1f} s)")
    say(f"  in all    ~{total + back:.0f} s of motion and pauses, plus the time you take to press Enter")
    return stages


def check_stages(run_dir: Path, cal: Calibration, info: dict, say=print) -> list[dict]:
    """Each stage still: where the calibration puts the can's top centre and, with Grok, where it is seen."""
    snaps = run_dir / "snapshots.json"
    rows = json.loads(snaps.read_text(encoding="utf-8"))["snapshots"]["rows"] if snaps.exists() else []
    out = []
    for k, row in enumerate(r for r in rows if r["view"] != "start"):
        img = cv2.imread(str(run_dir / row["file"]))
        if img is None:
            continue
        say(f"stage {k + 1} ({STAGES[k] if k < len(STAGES) else 'view'}) still {row['file']}:")
        res, vis = hover_check.analyse_still(img, row["q"], info["target_mm"], cal, info, say=say)
        cv2.imwrite(str(run_dir / f"stage{k + 1}_check.jpg"), vis)
        out.append({"stage": k + 1, "still": row["file"], **res})
    (run_dir / "stage_check.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    if not out:
        say(f"no stage stills in {run_dir}")
    elif info.get("find") != "grok":
        say("(picked by hand: no automatic re-find; the green + in stage*_check.jpg is where the can's top should be)")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m scan3d.point_at", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("what", nargs="?", default="red bull can", help='what to point at (default "red bull can")')
    ap.add_argument("--ip", default=os.environ.get("XARM_IP"), help="xArm IP (or env XARM_IP)")
    ap.add_argument("--go", action="store_true", help="MOVE (orbit_here.py still asks y/N): someone at the e-stop")
    ap.add_argument("--q0", type=lambda t: [float(v) for v in t.split(",")],
                    help="offline rehearsal: start joints J1..J5 (with --still; no arm, no motion). --q0=-93.5,... ")
    ap.add_argument("--still", help="offline rehearsal: a picture taken from the --q0 joints")
    ap.add_argument("--height", type=float, default=134,
                    help="object height above the table, mm (250 ml can standing 134, 355 ml 168, lying down 53)")
    ap.add_argument("--radius", type=float, default=60, help="object keep-out radius, mm (can 27 + margin)")
    ap.add_argument("--far", type=float, default=300, help="stage 1: lens above the can's top, mm")
    ap.add_argument("--near", type=float, default=150, help="stage 2: lens above the can's top, mm (>= 120)")
    ap.add_argument("--speed", type=float, default=10, help="joint speed deg/s")
    ap.add_argument("--dwell", type=float, default=2, help="seconds still at each stage before its picture")
    ap.add_argument("--pixel", type=lambda t: [float(v) for v in t.split(",")], help="object centre u,v (no Grok)")
    ap.add_argument("--click", action="store_true", help="click the object in the still (no Grok)")
    ap.add_argument("--no-cubes", action="store_true", help="the marker cubes are off the table")
    ap.add_argument("--cube-half", type=float, default=32, help="half size of each cube's keep-out box, mm")
    ap.add_argument("--calib", default=str(CALIB))
    ap.add_argument("--runs", default=str(KIT / "runs"), help="where the run folder goes")
    a = ap.parse_args(argv)
    a.calib = str(Path(a.calib).resolve())                  # orbit_here.py runs in arm_pc/
    a.above = a.far
    if a.near < 120:
        ap.error("--near below 120 mm brings the camera head within ~10 cm of the can's top: not in a first test")
    if a.near >= a.far:
        ap.error("--near must be smaller than --far (stage 2 moves closer)")
    if a.q0 is not None:
        if len(a.q0) != 5 or not a.still:
            ap.error("offline rehearsal needs --q0 with 5 joints and --still")
        if a.go:
            ap.error("--go needs the real arm (--ip), not --q0")
        a.ip = None
    elif not a.ip:
        ap.error("--ip <arm ip> (or set XARM_IP), or --q0 + --still for an offline rehearsal")
    cfg = config_from_args(a)
    try:
        sys.stdout.reconfigure(errors="replace")
    except AttributeError:
        pass

    url = f"http://{cfg['pi']['host']}:{cfg['pi']['port']}/snapshot.jpg"
    cal = Calibration.load(a.calib)
    if a.q0 is not None:
        q0, jpeg = np.array(a.q0, float), Path(a.still).read_bytes()
    else:
        try:
            q0, jpeg = read_still(a.ip, url)
        except (RuntimeError, OSError) as e:
            print(f"could not read the arm and the camera: {e}")
            return 2
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print("the still is not a picture")
        return 2
    if img.shape[1::-1] != (cal.width, cal.height):
        cal = cal.for_size(img.shape[1], img.shape[0])
    run_dir = Path(a.runs).resolve() / (time.strftime("%Y%m%d-%H%M%S") + "_point")
    run_dir.mkdir(parents=True)
    (run_dir / "before.jpg").write_bytes(jpeg)
    print(f"1. still {img.shape[1]}x{img.shape[0]} at joints {np.round(q0, 2).tolist()} -> {run_dir / 'before.jpg'}")

    print(f'2. where is "{a.what}"?')
    det = detect(img, jpeg, a)
    if not det or not det.get("pixel"):
        print("nothing to point at: run again with --click, or --pixel u,v")
        return 2
    try:
        top, mid = can_target(cal, det["pixel"], q0, a.height)
    except ValueError as e:
        print(f"the camera does not look down at the table from these joints ({e}): put the arm in a start pose above "
              "the can first")
        return 2
    reach = math.hypot(top[0], top[1])
    print(f"   -> robot mm: can middle ({mid[0]:.0f}, {mid[1]:.0f}, {mid[2]:.0f}), top centre ({top[0]:.0f}, "
          f"{top[1]:.0f}, {top[2]:.0f}); {reach:.0f} mm from the base axis (table z {cal.table_z:.0f}, "
          f"can {a.height:g} mm)")
    if reach > 600:
        print("   ! more than 600 mm from the base: the planner will probably not reach it; move the can closer")
    boxes = cube_boxes(img, q0, cal, cfg, a.cube_half, say=lambda t: print("   " + t), remembered=not a.no_cubes)
    cv2.imwrite(str(run_dir / "before_annotated.jpg"), annotate(img, det, cal, q0, top, mid))
    info = {"what": a.what, "q0": q0.tolist(), "detection": {k: v for k, v in det.items() if k != "how"},
            "how": det["how"], "pixel": det["pixel"], "height_mm": a.height, "middle_mm": mid.tolist(),
            "target_mm": top.tolist(), "boxes": boxes, "far_mm": a.far, "near_mm": a.near, "calib": a.calib,
            "table_z_mm": cal.table_z, "find": "grok" if det["how"] == "grok" else "none", "prompt": a.what}

    print("3. planning (orbit_here.py dry run: nothing moves)")
    dists = [a.far, a.near]
    code = hover_check.run_child(hover_check.orbit_cmd(a, a.calib, top, boxes, run_dir, url, go=False, dists=dists))
    plan_path = run_dir / "plan.json"
    if code != 0 or not plan_path.exists():
        print(f"dry run failed (exit {code})")
        return 2
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if np.abs(np.array(plan["q0"]) - q0).max() > 0.1:
        print("the arm moved between the still and the plan: run again")
        return 2
    subprocess.run(py(KIT / "tools" / "render_plan.py", plan_path), cwd=KIT, env=CHILD_ENV, capture_output=True)
    info["stages"] = explain(q0, plan, cal, top, boxes, a)
    (run_dir / "point_target.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    png = plan_path.with_suffix(".png")
    print(f"\npictures: {run_dir / 'before_annotated.jpg'} (blue = seen, green + = target top)"
          + (f", {png}" if png.exists() else ""))
    if not all(s.get("q") for s in info["stages"]):
        print("not every stage is reachable: move the can (nearer the start view's centre) or change --far/--near")
        return 2
    if not a.go:
        print("\ndry run only: nothing moved." + ("" if a.ip else " (offline rehearsal)")
              + " Add --go (someone at the e-stop) to move.")
        return 0

    print("\n4. moving: answer y, then press Enter before each stage. Ctrl+C stops the arm.")
    code = hover_check.run_child(hover_check.orbit_cmd(a, a.calib, top, boxes, run_dir, url, go=True, dists=dists,
                                                       pause=True), stdin=None)
    if code != 0:
        print(f"orbit_here.py exit {code}")
        return code
    print("5. how close did it get?")
    check_stages(run_dir, cal, info)
    return 0


if __name__ == "__main__":
    sys.exit(main())
