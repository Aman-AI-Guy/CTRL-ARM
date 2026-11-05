"""One command on THIS PC: camera aimed at the object the whole time while the arm circles it, recorded by the Pi.

    python -m scan3d.orbit_scan --ip 192.168.1.xxx                  # DRY RUN: aim check + plan + picture, no motion
    python -m scan3d.orbit_scan --ip ... --go --max-views 2 --pause --speed 10   # first real test (e-stop!)
    python -m scan3d.orbit_scan --ip ... --go --recon               # whole orbit, recorded, then preview mesh
    python -m scan3d.orbit_scan --ip ... --find disk --diameter 44 --obj-height 55   # the white cup upside down

What it does:
  1. where the object is: with calib/camera_robot.json (default), scan3d.locate finds it in a still (largest red blob,
     --find disk: a white disk on its top, or --pixel) in robot mm, and the marker cubes as keep-out boxes; the orbit
     circles that point (orbit_here.py --calib --target --box). Without it (--no-calib, or --dist given): the aim
     check (scan3d.aim_check), the object must be on the crosshair, --dist from the markers;
  2. dry run of arm_pc/orbit_here.py (reads the joints, checks our kinematics against the controller, plans the
     cone of views with the camera head's CAD, collision checks) + arm_pc/tools/render_plan.py picture;
  3. with --go: starts the Pi scan (scan3d.capture), then orbit_here.py --go with the SAME start azimuth and
     direction as the dry run. orbit_here.py asks its own "Move the arm? [y/N]": that answer is the only one that
     moves the arm. Its START / ORBIT END lines mark the motion window in the capture (no Enter presses);
  4. the capture collects the frames; arm.json (orbit_here's log: when the arm stood still at each view, and its
     joints there, + the calibration) goes into the scan folder, so the reconstruction takes its scale and the robot
     frame from the arm, not from markers (scan3d/armpose.py); --recon runs python -m scan3d.reconstruct --scan <id>
     --mode preview. --no-cubes: the cubes are off the table.

The arm is driven only by arm_pc/orbit_here.py (owned by the arm kit; its CLAUDE.md has the safety rules).
"""

from __future__ import annotations

import argparse
import codecs
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from .config import ROOT, add_config_args, config_from_args

KIT = ROOT / "arm_pc"
ORBIT_PY = Path(os.environ.get("ORBIT_SCAN_ORBIT_PY", KIT / "orbit_here.py"))     # env override: tests


def aim(cfg, a) -> dict | None:
    """aim_check on one snapshot; None when the camera is not reachable."""
    import cv2
    import numpy as np

    from .aim_check import line, measure

    url = f"http://{cfg['pi']['host']}:{cfg['pi']['port']}/snapshot.jpg"
    try:
        data = urllib.request.urlopen(url, timeout=10).read()
    except OSError as e:
        print(f"camera: {e}")
        return None
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    sc = cfg["aruco_scan"]
    m = measure(img, sc["dictionary"], float(sc["marker_mm"]), a.marker_top, a.obj_height)
    print("aim:", line(m))
    return m


def orbit_args(a, dist: float | None, loc: dict | None = None) -> list[str]:
    out = ["--ip", a.ip, "--obj-height", str(a.obj_height)]
    if loc:                                       # located with the calibration: orbit around that point
        out += ["--calib", loc["calib"], "--target=" + ",".join(f"{v:.1f}" for v in loc["center_mm"])]
        out += ["--box=" + ",".join(f"{v:g}" for v in b) for b in loc["boxes"]]
    else:
        out += ["--dist", f"{dist:.0f}"]
    for k in ("orbit_dist", "step", "span", "speed", "dwell", "max_views", "head_turn"):
        v = getattr(a, k)
        if v:
            out += ["--" + k.replace("_", "-"), str(v)]
    if a.tilts:
        out += ["--tilts", a.tilts]
    if a.pause:
        out.append("--pause")
    return out


class Streamer(threading.Thread):
    """Forwards a child's output as it comes (prompts without a newline too) and calls on_line for full lines."""

    def __init__(self, proc, prefix, on_line=None):
        super().__init__(daemon=True)
        self.proc, self.prefix, self.on_line, self.lines = proc, prefix, on_line, []

    def run(self):
        buf, fd, start = "", self.proc.stdout.fileno(), True
        dec = codecs.getincrementaldecoder("utf-8")(errors="replace")      # a char may span two reads
        while True:
            chunk = os.read(fd, 4096)
            if not chunk:
                break
            text = dec.decode(chunk).replace("\r\n", "\n")
            for ch in text:
                if start:
                    sys.stdout.write(self.prefix)
                    start = False
                sys.stdout.write(ch)
                if ch == "\n":
                    start = True
            sys.stdout.flush()
            buf += text
            while "\n" in buf:
                ln, buf = buf.split("\n", 1)
                self.lines.append(ln)
                if self.on_line:
                    self.on_line(ln)
        if buf:
            self.lines.append(buf)
            if self.on_line:
                self.on_line(buf)


def py(*args) -> list[str]:
    return [sys.executable, "-u", *map(str, args)]


CHILD_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}     # pipes are cp1252 otherwise


def dry_run(a, dist, loc=None) -> dict | None:
    out = KIT / "runs" / (time.strftime("%Y%m%d-%H%M%S") + "_scan_plan.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = py(ORBIT_PY, *orbit_args(a, dist, loc), "--out", out)
    print("$", " ".join(cmd[1:]), flush=True)
    p = subprocess.Popen(cmd, cwd=KIT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=CHILD_ENV)
    s = Streamer(p, "  [arm] ")
    s.start()
    code = p.wait()
    s.join()
    if code != 0 or not out.exists():
        print(f"dry run failed (exit {code})")
        return None
    plan = json.loads(out.read_text(encoding="utf-8"))
    plan["_path"] = str(out)
    pic = subprocess.run(py(KIT / "tools" / "render_plan.py", out), cwd=KIT, capture_output=True, text=True,
                         encoding="utf-8", errors="replace", env=CHILD_ENV)
    plan["_png"] = str(out.with_suffix(".png")) if pic.returncode == 0 else None
    print(f"plan picture: {plan['_png'] or 'not made: ' + (pic.stdout + pic.stderr).strip()[-200:]}")
    return plan


def attach_arm(scan_dir: Path, arm_log: Path, calib: str) -> Path | None:
    """arm.json in the scan folder: scan3d.reconstruct takes the scale and the robot frame from it."""
    from .armpose import write

    if not arm_log.exists() or not Path(calib).exists():
        print(f"! no arm.json ({arm_log.name if not arm_log.exists() else calib} missing): scale from the markers")
        return None
    try:
        p = write(scan_dir, arm_log, calib)
    except (OSError, KeyError, ValueError) as e:
        print(f"! arm.json not written ({e}): scale from the markers")
        return None
    print(f"arm poses -> {p}")
    return p


def go(a, cfg, dist, plan, loc=None) -> int:
    total_s = float(plan.get("est_s") or 150) + 60
    how = f"calib target {[round(v) for v in loc['center_mm']]}" if loc else f"dist {dist:.0f}"
    cap_cmd = py("-m", "scan3d.capture", "--note", a.note or f"orbit_scan {how}", "--label", "orbit",
                 "--set", f"capture.max_s={int(min(600, max(180, total_s)))}")
    for o in a.overrides:
        cap_cmd += ["--set", o]
    print("$", " ".join(cap_cmd[1:]), flush=True)
    cap = subprocess.Popen(cap_cmd, cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           env=CHILD_ENV)
    recording = threading.Event()
    marks = {"start": 0, "end": 0}

    def cap_line(ln):
        if " recording (" in ln:
            recording.set()

    cs = Streamer(cap, "  [cam] ", cap_line)
    cs.start()
    if not recording.wait(60) or cap.poll() is not None:
        print("the scan did not start: not moving the arm")
        cap.kill()
        return 2

    def enter():
        if cap.poll() is None:
            try:
                cap.stdin.write(b"\n")
                cap.stdin.flush()
            except OSError:
                pass

    def arm_line(ln):
        if ln.startswith("START ") and not marks["start"]:
            marks["start"] = 1
            enter()                                   # capture: motion starts
        elif ln.startswith("ORBIT END") and not marks["end"]:
            marks["end"] = 1
            enter()                                   # capture: motion ends

    arm_log = KIT / "runs" / (time.strftime("%Y%m%d-%H%M%S") + "_log.json")
    args = orbit_args(a, dist, loc) + ["--go", "--log", str(arm_log)]
    if plan.get("start_az") is not None:
        args += ["--start-az", f"{float(plan['start_az']):.3f}"]
    if plan.get("direction"):
        args += ["--direction", str(int(plan["direction"]))]
    arm_cmd = py(ORBIT_PY, *args)
    print("$", " ".join(arm_cmd[1:]), flush=True)
    arm = subprocess.Popen(arm_cmd, cwd=KIT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,   # stdin: the user
                           env=CHILD_ENV)
    s = Streamer(arm, "  [arm] ", arm_line)
    s.start()
    try:
        code = arm.wait()
    except KeyboardInterrupt:
        print("\nCtrl+C: orbit_here.py stops the arm (it got the Ctrl+C too)")
        code = arm.wait()
    s.join()
    for _ in range(2 - marks["start"] - marks["end"]):     # no orbit (declined / error): stop the scan anyway
        enter()
        time.sleep(0.3)
    cap.wait()
    cs.join()
    saved = next((ln.split("Saved:", 1)[1].strip() for ln in cs.lines if ln.startswith("Saved:")), None)
    print(f"\norbit exit {code}, motion window {'complete' if marks['end'] else 'INCOMPLETE'}, scan saved: {saved}")
    if saved:
        attach_arm(Path(saved), arm_log, a.calib)
    if saved and code == 0 and marks["end"] and a.recon:
        sid = Path(saved).name
        r = subprocess.run(py("-m", "scan3d.reconstruct", "--scan", sid, "--mode", "preview"), cwd=ROOT, env=CHILD_ENV)
        return r.returncode
    return 0 if code == 0 else code


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m scan3d.orbit_scan", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--ip", default=os.environ.get("XARM_IP"), help="xArm IP (or env XARM_IP)")
    ap.add_argument("--go", action="store_true", help="record + MOVE (orbit_here.py still asks y/N)")
    ap.add_argument("--dist", type=float, help="camera to the object's mid height; default: from the markers, else 432")
    ap.add_argument("--marker-top", type=float, default=53.0, help="cube-top markers above the table (53 mm)")
    ap.add_argument("--obj-height", type=float, default=35.0, help="object height (red mouse 35 mm)")
    ap.add_argument("--orbit-dist", type=float, help="camera to the object during the orbit (kit default 350)")
    ap.add_argument("--tilts", help="rings in degrees from vertical, e.g. 40 or 35,60 (kit default 40)")
    ap.add_argument("--step", type=float)
    ap.add_argument("--span", type=float)
    ap.add_argument("--speed", type=float)
    ap.add_argument("--dwell", type=float)
    ap.add_argument("--head-turn", type=float)
    ap.add_argument("--max-views", type=int)
    ap.add_argument("--pause", action="store_true")
    ap.add_argument("--recon", action="store_true", help="preview reconstruction after a complete orbit")
    ap.add_argument("--note", default="")
    ap.add_argument("--calib", default=str(ROOT / "calib" / "camera_robot.json"),
                    help="camera <-> robot calibration: the object is located in robot mm from a still (scan3d.locate) "
                    "and the orbit circles that point, with the marker cubes as keep-out boxes")
    ap.add_argument("--no-calib", action="store_true", help="old way: object on the crosshair, --dist from the markers")
    ap.add_argument("--pixel", type=lambda t: [float(v) for v in t.split(",")],
                    help="object centre u,v in the still (default: --find)")
    ap.add_argument("--find", choices=("red", "disk"), default="red",
                    help="red: the largest red blob (mouse); disk: a white disk of --diameter mm on top (white cup "
                    "upside down: 44, with --obj-height 55)")
    ap.add_argument("--diameter", type=float, help="--find disk: diameter of the disk, mm")
    ap.add_argument("--no-cubes", action="store_true",
                    help="the marker cubes are off the table: no keep-out box where the calibration last saw them "
                    "(a cube seen in the still still gets one); the scale comes from the arm (arm.json)")
    a = ap.parse_args(argv)
    a.calib = str(Path(a.calib).resolve())                  # orbit_here.py runs in arm_pc/
    cfg = config_from_args(a)
    try:
        sys.stdout.reconfigure(errors="replace")
    except AttributeError:
        pass
    if not a.ip:
        ap.error("--ip <arm ip> (or set XARM_IP)")
    if a.find == "disk" and not a.diameter:
        ap.error("--find disk needs --diameter")
    loc, dist = None, None
    if not a.no_calib and Path(a.calib).exists() and not a.dist:
        from .locate import locate
        try:
            loc = locate(a.ip, a.obj_height, a.pixel, cfg, a.calib, find=a.find, diameter_mm=a.diameter,
                         remembered_cubes=not a.no_cubes)
            print(f"orbit centre {[round(v, 1) for v in loc['center_mm']]} (object mid height, {Path(a.calib).name}); "
                  "the object does not need to be on the crosshair")
        except (RuntimeError, OSError) as e:
            print(f"locating with the calibration failed ({e}): the old way (markers + crosshair)")
    if loc is None:
        m = aim(cfg, a)
        dist = a.dist or (m or {}).get("dist_mm") or 432.0
        if m and "object_off_mm" in m and max(map(abs, m["object_off_mm"])) > 30:
            print(f"  ! the object is {m['object_off_mm']} mm off the picture centre: the orbit will circle the wrong "
                  "point. Centre it (move the object or the arm) first")
        print(f"--dist {dist:.0f} ({'given' if a.dist else 'from the markers' if m and 'dist_mm' in m else 'default'})")
    plan = dry_run(a, dist, loc)
    if plan is None:
        return 2
    if not a.go:
        print("\ndry run only: nothing moved. Add --go (someone at the e-stop) to record + orbit.")
        return 0
    return go(a, cfg, dist, plan, loc)


if __name__ == "__main__":
    sys.exit(main())
