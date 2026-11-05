"""Guided camera <-> robot calibration (v2): close views of the cubes, ruler checkpoints, cubes moved between legs.

    python -m scan3d.calib_session plan --ip 192.168.1.111 --leg A      # still -> cubes -> segments + dry runs, no motion
    python -m scan3d.calib_session go --ip ... --leg A --seg far         # MOVES one segment (orbit_here.py asks y/N)
    python -m scan3d.calib_session answer --leg A --seg check-2 --gap 141   # ruler: head's lowest point -> cube top
    python -m scan3d.calib_session status                                # what is done / missing
    python -m scan3d.calib_session solve                                 # every leg + ruler answers -> calib/v2/
    python -m scan3d.calib_session promote                               # calib/v2 -> calib/camera_robot.json

Segments of a leg; each starts and ends at the leg's start pose (camera looking down at the table, both cubes in view):
  far          first leg only: rings 15/35/50 deg from vertical, 350 mm from the middle of the cubes, every 40 deg
  ring-<id>    4 views 30 deg from vertical, 170 mm from cube <id>'s top marker (pins the camera's place on the flange)
  check-<id>   straight above cube <id>'s top marker, lens 150 mm up, 20 s still: ruler from the head's lowest point
               down to the cube's top
  check-table  first leg only: straight above bare table between the cubes, lens 200 mm up, 20 s: ruler to the table
Between legs, move both cubes (e.g. 8 cm towards the arm, then 10 cm away from it) and keep them in the picture:
their feet give the table as a plane. The calibration in use (calib/camera_robot.json) only plans the moves; the new
one goes to calib/v2/ and replaces it only with `promote` (the old one is kept as calib/camera_robot_v1.json).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .config import ROOT, add_config_args, config_from_args
from .locate import CALIB, Calibration, cube_boxes, read_still
from .orbit_scan import CHILD_ENV, KIT, ORBIT_PY, Streamer, py

SESSION = KIT / "runs" / "calib2"
V2 = ROOT / "calib" / "v2"
CUBE_MM = 53.0
SEGMENTS = {   # orbit_here options per kind of segment; the target and the keep-outs come from the leg's still
    "far": ["--tilts", "15,35,50", "--step", "40", "--span", "200", "--orbit-dist", "350", "--dwell", "1.5", "--speed", "15"],
    # 3 sides (the view from the base side needs a wrist flip next to the cube), 2 tilts
    "ring": ["--tilts", "25,45", "--step", "90", "--span", "180", "--orbit-dist", "170", "--dwell", "1.5", "--speed", "10"],
    "check": ["--tilts", "0", "--span", "0", "--orbit-dist", "150", "--dwell", "20", "--speed", "10"],
    "check-table": ["--tilts", "0", "--span", "0", "--orbit-dist", "200", "--dwell", "20", "--speed", "10"],
}


def load_session() -> dict:
    f = SESSION / "session.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {"legs": {}, "order": [], "ruler": []}


def save_session(s: dict) -> None:
    SESSION.mkdir(parents=True, exist_ok=True)
    (SESSION / "session.json").write_text(json.dumps(s, indent=1), encoding="utf-8")


def run(cmd, stdin: str = "user") -> tuple[int, str]:
    """orbit_here.py as a child; stdin "user" = the person at the keyboard answers its y/N, "none" = no input."""
    print("$", " ".join(str(c) for c in cmd[1:]), flush=True)
    p = subprocess.Popen(cmd, cwd=KIT, stdin=subprocess.DEVNULL if stdin == "none" else None, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, env=CHILD_ENV)
    s = Streamer(p, "  [arm] ")
    s.start()
    try:
        code = p.wait()
    except KeyboardInterrupt:
        print("\nCtrl+C: orbit_here.py stops the arm (it got the Ctrl+C too)")
        code = p.wait()
    s.join()
    return code, "\n".join(s.lines)


def cmd_plan(a, cfg) -> int:
    from .handeye import head_bottom

    cal = Calibration.load(CALIB)
    url = f"http://{cfg['pi']['host']}:{cfg['pi']['port']}/snapshot.jpg"
    q0, data = read_still(a.ip, url)
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    leg_dir = SESSION / a.leg
    leg_dir.mkdir(parents=True, exist_ok=True)
    (leg_dir / "start.jpg").write_bytes(data)
    cubes = cube_boxes(img, q0, cal, cfg, 32.0, with_ids=True)
    if not all(seen for _, seen, _ in cubes):
        print("a cube-top marker is not in the picture: move the cube (or the arm) so both are seen")
        return 2
    s = load_session()
    first = not s["order"] or s["order"][0] == a.leg
    boxes = [b for _, _, b in cubes]
    tops = {mid: [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[4]] for mid, _, b in cubes}
    mid_xy = np.mean([t[:2] for t in tops.values()], axis=0)
    common = ["--calib", str(CALIB), "--obj-height", f"{CUBE_MM:g}", "--obj-radius", "32", "--self-margin", "10",
              "--snapshot-url", url] + ["--box=" + ",".join(f"{v:g}" for v in b) for b in boxes]
    segs = {}
    if first:
        z = float(np.mean([t[2] for t in tops.values()]))
        segs["far"] = ([*mid_xy, z], SEGMENTS["far"])
        segs["check-table"] = ([*mid_xy, cal.table_z_at(*mid_xy)], SEGMENTS["check-table"])
    check = [a.check_dist if v == "150" and k > 0 and SEGMENTS["check"][k - 1] == "--orbit-dist" else v
             for k, v in enumerate(SEGMENTS["check"])]
    for mid, t in tops.items():
        segs[f"ring-{mid}"] = (t, SEGMENTS["ring"])
        segs[f"check-{mid}"] = (t, check)
    if a.skip:
        segs = {k: v for k, v in segs.items() if k not in a.skip}
    leg = {"q0": [float(v) for v in q0], "time": time.strftime("%H:%M:%S"), "tops_mm": tops, "boxes": boxes,
           "segments": {}}
    print(f"\nleg {a.leg}: start joints {np.round(q0, 2).tolist()}, cube tops "
          + ", ".join(f"{m}: {np.round(t, 0).tolist()}" for m, t in tops.items()))
    for name, (tgt, opts) in segs.items():
        folder = leg_dir / name
        args = ["--target=" + ",".join(f"{v:.1f}" for v in tgt), *opts, *common]
        out = leg_dir / f"{name}_dry.json"
        code, _ = run(py(ORBIT_PY, "--q0=" + ",".join(f"{v:.4f}" for v in q0), *args, "--out", out), stdin="none")
        plan = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
        poses = plan.get("poses", [])
        ok = [p for p in poses if p["q"]]
        info = {"target_mm": [round(v, 1) for v in tgt], "args": args, "folder": str(folder), "done": False,
                "views": f"{len(ok)}/{len(poses)}", "j5_span_deg": round(plan.get("j5_span_deg", 0), 0),
                "max_jump_deg": round(max([p["jump_deg"] for p in ok] or [0]), 1)}
        if name.startswith("check") and ok:              # kept from the screen: the ruler reading comes first
            hb = head_bottom(ok[0]["q"], cal.T_fc[:3, 2])
            info["predicted_gap_mm"] = round(float(hb[2] - tgt[2]), 1)
        leg["segments"][name] = info
        if code == 0 and out.exists():
            subprocess.run(py(KIT / "tools" / "render_plan.py", out), cwd=KIT, env=CHILD_ENV, capture_output=True)
    s["legs"][a.leg] = leg
    if a.leg not in s["order"]:
        s["order"].append(a.leg)
    save_session(s)
    print(f"\nleg {a.leg} planned (dry runs, nothing moved):")
    for name, info in leg["segments"].items():
        print(f"  {name:12s} views {info['views']:5s}  max joint jump {info['max_jump_deg']:5.1f} deg  "
              f"J5 {info['j5_span_deg']:4.0f} deg   picture {leg_dir / (name + '_dry.png')}"
              + ("   ! WRIST FLIP: do not run as is" if info["max_jump_deg"] > 90 else ""))
    return 0


def cmd_go(a, cfg) -> int:
    s = load_session()
    seg = s["legs"].get(a.leg, {}).get("segments", {}).get(a.seg)
    if seg is None:
        print(f"no segment {a.seg} in leg {a.leg}: run plan first")
        return 2
    folder = Path(seg["folder"])
    if folder.exists() and (folder / "snapshots.json").exists() and not a.again:
        print(f"{a.seg} already done ({folder}); --again to redo it")
        return 2
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    if a.seg.startswith("check"):
        what = "the TABLE (bare wood)" if a.seg == "check-table" else f"the TOP of cube {a.seg.split('-')[1]}"
        print(f"\n>>> at the stop (20 s): measure with a ruler from the LOWEST point of the camera head straight down "
              f"to {what}, in mm. Then: calib_session answer --leg {a.leg} --seg {a.seg} --gap <mm>\n")
    code, _ = run(py(ORBIT_PY, "--ip", a.ip, *seg["args"], "--go", "--snapshots", folder, "--out", folder / "plan.json",
                     "--log", folder / "log.json"))
    seg["done"] = code == 0 and (folder / "snapshots.json").exists()
    seg["time"] = time.strftime("%H:%M:%S")
    save_session(s)
    print(f"{a.seg}: {'done' if seg['done'] else f'NOT done (exit {code})'}")
    return code


def cmd_answer(a, cfg) -> int:
    s = load_session()
    seg = s["legs"].get(a.leg, {}).get("segments", {}).get(a.seg)
    if seg is None or not a.seg.startswith("check"):
        print("answers go with a check-* segment of a planned leg")
        return 2
    if not seg.get("done"):
        print(f"{a.seg} has not run yet")
        return 2
    row = {"folder": seg["folder"], "view": 0, "gap_mm": float(a.gap), "label": f"{a.leg}/{a.seg}",
           "to": "table" if a.seg == "check-table" else "marker"}
    if row["to"] == "marker":
        row["marker"] = int(a.seg.split("-")[1])
    s["ruler"] = [r for r in s["ruler"] if r["label"] != row["label"]] + [row]
    save_session(s)
    (SESSION / "ruler.json").write_text(json.dumps(s["ruler"], indent=1), encoding="utf-8")
    pred = seg.get("predicted_gap_mm")
    print(f"{row['label']}: ruler {a.gap:g} mm" + (f"; the calibration in use expected {pred:g} mm "
                                                    f"(difference {a.gap - pred:+.1f})" if pred is not None else ""))
    return 0


def cmd_status(a, cfg) -> int:
    s = load_session()
    if not s["order"]:
        print("no leg planned yet")
    for leg in s["order"]:
        print(f"leg {leg} (planned {s['legs'][leg]['time']}):")
        for name, info in s["legs"][leg]["segments"].items():
            ans = next((r["gap_mm"] for r in s["ruler"] if r["label"] == f"{leg}/{name}"), None)
            need = name.startswith("check")
            print(f"  {name:12s} {'done' if info['done'] else 'to do':6s} views {info['views']:5s}"
                  + (f"  ruler {ans:g} mm" if ans is not None else "  ruler: MISSING" if need and info["done"] else ""))
    return 0


def cmd_solve(a, cfg) -> int:
    from . import handeye

    s = load_session()
    argv = ["solve", "--out", str(V2)]
    for leg in s["order"]:
        folders = [i["folder"] for i in s["legs"][leg]["segments"].values()
                   if i["done"] and (Path(i["folder"]) / "snapshots.json").exists()]
        if folders:
            argv += ["--leg", *folders]
    if "--leg" not in argv:
        print("nothing recorded yet")
        return 2
    if s["ruler"]:
        argv += ["--ruler", str(SESSION / "ruler.json")]
    if a.verbose:
        argv.append("--verbose")
    return handeye.main(argv)


def cmd_promote(a, cfg) -> int:
    new = V2 / "camera_robot.json"
    if not new.exists():
        print("no calib/v2/camera_robot.json: run solve first")
        return 2
    cur = ROOT / "calib" / "camera_robot.json"
    old = ROOT / "calib" / "camera_robot_v1.json"
    if not old.exists():
        shutil.copy2(cur, old)
    shutil.copy2(new, cur)
    if (V2 / "report.json").exists():
        shutil.copy2(V2 / "report.json", ROOT / "calib" / "report.json")
    print(f"{new} -> {cur} (the previous one is {old})")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m scan3d.calib_session", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "go"):
        p = sub.add_parser(name)
        p.add_argument("--ip", default=os.environ.get("XARM_IP"), required=os.environ.get("XARM_IP") is None)
        p.add_argument("--leg", required=True, help="name of this cube placement, e.g. A, B, C")
        if name == "go":
            p.add_argument("--seg", required=True)
            p.add_argument("--again", action="store_true", help="redo a segment already done")
        else:
            p.add_argument("--check-dist", default="150", help="lens above the cube top at the check-* stops, mm "
                           "(cubes close to the base: 180, else the arm needs a wrist flip)")
            p.add_argument("--skip", type=lambda t: t.split(","), help="segments to leave out, e.g. ring-4")
    p = sub.add_parser("answer")
    p.add_argument("--leg", required=True)
    p.add_argument("--seg", required=True)
    p.add_argument("--gap", type=float, required=True, help="mm from the head's lowest point down to the cube top / table")
    sub.add_parser("status")
    p = sub.add_parser("solve")
    p.add_argument("--verbose", action="store_true")
    sub.add_parser("promote")
    a = ap.parse_args(argv)
    cfg = config_from_args(a)
    try:
        sys.stdout.reconfigure(errors="replace")
    except AttributeError:
        pass
    return {"plan": cmd_plan, "go": cmd_go, "answer": cmd_answer, "status": cmd_status, "solve": cmd_solve,
            "promote": cmd_promote}[a.cmd](a, cfg)


if __name__ == "__main__":
    sys.exit(main())
