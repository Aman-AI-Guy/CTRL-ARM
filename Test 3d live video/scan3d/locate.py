"""Where is the object, in robot mm? One still from the arm camera + the arm's joints + calib/camera_robot.json.

    python -m scan3d.locate --ip 192.168.1.111                    # red object (the mouse) + marker cubes -> JSON
    python -m scan3d.locate --ip ... --pixel 812,455 --height 60   # an object found by another detector

    from scan3d.locate import locate
    r = locate("192.168.1.111", height_mm=35)    # r["top_mm"], r["center_mm"], r["boxes"], r["q"], r["image"]

Used by scan3d.hover_check (straight above it) and scan3d.orbit_scan (orbit around it). Reads only, never moves:
the arm must stand still, the camera looking down at the table (tilt <= 20 deg for a good centre).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from .config import ROOT, add_config_args, config_from_args

sys.path.insert(0, str(ROOT / "calib"))
from pix2robot import Calibration  # noqa: E402

CALIB = ROOT / "calib" / "camera_robot.json"


def red_blob(img: np.ndarray) -> dict | None:
    """Largest red region (the red mouse): centre pixel, area, box, and the picture edges it touches."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = (((hsv[..., 0] < 10) | (hsv[..., 0] > 170)) & (hsv[..., 1] > 120) & (hsv[..., 2] > 60)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, _, st, cen = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None
    j = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    if st[j, cv2.CC_STAT_AREA] < 2000:
        return None
    x, y, w, h = (int(v) for v in st[j, :4])
    H, W = m.shape
    cut = [s for s, c in (("left", x <= 0), ("top", y <= 0), ("right", x + w >= W), ("bottom", y + h >= H)) if c]
    return {"pixel": [float(cen[j][0]), float(cen[j][1])], "area_px": int(st[j, 4]), "box_px": [x, y, w, h], "cut": cut}


def white_disk(img: np.ndarray, r_px: tuple[float, float]) -> dict | None:
    """A round bright blob whose radius is in r_px = (min, max), closest to their middle: e.g. the upturned bottom of
    a white cup. Several brightness thresholds: a low one also takes in the cup's lit rim further down (seen from a
    tilted view, its centre is off by parallax), so the circle of the expected size wins. The cube labels are white
    too, but square (circularity 0.79 < 0.83) and of another size."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    r_mid = (r_px[0] + r_px[1]) / 2
    best = None
    for vmin in (170, 185, 200, 215, 230):
        m = ((hsv[..., 2] > vmin) & (hsv[..., 1] < 70)).astype(np.uint8)  # no opening: it eats a thin rim
        for c in cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]:
            a, p = cv2.contourArea(c), cv2.arcLength(c, True)
            (x, y), r = cv2.minEnclosingCircle(c)
            circ = 4 * np.pi * a / p ** 2 if p else 0.0
            if r_px[0] <= r <= r_px[1] and circ > 0.83 and (best is None or abs(r - r_mid) < abs(best[1] - r_mid)):
                best = (circ, r, c)
    if best is None:
        return None
    circ, r, c = best
    mo = cv2.moments(c)
    x, y, w, h = cv2.boundingRect(c)
    H, W = m.shape
    cut = [s for s, k in (("left", x <= 0), ("top", y <= 0), ("right", x + w >= W), ("bottom", y + h >= H)) if k]
    return {"pixel": [mo["m10"] / mo["m00"], mo["m01"] / mo["m00"]], "radius_px": float(r),
            "circularity": round(float(circ), 3), "area_px": int(mo["m00"]), "box_px": [x, y, w, h], "cut": cut}


def find_object(img, q, cal: Calibration, find: str, height_mm: float, diameter_mm: float | None) -> dict | None:
    """find = "red" (largest red blob) or "disk" (a white disk of diameter_mm on top of the object)."""
    if find == "red":
        return red_blob(img)
    cam_z = float(cal.camera_pose(q)[2, 3])
    c = cal.pixel_to_object_top(cal.width / 2, cal.height / 2, q, height_mm)
    r = (diameter_mm / 2) * cal.K[0, 0] / max(cam_z - c[2], 30.0)
    return white_disk(img, (0.75 * r, 1.3 * r))


def read_still(ip: str, url: str) -> tuple[np.ndarray, bytes]:
    """Joints (read before and after the still: the arm must stand still) and the still. Reads only."""
    from xarm.wrapper import XArmAPI

    arm = XArmAPI(ip, is_radian=False)
    try:
        c1, q1 = arm.get_servo_angle()
        data = urllib.request.urlopen(url, timeout=10).read()
        c2, q2 = arm.get_servo_angle()
    finally:
        arm.disconnect()
    if c1 or c2:
        raise RuntimeError(f"get_servo_angle codes {c1}, {c2}")
    q1, q2 = np.array(q1[:5], float), np.array(q2[:5], float)
    if np.abs(q1 - q2).max() > 0.05:
        raise RuntimeError("the arm moved while the still was taken: run again with the arm standing still")
    return q2, data


def cube_boxes(img, q, cal: Calibration, cfg: dict, half: float, say=print, with_ids=False,
               remembered=True) -> list:
    """Keep-out boxes [x0, y0, x1, y1, z_top] for the cubes whose top markers are in the calibration: where they are
    in this still, else (not seen) where the calibration saw them; remembered=False: no box for a cube not seen
    (the cubes are off the table). with_ids: [(marker id, seen now, box)]."""
    from .handeye import detect

    sc = cfg["aruco_scan"]
    seen = detect(img, sc["dictionary"], sc.get("ids") or [])
    z_cube = cal.table_z + float(cal.data.get("cube_height_mm", 53.0))
    boxes = []
    for mid, m in sorted(cal.data.get("markers", {}).items()):
        if m["normal"][2] < 0.9:
            continue                                              # side marker: on a cube that has a top one
        corners = np.array(m["corners_mm"], float)
        z = max(z_cube, float(corners[:, 2].max()))
        if int(mid) in seen:
            c, how = cal.pixel_to_plane(*seen[int(mid)].mean(0), q, z), "seen now"
        elif not remembered:
            say(f"cube (marker {mid}): not seen, no keep-out box (--no-cubes: the cubes are off the table)")
            continue
        else:
            c, how = corners.mean(0), "NOT seen now: where the calibration saw it"
        box = [round(float(c[0] - half), 1), round(float(c[1] - half), 1), round(float(c[0] + half), 1),
               round(float(c[1] + half), 1), round(z, 1)]
        boxes.append((int(mid), int(mid) in seen, box) if with_ids else box)
        say(f"cube (marker {mid}): centre ({c[0]:.0f}, {c[1]:.0f}), top z {z:.0f} mm, keep-out +-{half:g} mm ({how})")
    return boxes


def locate(ip: str, height_mm: float, pixel=None, cfg: dict | None = None, calib: str | Path = CALIB,
           cube_half: float = 32.0, say=print, find: str = "red", diameter_mm: float | None = None,
           remembered_cubes: bool = True) -> dict:
    """The object's top centre and mid-height centre in robot mm, keep-out boxes for the marker cubes, the joints and
    the still they come from. pixel = the object's centre (u, v) in the still; default: found with `find` ("red" =
    the largest red blob, "disk" = a white disk of diameter_mm on its top, e.g. an upturned white cup).
    remembered_cubes=False: no keep-out box for a cube not seen in the still (see cube_boxes)."""
    if cfg is None:
        from .config import load_config
        cfg = load_config([])
    cal = Calibration.load(calib)
    url = f"http://{cfg['pi']['host']}:{cfg['pi']['port']}/snapshot.jpg"
    q, data = read_still(ip, url)
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img.shape[1::-1] != (cal.width, cal.height):
        cal = cal.for_size(img.shape[1], img.shape[0])
    blob = None
    try:
        if pixel is None:
            blob = find_object(img, q, cal, find, height_mm, diameter_mm)
            if blob is None:
                raise RuntimeError(f"no {'red object' if find == 'red' else 'white disk'} in the still: give the "
                                   "object's pixel")
            pixel = blob["pixel"]
            say(f"{'red object' if find == 'red' else 'white disk'}: centre pixel ({pixel[0]:.0f}, {pixel[1]:.0f}), "
                f"{blob['area_px']} px" + (f"  ! runs off the picture ({', '.join(blob['cut'])})" if blob["cut"] else ""))
        top = cal.pixel_to_object_top(pixel[0], pixel[1], q, height_mm)
    except ValueError as e:
        raise RuntimeError(f"the camera does not look down at the table from joints {np.round(q, 1).tolist()} ({e}): "
                           "put the arm in a start pose above the object first") from e
    say(f"object top centre (robot mm) {np.round(top, 1).tolist()}  (table z {cal.table_z:.1f} + {height_mm:g})")
    return {"q": [float(v) for v in q], "pixel": [float(v) for v in pixel], "blob": blob, "height_mm": float(height_mm),
            "top_mm": [float(v) for v in top], "center_mm": [float(top[0]), float(top[1]), float(top[2] - height_mm / 2)],
            "boxes": cube_boxes(img, q, cal, cfg, cube_half, say, remembered=remembered_cubes),
            "table_z_mm": cal.table_z, "calib": str(calib),
            "snapshot_url": url, "jpeg": data, "image": img}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m scan3d.locate", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--ip", default=os.environ.get("XARM_IP"), help="xArm IP (or env XARM_IP)")
    ap.add_argument("--height", type=float, default=35, help="object height above the table, mm (red mouse 35)")
    ap.add_argument("--pixel", type=lambda t: [float(v) for v in t.split(",")], help="object centre u,v (default: --find)")
    ap.add_argument("--find", choices=("red", "disk"), default="red", help="red blob, or a white disk (--diameter) on top")
    ap.add_argument("--diameter", type=float, help="--find disk: the disk's diameter, mm (white cup upside down: 44)")
    ap.add_argument("--calib", default=str(CALIB))
    ap.add_argument("--no-cubes", action="store_true",
                    help="the marker cubes are off the table: no keep-out box for a cube not seen in the still")
    ap.add_argument("--save", help="also save the still (jpg) here")
    a = ap.parse_args(argv)
    if not a.ip:
        ap.error("--ip <arm ip> (or set XARM_IP)")
    if a.find == "disk" and not a.diameter:
        ap.error("--find disk needs --diameter")
    r = locate(a.ip, a.height, a.pixel, config_from_args(a), a.calib, say=lambda *t: print(*t, file=sys.stderr),
               find=a.find, diameter_mm=a.diameter, remembered_cubes=not a.no_cubes)
    if a.save:
        Path(a.save).write_bytes(r["jpeg"])
    print(json.dumps({k: v for k, v in r.items() if k not in ("jpeg", "image")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
