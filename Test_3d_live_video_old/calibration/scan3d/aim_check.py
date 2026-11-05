"""Live check of the orbit START pose from the Pi camera (no arm access): is the camera looking straight down at
the object, and how far is it? Prints the numbers arm_pc/orbit_here.py needs.

    python -m scan3d.aim_check                 # every second until Ctrl+C
    python -m scan3d.aim_check --once --save data/checks/aim.jpg

Tilt comes from the scale markers on top of the cubes (solvePnP, their top faces are parallel to the table),
distance from their size, the object position from a colour mask (red mouse by default). Intrinsics are the
Camera v2 nominal ones (62.2 deg horizontal FOV over the full frame width), good to a few percent.
"""

from __future__ import annotations

import argparse
import math
import time
import urllib.request

import cv2
import numpy as np

from .config import add_config_args, config_from_args


def measure(img: np.ndarray, dictionary: str, marker_mm: float, marker_top_mm: float, obj_height_mm: float,
            red: bool = True) -> dict:
    h, w = img.shape[:2]
    f = (w / 2) / math.tan(math.radians(62.2 / 2))
    k = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary)),
                                  cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(gray)
    s = marker_mm
    obj = np.array([[-s / 2, s / 2, 0], [s / 2, s / 2, 0], [s / 2, -s / 2, 0], [-s / 2, -s / 2, 0]], np.float32)
    tops, out = [], {"size": [w, h], "markers": []}
    for c, i in zip(corners, [] if ids is None else ids.ravel()):
        ok, rv, tv = cv2.solvePnP(obj, c.reshape(4, 2), k, np.zeros(5), flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            continue
        n = cv2.Rodrigues(rv)[0][:, 2]
        n = n if n[2] < 0 else -n                                    # towards the camera
        tilt = math.degrees(math.acos(min(1.0, abs(n[2]))))
        out["markers"].append({"id": int(i), "tilt_deg": round(tilt, 1), "dist_mm": round(float(tv[2, 0]), 0)})
        if tilt < 30:                                                # a marker on a cube TOP (not a side)
            tops.append((tv.ravel(), n))
    if tops:
        normal = np.mean([n for _, n in tops], axis=0)
        normal /= np.linalg.norm(normal)
        point = np.mean([t for t, _ in tops], axis=0)
        # distance along the optical axis to the marker-top plane, then down to the table
        d_top = float(normal @ point / normal[2])
        tilt = math.degrees(math.acos(abs(normal[2])))
        # direction in the IMAGE the optical axis leans to (towards where the table is farther)
        lean = math.degrees(math.atan2(-normal[1], -normal[0]))
        out.update(tilt_deg=round(tilt, 1), lean_image_deg=round(lean, 0), top_plane_mm=round(d_top, 0),
                   table_mm=round(d_top + marker_top_mm, 0), dist_mm=round(d_top + marker_top_mm - obj_height_mm / 2, 0))
    if red:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        m = (((hsv[..., 0] < 10) | (hsv[..., 0] > 170)) & (hsv[..., 1] > 120) & (hsv[..., 2] > 60)).astype(np.uint8)
        n_lab, lab, stats, cent = cv2.connectedComponentsWithStats(m)
        if n_lab > 1:
            j = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            if stats[j, cv2.CC_STAT_AREA] > 500:
                cx, cy = cent[j]
                z = out.get("dist_mm", 400)
                out["object_px"] = [round(float(cx)), round(float(cy))]
                out["object_off_mm"] = [round((cx - w / 2) * z / f), round((cy - h / 2) * z / f)]
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    out["sharpness"] = round(float(lap.var()), 0)
    out["brightness"] = round(float(gray.mean()), 0)
    return out


def line(m: dict) -> str:
    parts = [f"{len(m['markers'])} markers"]
    if "tilt_deg" in m:
        parts.append(f"tilt {m['tilt_deg']:.1f} deg (leans to image {m['lean_image_deg']:.0f} deg; 0 = right, 90 = down)")
        parts.append(f"table {m['table_mm']:.0f} mm, --dist {m['dist_mm']:.0f}")
    else:
        parts.append("no cube-top marker: tilt/distance unknown")
    if "object_off_mm" in m:
        dx, dy = m["object_off_mm"]
        parts.append(f"object {abs(dx)} mm {'right' if dx > 0 else 'left'}, {abs(dy)} mm {'down' if dy > 0 else 'up'} "
                     f"of centre (image)")
    parts.append(f"sharp {m['sharpness']:.0f}, bright {m['brightness']:.0f}")
    verdict = "OK" if m.get("tilt_deg", 99) < 3 and max(map(abs, m.get("object_off_mm", [99, 99]))) < 15 else "adjust"
    return f"[{verdict}] " + " | ".join(parts)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m scan3d.aim_check", description=__doc__.split("\n\n")[0])
    add_config_args(ap)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--every", type=float, default=1.0)
    ap.add_argument("--marker-top", type=float, default=53.0,
                    help="height of the cube-top markers above the table, mm (cubes: 53, measured)")
    ap.add_argument("--obj-height", type=float, default=35.0, help="object height, mm (red mouse: 35)")
    ap.add_argument("--no-red", action="store_true", help="do not look for a red object")
    ap.add_argument("--save", help="save the last snapshot here")
    args = ap.parse_args(argv)
    cfg = config_from_args(args)
    url = f"http://{cfg['pi']['host']}:{cfg['pi']['port']}/snapshot.jpg"
    sc = cfg["aruco_scan"]
    while True:
        try:
            data = urllib.request.urlopen(url, timeout=10).read()
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            m = measure(img, sc["dictionary"], float(sc["marker_mm"]), args.marker_top, args.obj_height, not args.no_red)
            print(time.strftime("%H:%M:%S"), line(m), flush=True)
            if args.save:
                with open(args.save, "wb") as fh:
                    fh.write(data)
        except Exception as e:  # noqa: BLE001  (network hiccups: keep looping)
            print(time.strftime("%H:%M:%S"), f"error: {e}", flush=True)
        if args.once:
            return 0
        time.sleep(args.every)


if __name__ == "__main__":
    raise SystemExit(main())
