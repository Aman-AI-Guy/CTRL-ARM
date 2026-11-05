"""Synthetic scan with a known answer, to test scan3d.reconstruct without moving the arm.

A textured box (default 80 x 50 x 35 mm) on a textured table with ArUco markers (dictionary and size from
aruco_scan in scan3d/config.json), seen by a pinhole camera like the Pi's (1640x1232, ~62 deg HFOV) on an orbit
around the box (default 270 deg, the xArm cannot do a full circle). Each planar face is rendered with a homography,
so no 3D renderer is needed. Writes JPEGs (quality 85) and truth.json.

    .venv\\Scripts\\python tests\\make_synthetic_scan.py [--out data/synthetic/box] [--views 40] [--arc 270]
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scan3d.config import load_config  # noqa: E402

W, H = 1640, 1232
F = W / (2 * np.tan(np.radians(62.2) / 2))           # IMX219 binned mode horizontal field of view


def texture(size, seed, base):
    """Busy, non-repeating texture (blobs, lines, digits, noise): plenty of SIFT features."""
    rng = np.random.default_rng(seed)
    t = np.zeros((size, size, 3), np.float32)
    for s in (size // 4, size // 16, size // 64):
        n = rng.random((max(2, size // s), max(2, size // s), 3)).astype(np.float32)
        t += cv2.resize(n, (size, size), interpolation=cv2.INTER_CUBIC) * (0.5 if s > size // 8 else 0.35)
    t = (t / t.max() * 160 + np.array(base, np.float32) * 0.4)
    img = np.clip(t, 0, 255).astype(np.uint8)
    for _ in range(size // 12):
        c = tuple(int(v) for v in rng.integers(0, 255, 3))
        p = tuple(int(v) for v in rng.integers(0, size, 2))
        k = rng.integers(0, 3)
        if k == 0:
            cv2.circle(img, p, int(rng.integers(3, size // 25)), c, -1, cv2.LINE_AA)
        elif k == 1:
            cv2.line(img, p, tuple(int(v) for v in rng.integers(0, size, 2)), c, int(rng.integers(1, 4)), cv2.LINE_AA)
        else:
            cv2.putText(img, str(int(rng.integers(0, 999))), p, cv2.FONT_HERSHEY_SIMPLEX, size / 600, c, 2, cv2.LINE_AA)
    return img


def look_at(c, target):
    f = target - c
    f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1.0])
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    R = np.stack([r, d, f])
    return R, -R @ c


def warp_quad(canvas, tex, origin, e1, e2, K, R, t):
    """Draw a textured planar rectangle (origin + a*e1 + b*e2, a,b in [0,1]) onto canvas."""
    h, w = tex.shape[:2]
    M = K @ np.column_stack([R @ (e1 / w), R @ (e2 / h), R @ origin + t])
    corners = [origin, origin + e1, origin + e1 + e2, origin + e2]
    if min((R @ p + t)[2] for p in corners) <= 1:
        return
    img = cv2.warpPerspective(tex, M, (W, H), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, (W, H), flags=cv2.INTER_NEAREST)
    canvas[mask > 0] = img[mask > 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "synthetic", "box"))
    ap.add_argument("--size", default="80,50,35", help="box L,W,H in mm")
    ap.add_argument("--views", type=int, default=40)
    ap.add_argument("--arc", type=float, default=270, help="orbit arc in degrees")
    ap.add_argument("--radius", type=float, default=320, help="camera distance from the box centre, mm")
    ap.add_argument("--elevation", type=float, default=40, help="degrees above the table")
    a = ap.parse_args()
    L, Wd, Ht = (float(v) for v in a.size.split(","))
    ac = load_config()["aruco_scan"]
    dic = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ac["dictionary"]))
    msize = float(ac["marker_mm"])

    table_mm, tpx = 420.0, 2520                         # table texture: 6 px per mm
    table = texture(tpx, 1, (120, 110, 100))
    ppm = tpx / table_mm
    markers = [(4, -110.0, 15.0), (2, 110.0, -20.0), (3, 20.0, 95.0)]
    for mid, x, y in markers:                           # white label 1.3x the black square, like the real ones
        side = int(round(msize * ppm))
        img = cv2.aruco.generateImageMarker(dic, mid, side)
        lab = int(round(msize * 1.3 * ppm))
        cx, cy = int((x + table_mm / 2) * ppm), int((table_mm / 2 - y) * ppm)
        table[cy - lab // 2:cy - lab // 2 + lab, cx - lab // 2:cx - lab // 2 + lab] = 255
        table[cy - side // 2:cy - side // 2 + side, cx - side // 2:cx - side // 2 + side] = img[..., None]
    faces_tex = [texture(800, 10 + k, base) for k, base in
                 enumerate([(200, 60, 60), (60, 160, 60), (60, 60, 200), (200, 180, 40), (160, 60, 160)])]

    K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1]])
    # box corners: x in [-L/2, L/2], y in [-W/2, W/2], z in [0, H]
    x0, x1, y0, y1 = -L / 2, L / 2, -Wd / 2, Wd / 2
    faces = [  # (origin, e1, e2, outward normal, shade, texture)
        (np.array([x0, y1, Ht]), np.array([L, 0, 0]), np.array([0, -Wd, 0]), np.array([0, 0, 1.0]), 1.00, 0),
        (np.array([x0, y0, Ht]), np.array([L, 0, 0]), np.array([0, 0, -Ht]), np.array([0, -1.0, 0]), 0.85, 1),
        (np.array([x1, y0, Ht]), np.array([0, Wd, 0]), np.array([0, 0, -Ht]), np.array([1.0, 0, 0]), 0.75, 2),
        (np.array([x1, y1, Ht]), np.array([-L, 0, 0]), np.array([0, 0, -Ht]), np.array([0, 1.0, 0]), 0.65, 3),
        (np.array([x0, y1, Ht]), np.array([0, -Wd, 0]), np.array([0, 0, -Ht]), np.array([-1.0, 0, 0]), 0.80, 4),
    ]
    os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
    target = np.array([0, 0, Ht / 2])
    el = np.radians(a.elevation)
    rng = np.random.default_rng(3)
    for k in range(a.views):
        az = np.radians(-a.arc / 2 + a.arc * k / max(1, a.views - 1))
        c = target + a.radius * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        R, t = look_at(c, target + rng.normal(0, 3, 3))      # the arm never aims perfectly
        canvas = np.full((H, W, 3), 60, np.uint8)
        warp_quad(canvas, table, np.array([-table_mm / 2, table_mm / 2, 0]), np.array([table_mm, 0, 0]),
                  np.array([0, -table_mm, 0]), K, R, t)
        for o, e1, e2, n, shade, ti in faces:
            if n @ (c - (o + (e1 + e2) / 2)) > 0:
                warp_quad(canvas, (faces_tex[ti] * shade).astype(np.uint8), o, e1, e2, K, R, t)
        canvas = np.clip(canvas.astype(np.int16) + rng.normal(0, 3, canvas.shape), 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(a.out, "images", f"{k:06d}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])
    truth = {"box_mm": [L, Wd, Ht], "marker_mm": msize, "dictionary": ac["dictionary"], "markers": markers,
             "views": a.views, "arc_deg": a.arc, "radius_mm": a.radius, "elevation_deg": a.elevation,
             "focal_px": F, "size": [W, H]}
    with open(os.path.join(a.out, "truth.json"), "w") as f:
        json.dump(truth, f, indent=1)
    print(f"wrote {a.views} views to {a.out}: box {L:g} x {Wd:g} x {Ht:g} mm, markers {[m[0] for m in markers]}")


if __name__ == "__main__":
    main()
