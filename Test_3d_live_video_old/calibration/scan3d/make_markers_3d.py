"""Two-colour 3D-printable ArUco markers (white tile + black modules), for printing without paper.

    python -m scan3d.make_markers_3d                 # scan-scale ids (aruco.scan_ids) -> scan3d/print3d/scan_*.stl
    python -m scan3d.make_markers_3d --sheet calibration      # Trova calibration ids 0-3

Writes two STL files per sheet, one per colour, in the same coordinates: load BOTH into the slicer at once as ONE
object with two parts, then give each part its filament. The tiles are modelled FACE DOWN: the marker pattern is
in the first `pattern_depth_mm` on the build plate (flat, sharp, matte on a textured plate) and a white backing is
printed above it. Turned face down means rotated 180 degrees about X, so the printed face reads correctly (a mirrored
ArUco would decode as a different id or not at all).

Check: the script reads the STLs back, turns them face up again, rasterises the visible face and decodes it with
OpenCV. It also checks that the unflipped (mirrored) view does NOT decode to the same ids.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import cv2
import numpy as np

from .config import HERE, add_config_args, config_from_args
from .make_markers import marker_bits, sheet_spec

# 12 triangles of an axis-aligned box, as corner indices (corner k: x = k&1, y = k>>1 & 1, z = k>>2 & 1),
# wound counter-clockwise seen from outside.
_BOX_TRIS = [(0, 2, 3), (0, 3, 1), (4, 5, 7), (4, 7, 6), (0, 1, 5), (0, 5, 4),
             (2, 6, 7), (2, 7, 3), (0, 4, 6), (0, 6, 2), (1, 3, 7), (1, 7, 5)]


def box_triangles(b) -> np.ndarray:
    x0, x1, y0, y1, z0, z1 = b
    c = np.array([[(x0, x1)[k & 1], (y0, y1)[(k >> 1) & 1], (z0, z1)[(k >> 2) & 1]] for k in range(8)], float)
    return c[np.array(_BOX_TRIS)]


def write_stl(path: Path, boxes: list[tuple]) -> int:
    tris = np.concatenate([box_triangles(b) for b in boxes]).astype(np.float32)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    with open(path, "wb") as f:
        f.write(b"trova scan3d aruco marker".ljust(80, b" "))
        f.write(struct.pack("<I", len(tris)))
        rec = np.zeros(len(tris), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
        rec["n"], rec["v"] = n, tris
        f.write(rec.tobytes())
    return len(tris)


def read_stl(path: Path) -> np.ndarray:
    data = path.read_bytes()
    count = struct.unpack("<I", data[80:84])[0]
    rec = np.frombuffer(data[84:84 + 50 * count], dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    return rec["v"].astype(float)


def runs(row) -> list[tuple[int, int]]:
    """[start, end) column ranges of consecutive True values."""
    out, c = [], 0
    while c < len(row):
        if row[c]:
            s = c
            while c < len(row) and row[c]:
                c += 1
            out.append((s, c))
        else:
            c += 1
    return out


def tile_boxes(bits: np.ndarray, x0: float, y0: float, side: float, border: float, t: float, d: float):
    """White and black boxes of one tile, FACE UP (pattern on top, z in [t-d, t]); image row 0 = largest y."""
    n = bits.shape[0]
    m = side / n
    w = side + 2 * border
    mx, my = x0 + border, y0 + border                       # marker's lower-left corner
    white = [(x0, x0 + w, y0, y0 + w, 0.0, t - d),           # backing slab
             (x0, mx, y0, y0 + w, t - d, t), (mx + side, x0 + w, y0, y0 + w, t - d, t),    # quiet zone left/right
             (mx, mx + side, y0, my, t - d, t), (mx, mx + side, my + side, y0 + w, t - d, t)]  # bottom/top
    black = []
    for r in range(n):
        ylo, yhi = my + (n - 1 - r) * m, my + (n - r) * m
        black += [(mx + a * m, mx + b * m, ylo, yhi, t - d, t) for a, b in runs(bits[r])]
        white += [(mx + a * m, mx + b * m, ylo, yhi, t - d, t) for a, b in runs(~bits[r])]
    return white, black


def face_down(b, t):
    """Rotate a box 180 degrees about the X axis (x, y, z) -> (x, -y, t - z): a rotation, not a mirror."""
    x0, x1, y0, y1, z0, z1 = b
    return (x0, x1, -y1, -y0, t - z1, t - z0)


def facing(tris: np.ndarray, sign: float) -> np.ndarray:
    """Triangles whose outward normal points to +z (sign=1) or -z (sign=-1)."""
    nz = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])[:, 2]
    return tris[sign * nz > 1e-9]


def rasterise(tri_sets: list[tuple[np.ndarray, int]], px_per_mm: float) -> np.ndarray:
    """Top view (x right, y up) of triangle footprints, one triangle at a time, later sets painted on top."""
    allv = np.concatenate([t.reshape(-1, 3) for t, _ in tri_sets])
    lo, hi = allv[:, :2].min(0) - 15, allv[:, :2].max(0) + 15
    size = np.ceil((hi - lo) * px_per_mm).astype(int)
    img = np.full((size[1], size[0]), 128, np.uint8)
    for tris, value in tri_sets:
        p = np.empty(tris.shape[:2] + (2,))
        p[..., 0] = (tris[..., 0] - lo[0]) * px_per_mm
        p[..., 1] = (hi[1] - tris[..., 1]) * px_per_mm
        for tri in np.round(p * 16).astype(np.int32):
            cv2.fillConvexPoly(img, tri, value, shift=4)
    return img


def detect(img: np.ndarray, dictionary: str, px_per_mm: float) -> dict[int, float]:
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary)),
                                  cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(img)
    if ids is None:
        return {}
    return {int(i): float(np.mean([np.linalg.norm(c.reshape(4, 2)[k] - c.reshape(4, 2)[(k + 1) % 4])
                                   for k in range(4)])) / px_per_mm for c, i in zip(corners, ids.ravel())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_config_args(ap)
    ap.add_argument("--sheet", choices=("scan", "calibration"), default="scan")
    ap.add_argument("--out-dir", default=str(HERE / "print3d"))
    args = ap.parse_args()
    cfg = config_from_args(args)
    a, pr = cfg["aruco"], cfg["aruco_print3d"]
    dictionary, side = a["dictionary"], float(a["marker_mm"])
    border, t, d, gap = (float(pr[k]) for k in ("white_border_mm", "thickness_mm", "pattern_depth_mm", "gap_mm"))
    ids, _ = sheet_spec(cfg, args.sheet)
    w = side + 2 * border
    cols = 2 if len(ids) > 1 else 1

    white, black = [], []
    for k, i in enumerate(ids):
        r, c = divmod(k, cols)
        wb, bb = tile_boxes(marker_bits(dictionary, i), c * (w + gap), -r * (w + gap), side, border, t, d)
        white += wb
        black += bb
    white = [face_down(b, t) for b in white]
    black = [face_down(b, t) for b in black]
    ymin = min(b[2] for b in white)                              # move onto positive y
    white = [(b[0], b[1], b[2] - ymin, b[3] - ymin, b[4], b[5]) for b in white]
    black = [(b[0], b[1], b[2] - ymin, b[3] - ymin, b[4], b[5]) for b in black]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pw, pb = out / f"{args.sheet}_markers_white.stl", out / f"{args.sheet}_markers_black.stl"
    nw, nb = write_stl(pw, white), write_stl(pb, black)

    # --- verify from the files: turn face up (same rotation again), look at the printed face from above
    tw, tb = read_stl(pw), read_stl(pb)

    def flip(tris):
        f = tris.copy()
        f[..., 1], f[..., 2] = -tris[..., 1], t - tris[..., 2]
        return f

    px = 10.0
    # the printed face is the one that lay on the plate (z = 0, facing -z); turned face up it faces +z
    def plate_face(tris):
        return facing(tris[np.all(np.abs(tris[..., 2]) < 1e-6, axis=1)], -1)

    face = rasterise([(flip(plate_face(tw)), 255), (flip(plate_face(tb)), 0)], px)
    found = detect(face, dictionary, px)
    preview = out / f"{args.sheet}_markers_face_preview.png"
    cv2.imwrite(str(preview), face)
    # negative control: the same face seen from above without turning the part (a mirror image)
    mirrored = detect(rasterise([(plate_face(tw), 255), (plate_face(tb), 0)], px), dictionary, px)

    ext = np.concatenate([tw.reshape(-1, 3), tb.reshape(-1, 3)])
    size = ext.max(0) - ext.min(0)
    print(f"wrote {pw.name} ({nw} triangles) and {pb.name} ({nb} triangles) in {out}")
    print(f"object size {size[0]:.1f} x {size[1]:.1f} x {size[2]:.2f} mm, {len(ids)} tiles of {w:g} x {w:g} mm, "
          f"marker {side:g} mm, pattern {d:g} mm deep on the plate")
    print("decoded on the printed face:", {i: f"{s:.2f} mm" for i, s in sorted(found.items())})
    print("decoded on the mirrored view (must not match):", sorted(mirrored))
    ok = sorted(found) == sorted(ids) and all(abs(s - side) < 0.3 for s in found.values()) \
        and not set(mirrored) & set(ids)
    print("check: OK" if ok else "check FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
