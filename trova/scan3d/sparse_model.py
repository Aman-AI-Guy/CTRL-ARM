"""Original COLMAP text model reader; poses stay in the arbitrary SfM frame."""
import numpy as np
from pathlib import Path

def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])

def read_model(txt: Path) -> dict:
    cams = {}
    for line in (txt / "cameras.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            p = line.split()
            cams[int(p[0])] = (p[1], int(p[2]), int(p[3]), [float(v) for v in p[4:]])
    lines = [ln for ln in (txt / "images.txt").read_text().splitlines() if not ln.startswith("#")]
    images = {}
    for ln in lines[0::2]:
        p = ln.split()
        if len(p) < 10:
            continue
        R = quat_to_R([float(v) for v in p[1:5]])
        t = np.array([float(v) for v in p[5:8]])
        images[p[9]] = {"R": R, "t": t, "camera": int(p[8]), "center": -R.T @ t, "axis": R.T @ np.array([0, 0, 1.0])}
    errs = []
    for ln in (txt / "points3D.txt").read_text().splitlines():
        if ln and not ln.startswith("#"):
            errs.append(float(ln.split()[7]))
    return {"cameras": cams, "images": images, "n_points": len(errs),
            "mean_reproj_px": round(float(np.mean(errs)), 3) if errs else None}
