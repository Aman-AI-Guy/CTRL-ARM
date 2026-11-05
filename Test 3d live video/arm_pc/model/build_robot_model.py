"""Build ../robot_model.json: the real shapes of the xArm 5 links and of the camera head as sets of spheres.

    python model/build_robot_model.py          # from the kit folder: writes robot_model.json + model/robot_model.png

Only needed when a shape changes (new head CAD: export it to model/camera_head.stl, mm, same axes). Needs numpy
(+ matplotlib for the picture). Sources, all in model/:
  xarm5_meshes/*.stl   UFactory's visual meshes (xArm-Developer/xarm_ros, xarm_description/meshes/xarm5/visual, BSD-3),
                       metres, each in its own URDF link frame (the frames of orbit_here.fk_frames)
  camera_head.stl      the camera head, tessellated from camera_head.step ("Camera Gatto.step", Fusion 360, mm)

Camera head, measured in the STEP file:
  - mounting plate: bottom face at CAD z = -7, outer diameter 66.2, 4 x M6 clearance holes (dia 6.2) on a 50 mm
    circle centred at CAD (36, 0): the ISO 9409-1-50-4-M6 tool flange of the xArm. So the flange centre is CAD
    (36, 0, -7) and the flange axis (out of the arm) is CAD +z.
  - lens hole: dia 6.3, axis along CAD -y, centred at CAD x 36, z 33.07; it goes through the front wall from
    y -49.90 (outside) to y -39.64 (inside face, where the camera board sits).
  - the optical centre is taken 2 mm inside the outer face: CAD (36, -42, 33.07). +-3 mm does not matter here.
Head frame H used in the JSON: origin = flange centre, z = flange axis, x = the camera's viewing direction.
  p_H = (-y_cad, x_cad - 36, z_cad + 7)
so the camera is at H (42, 0, 40.07) looking along H +x, perpendicular to the flange axis.

Each shape becomes spheres that cover all of it: surface voxels (5 mm) + the inside (flood fill from outside), k-means
into k clusters, one sphere per cluster with the radius that covers every voxel of the cluster. Spheres stick out of
the real surface by a few mm; that is extra margin, reported per shape.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

MODEL_DIR = Path(__file__).resolve().parent
MESHES = MODEL_DIR / "xarm5_meshes"
HEAD_STL = MODEL_DIR / "camera_head.stl"
OUT = MODEL_DIR.parent / "robot_model.json"
PNG = MODEL_DIR / "robot_model.png"
VOXEL = 5.0
CLOSE = 3            # voxels: openings up to ~30 mm are sealed before filling the inside

# spheres per shape: more where the shape is long or has lumps (joint housings)
K = {"link_base": 16, "link1": 16, "link2": 44, "link3": 32, "link4": 18, "link5": 8, "head": 28}
HEAD_FLANGE_CENTRE_CAD = np.array([36.0, 0.0, -7.0])
HEAD_CAMERA_CAD = np.array([36.0, -42.0, 33.0719])


def load_stl(path: Path) -> np.ndarray:
    """Binary STL -> (n, 3, 3) triangles."""
    b = path.read_bytes()
    n = int(np.frombuffer(b, "<u4", 1, 80)[0])
    if 84 + 50 * n != len(b):
        raise ValueError(f"{path}: not a binary STL")
    rec = np.frombuffer(b, np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), n, 84)
    return rec["v"].astype(float)


def cad_to_head(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    return np.stack([-p[..., 1], p[..., 0] - HEAD_FLANGE_CENTRE_CAD[0], p[..., 2] - HEAD_FLANGE_CENTRE_CAD[2]], -1)


def surface_points(tris: np.ndarray, spacing: float) -> np.ndarray:
    """Points on the triangles, about `spacing` apart (vertices + area-weighted samples)."""
    area = 0.5 * np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1)
    n = np.maximum(1, np.ceil(area / (spacing * spacing * 0.4))).astype(int)
    idx = np.repeat(np.arange(len(tris)), n)
    rng = np.random.default_rng(0)
    u, v = rng.random(len(idx)), rng.random(len(idx))
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    t = tris[idx]
    pts = t[:, 0] + (t[:, 1] - t[:, 0]) * u[:, None] + (t[:, 2] - t[:, 0]) * v[:, None]
    return np.concatenate([pts, tris.reshape(-1, 3)])


def _grow(mask: np.ndarray, axes=(0, 1, 2)) -> np.ndarray:
    """Mask plus its neighbours along the given axes."""
    out = mask.copy()
    for ax in axes:
        hi = tuple(slice(1, None) if a == ax else slice(None) for a in range(3))
        lo = tuple(slice(None, -1) if a == ax else slice(None) for a in range(3))
        out[hi] |= mask[lo]
        out[lo] |= mask[hi]
    return out


def solid_voxels(tris: np.ndarray, close: int = CLOSE) -> tuple[np.ndarray, int]:
    """Centres of the voxels on or inside the surface; also how many are inside (not on the surface).

    The meshes are shells with openings: the lens hole and LED slots, the head's open bottom (closed by the flange
    in reality), the link ends (closed by the next link). So the inside is filled slice by slice: in every 2D slice
    along x, y and z, what the walls enclose is solid (a tube open at its ends is closed in every cross-section).
    Gaps up to ~2 * close voxels inside a slice are sealed first (closing)."""
    pts = surface_points(tris, VOXEL / 2)
    pad = close + 2
    lo = pts.min(0) - pad * VOXEL
    ijk = np.floor((pts - lo) / VOXEL).astype(int)
    shape = ijk.max(0) + pad + 1
    surf = np.zeros(shape, bool)
    surf[tuple(ijk.T)] = True
    solid = surf.copy()
    for normal in range(3):
        plane = tuple(a for a in range(3) if a != normal)
        wall = surf
        for _ in range(close):
            wall = _grow(wall, plane)
        outside = np.zeros(shape, bool)                  # seeds: the border of every slice
        for ax in plane:
            idx0 = tuple(0 if a == ax else slice(None) for a in range(3))
            idx1 = tuple(-1 if a == ax else slice(None) for a in range(3))
            outside[idx0] = True
            outside[idx1] = True
        outside &= ~wall
        while True:
            grown = _grow(outside, plane) & ~wall
            if (grown == outside).all():
                break
            outside = grown
        for _ in range(close):                           # give back the thickening on the outside
            outside = _grow(outside, plane)
        solid |= ~outside
    centres = lo + (np.argwhere(solid) + 0.5) * VOXEL
    return centres, int((solid & ~surf).sum())


def spheres(points: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """k-means (farthest-point start) -> centres, radii covering every point (+ half a voxel diagonal)."""
    rng = np.random.default_rng(1)
    cent = [points[rng.integers(len(points))]]
    d = np.linalg.norm(points - cent[0], axis=1)
    for _ in range(k - 1):
        cent.append(points[int(np.argmax(d))])
        d = np.minimum(d, np.linalg.norm(points - cent[-1], axis=1))
    cent = np.array(cent)
    for _ in range(40):
        lab = np.argmin(((points[:, None, :] - cent[None]) ** 2).sum(-1), axis=1)
        new = np.array([points[lab == j].mean(0) if (lab == j).any() else cent[j] for j in range(k)])
        if np.allclose(new, cent):
            break
        cent = new
    lab = np.argmin(((points[:, None, :] - cent[None]) ** 2).sum(-1), axis=1)
    # centre each sphere on its cluster's bounding box (smaller covering radius than the mean)
    out_c, out_r = [], []
    for j in range(k):
        m = points[lab == j]
        if not len(m):
            continue
        c = (m.min(0) + m.max(0)) / 2
        out_c.append(c)
        out_r.append(float(np.linalg.norm(m - c, axis=1).max()) + VOXEL * math.sqrt(3) / 2)
    return np.array(out_c), np.array(out_r)


def bulge(tris: np.ndarray, c: np.ndarray, r: np.ndarray) -> float:
    """How far (mm) the spheres reach beyond the shape, 95th percentile over points on the sphere surfaces."""
    surf = surface_points(tris, 4.0)
    rng = np.random.default_rng(2)
    worst = []
    for ci, ri in zip(c, r):
        d = rng.normal(size=(60, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        p = ci + d * ri
        inside_other = (np.linalg.norm(p[:, None] - c[None], axis=2) < r[None] - 1e-6).any(1)
        p = p[~inside_other]                         # only the outer skin of the union
        if len(p):
            dist = np.sqrt(((p[:, None] - surf[None, ::2]) ** 2).sum(-1)).min(1)
            worst.extend(dist.tolist())
    return float(np.percentile(worst, 95)) if worst else 0.0


def build() -> dict:
    model = {"_note": "Built by model/build_robot_model.py. Spheres (centre mm, radius mm) per shape. Arm links in their "
                      "URDF link frames (link_base = robot base, link1..link5 = frames after joints 1..5; link5 = "
                      "flange). Head in the head frame H: origin = flange centre, z = flange axis, x = camera view "
                      "direction. The head is turned about the flange axis to match the camera axis in use.",
             "links": {}, "report": {}}
    for name in ("link_base", "link1", "link2", "link3", "link4", "link5"):
        tris = load_stl(MESHES / f"{name}.stl") * 1000.0
        vox, inner = solid_voxels(tris)
        c, r = spheres(vox, K[name])
        model["links"][name] = {"c": np.round(c, 1).tolist(), "r": np.round(r, 1).tolist()}
        model["report"][name] = {"spheres": len(r), "voxels": len(vox), "inside_voxels": inner,
                                 "max_radius_mm": round(float(r.max()), 1), "bulge_p95_mm": round(bulge(tris, c, r), 1)}
        print(f"{name:9s} {len(r):2d} spheres, r {r.min():5.1f}..{r.max():5.1f} mm, {len(vox)} voxels "
              f"({inner} inside), bulge p95 {model['report'][name]['bulge_p95_mm']} mm")
    tris_cad = load_stl(HEAD_STL)
    tris = cad_to_head(tris_cad)
    vox, inner = solid_voxels(tris)
    c, r = spheres(vox, K["head"])
    v = tris.reshape(-1, 3)
    cam = cad_to_head(HEAD_CAMERA_CAD)
    model["head"] = {
        "c": np.round(c, 1).tolist(), "r": np.round(r, 1).tolist(),
        "camera_center_mm": np.round(cam, 2).tolist(), "camera_axis": [1.0, 0.0, 0.0],
        "bbox_mm": [np.round(v.min(0), 1).tolist(), np.round(v.max(0), 1).tolist()],
        "source": "Camera Gatto.step (Fusion 360) = model/camera_head.step -> model/camera_head.stl; mount ISO 9409-1-50-4-M6 "
                  "(4 holes dia 6.2 on PCD 50), plate face at CAD z=-7, centre CAD (36,0); lens hole dia 6.3 along "
                  "CAD -y at x 36, z 33.07",
    }
    model["report"]["head"] = {"spheres": len(r), "voxels": len(vox), "inside_voxels": inner,
                               "max_radius_mm": round(float(r.max()), 1), "bulge_p95_mm": round(bulge(tris, c, r), 1)}
    print(f"head      {len(r):2d} spheres, r {r.min():5.1f}..{r.max():5.1f} mm, {len(vox)} voxels ({inner} inside), "
          f"bulge p95 {model['report']['head']['bulge_p95_mm']} mm")
    print(f"head size (H frame): x {v[:, 0].min():.1f}..{v[:, 0].max():.1f}, y {v[:, 1].min():.1f}..{v[:, 1].max():.1f},"
          f" z {v[:, 2].min():.1f}..{v[:, 2].max():.1f} mm; camera at {np.round(cam, 1).tolist()}")
    return model


def plot(model: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shapes = [(n, load_stl(MESHES / f"{n}.stl") * 1000.0, model["links"][n]) for n in model["links"]]
    shapes.append(("head", cad_to_head(load_stl(HEAD_STL)), model["head"]))
    fig, axes = plt.subplots(2, 7, figsize=(21, 6.5))
    t = np.linspace(0, 2 * np.pi, 48)
    for col, (name, tris, sph) in enumerate(shapes):
        v = tris.reshape(-1, 3)[::3]
        for row, (i, j) in enumerate(((0, 2), (1, 2))):
            ax = axes[row, col]
            ax.scatter(v[:, i], v[:, j], s=0.2, c="0.35", alpha=0.4)
            for c, r in zip(sph["c"], sph["r"]):
                ax.plot(c[i] + r * np.cos(t), c[j] + r * np.sin(t), color="tab:orange", lw=0.7)
            if name == "head":
                cam = model["head"]["camera_center_mm"]
                ax.plot(cam[i], cam[j], "b*", ms=9)
            ax.set_aspect("equal")
            ax.set_title(f"{name}  ({'xz' if row == 0 else 'yz'})", fontsize=9)
            ax.tick_params(labelsize=6)
    fig.suptitle("collision spheres (orange) over the real meshes (grey); head: blue star = camera optical centre")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main() -> int:
    model = build()
    OUT.write_text(json.dumps(model, indent=1), encoding="utf-8")
    png = PNG
    plot(model, png)
    print(f"-> {OUT}  ({OUT.stat().st_size // 1024} kB)\n-> {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
