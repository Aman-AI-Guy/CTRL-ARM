"""Reused COLMAP/Open3D algorithms from scan3d/reconstruct.py. No camera or robot controller."""
import json, os, re, shutil, struct, time
from pathlib import Path
from collections import defaultdict
from contextlib import contextmanager
from .sparse_model import read_model, quat_to_R
import cv2
import numpy as np

def colmap(cfg, args, log):
    return cfg['_command'](args)


class ReconError(RuntimeError):
    pass

class Stages:
    """Times each stage and prints progress."""

    def __init__(self):
        self.times = {}

    @contextmanager
    def __call__(self, name):
        t = time.perf_counter()
        print(f"[{name}] ...", flush=True)
        try:
            yield
        finally:
            self.times[name] = round(time.perf_counter() - t, 2)
            print(f"[{name}] {self.times[name]:.1f} s", flush=True)

def settings(cfg: dict, mode: str) -> tuple[dict, dict]:
    r, m = cfg["reconstruct"], cfg["mesh"]
    s = {k: v for k, v in r["common"].items() if not k.startswith("_")} | r.get(mode, {})
    ms = {k: v for k, v in m["common"].items() if not k.startswith("_")} | m.get(mode, {})
    return s, ms



def run_sparse(cfg, s, ws: Path, imgs: Path, log: Path, st: Stages, masks: Path | None = None) -> tuple[dict, Path]:
    db = ws / "database.db"
    if db.exists():
        db.unlink()
    with st("features"):
        colmap(cfg, ["feature_extractor", "--database_path", db, "--image_path", imgs,
                     "--ImageReader.single_camera", 1, "--ImageReader.camera_model", s["camera_model"],
                     "--FeatureExtraction.use_gpu", 1, "--FeatureExtraction.max_image_size", s["sift_max_image_size"],
                     "--SiftExtraction.max_num_features", s["max_num_features"]]
               + (["--ImageReader.mask_path", masks] if masks else []), log)
    with st("matching"):
        if s["matcher"] == "sequential":
            colmap(cfg, ["sequential_matcher", "--database_path", db, "--FeatureMatching.use_gpu", 1,
                         "--SequentialMatching.overlap", s["sequential_overlap"]], log)
        else:
            colmap(cfg, ["exhaustive_matcher", "--database_path", db, "--FeatureMatching.use_gpu", 1], log)
    sparse = ws / "sparse"
    if sparse.exists():
        shutil.rmtree(sparse)
    sparse.mkdir()
    with st("mapper"):
        if s["mapper"] == "global_mapper":
            colmap(cfg, ["global_mapper", "--database_path", db, "--image_path", imgs, "--output_path", sparse], log)
        else:
            colmap(cfg, ["mapper", "--database_path", db, "--image_path", imgs, "--output_path", sparse], log)
    best, best_dir = None, None
    for d in sorted(p for p in sparse.iterdir() if p.is_dir() and p.name.isdigit()):
        txt = ws / "sparse_txt" / d.name
        txt.mkdir(parents=True, exist_ok=True)
        colmap(cfg, ["model_converter", "--input_path", d, "--output_path", txt, "--output_type", "TXT"], log)
        m = read_model(txt)
        m["n_models"] = 1
        if best is None or len(m["images"]) > len(best["images"]):
            best, best_dir = m, d
    if best is None:
        raise ReconError("COLMAP registered no images: not enough motion, overlap or texture")
    best["n_models"] = sum(1 for p in sparse.iterdir() if p.is_dir() and p.name.isdigit())
    return best, best_dir

def intrinsics(model: str, params: list[float]):
    p = params
    if model == "SIMPLE_PINHOLE":
        return np.array([[p[0], 0, p[1]], [0, p[0], p[2]], [0, 0, 1]]), np.zeros(4)
    if model == "PINHOLE":
        return np.array([[p[0], 0, p[2]], [0, p[1], p[3]], [0, 0, 1]]), np.zeros(4)
    if model == "SIMPLE_RADIAL":
        return np.array([[p[0], 0, p[1]], [0, p[0], p[2]], [0, 0, 1]]), np.array([p[3], 0, 0, 0])
    if model == "RADIAL":
        return np.array([[p[0], 0, p[1]], [0, p[0], p[2]], [0, 0, 1]]), np.array([p[3], p[4], 0, 0])
    if model in ("OPENCV", "FULL_OPENCV"):
        K = np.array([[p[0], 0, p[2]], [0, p[1], p[3]], [0, 0, 1]])
        return K, np.array(p[4:8] if model == "OPENCV" else [p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11]])
    raise ReconError(f"camera model {model} not supported for marker scaling")

def triangulate(views):
    """Linear triangulation from [(P 3x4 normalised camera, (x, y) normalised)]: (X, rms error in normalised units)."""
    A = []
    for P, (x, y) in views:
        A += [x * P[2] - P[0], y * P[2] - P[1]]
    X = np.linalg.svd(np.array(A))[2][-1]
    X = X[:3] / X[3]
    err = []
    for P, (x, y) in views:
        q = P @ np.append(X, 1)
        err.append(np.hypot(q[0] / q[2] - x, q[1] / q[2] - y) if q[2] > 0 else 1.0)
    return X, np.array(err)

def marker_scale(imgs: Path, model: dict, acfg: dict) -> dict:
    """Triangulate each marker corner from every registered view; scale = real side / reconstructed side."""
    det_params = cv2.aruco.DetectorParameters()
    det_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, acfg["dictionary"])), det_params)
    wanted = set(acfg.get("ids") or [])
    obs = defaultdict(list)
    seen = defaultdict(int)
    for name, im in model["images"].items():
        gray = cv2.imread(str(imgs / name), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        corners, ids, _ = det.detectMarkers(gray)
        if ids is None:
            continue
        cam = model["cameras"][im["camera"]]
        K, dist = intrinsics(cam[0], cam[3])
        P = np.column_stack([im["R"], im["t"]])
        for c, i in zip(corners, ids.ravel()):
            if wanted and int(i) not in wanted:
                continue
            seen[int(i)] += 1
            und = cv2.undistortPoints(c.reshape(-1, 1, 2).astype(np.float64), K, dist).reshape(-1, 2)
            for k in range(4):
                obs[(int(i), k)].append((P, und[k], K[0, 0]))
    corners3d, edges, per_marker = {}, [], {}
    for key, views in obs.items():
        if len(views) < 2:
            continue
        vv = [(P, x) for P, x, _ in views]
        X, err = triangulate(vv)
        f = views[0][2]
        keep = err * f < 3.0                                 # drop views more than 3 px off, then redo
        if keep.sum() >= 2 and not keep.all():
            X, err = triangulate([v for v, k in zip(vv, keep) if k])
        if np.median(err) * f < 2.0:
            corners3d[key] = X
    for mid in sorted({k[0] for k in corners3d}):
        if all((mid, k) in corners3d for k in range(4)):
            e = [float(np.linalg.norm(corners3d[(mid, k)] - corners3d[(mid, (k + 1) % 4)])) for k in range(4)]
            per_marker[mid] = e
            edges += e
    out = {"dictionary": acfg["dictionary"], "marker_mm": acfg["marker_mm"], "views_per_marker": dict(seen),
           "markers_used": sorted(per_marker)}
    if not edges:
        out.update(scale=None, note="no marker triangulated in 2+ registered views")
        return out
    scale = float(acfg["marker_mm"]) / float(np.median(edges))
    mm = np.array(edges) * scale
    out.update(scale=scale, edges_mm=[round(float(v), 2) for v in mm],
               spread_pct=round(float(np.std(mm) / acfg["marker_mm"] * 100), 2),
               per_marker_mm={m: round(float(np.mean(e)) * scale, 2) for m, e in per_marker.items()})
    return out

def undistort_masks(cfg, model: dict, dense: Path, masks: Path, log: Path) -> Path:
    """Masks for the undistorted (and resized) dense images: same warp as COLMAP's image_undistorter."""
    txt = dense / "sparse_txt"
    txt.mkdir(exist_ok=True)
    colmap(cfg, ["model_converter", "--input_path", dense / "sparse", "--output_path", txt, "--output_type", "TXT"], log)
    und = read_model(txt)
    out = dense / "masks"
    out.mkdir(exist_ok=True)
    for name, im in und["images"].items():
        src = cv2.imread(str(masks / (name + ".png")), cv2.IMREAD_GRAYSCALE)
        if src is None:
            continue
        o_model, o_w, o_h, o_par = model["cameras"][model["images"][name]["camera"]]
        K, dist = intrinsics(o_model, o_par)
        _, u_w, u_h, u_par = und["cameras"][im["camera"]]          # PINHOLE fx fy cx cy
        uu, vv = np.meshgrid(np.arange(u_w, dtype=np.float64), np.arange(u_h, dtype=np.float64))
        pts = np.stack([(uu - u_par[2]) / u_par[0], (vv - u_par[3]) / u_par[1], np.ones_like(uu)], -1).reshape(-1, 1, 3)
        proj, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), K, dist)
        proj = proj.reshape(u_h, u_w, 2).astype(np.float32)
        if src.shape[:2] != (o_h, o_w):
            src = cv2.resize(src, (o_w, o_h), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(out / (name + ".png")), cv2.remap(src, proj[..., 0], proj[..., 1], cv2.INTER_NEAREST,
                                                          borderValue=0))
    return out

def run_dense(cfg, s, ws: Path, imgs: Path, model_dir: Path, log: Path, st: Stages,
              masks: Path | None = None, model: dict | None = None) -> Path:
    dense = ws / "dense"
    if dense.exists():
        shutil.rmtree(dense)
    with st("undistort"):
        colmap(cfg, ["image_undistorter", "--image_path", imgs, "--input_path", model_dir, "--output_path", dense,
                     "--output_type", "COLMAP", "--max_image_size", s["dense_max_image_size"]], log)
        dense_masks = undistort_masks(cfg, model, dense, masks, log) if masks and model else None
    geom = int(s["geom_consistency"])
    pm_cfg = dense / "stereo" / "patch-match.cfg"           # source images per view: "__auto__, N"
    pm_cfg.write_text(re.sub(r"__auto__, *\d+", f"__auto__, {int(s['patchmatch_src_images'])}", pm_cfg.read_text()))
    with st("patchmatch"):
        colmap(cfg, ["patch_match_stereo", "--workspace_path", dense, "--workspace_format", "COLMAP",
                     "--PatchMatchStereo.max_image_size", s["dense_max_image_size"],
                     "--PatchMatchStereo.geom_consistency", geom,
                     "--PatchMatchStereo.num_iterations", s["patchmatch_iterations"],
                     "--PatchMatchStereo.window_radius", s["patchmatch_window_radius"],
                     "--PatchMatchStereo.window_step", s["patchmatch_window_step"],
                     "--PatchMatchStereo.num_samples", s["patchmatch_num_samples"]], log)
    fused = dense / "fused.ply"
    with st("fusion"):
        colmap(cfg, ["stereo_fusion", "--workspace_path", dense, "--workspace_format", "COLMAP",
                     "--input_type", "geometric" if geom else "photometric", "--output_path", fused,
                     "--StereoFusion.min_num_pixels", s["fusion_min_num_pixels"]]
               + (["--StereoFusion.mask_path", dense_masks] if dense_masks else []), log)
    return fused

def orbit_center(centers: np.ndarray, axes: np.ndarray) -> np.ndarray:
    """Point closest to every camera's optical axis: where the orbit was aimed."""
    A, b = np.zeros((3, 3)), np.zeros(3)
    for c, d in zip(centers, axes):
        M = np.eye(3) - np.outer(d, d)
        A += M
        b += M @ c
    return np.linalg.lstsq(A, b, rcond=None)[0]

def rotation_to_z(n: np.ndarray) -> np.ndarray:
    n = n / np.linalg.norm(n)
    v, c = np.cross(n, [0, 0, 1.0]), float(n @ [0, 0, 1.0])
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1, -1])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))

def _lab(colors: np.ndarray) -> np.ndarray:
    rgb = (np.clip(colors, 0, 1) * 255).astype(np.uint8)[None]
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[0].astype(np.float64)

def drop_table_colour(pcd, table_idx, obj, m: dict):
    """Drop object points that have the table's colour: in preview mode the table/object junction smears table
    points upwards (a grey 'skirt' that widens L and W). Table colour = Lab mean + covariance of the plane inliers
    near the orbit centre; a point goes when its Mahalanobis distance is below table_colour_sigma. Skipped when
    the object itself looks like the table (more than table_colour_max_drop of it would go)."""
    sigma = float(m.get("table_colour_sigma", 0) or 0)
    if sigma <= 0 or not obj.has_colors() or not pcd.has_colors():
        return obj, {"applied": False, "reason": "off or no colours"}
    P = np.asarray(pcd.points)[table_idx]
    near = np.hypot(P[:, 0], P[:, 1]) < m["crop_radius_mm"]
    tab = _lab(np.asarray(pcd.colors)[table_idx][near])
    if len(tab) < 200:
        return obj, {"applied": False, "reason": f"only {len(tab)} table points near the object"}
    mu, cov = tab.mean(0), np.cov(tab.T) + np.eye(3) * 4.0
    lab = _lab(np.asarray(obj.colors))
    dd = lab - mu
    dist = np.sqrt(np.einsum("ij,jk,ik->i", dd, np.linalg.inv(cov), dd))
    drop = dist < sigma
    frac = float(drop.mean())
    rep = {"sigma": sigma, "table_lab_mean": [round(float(v), 1) for v in mu], "dropped_frac": round(frac, 3)}
    if frac > float(m.get("table_colour_max_drop", 0.6)):
        return obj, {**rep, "applied": False, "reason": "the object looks like the table"}
    return obj.select_by_index(np.where(~drop)[0]), {**rep, "applied": True}

def build_mesh(fused: Path, scale: float, model: dict, m: dict, out: Path, st: Stages) -> dict:
    import open3d as o3d

    info = {}
    with st("cloud"):
        pcd = o3d.io.read_point_cloud(str(fused))
        info["dense_points"] = len(pcd.points)
        if len(pcd.points) < 200:
            raise ReconError(f"dense cloud has only {len(pcd.points)} points")
        cams = np.array([im["center"] for im in model["images"].values()]) * scale
        axes = np.array([im["axis"] for im in model["images"].values()])
        pcd.scale(scale, center=(0, 0, 0))
        plane, inl = pcd.segment_plane(m["plane_distance_mm"], 3, 3000)
        n, d = np.array(plane[:3]), plane[3]
        if (cams.mean(0) @ n + d) < 0:                     # normal towards the cameras = up
            n, d = -n, -d
        info["table_inliers"] = len(inl)
        center = orbit_center(cams, axes)
        c0 = center - (center @ n + d) * n                # orbit centre on the table
        Rz = rotation_to_z(n)
        pcd.translate(-c0)
        pcd.rotate(Rz, center=(0, 0, 0))
        cams_t = (cams - c0) @ Rz.T
        P = np.asarray(pcd.points)
        r = np.hypot(P[:, 0], P[:, 1])
        keep = np.where((P[:, 2] > m["table_margin_mm"]) & (P[:, 2] < m["max_height_mm"]) & (r < m["crop_radius_mm"]))[0]
        obj = pcd.select_by_index(keep)
        info["cropped_points"] = len(obj.points)
        if len(obj.points) < 100:
            raise ReconError("nothing above the table near the orbit centre (check the crop radius / scale)")
        obj = obj.voxel_down_sample(m["voxel_mm"])
        obj, _ = obj.remove_statistical_outlier(m["outlier_neighbors"], m["outlier_std"])
        labels = np.array(obj.cluster_dbscan(m["dbscan_eps_mm"], m["dbscan_min_points"]))
        if labels.max() < 0:
            raise ReconError("no cluster found above the table")
        P = np.asarray(obj.points)
        clusters = []
        for lab in range(labels.max() + 1):
            idx = np.where(labels == lab)[0]
            cxy = P[idx, :2].mean(0)
            clusters.append((len(idx), float(np.hypot(*cxy)), idx))
        near = [c for c in clusters if c[1] < m["center_radius_mm"]]
        pool = near if (m["pick"] == "center" and near) else clusters
        size, dist, idx = max(pool, key=lambda c: c[0])
        obj = obj.select_by_index(idx)
        info.update(clusters=len(clusters), cluster_points=int(size), cluster_offset_mm=round(dist, 1))
        obj, info["table_colour"] = drop_table_colour(pcd, inl, obj, m)
        if not obj.has_normals():
            obj.estimate_normals()
            obj.orient_normals_towards_camera_location(cams_t.mean(0))

    # size: footprint rectangle on the table and height above it; then turn the long side onto x
    # robust extents: orientation from the minimum-area rectangle, size from percentiles along its axes
    P = np.asarray(obj.points)
    (_, _), (w, h), ang = cv2.minAreaRect(P[:, :2].astype(np.float32))
    if w < h:
        ang += 90
    a = -np.radians(ang)
    Ra = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    q = float(m["size_percentile"])
    Q = P @ Ra.T
    top = np.percentile(Q[:, 2], 100 - q)
    band = Q[Q[:, 2] >= max(m["table_margin_mm"], m["size_skip_bottom_frac"] * top)]
    lo, hi = np.percentile(band, q, axis=0), np.percentile(band, 100 - q, axis=0)
    hi[2] = top
    mid = (lo + hi) / 2
    obj.rotate(Ra, center=(0, 0, 0))
    obj.translate([-mid[0], -mid[1], 0])
    info["size_mm"] = {"L": round(float(hi[0] - lo[0]), 1), "W": round(float(hi[1] - lo[1]), 1),
                       "H": round(float(hi[2]), 1)}
    info["size_extremes_mm"] = {"L": round(float(max(w, h)), 1), "W": round(float(min(w, h)), 1),
                                "H": round(float(P[:, 2].max()), 1)}

    with st("poisson"):
        mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(obj, depth=int(m["poisson_depth"]))
        dens = np.asarray(dens)
        mesh.remove_vertices_by_mask(dens < np.quantile(dens, m["density_trim_quantile"]))
        bb = obj.get_axis_aligned_bounding_box()
        lo, hi = np.asarray(bb.min_bound) - 2 * m["voxel_mm"], np.asarray(bb.max_bound) + 2 * m["voxel_mm"]
        lo[2] = 0.0
        mesh = mesh.crop(o3d.geometry.AxisAlignedBoundingBox(lo, hi))
        tri_cl, n_tri, _ = mesh.cluster_connected_triangles()
        tri_cl, n_tri = np.asarray(tri_cl), np.asarray(n_tri)
        if len(n_tri):
            mesh.remove_triangles_by_mask(tri_cl != int(np.argmax(n_tri)))
        mesh.remove_unreferenced_vertices()
    with st("decimate"):
        if len(mesh.triangles) > m["target_triangles"]:
            mesh = mesh.simplify_quadric_decimation(int(m["target_triangles"]))
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
    if len(mesh.triangles) == 0:
        raise ReconError("Poisson produced an empty mesh")
    mb = mesh.get_axis_aligned_bounding_box()
    ext = np.asarray(mb.max_bound) - np.asarray(mb.min_bound)
    info["mesh_size_mm"] = {"L": round(float(ext[0]), 1), "W": round(float(ext[1]), 1),
                            "H": round(float(mb.max_bound[2]), 1)}
    obb = obj.get_minimal_oriented_bounding_box() if hasattr(obj, "get_minimal_oriented_bounding_box") \
        else obj.get_oriented_bounding_box()
    info["obb3d_mm"] = sorted((round(float(v), 1) for v in obb.extent), reverse=True)
    info["triangles"], info["vertices"] = len(mesh.triangles), len(mesh.vertices)

    with st("export"):
        md = out / "mesh"
        md.mkdir(parents=True, exist_ok=True)
        o3d.io.write_triangle_mesh(str(md / "object.stl"), mesh)
        o3d.io.write_triangle_mesh(str(md / "object.ply"), mesh, write_vertex_colors=True)
        o3d.io.write_point_cloud(str(md / "cloud.ply"), obj)
        write_glb(md / "object.glb", mesh)
        info["files"] = {k: str((md / f).relative_to(out)) for k, f in
                         (("stl", "object.stl"), ("ply", "object.ply"), ("glb", "object.glb"), ("cloud", "cloud.ply"))}
    with st("renders"):
        info["renders"] = render_views(mesh, out / "renders", info["size_mm"], int(m["render_px"]))
    return info

def write_glb(path: Path, mesh) -> None:
    """Minimal glTF 2.0 binary: positions in metres, y up (from mm, z up), normals, vertex colours."""
    v = np.asarray(mesh.vertices) / 1000.0
    v = np.column_stack([v[:, 0], v[:, 2], -v[:, 1]]).astype(np.float32)
    nrm = np.asarray(mesh.vertex_normals)
    nrm = np.column_stack([nrm[:, 0], nrm[:, 2], -nrm[:, 1]]).astype(np.float32)
    col = (np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else np.full((len(v), 3), 0.7)).astype(np.float32)
    idx = np.asarray(mesh.triangles).astype(np.uint32).ravel()
    blobs = [v.tobytes(), nrm.tobytes(), col.tobytes(), idx.tobytes()]
    views, offset = [], 0
    for k, b in enumerate(blobs):
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(b), "target": 34963 if k == 3 else 34962})
        offset += (len(b) + 3) // 4 * 4
    binary = b"".join(b + b"\0" * ((4 - len(b) % 4) % 4) for b in blobs)
    gltf = {"asset": {"version": "2.0", "generator": "scan3d"}, "scene": 0, "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0, "name": "object"}],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1, "COLOR_0": 2}, "indices": 3,
                                        "material": 0}]}],
            "materials": [{"pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": 0,
                                                    "roughnessFactor": 1}, "doubleSided": True}],
            "accessors": [
                {"bufferView": 0, "componentType": 5126, "count": len(v), "type": "VEC3",
                 "min": v.min(0).tolist(), "max": v.max(0).tolist()},
                {"bufferView": 1, "componentType": 5126, "count": len(v), "type": "VEC3"},
                {"bufferView": 2, "componentType": 5126, "count": len(v), "type": "VEC3"},
                {"bufferView": 3, "componentType": 5125, "count": len(idx), "type": "SCALAR"}],
            "bufferViews": views, "buffers": [{"byteLength": len(binary)}]}
    js = json.dumps(gltf).encode()
    js += b" " * ((4 - len(js) % 4) % 4)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(binary)))
        f.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
        f.write(struct.pack("<II", len(binary), 0x004E4942) + binary)

def render_views(mesh, out: Path, size_mm: dict, px: int) -> list[str]:
    """Orthographic shaded renders with a 10 mm grid, by painter's algorithm (no GPU or window needed)."""
    out.mkdir(parents=True, exist_ok=True)
    V = np.asarray(mesh.vertices)
    T = np.asarray(mesh.triangles)
    N = np.asarray(mesh.triangle_normals)
    C = np.asarray(mesh.vertex_colors)[T].mean(1) if mesh.has_vertex_colors() else np.full((len(T), 3), 0.7)
    span = float(np.max(V.max(0) - V.min(0))) * 1.15 + 10
    k = px / span
    paths = []
    for name, (u, v, w) in VIEWS.items():
        u, v, w = (np.array(a, float) for a in (u, v, w))
        cu, cv = (V @ u).mean(), (V @ v).mean()
        img = np.full((px, px, 3), 245, np.uint8)
        g0 = int(np.floor((cu - span / 2) / 10)) * 10
        for gmm in range(g0, int(cu + span / 2) + 10, 10):         # 10 mm grid
            x = int((gmm - cu) * k + px / 2)
            cv2.line(img, (x, 0), (x, px), (225, 225, 225), 1)
        g0 = int(np.floor((cv - span / 2) / 10)) * 10
        for gmm in range(g0, int(cv + span / 2) + 10, 10):
            y = int(px / 2 - (gmm - cv) * k)
            cv2.line(img, (0, y), (px, y), (225, 225, 225), 1)
        facing = N @ w
        order = np.argsort((V[T] @ w).mean(1))                   # far first
        for t in order:
            if facing[t] <= 0:
                continue
            p = V[T[t]]
            pts = np.column_stack([(p @ u - cu) * k + px / 2, px / 2 - (p @ v - cv) * k])
            shade = 0.35 + 0.65 * facing[t]
            col = tuple(int(c) for c in np.clip(C[t][::-1] * 255 * shade, 0, 255))
            cv2.fillConvexPoly(img, np.round(pts * 16).astype(np.int32), col, cv2.LINE_AA, 4)
        txt = f"{name}   L {size_mm['L']} x W {size_mm['W']} x H {size_mm['H']} mm   (grid 10 mm)"
        cv2.putText(img, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1, cv2.LINE_AA)
        p = out / f"{name}.png"
        cv2.imwrite(str(p), img)
        paths.append(str(p.relative_to(out.parent)))
    return paths

VIEWS = {  # name: (u axis, v axis, towards-the-viewer axis), world mm, z up
    "front": ((1, 0, 0), (0, 0, 1), (0, -1, 0)),
    "side": ((0, 1, 0), (0, 0, 1), (1, 0, 0)),
    "back": ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
    "top": ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
}
