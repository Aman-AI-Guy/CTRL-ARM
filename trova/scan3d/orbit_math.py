"""Pure planning/IK extracted from the retained arm_pc/orbit_here.py. No SDK or execution code."""
import math
import numpy as np
from .flange import D2R, _FIXED, _rz
from .robot_geometry import MOVING
XARM5_LIMITS = np.array([(-360,360),(-118,120),(-225,11),(-97,180),(-360,360)], float)
LIMITS = XARM5_LIMITS.copy()
LIMITS[1,0] = -60

def fk_frames(q):
    t, out = np.eye(4), []
    for fixed, qi in zip(_FIXED, q):
        t = t @ fixed @ _rz(float(qi) * D2R)
        out.append(t)
    return out

def fk(q):
    return fk_frames(q)[-1]

def clip(q):
    return np.clip(np.asarray(q, float), LIMITS[:, 0], LIMITS[:, 1])

def rpy_matrix(roll, pitch, yaw):
    r, p, y = roll * D2R, pitch * D2R, yaw * D2R
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])

def pose_matrix(pose):
    t = np.eye(4)
    t[:3, :3] = rpy_matrix(*pose[3:6])
    t[:3, 3] = pose[:3]
    return t

class Camera:
    """Optical centre (offset) and optical axis in the FLANGE frame."""

    def __init__(self, offset, axis):
        self.offset = np.asarray(offset, float)
        self.axis = np.asarray(axis, float) / np.linalg.norm(axis)

    def ray(self, q):
        f = fk(q)
        return f[:3, :3] @ self.offset + f[:3, 3], f[:3, :3] @ self.axis

def solve_view(cam, pos, direction, seed, scale=250.0, iters=60, tol_mm=1.0, tol_deg=0.5):
    """Joints putting the camera at pos looking along direction (Levenberg-Marquardt from seed), or None."""
    pos, direction = np.asarray(pos, float), np.asarray(direction, float)
    tol_dir = 2 * math.sin(tol_deg * D2R / 2)

    def res(q):
        p, v = cam.ray(q)
        return np.concatenate([p - pos, (v - direction) * scale])

    q = clip(seed)
    r = res(q)
    cost, lam = float(r @ r), 1e-3
    for _ in range(iters):
        if np.linalg.norm(r[:3]) < tol_mm and np.linalg.norm(r[3:]) / scale < tol_dir:
            return q
        jac = np.empty((6, 5))
        for j in range(5):
            dq = np.zeros(5)
            dq[j] = 1e-3
            jac[:, j] = (res(q + dq) - r) / 1e-3
        jtj, g = jac.T @ jac, jac.T @ r
        while True:
            step = -np.linalg.solve(jtj + lam * (np.diag(np.diag(jtj)) + 1e-6 * np.eye(5)), g)
            big = np.abs(step).max()
            if big > 15:
                step *= 15 / big
            qn = clip(q + step)
            rn = res(qn)
            cn = float(rn @ rn)
            if cn < cost:
                q, r, cost, lam = qn, rn, cn, max(lam * 0.3, 1e-9)
                break
            lam *= 10
            if lam > 1e9:
                return None
    ok = np.linalg.norm(r[:3]) < tol_mm and np.linalg.norm(r[3:]) / scale < tol_dir
    return q if ok else None

def view_condition(cam, q, scale=250.0):
    """Numerical conditioning of the five camera-position/axis task variables."""
    q=np.asarray(q,float)
    position,direction=cam.ray(q)
    jac=np.zeros((6,5))
    for j in range(5):
        shifted=q.copy();shifted[j]+=.001
        p,d=cam.ray(shifted)
        jac[:,j]=np.r_[p-position,(d-direction)*scale]/.001
    singular=np.linalg.svd(jac,compute_uv=False)
    rank=int((singular>max(singular[0]*1e-6,1e-8)).sum())
    return {'rank':rank,'condition_number':float(singular[0]/singular[-1]) if rank==5 else None,
            'task':'camera_position_and_optical_axis_free_image_roll'}


class Scene:
    def __init__(self, obj_xy, obj_radius, obj_top, table_z, table_margin=40.0, obj_margin=30.0,
                 link_radius=45.0, camera_radius=45.0, base_radius=80.0, base_height=330.0,
                 geometry=None, head_turn=None, self_margin=5.0):
        self.obj_xy, self.obj_radius, self.obj_top, self.table_z = np.asarray(obj_xy, float), obj_radius, obj_top, table_z
        self.table_margin, self.obj_margin = table_margin, obj_margin
        self.link_radius, self.camera_radius = link_radius, camera_radius
        self.base_radius, self.base_height = base_radius, base_height
        self.geometry, self.head_turn, self.self_margin = geometry, head_turn, self_margin
        self.boxes = []          # --box keep-outs: (lo xyz, hi xyz), kept obj_margin away like the object
        self.notes = {}

def box_gap(c, r, lo, hi):
    """Gaps (mm) between spheres (centres c, radii r) and an axis-aligned box."""
    c = np.atleast_2d(c)
    return np.linalg.norm(np.clip(c, lo, hi) - c, axis=1) - r

def collisions(q, cam, sc):
    if sc.geometry is not None:
        frames = fk_frames(q)
        hits = sc.geometry.collisions(frames, sc.head_turn, sc.table_z, sc.table_margin, sc.obj_xy,
                                      sc.obj_radius, sc.obj_top, sc.obj_margin, sc.self_margin)
        if sc.boxes:
            w = sc.geometry.world(frames, sc.head_turn)
            for k, (lo, hi) in enumerate(sc.boxes):
                if any((box_gap(*w[n], lo, hi) < sc.obj_margin).any() for n in MOVING):
                    hits.append(f"box {k}")
        return sorted(hits)
    f = fk_frames(q)
    shoulder, elbow, wrist, flange = (f[i][:3, 3] for i in (1, 2, 3, 4))
    cam_p = f[4][:3, :3] @ cam.offset + flange
    hits = set()
    for k, (a, b, rad) in enumerate(((shoulder, elbow, sc.link_radius), (elbow, wrist, sc.link_radius),
                                     (wrist, flange, sc.link_radius), (flange, cam_p, sc.camera_radius))):
        for t in np.linspace(0, 1, 8):
            p = a + (b - a) * t
            if p[2] - rad < sc.table_z + sc.table_margin:
                hits.add("table")
            if k > 0 and math.hypot(p[0], p[1]) < sc.base_radius + rad and p[2] - rad < sc.base_height:
                hits.add("robot base")
            if (math.hypot(p[0] - sc.obj_xy[0], p[1] - sc.obj_xy[1]) < sc.obj_radius + rad + sc.obj_margin
                    and p[2] - rad < sc.obj_top + sc.obj_margin):
                hits.add("object")
            for kb, (lo, hi) in enumerate(sc.boxes):
                if box_gap(p, rad, lo, hi)[0] < sc.obj_margin:
                    hits.add(f"box {kb}")
    return sorted(hits)

def path_collisions(q0, q1, cam, sc):
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    steps = max(4, int(math.ceil(np.abs(q1 - q0).max() / 5.0)))
    hits = set()
    for t in np.linspace(0, 1, steps + 1):
        hits.update(collisions(q0 + (q1 - q0) * t, cam, sc))
    return sorted(hits)

def generic_seeds(pos):
    a = math.atan2(pos[1], pos[0]) / D2R
    out = []
    for q2, q3 in ((0, -60), (25, -85), (45, -115), (-15, -35)):
        for tilt in (0, 45, -45, 90, -90):
            for q5 in (-180, -90, 0, 90, 180):
                out.append(np.array([a, q2, q3, -(q2 + q3) + tilt, q5], float))
    return out

def view_pose(target, az, el, dist):
    u = np.array([math.cos(el * D2R) * math.cos(az * D2R), math.cos(el * D2R) * math.sin(az * D2R), math.sin(el * D2R)])
    return target + dist * u, -u

def solve_one(cam, sc, target, az, el, dist, prev, q0):
    """Collision-free solutions for one view, nearest to prev first (J5 +-360 variants included)."""
    pos, view = view_pose(target, az, el, dist)
    sols = []
    seeds = [prev, q0]
    for stage in (seeds, generic_seeds(pos)):
        for s in stage:
            q = solve_view(cam, pos, view, s, scale=dist)
            if q is None or collisions(q, cam, sc):
                continue
            for k in (0, -360, 360):
                qq = q.copy()
                qq[4] += k
                if LIMITS[4, 0] <= qq[4] <= LIMITS[4, 1] and all(np.abs(qq - x).max() > 3 for x in sols):
                    sols.append(qq)
        if sols:
            break
    sols.sort(key=lambda x: float(np.abs(x - prev).max()) + 0.2 * abs(x[4] - q0[4]) / 10)
    return sols

def per(values, k):
    """--orbit-dist / --dwell take one value or one per ring / view: the k-th, or the last one."""
    v = values if isinstance(values, (list, tuple)) else [values]
    return float(v[min(k, len(v) - 1)])
