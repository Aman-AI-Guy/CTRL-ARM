"""Conservative interval bounds for the imported sphere model along joint-linear curves.

An analytic Lipschitz bound covers unsampled positions in each interval. This certifies
only the stated sphere/obstacle model and joint-linear interpolation. Controller tracking,
unknown obstacles and intended mechanical contact exclusions still require qualification.
"""

from itertools import combinations
import math
import numpy as np

from .flange import _FIXED
from .orbit_math import fk_frames, box_gap
from .robot_geometry import LINKS, MOVING

NAMES = (*LINKS, "head")
PAIRS = [
    (a, b)
    for a, b in combinations(NAMES, 2)
    if abs(NAMES.index(a) - NAMES.index(b)) > 1
]
# The head and link4 sphere covers overlap around their connected wrist assembly.
ASSEMBLY_CONTACT = ("link4", "head")


class IntervalValidator:
    def __init__(self, context):
        self.context = context
        self.geo = context.geometry
        self.tracking = float(
            (context.settings.motion_qualification or {}).get("tracking_error_mm", 0)
        )
        self.cache = {}
        self.lengths = [float(np.linalg.norm(t[:3, 3])) for t in _FIXED]
        self.count = {name: i for i, name in enumerate(LINKS)} | {
            "head": 5,
            "tcp": 5,
            "tool": 5,
            "camera": 5,
        }
        self.local = {
            name: float(np.linalg.norm(c, axis=1).max())
            for name, (c, _) in self.geo.shapes.items()
        }
        self.local.update(
            tcp=float(np.linalg.norm(context.tcp[:3, 3])),
            camera=float(np.linalg.norm(context.flange_camera[:3, 3])),
            tool=float(np.linalg.norm(context.tcp[:3, 3]))
            + max(
                [np.linalg.norm(s[:3]) for s in context.settings.tool_spheres_tcp_mm]
                or [0]
            ),
        )

    def weights(self, name, common=0):
        n = self.count[name]
        return np.array(
            [
                sum(self.lengths[j + 1 : n]) + self.local[name]
                if common <= j < n
                else 0
                for j in range(5)
            ]
        )

    def constraints(self, joints):
        key = tuple(float(q) for q in joints)
        if key in self.cache:
            return self.cache[key]
        ctx = self.context
        frames = fk_frames(joints)
        world = self.geo.world(frames, ctx.turn)
        tcp = frames[-1] @ ctx.tcp
        world["tcp"] = (tcp[:3, 3].reshape(1, 3), np.array([0.0]))
        world["camera"] = (
            (frames[-1] @ ctx.flange_camera)[:3, 3].reshape(1, 3),
            np.array([ctx.gateway.camera.radius_mm if ctx.gateway.camera else 0]),
        )
        if ctx.settings.tool_spheres_tcp_mm:
            tools = np.asarray(ctx.settings.tool_spheres_tcp_mm, float)
            world["tool"] = (tools[:, :3] @ tcp[:3, :3].T + tcp[:3, 3], tools[:, 3])
        constraints = []
        for name in [*MOVING, "camera", *(["tool"] if "tool" in world else [])]:
            centers, radii = world[name]
            weights = self.weights(name)
            table = (
                float(
                    (
                        centers[:, 2]
                        - radii
                        - ctx.scene.table_z
                        - ctx.settings.table_margin_mm
                    ).min()
                )
                - self.tracking
            )
            constraints.append((f"table/{name}", table, weights))
            radial = (
                np.linalg.norm(centers[:, :2] - ctx.foot[:2], axis=1)
                - radii
                - ctx.scene.obj_radius
                - ctx.settings.object_margin_mm
            )
            above = (
                centers[:, 2]
                - radii
                - ctx.scene.obj_top
                - ctx.settings.object_margin_mm
            )
            constraints.append(
                (
                    f"object/{name}",
                    float(np.maximum(radial, above).min()) - self.tracking,
                    weights,
                )
            )
        for a, b in PAIRS:
            if (a, b) == ASSEMBLY_CONTACT:
                continue
            ca, ra = world[a]
            cb, rb = world[b]
            gap = float(
                (
                    np.linalg.norm(ca[:, None] - cb[None, :], axis=2)
                    - ra[:, None]
                    - rb[None, :]
                ).min()
            )
            common = min(self.count[a], self.count[b])
            constraints.append(
                (
                    f"self/{a}/{b}",
                    gap - ctx.settings.self_margin_mm - 2 * self.tracking,
                    self.weights(a, common) + self.weights(b, common),
                )
            )
        if "tool" in world:
            ca, ra = world["tool"]
            for name in LINKS[:-1]:
                cb, rb = world[name]
                gap = float(
                    (
                        np.linalg.norm(ca[:, None] - cb[None, :], axis=2)
                        - ra[:, None]
                        - rb[None, :]
                    ).min()
                )
                common = min(self.count["tool"], self.count[name])
                constraints.append(
                    (
                        f"self/tool/{name}",
                        gap - ctx.settings.self_margin_mm - 2 * self.tracking,
                        self.weights("tool", common) + self.weights(name, common),
                    )
                )
        for name, (centers, radii) in world.items():
            weights = self.weights(name)
            for index, box in enumerate(ctx.gateway.keepouts):
                lo = [box.x[0], box.y[0], box.z[0]]
                hi = [box.x[1], box.y[1], box.z[1]]
                gap = (
                    float(box_gap(centers, radii, lo, hi).min())
                    - ctx.settings.object_margin_mm
                    - self.tracking
                )
                constraints.append((f"keepout/{index}/{name}", gap, weights))
            if name in {"tcp", "head", "camera", "tool"} and ctx.gateway.limits:
                limits = ctx.gateway.limits
                lo = np.array([limits.x[0], limits.y[0], limits.z[0]])
                hi = np.array([limits.x[1], limits.y[1], limits.z[1]])
                gap = (
                    float(
                        np.minimum(
                            centers - lo - radii[:, None], hi - centers - radii[:, None]
                        ).min()
                    )
                    - self.tracking
                )
                constraints.append((f"workspace/{name}", gap, weights))
                if limits.radius_mm is not None:
                    constraints.append(
                        (
                            f"cylinder/{name}",
                            float(
                                (
                                    limits.radius_mm
                                    - np.linalg.norm(centers[:, :2], axis=1)
                                    - radii
                                ).min()
                            )
                            - self.tracking,
                            weights,
                        )
                    )
        self.cache[key] = constraints
        return constraints

    def validate(self, start, target, *, max_depth=14, max_evaluations=4096):
        begin, end = np.asarray(start, float), np.asarray(target, float)
        if (
            begin.shape != (5,)
            or end.shape != (5,)
            or not np.isfinite([begin, end]).all()
        ):
            return {
                "ok": False,
                "reason": "Five finite start and target joints are required.",
            }
        intervals = [(begin, end, 0)]
        leaves = 0
        minimum = math.inf
        accepted = []
        while intervals:
            a, b, depth = intervals.pop()
            mid = (a + b) / 2
            rows = self.constraints(mid)
            failures = [name for name, gap, _ in rows if gap < 0]
            if failures:
                return {
                    "ok": False,
                    "reason": "Conservative robot model: " + ", ".join(failures),
                    "evaluations": len(self.cache),
                }
            half = np.abs(np.deg2rad(b - a)) / 2
            margins = [gap - float(weights @ half) for _, gap, weights in rows]
            if min(margins, default=0) >= 0:
                minimum = min(minimum, min(margins))
                leaves += 1
                accepted.append((a, b))
                continue
            if depth >= max_depth or len(self.cache) >= max_evaluations:
                return {
                    "ok": False,
                    "reason": "Unable to bound unsampled clearance within the validation budget.",
                    "evaluations": len(self.cache),
                }
            intervals.extend([(a, mid, depth + 1), (mid, b, depth + 1)])
        return {
            "ok": True,
            "reason": "All joint-linear intervals bounded by sphere-motion Lipschitz limits.",
            "evaluations": len(self.cache),
            "intervals": leaves,
            "minimum_model_clearance_mm": minimum,
            "joint_waypoints": [begin.tolist()]
            + [
                b.tolist()
                for _, b in sorted(
                    accepted, key=lambda ab: float(np.linalg.norm(ab[0] - begin))
                )
            ],
            "joint_linear_model_bounded": True,
            "controller_interpolation_proven": False,
            "self_pairs_checked": [list(p) for p in PAIRS if p != ASSEMBLY_CONTACT],
            "assembly_contact_exclusion": list(ASSEMBLY_CONTACT),
            "collision_avoidance": False,
        }
