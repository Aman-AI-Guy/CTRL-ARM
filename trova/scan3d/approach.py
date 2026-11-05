"""Camera-centered five-axis approaches using the imported camera IK and geometry.

Pure proposals: desired camera axis/position leave image roll free on the five-axis arm.
Every candidate is checked from the actual telemetry joints through MotionGateway.
"""

import math
import numpy as np

from . import orbit_math as orbit
from .flange import flange_pose
from ..motion import sample_joints


def propose(context, joints, clearance_mm, *, cancel=None):
    q0 = np.asarray(joints, float)
    gap = float(clearance_mm)
    if (
        q0.shape != (5,)
        or not np.isfinite(q0).all()
        or not math.isfinite(gap)
        or gap <= 0
    ):
        raise ValueError(
            "Five actual joint readings and positive measured camera clearance are required."
        )
    target = context.foot + [0, 0, context.settings.object_height_mm]
    camera, _ = context.cam.ray(q0)
    azimuth = math.degrees(math.atan2(camera[1] - target[1], camera[0] - target[0]))
    candidates = []
    for elevation in (90.0, 85.0, 75.0, 65.0):
        for azimuth_offset in (0.0, -20.0, 20.0):
            if cancel is not None and cancel.is_set():
                return {
                    "next": None,
                    "candidates": candidates,
                    "reason": "Planning cancelled.",
                }
            distance = gap / math.sin(math.radians(elevation))
            solutions = orbit.solve_one(
                context.cam,
                context.scene,
                target,
                azimuth + azimuth_offset,
                elevation,
                distance,
                q0,
                q0,
            )
            reasons = []
            for q in solutions:
                if hasattr(
                    context, "calibration_domain"
                ) and not context.calibration_domain(q):
                    reasons.append("Observation leaves the independently qualified calibration pose domain.")
                    candidates.append({'joints_deg':q.tolist(),'validated':False,'executable':False,
                        'status':'NOT_VERIFIED','reason':'Numerical IK only: this camera pose needs independent calibration-domain qualification.'})
                    continue
                condition = orbit.view_condition(context.cam, q, distance)
                if condition["rank"] < 5:
                    reasons.append(
                        "Near-singular five-axis camera task; evaluating another view."
                    )
                    continue
                path = sample_joints(q0, q)
                positions = [(flange_pose(p) @ context.tcp)[:3, 3] for p in path]
                ratios = [
                    math.dist(a, b) / max(1e-9, max(abs(x - y) for x, y in zip(qa, qb)))
                    for a, b, qa, qb in zip(positions, positions[1:], path, path[1:])
                ]
                speed, _ = context.rates(
                    min(
                        context.settings.joint_speed_deg_s,
                        context.gateway.max_joint_speed_deg_s,
                    )
                )
                if ratios and max(ratios) > 0:
                    speed = min(
                        speed, context.gateway.max_speed_mm_s / max(ratios) * 0.99
                    )
                decision = context.gateway.authorize_joints(
                    q,
                    speed,
                    source="vision_joint_approach",
                    geometry=context,
                    start_q=q0,
                )
                if not decision.ok:
                    reasons.append(decision.reason)
                    continue
                center, direction = context.cam.ray(q)
                item = {
                    "id": "object-approach",
                    "joints_deg": q.tolist(),
                    "target_mm": positions[-1].tolist(),
                    "camera_mm": center.tolist(),
                    "camera_direction": direction.tolist(),
                    "object_top_reference_mm": target.tolist(),
                    "ik_condition": condition,
                    "joint_path": [list(map(float, p)) for p in path],
                    "joint_speed_deg_s": speed,
                    "elevation_deg": elevation,
                    "clearance_mm": gap,
                    "gateway": decision.as_dict(),
                    "validated": True,
                    "executable": decision.allow_motion,
                    "reason": "Imported camera-axis IK and robot-body path checks passed; no motion sent.",
                }
                candidates.append(item)
                return {
                    "next": item,
                    "candidates": candidates,
                    "geometry": context.signature(),
                    "reason": item["reason"],
                    "collision_avoidance": False,
                    "orientation_goal": "camera_optical_axis_towards_object_with_free_image_roll",
                    "object_center_is_measured": False,
                }
            candidates.append(
                {
                    "elevation_deg": elevation,
                    "azimuth_deg": azimuth + azimuth_offset,
                    "validated": False,
                    "reason": "; ".join(sorted(set(reasons)))
                    or "Five-axis camera pose unreachable or obstructed.",
                }
            )
    return {
        "next": None,
        "candidates": candidates,
        "reason": "No qualified five-axis approach from the actual current joints.",
        "collision_avoidance": False,
    }
