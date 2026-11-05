"""Deterministic fixed-orientation viewpoint exploration; no driver calls here.

Coverage is viewpoint diversity, never a claim about observed mesh surfaces. The current
Cartesian driver cannot command arbitrary look-at orientations, so this planner keeps
the measured orientation and only proposes modest parallax translations.
"""

import math

import cv2
import numpy as np

from .geom import camera_pose_matrix, intersect_ray_plane, pixel_ray
from .motion import sample_segment

OFFSETS = [(0, 0), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]


def view_goal(host, location, viewpoint_id, yaw):
    """Derive a viewpoint from the calibrated object; an API cannot supply coordinates."""
    index = int(viewpoint_id.removeprefix('parallax-'))
    if not 0 <= index < len(OFFSETS):
        raise ValueError('unknown viewpoint')
    goal = host.vision.observation_target(location.get('point_mm'), host.observe_clearance_mm, yaw_deg=yaw)
    if not goal.get('ok'):
        return goal
    radius = min(30.0, float(host.observe_clearance_mm) * .15)
    dx, dy = OFFSETS[index]
    goal['target_mm'][0] += dx * radius
    goal['target_mm'][1] += dy * radius
    goal['pose6'][:3] = goal['target_mm']
    return goal


def projected_pixel(vision, pose6, point):
    transform = camera_pose_matrix(pose6, vision.hand_eye['matrix'])
    camera_point = np.linalg.inv(transform) @ np.array([*point, 1.0])
    if camera_point[2] <= 0:
        return None
    uv, _ = cv2.projectPoints(camera_point[:3].reshape(1, 3), np.zeros(3), np.zeros(3),
                              np.asarray(vision.intrinsics['k'], float),
                              np.asarray(vision.intrinsics.get('dist', [0] * 5), float))
    return uv.reshape(2).tolist()


def plane_footprint(host, transform):
    """Image footprint on the measured plane, not visibility of non-planar surfaces."""
    transform = np.asarray(transform, float)
    origin = np.asarray(host.vision.table['point_mm'], float)
    normal = np.asarray(host.vision.table['normal'], float)
    normal /= np.linalg.norm(normal)
    axis = np.cross(normal, [1, 0, 0] if abs(normal[0]) < .9 else [0, 1, 0])
    axis /= np.linalg.norm(axis)
    other = np.cross(normal, axis)
    width, height = host.image_size
    points = []
    for u, v in [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]:
        ray = pixel_ray(u, v, np.asarray(host.vision.intrinsics['k']), np.asarray(host.vision.intrinsics['dist']))
        point = intersect_ray_plane(transform[:3, 3], transform[:3, :3] @ ray, origin, normal)
        if point is None:
            return None
        delta = point - origin
        points.append([np.dot(delta, axis), np.dot(delta, other)])
    return cv2.convexHull(np.asarray(points, np.float32))


def overlap_fraction(first, second):
    if first is None or second is None:
        return None
    area = min(cv2.contourArea(first), cv2.contourArea(second))
    if area <= 0:
        return None
    intersection, _ = cv2.intersectConvexConvex(first, second)
    return min(1.0, max(0.0, float(intersection) / area))


def rank_views(host, task, views):
    robot = host.robot()
    pose = robot.get('pose6')
    if host.gateway is None or host.gateway.camera is None:
        return {'next': None, 'candidates': [], 'reason': 'Camera body clearance and a motion gateway are required.'}
    if not pose or robot.get('moving') or robot.get('source') != 'telemetry':
        return {'next': None, 'candidates': [], 'reason': 'Stationary controller telemetry is required.'}
    taken = [view['robot_tcp_mm'] for view in views if view.get('accepted') and view.get('robot_tcp_mm')]
    candidates = []
    point = (task.get('location') or {}).get('point_mm')
    for index in range(len(OFFSETS)):
        candidate = {'id': f'parallax-{index}', 'validated': False, 'executable': False}
        goal = view_goal(host, task['location'], candidate['id'], pose[5])
        reasons = []
        if not goal.get('ok'):
            candidate['reason'] = goal.get('reason')
            candidates.append(candidate)
            continue
        target = goal['target_mm']
        uv = projected_pixel(host.vision, goal['pose6'], point)
        width, height = host.image_size
        visible = uv is not None and .1 * width <= uv[0] <= .9 * width and .1 * height <= uv[1] <= .9 * height
        if not visible:
            reasons.append('The calibrated ray predicts the target outside the central image area.')
        if abs(abs(float(pose[3])) - 180) > .1 or abs(float(pose[4])) > .1:
            reasons.append('This Cartesian driver would change the current tool orientation.')
        novelty = min((math.dist(target, xyz) for xyz in taken), default=1e6)
        if novelty < 10:
            reasons.append('A capture already covers this viewpoint (within 10 mm).')
        overlap = None
        accepted = [view for view in views if view.get('accepted')]
        if accepted:
            footprint = plane_footprint(host, camera_pose_matrix(goal['pose6'], host.vision.hand_eye['matrix']))
            overlaps = [overlap_fraction(footprint, plane_footprint(host, view['camera_pose']))
                        for view in accepted if view.get('camera_pose')]
            known = [value for value in overlaps if value is not None]
            overlap = max(known) if known else None
            if overlap is None or overlap < .5:
                reasons.append('Insufficient measured-plane image overlap with accepted views (need 50%).')
        speed = min(50, host.gateway.max_speed_mm_s)
        decision = host.gateway.authorize_cartesian(target, speed, source='scan_viewpoint', start=pose[:3])
        if not decision.ok:
            reasons.append(decision.reason)
        radius = float(task['location'].get('uncertainty_mm') or 0)
        if not math.isfinite(radius) or radius < 0:
            reasons.append('Localization uncertainty is invalid.')
        else:
            for dx, dy in ((radius, 0), (-radius, 0), (0, radius), (0, -radius)):
                edge = [target[0] + dx, target[1] + dy, target[2]]
                check = host.gateway.authorize_cartesian(edge, speed, source='scan_uncertainty', start=pose[:3])
                if not check.ok:
                    reasons.append(check.reason)
                    break
        inverse = host.inverse(goal['pose6'], robot.get('joints'))
        if not inverse.get('ik_verified'):
            reasons.append(inverse.get('reason') or 'Controller IK has not validated this view.')
        travel = math.dist(pose[:3], target)
        candidate.update(target_mm=target, pose6=goal['pose6'], predicted_pixel=uv,
                         object_visibility_predicted=visible, ik_verified=bool(inverse.get('ik_verified')),
                         plane_overlap_fraction=overlap, overlap_is_not_object_surface_coverage=True,
                         trajectory_mm=sample_segment(pose[:3], target),
                         novelty_mm=round(min(novelty, 1000), 2), travel_mm=round(travel, 2),
                         score=round(min(novelty, 100) - .1 * travel, 2),
                         validated=not reasons, executable=not reasons and decision.allow_motion,
                         reason='; '.join(dict.fromkeys(reasons)) or 'Configured path and IK checks passed.')
        candidates.append(candidate)
    candidates.sort(key=lambda item: (item['validated'], item.get('score', -1e6)), reverse=True)
    best = next((item for item in candidates if item['validated']), None)
    return {'next': best, 'candidates': candidates, 'coverage_kind': 'viewpoint_diversity_only',
            'surface_coverage_known': False, 'collision_avoidance': False,
            'requires_confirmation': True, 'reconstructed': False,
            'human_assistance_requested': best is None,
            'reason': 'A useful viewpoint is available.' if best else
                      'No additional validated viewpoint is available. Review the blockers; a human may need to reorient the object.'}
