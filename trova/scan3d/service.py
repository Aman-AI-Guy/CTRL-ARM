"""Trova adapter: current calibrated target -> imported five-axis orbit proposals.

Settings belong to the operator. Neither Grok nor the planner can activate calibration,
unlock hardware or replace a target with model-generated coordinates.
"""
from dataclasses import asdict
import json
from pathlib import Path
import math
import numpy as np
from .orbit_planner import OrbitSettings, JointGeometry, plan_orbit
from .flange import flange_pose


class OrbitStore:
    def __init__(self, path):
        self.path = Path(path)
        self.settings = OrbitSettings()
        if self.path.is_file():
            self.settings = OrbitSettings(**json.loads(self.path.read_text()))
            self.settings.validate()

    def public(self):
        return asdict(self.settings)

    def configure(self, values, confirmed):
        if not confirmed:
            raise ValueError('Confirm measured orbit/model settings as the operator.')
        candidate = OrbitSettings(**values)
        candidate.validate()
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.path.write_text(json.dumps(asdict(candidate),indent=2))
        self.settings = candidate
        return self.public()

    def context(self, host, task):
        if not host.vision.assess(host.image_size)['valid']:
            raise ValueError('Active camera calibration is missing or invalid; historical calibration is not active.')
        if not (task.get('location') or {}).get('ok'):
            raise ValueError('A currently localized selected object is required.')
        robot = host.robot()
        if robot.get('source') != 'telemetry' or robot.get('moving') or robot.get('error_code') or robot.get('error'):
            raise ValueError('Stationary fault-free robot telemetry is required.')
        if not host.gateway or host.gateway.camera is None:
            raise ValueError('Configure the camera body and MotionGateway first.')
        self.settings.validate()
        expected = (flange_pose(robot['joints']) @ __import__('trova.geom',fromlist=['tool_pose_matrix']).tool_pose_matrix(self.settings.tcp_offset_mm))[:3,3]
        if np.linalg.norm(expected - np.asarray(robot['pose6'][:3])) > 2:
            raise ValueError('The measured TCP offset/model disagrees with live controller telemetry by more than 2 mm.')
        if task['location'].get('reference_height_mm'):
            # JointGeometry expects a support point, not the measured top point.
            reference_point = np.asarray(task['location']['point_mm'],float).copy()
            reference_point[2] -= task['location']['reference_height_mm']
        else:
            reference_point = task['location']['point_mm']
        expected_pose = flange_pose(robot['joints']) @ __import__('trova.geom',fromlist=['tool_pose_matrix']).tool_pose_matrix(self.settings.tcp_offset_mm)
        actual_pose = __import__('trova.geom',fromlist=['tool_pose_matrix']).tool_pose_matrix(robot['pose6'])
        angular_error = np.degrees(np.arccos(np.clip((np.trace(expected_pose[:3,:3].T @ actual_pose[:3,:3])-1)/2,-1,1)))
        if angular_error > 2:
            raise ValueError('Controller/base/tool orientation differs from imported flange FK by more than 2 degrees.')
        context = JointGeometry(host.vision.hand_eye['matrix'],host.vision.table,
                                reference_point,self.settings,host.gateway,
                                qualified=self.settings.geometry_confirmed)
        if hasattr(host.vision,'configuration_compatible'):
            context.calibration_domain = host.vision.configuration_compatible
        uncertainty = float(task['location'].get('uncertainty_mm',float('inf')))
        if not math.isfinite(uncertainty) or uncertainty < 0:
            raise ValueError('Localization uncertainty is invalid.')
        # Inflate object keep-clear cylinder instead of treating an uncertain point as exact.
        context.scene.obj_radius += uncertainty
        anchor=task.get('orbit_anchor_joints')
        if anchor is not None:
            if len(anchor)!=5 or not np.isfinite(anchor).all():raise ValueError('Invalid scan cable reference.')
            context.cable_reference_deg=float(anchor[4])
        return context

    def approach(self, host, task):
        from .approach import propose
        try:
            context = self.context(host,task)
            report = propose(context,host.robot()['joints'],host.observe_clearance_mm,cancel=host.cancel)
            if report.get('next'):
                report['next']['operation'] = 'approach'
            return report
        except (ValueError,TypeError,KeyError) as exc:
            return {'next':None,'candidates':[],'reason':str(exc),'collision_avoidance':False}

    def plan(self, host, task, views=()):
        from ..scanning import plane_footprint, overlap_fraction
        try:
            context = self.context(host,task)
            robot = host.robot()
            result = plan_orbit(robot['joints'],context,cancel=host.cancel)
            captured = [v for v in views if v.get('accepted') and v.get('camera_pose')]
            for candidate in result['candidates']:
                if not candidate.get('validated'): continue
                transform = flange_pose(candidate['joints_deg']) @ context.flange_camera
                axis = transform[:3,2]
                novelty = min((math.degrees(math.acos(np.clip(axis @ np.asarray(v['camera_pose'])[:3,2],-1,1))) for v in captured),default=180)
                footprint = plane_footprint(host,transform)
                overlaps = [overlap_fraction(footprint,plane_footprint(host,v['camera_pose'])) for v in captured]
                overlaps = [value for value in overlaps if value is not None]
                overlap = max(overlaps) if overlaps else None
                if captured and (novelty < 5 or overlap is None or overlap < .5):
                    candidate.update(validated=False,executable=False,reason='Insufficient new angle or measured-plane overlap (need 50%).')
                candidate.update(novel_angle_deg=novelty,plane_overlap_fraction=overlap,
                                 camera_pose=transform.tolist(),score=novelty-.1*candidate.get('travel_deg',0))
            result['candidates'].sort(key=lambda c:(bool(c.get('validated')),c.get('score',-1e6)),reverse=True)
            result['next'] = next((c for c in result['candidates'] if c.get('validated')),None)
            result.update(surface_coverage_known=False, coverage_kind='viewpoint_angle_and_measured_plane_overlap',
                          expected_new_surface_area=None, pose_uncertainty_mm=task['location'].get('uncertainty_mm'),
                          reason='Joint orbit proposal available.' if result['next'] else 'No further validated view. Review geometry and overlap blockers; reorientation may be needed.')
            return result
        except (ValueError,KeyError,TypeError) as exc:
            return {'next':None,'candidates':[],'reason':str(exc),'collision_avoidance':False,'surface_coverage_known':False}
