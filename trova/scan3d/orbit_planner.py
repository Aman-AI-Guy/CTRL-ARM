"""Imported xArm 5 camera IK/geometry adapted to Trova's single motion gateway.

Pure proposals only. No SDK imports, controller connection, enable or actuator call.
The imported sphere model is sampled along joint-linear paths; continuous collision
avoidance, a complete self-pair set and controller trajectory equivalence are not claimed.
"""
from dataclasses import dataclass, asdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import orbit_math as orbit
from .robot_geometry import RobotGeometry, MOVING
from .flange import flange_pose
from ..geom import tool_pose_matrix
from ..motion import sample_joints

MODEL = Path(__file__).with_name('robot_model.json')


@dataclass
class OrbitSettings:
    distance_mm: float = 350.0
    elevation_deg: float = 50.0
    span_deg: float = 220.0
    step_deg: float = 20.0
    start_azimuth_deg: float = 176.0
    direction: int = 1
    object_height_mm: float | None = None
    object_radius_mm: float | None = None
    tcp_offset_mm: list | None = None
    joint_speed_deg_s: float = 10.0
    joint_accel_deg_s2: float = 30.0
    max_joint_step_deg: float = 45.0
    cable_span_deg: float = 180.0
    table_margin_mm: float = 15.0
    object_margin_mm: float = 20.0
    self_margin_mm: float = 5.0
    geometry_confirmed: bool = False

    def validate(self):
        for key, lo, hi in [('distance_mm', 100, 700), ('elevation_deg', 15, 85),
                            ('span_deg', 0, 360), ('step_deg', 10, 90),
                            ('joint_speed_deg_s', .1, 30), ('joint_accel_deg_s2', .1, 60),
                            ('max_joint_step_deg', .1, 45), ('cable_span_deg', 1, 360)]:
            value = float(getattr(self, key))
            if not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f'{key} must be finite in [{lo}, {hi}]')
        for key in ['object_height_mm', 'object_radius_mm']:
            value = getattr(self, key)
            if value is None or not math.isfinite(float(value)) or not 0 < float(value) <= 500:
                raise ValueError(f'Measure {key} before planning an object orbit.')
        if self.tcp_offset_mm is None or np.asarray(self.tcp_offset_mm).shape != (6,) or not np.isfinite(self.tcp_offset_mm).all():
            raise ValueError('The current measured six-value flange-to-TCP offset is required.')
        if self.direction not in (-1, 1):
            raise ValueError('Orbit direction must be -1 or 1.')
        if not math.isfinite(self.start_azimuth_deg):
            raise ValueError('Start azimuth must be finite.')
        for key, minimum in [('table_margin_mm',15), ('object_margin_mm',20), ('self_margin_mm',5)]:
            if not math.isfinite(getattr(self,key)) or getattr(self,key) < minimum:
                raise ValueError(f'{key} cannot reduce the imported conservative margin below {minimum} mm.')


class JointGeometry:
    def __init__(self, tool_camera, table, target_foot, settings, gateway, *, qualified=False):
        settings.validate()
        self.settings, self.gateway = settings, gateway
        self.geometry = RobotGeometry.load(MODEL)
        self.tcp = tool_pose_matrix(settings.tcp_offset_mm)
        self.flange_camera = self.tcp @ np.asarray(tool_camera, float)
        if self.flange_camera.shape != (4,4) or not np.isfinite(self.flange_camera).all():
            raise ValueError('Invalid active camera mount transform.')
        self.cam = orbit.Camera(self.flange_camera[:3,3], self.flange_camera[:3,2])
        if np.hypot(*self.cam.axis[:2]) < .5:
            raise ValueError('The xArm 5 orbit requires the calibrated sideways camera mount, not an axial camera assumption.')
        self.turn = self.geometry.head_turn(self.cam.axis)
        self.foot = np.asarray(target_foot, float)
        if self.foot.shape != (3,) or not np.isfinite(self.foot).all():
            raise ValueError('A valid calibrated object foot is required.')
        self.target = self.foot + np.array([0,0,settings.object_height_mm / 2])
        normal = np.asarray(table['normal'], float)
        if abs(normal[2]) < .5:
            raise ValueError('The table plane is too steep for the imported collision model.')
        origin = np.asarray(table['point_mm'], float)
        # The original collision model is horizontal. Use the maximum plane height
        # over the TCP workspace, never flatten a slope downwards.
        limits = gateway.limits
        xy = [(self.foot[0],self.foot[1])]
        if limits:
            xy += [(x,y) for x in limits.x for y in limits.y]
        heights = [origin[2] - (normal[0]*(x-origin[0])+normal[1]*(y-origin[1])) / normal[2] for x,y in xy]
        floor = max(heights)
        self.scene = orbit.Scene(self.foot[:2], settings.object_radius_mm, self.foot[2]+settings.object_height_mm,
                                 floor, table_margin=settings.table_margin_mm, obj_margin=settings.object_margin_mm,
                                 geometry=self.geometry, head_turn=self.turn, self_margin=settings.self_margin_mm)
        self.qualified = bool(qualified and settings.geometry_confirmed)
        self.model_sha256 = hashlib.sha256(MODEL.read_bytes()).hexdigest()

    def signature(self):
        return {'model': self.model_sha256, 'settings': asdict(self.settings),
                'target_mm': self.target.tolist(), 'table_z_mm': self.scene.table_z,
                'T_flange_camera': self.flange_camera.tolist(), 'qualified': self.qualified}

    def check(self, joints):
        q = np.asarray(joints, float)
        if q.shape != (5,) or not np.isfinite(q).all():
            return ['invalid five-joint pose']
        hits = orbit.collisions(q, self.cam, self.scene)
        tcp=(flange_pose(q) @ self.tcp)[:3,3]
        if self.gateway.limits and self.gateway.limits.check(*tcp):
            hits.append('TCP workspace')
        if any(box.contains(tcp) for box in self.gateway.keepouts):
            hits.append('TCP keep-out')
        shapes = self.geometry.world(orbit.fk_frames(q), self.turn)
        # Keep-outs cover every supplied link, including static/base shapes.
        for name, (centers, radii) in shapes.items():
            for box in self.gateway.keepouts:
                lo, hi = np.array([box.x[0],box.y[0],box.z[0]]), np.array([box.x[1],box.y[1],box.z[1]])
                if (orbit.box_gap(centers, radii, lo, hi) < self.settings.object_margin_mm).any():
                    hits.append(f'keep-out / {name}')
        if self.gateway.limits:
            centers,radii = shapes['head']
            for center,radius in zip(centers,radii):
                if self.gateway.limits.radius_mm is not None and math.hypot(*center[:2])+radius > self.gateway.limits.radius_mm:
                    hits.append('camera head cylindrical workspace')
                if self.gateway.limits.check(*center):
                    hits.append('camera head workspace')
                for axis,(lo,hi) in enumerate((self.gateway.limits.x,self.gateway.limits.y,self.gateway.limits.z)):
                    if not lo + radius <= center[axis] <= hi - radius:
                        hits.append('camera head workspace')
        return sorted(set(hits))

    def path(self, start, target):
        q0,q1 = np.asarray(start,float),np.asarray(target,float)
        hits=set()
        for q in sample_joints(q0,q1):
            hits.update(self.check(q))
        return sorted(hits)


def plan_orbit(joints, context, *, start_azimuth=None, sequential=False, cancel=None):
    """Reuse original branch/IK candidate generation; reject excessive jumps/cable travel.

    Live candidates are checked from the actual starting pose independently. Offline
    sequential replay additionally validates the retained ordering between viewpoints.
    """
    q0=np.asarray(joints,float)
    if q0.shape != (5,) or not np.isfinite(q0).all():
        raise ValueError('Five finite stationary telemetry joints are required.')
    settings=context.settings
    start=settings.start_azimuth_deg if start_azimuth is None else start_azimuth
    n=int(round(settings.span_deg/settings.step_deg))+1
    if settings.span_deg == 360:
        n-=1
    candidates=[]
    previous=q0.copy()
    for index in range(n):
        if cancel is not None and cancel.is_set():
            break
        az=start+settings.direction*index*settings.step_deg
        solutions=orbit.solve_one(context.cam,context.scene,context.target,az,settings.elevation_deg,
                                  settings.distance_mm,previous if sequential else q0,q0)
        chosen=None; reasons=[]
        for q in solutions:
            begin=previous if sequential else q0
            jump=float(np.max(np.abs(q-begin)))
            if jump > settings.max_joint_step_deg:
                reasons.append('joint jump exceeds the configured 45-degree-or-smaller limit'); continue
            if abs(q[4]-q0[4]) > settings.cable_span_deg:
                reasons.append('camera cable / J5 travel exceeds its configured budget'); continue
            hits=context.path(begin,q)
            if hits:
                reasons.extend(hits); continue
            target_tcp=(flange_pose(q) @ context.tcp)[:3,3]
            if context.gateway.limits and context.gateway.limits.check(*target_tcp):
                reasons.append('TCP workspace'); continue
            chosen=q; break
        item={'id':f'orbit-{index}', 'azimuth_deg':az%360, 'elevation_deg':settings.elevation_deg,
              'validated':chosen is not None, 'reason':'; '.join(sorted(set(reasons))) or 'unreachable / collision'}
        if chosen is not None:
            begin=previous if sequential else q0
            camera,direction=context.cam.ray(chosen)
            tcp=(flange_pose(chosen) @ context.tcp)[:3,3]
            item.update(joints_deg=chosen.tolist(), target_mm=tcp.tolist(), camera_mm=camera.tolist(),
                        camera_direction=direction.tolist(), joint_path=[list(map(float,q)) for q in sample_joints(begin,chosen)],
                        travel_deg=float(np.max(np.abs(chosen-begin))), reason='IK and sampled CAD-derived sphere path passed.')
            expected=[(flange_pose(q) @ context.tcp)[:3,3] for q in item['joint_path']]
            ratios=[math.dist(a,b)/max(1e-9,max(abs(x-y) for x,y in zip(qa,qb)))
                    for a,b,qa,qb in zip(expected,expected[1:],item['joint_path'],item['joint_path'][1:])]
            speed=min(settings.joint_speed_deg_s,context.gateway.max_joint_speed_deg_s)
            if ratios and max(ratios)>0:
                speed=min(speed,context.gateway.max_speed_mm_s/max(ratios)*.99)
            item['joint_speed_deg_s']=speed
            decision=context.gateway.authorize_joints(chosen,speed,source='scan_orbit',geometry=context,start_q=begin)
            item['gateway']=decision.as_dict()
            item['validated']=decision.ok
            item['executable']=decision.allow_motion
            if not decision.ok: item['reason']=decision.reason
            if sequential and decision.ok: previous=chosen
        candidates.append(item)
    return {'candidates':candidates, 'reachable':sum(c['validated'] for c in candidates),
            'target_mm':context.target.tolist(), 'provenance':'live_proposal' if context.qualified else 'offline_proposal',
            'geometry':context.signature(), 'sample_step_deg':2, 'collision_avoidance':False,
            'continuous_path_checked':False, 'full_self_collision_pairs':False,
            'reason':'No motion was sent. Every chosen view must still pass MotionGateway and proposal confirmation.'}


def recorded_context(root=None):
    from .evidence import workspace
    from ..arm import Limits
    from ..motion import MotionGateway
    retained=json.loads((workspace(root)/'arm_pc/runs/20261105-021400_plan.json').read_text())
    settings=OrbitSettings(object_height_mm=retained['args']['obj_height'],
                           object_radius_mm=retained['args']['obj_radius'],tcp_offset_mm=[0]*6,
                           start_azimuth_deg=retained['start_az'])
    axis=np.asarray(retained['camera_axis_flange'])
    x=np.cross([0,0,1],axis);x/=np.linalg.norm(x)
    transform=np.eye(4)
    transform[:3,:3]=np.column_stack([x,np.cross(axis,x),axis])
    transform[:3,3]=retained['camera_offset_flange']
    gateway=MotionGateway(None,mode='dry_run',simulated=True,
                          limits=Limits.parse('-1000,1000,-1000,1000,-100,1200'),jog_max_mm=100,
                          max_speed_mm_s=20,max_accel_mm_s2=100,max_joint_speed_deg_s=20,
                          joint_limits=tuple(map(tuple,orbit.LIMITS)),keepouts=[],camera=None)
    foot=np.asarray(retained['target_mm'])-[0,0,settings.object_height_mm/2]
    context=JointGeometry(transform,{'normal':[0,0,1],'point_mm':[0,0,retained['table_z_mm']]},
                          foot,settings,gateway,qualified=False)
    return retained,context
