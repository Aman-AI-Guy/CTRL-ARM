import ast
import json
from pathlib import Path
import importlib.util

import numpy as np
import pytest

from trova.scan3d.evidence import workspace
from trova.scan3d.orbit_planner import recorded_context, plan_orbit, JointGeometry
from trova.scan3d.robot_geometry import RobotGeometry
from trova.scan3d.flange import flange_pose
from trova.motion import KeepOut


def recorded():
    try:
        return recorded_context()
    except FileNotFoundError:
        pytest.skip('Optional retained workspace is absent')


def test_imported_planner_has_no_controller_or_actuator_code():
    root=Path(__file__).resolve().parents[1]/'trova/scan3d'
    for name in ['orbit_math.py','orbit_planner.py','robot_geometry.py','flange.py']:
        code=(root/name).read_text()
        assert 'XArmAPI' not in code and 'set_servo_angle' not in code and 'motion_enable' not in code


def test_spheres_cover_retained_mesh_vertices_in_the_correct_units():
    root=workspace()
    file=root/'arm_pc/model/build_robot_model.py'
    spec=importlib.util.spec_from_file_location('m5_geometry_benchmark',file)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    geometry=RobotGeometry.load(Path(__file__).resolve().parents[1]/'trova/scan3d/robot_model.json')
    for name in geometry.data['links']:
        triangles=module.load_stl(root/'arm_pc/model/xarm5_meshes'/f'{name}.stl')*1000
        centers,radii=geometry.shapes[name]
        vertices=np.unique(np.round(triangles.reshape(-1,3),1),axis=0)
        gaps=np.linalg.norm(vertices[:,None,:]-centers[None,:,:],axis=2)-radii
        assert gaps.min(axis=1).max() < .6,name
    vertices=module.cad_to_head(module.load_stl(root/'arm_pc/model/camera_head.stl')).reshape(-1,3)
    centers,radii=geometry.shapes['head']
    assert (np.linalg.norm(vertices[:,None,:]-centers[None,:,:],axis=2)-radii).min(axis=1).max()<.6


def test_retained_orbit_replans_through_the_gateway_without_motion():
    original,context=recorded()
    result=plan_orbit(original['q0'],context,sequential=True)
    assert result['reachable'] >= 8,[(v['id'],v['reason']) for v in result['candidates']]
    assert not result['collision_avoidance'] and not result['continuous_path_checked']
    for candidate in result['candidates']:
        if not candidate['validated']:continue
        assert candidate['gateway']['ok'] and not candidate['gateway']['allow_motion']
        assert len(candidate['joint_path'])>1
        for q in candidate['joint_path']:
            assert not context.check(q)
        position,direction=context.cam.ray(candidate['joints_deg'])
        desired=context.target-position;desired/=np.linalg.norm(desired)
        assert direction@desired > .999


def test_gateway_enforces_link_keepout_and_joint_boundaries():
    original,context=recorded()
    q=original['q0']
    w=context.geometry.world(__import__('trova.scan3d.orbit_math',fromlist=['fk_frames']).fk_frames(q),context.turn)
    center=w['link3'][0][0]
    context.gateway.keepouts=[KeepOut((center[0]-5,center[0]+5),(center[1]-5,center[1]+5),(center[2]-5,center[2]+5))]
    decision=context.gateway.authorize_joints(q,1,source='scan_orbit',geometry=context,start_q=q)
    assert not decision.ok and 'geometry' in decision.reason.lower()
    context.gateway.keepouts=[]
    target=list(q);target[2]=60
    assert not context.gateway.authorize_joints(target,1,source='scan_orbit',geometry=context,start_q=q).ok
    assert not context.gateway.authorize_joints([float('nan')]*5,1,geometry=context,start_q=q).ok


def test_head_self_collision_is_not_a_tcp_only_check():
    _original,context=recorded()
    q=[0,-20,-60,180,0]
    assert any('self' in reason for reason in context.check(q))
