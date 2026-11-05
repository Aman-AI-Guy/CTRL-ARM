import json
from pathlib import Path

import numpy as np
import pytest

from trova.geom import VisionStore, tool_pose_matrix
from trova.scan3d.flange import FlangeCalibration, flange_pose
from trova.scan3d.evidence import workspace, calibration_report, localize_recorded, flange_to_tool_camera
from trova.scan3d.marker_fit import fit_plane


def archive():
    try:
        return workspace()
    except FileNotFoundError:
        pytest.skip('Optional retained experiment archive is absent')


def test_flange_is_not_tcp_and_rotation_conventions():
    flange = flange_pose([0] * 5)
    np.testing.assert_allclose(flange[:3, 3], [207, 0, 112], atol=1e-8)
    np.testing.assert_allclose(flange[:3, 2], [0, 0, -1], atol=1e-8)
    camera = np.eye(4)
    camera[:3, 3] = [12, 20, 35]
    offset = [0, 0, 100, 0, 0, 15]
    tool_camera = flange_to_tool_camera(camera, offset)
    np.testing.assert_allclose(flange @ tool_pose_matrix(offset) @ tool_camera, flange @ camera, atol=1e-8)


def test_historical_calibration_has_no_activation_side_effect():
    path = archive() / 'calib/camera_robot.json'
    before = path.read_bytes()
    store = VisionStore()
    report = store.historical_compatibility([1640, 1232])
    assert report['image_size'] == [1600, 900]
    assert not report['valid_for_motion'] and not report['active']
    assert any('profile' in reason for reason in report['reasons'])
    assert any('1.5 px' in reason for reason in report['reasons'])
    assert not store.assess([1640, 1232])['valid'] and store.hand_eye is None
    assert path.read_bytes() == before


def test_recorded_pixel_roundtrip_and_distortion():
    cal = FlangeCalibration.load(archive() / 'calib/camera_robot.json')
    q = [-93.544069, -.240929, -43.002889, -43.494544, 48.566366]
    for uv in [(834, 430), (700, 500), (900, 390)]:
        point = localize_recorded(*uv, q, [1600, 900], height_mm=35)
        assert point['provenance'] == 'recorded' and not point['valid_for_motion']
        np.testing.assert_allclose(cal.robot_to_pixel(point['point_mm'], q), uv, atol=.001)
    with pytest.raises(ValueError, match='resolution'):
        localize_recorded(834, 430, q, [1640, 1232])
    with pytest.raises(ValueError, match='aspect ratio'):
        cal.for_size(1640, 1232)


def test_tilted_plane_math_is_reused():
    points = [[x, y, .01*x-.02*y-40] for x, y in [(0,0),(200,0),(0,200),(200,200)]]
    plane = fit_plane(points)
    np.testing.assert_allclose([plane['a'], plane['b'], plane['c']], [.01,-.02,-40], atol=1e-8)


def test_invalid_offset_is_not_treated_as_zero():
    with pytest.raises(ValueError, match='six-value'):
        flange_to_tool_camera(np.eye(4), [0,0,0])
