"""Read-only discovery and validation of retained physical experiments.

Historical robot-base, SfM, and object coordinates are kept separate. No archive is
ever an active calibration and no method in this module opens a robot connection.
"""

import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np

from .flange import FlangeCalibration, flange_pose
from . import marker_fit


def workspace(path=None):
    if path or os.getenv('TROVA_3D_WORKSPACE'):
        root = Path(path or os.environ['TROVA_3D_WORKSPACE']).resolve()
        if (root / 'calibration/calib/pix2robot.py').is_file():
            root /= 'calibration'
    else:
        repository = Path(__file__).resolve().parents[2]
        candidates = sorted(repository.glob('*/calibration/calib/pix2robot.py'))
        if len(candidates) != 1:
            raise FileNotFoundError('Retained 3D workspace not found unambiguously; set TROVA_3D_WORKSPACE.')
        root = candidates[0].parents[1]
    if not (root / 'calib/pix2robot.py').is_file():
        raise FileNotFoundError('This is not the retained calibration workspace.')
    return root


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def calibration_report(image_size=None, rotation=180, root=None):
    root = workspace(root)
    path = root / 'calib/camera_robot.json'
    data = json.loads(path.read_text())
    reasons = ['Historical mounting/camera identity and the current flange-to-TCP offset are not verified.',
               'Current independent physical reference measurements are missing.']
    if image_size is not None and list(image_size) != data['image_size']:
        reasons.append(f"Camera profile {list(image_size)} differs from historical {data['image_size']}; no automatic crop/scale conversion.")
    if rotation != 180:
        reasons.append('The retained images used the 180-degree server rotation; extra rotation is incompatible.')
    rms = data['quality']['reprojection_rms_px']
    if rms > 1.5:
        reasons.append(f'Historical reprojection RMS {rms:g} px exceeds the existing 1.5 px active-calibration gate.')
    return {'available': True, 'active': False, 'valid_for_motion': False, 'provenance': 'historical_recording',
            'image_size': data['image_size'], 'rotation_deg': 180, 'distortion': 'OpenCV k1,k2,p1,p2,k3',
            'units': 'mm', 'frame': 'robot_base', 'pose_relation': 'T_base_camera = FK_flange(joints_deg) @ T_flange_camera',
            'tool_pose_is_not_flange_pose': True, 'quality_reported': data['quality'],
            'table_z_mm': data['table_z_mm'], 'sha256': sha256(path), 'reasons': reasons}


def flange_to_tool_camera(transform, tcp_offset):
    """Convert an explicitly known flange->TCP transform; never assume a zero TCP."""
    from ..geom import tool_pose_matrix
    tcp = np.asarray(tcp_offset, float)
    if tcp.shape != (6,) or not np.isfinite(tcp).all():
        raise ValueError('A measured six-value flange-to-TCP offset is required.')
    matrix = np.asarray(transform, float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError('Invalid flange-camera transform.')
    return np.linalg.inv(tool_pose_matrix(tcp)) @ matrix


def localize_recorded(u, v, joints, image_size, height_mm=0, root=None):
    root = workspace(root)
    cal = FlangeCalibration.load(root / 'calib/camera_robot.json')
    q = np.asarray(joints, float)
    if q.shape != (5,) or not np.isfinite(q).all():
        raise ValueError('Five finite recorded joint angles in degrees are required.')
    if list(image_size) != [cal.width, cal.height]:
        raise ValueError('Recorded image resolution does not match calibration; no automatic rescaling.')
    point = cal.pixel_to_object_top(float(u), float(v), q, float(height_mm))
    return {'point_mm': point.tolist(), 'camera_pose': cal.camera_pose(q).tolist(),
            'frame': 'historical_robot_base', 'units': 'mm', 'provenance': 'recorded', 'active': False,
            'valid_for_motion': False, 'height_source': 'operator_or_recorded_measurement',
            'uncertainty_mm': cal.data['quality']['holdout_p95_mm'],
            'uncertainty_kind': 'historical_heldout_p95_not_current_guarantee',
            'calibration_sha256': sha256(root / 'calib/camera_robot.json')}


def calibration_replay(root=None):
    root = workspace(root)
    stored = json.loads((root / 'calib/camera_robot.json').read_text())
    cfg = {'aruco_scan': {'dictionary': 'DICT_ARUCO_MIP_36h12', 'ids': [2, 3, 4], 'marker_mm': 39.0}}
    images, obs, size, side = marker_fit.load_snaps(root / 'arm_pc/runs/20261105-023325_snaps', cfg)
    prob = marker_fit.Problem([flange_pose(row['q']) for row in images], obs, side, size,
                             (size[0] / 2) / np.tan(np.radians(62.2 / 2)))
    camera = np.asarray(stored['T_flange_camera'])
    x = [stored['K'][0][0], stored['K'][0][2], stored['K'][1][2], *stored['dist'][:2], *marker_fit.rt_of(camera)]
    for key in prob.mids:
        corners = np.asarray(stored['markers'][str(key)]['corners_mm'])
        center = corners.mean(0)
        rotation = (corners - center).T @ prob.obj
        u, _, vt = np.linalg.svd(rotation)
        rotation = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
        matrix = np.eye(4)
        matrix[:3, :3], matrix[:3, 3] = rotation, center
        x.extend(marker_fit.rt_of(matrix))
    x = np.asarray(x)
    # Same outlier policy as solve_all, preserving the recorded dropped observation.
    dropped = {(row[0], int(row[1])) for row in stored['source'].get('dropped_observations', [])}
    prob.obs = [row for row in obs if (images[row[0]]['file'], row[1]) not in dropped]
    x = marker_fit.lm(prob, x, iters=60)
    residuals = np.concatenate([np.sum(r ** 2, axis=1) for r in prob.project(x)])
    geometry = marker_fit.marker_geometry(prob, x)
    heldout = []
    for part in (0, 1):
        fit = [row for row in prob.obs if row[0] % 2 == part]
        test = [row for row in prob.obs if row[0] % 2 != part]
        fitted = marker_fit.Problem(prob.Tbf, fit, side, size, prob.f0)
        fitted.mids = prob.mids
        xf = marker_fit.lm(fitted, x, iters=30)
        heldout.extend(marker_fit.backproject_errors(fitted, xf, prob.Tbf, test, geometry))
    return {'provenance': 'recorded_recomputed', 'images': len(images), 'observations': len(prob.obs),
            'reprojection_rms_px': float(np.sqrt(np.mean(residuals))),
            'holdout_median_mm': float(np.median(heldout)), 'holdout_p95_mm': float(np.percentile(heldout, 95)),
            'reported': stored['quality'], 'valid_for_motion': False,
            'evaluation_limit': 'Original half-split procedure initializes from the full fit and uses full-fit marker planes; this is not independent ruler validation.'}


def catalog(root=None):
    root = workspace(root)
    datasets = []
    for capture in sorted((root / 'data/scans').glob('*/capture.json')):
        folder = capture.parent
        images = [p for p in (folder / 'images').glob('*.jpg') if p.is_file()]
        datasets.append({'id': folder.name, 'images': len(images), 'provenance': 'recorded',
                         'capture_sha256': sha256(capture)})
    return {'workspace': str(root), 'datasets': datasets, 'calibration': calibration_report(root=root),
            'originals_read_only': True, 'historical_results_are_not_integration_measurements': True}
