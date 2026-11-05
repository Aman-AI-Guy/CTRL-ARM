"""Offline scanning and reconstruction contracts. No robot or external COLMAP required."""

import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import pytest

from trova.acquire import ScanBook
from trova.reconstruction import ReconstructionJobs, prepare_dataset
from trova.scanning import rank_views


def jpeg(seed=0):
    frame = np.random.default_rng(seed).integers(20, 235, (64, 80, 3), dtype=np.uint8)
    return cv2.imencode('.jpg', frame)[1].tobytes()


def meta(x=0):
    return {'width': 80, 'height': 64, 'pose6': [x, 0, 150, 180, 0, 0],
            'robot_tcp_mm': [x, 0, 150], 'joints_deg': [0] * 5,
            'pose_source': 'telemetry', 'moving': False, 'error_code': 0,
            'requested_at': 20, 'stopped_at': 19, 'received_at': 21, 'read_finished_at': 21.1,
            'tool_camera': np.eye(4).tolist(), 'calibration_version': 1,
            'calibration_valid': True, 'quality_required': True, 'object_visible': True}


def dataset(tmp_path):
    book = ScanBook(tmp_path / 'scan')
    started = book.start('7')
    session = book._session(started['id'])
    session['intrinsics'] = {'k': [[100, 0, 40], [0, 100, 32], [0, 0, 1]], 'dist': [0] * 5,
                             'width': 80, 'height': 64}
    for index in range(3):
        view = book.capture(jpeg(index), meta(index * 20), {'track_id': 7, 'box': [5, 5, 75, 59]})
        assert view['accepted']
    return book, session


def test_quality_rejects_blur_identity_loss_and_stale_geometry(tmp_path):
    book = ScanBook(tmp_path)
    book.start('pen')
    blank = cv2.imencode('.jpg', np.zeros((64, 80, 3), np.uint8))[1].tobytes()
    blurry = book.capture(blank, meta(), {'box': [0, 0, 80, 64]})
    assert not blurry['accepted'] and blurry['camera_pose'] is None
    assert 'blurry' in blurry['reject_reason']
    missing = book.capture(jpeg(), {**meta(), 'object_visible': False})
    assert not missing['accepted'] and 'selected object' in missing['reject_reason']
    invalid = book.capture(jpeg(), {**meta(), 'calibration_valid': False})
    assert not invalid['accepted'] and invalid['camera_pose'] is None
    mismatched = book.capture(jpeg(), {**meta(), 'width': 90})
    assert not mismatched['accepted'] and 'size' in mismatched['reject_reason']
    restored = ScanBook(tmp_path)
    assert len(restored.list_views()) == 4
    assert Path(blurry['image']).exists()  # Rejections preserve the original artifact.
    bad_pose = book.capture(jpeg(), {**meta(), 'pose6': ['bad'] * 6})
    assert not bad_pose['accepted'] and bad_pose['camera_pose'] is None


def test_view_ranking_checks_visibility_ik_gateway_and_coverage():
    from test_agent import _Host, _arm, _task
    from test_geom import _fixture_store
    host = _Host(_arm(), _fixture_store())
    host.inverse = lambda *args: {'ik_verified': True, 'reason': 'fake controller'}
    task = _task()
    before = list(host.arm.tcp_mm())
    plan = rank_views(host, task, [])
    assert plan['next'] and plan['next']['validated']
    assert plan['surface_coverage_known'] is False
    assert all(v['ik_verified'] for v in plan['candidates'])
    assert list(host.arm.tcp_mm()) == before
    views = [{'accepted': True, 'robot_tcp_mm': c['target_mm']} for c in plan['candidates']]
    exhausted = rank_views(host, task, views)
    assert exhausted['next'] is None and exhausted['human_assistance_requested']
    host.inverse = lambda *args: {'ik_verified': False, 'reason': 'unreachable'}
    assert rank_views(host, task, [])['next'] is None


def test_scan_speech_plans_without_execution_and_flip_invalidates_context(tmp_path, monkeypatch):
    from test_voice_workflow import setup_scene
    from trova.server import AgentHost
    st, c = setup_scene(tmp_path, monkeypatch, True)
    monkeypatch.setattr(AgentHost, 'inverse', lambda *args: {'ik_verified': True, 'reason': 'fixture', 'joints': [0] * 5})
    c.post('/api/select', data={'text': 'oggetto 7'})
    before = list(st.arm.tcp_mm())
    planned = c.post('/api/agent/task', json={'text': 'Scansionalo.'}).json()
    assert planned['confirmation'] is None and planned['scan_plan']['next'] is None
    assert 'Measure object_height_mm' in planned['reason']
    assert st.scans.active and list(st.arm.tcp_mm()) == before
    flipped = c.post('/api/scan/orientation', json={'orientation': 'upside_down'}).json()
    assert flipped['requires_reselection'] and not flipped['old_object_frame_valid']
    assert st.selected_context is None
    denied = c.post('/api/agent/confirm', json={'proposal_id': 'absent-orbit-proposal'}).json()
    assert denied['status'] == 'refused' and list(st.arm.tcp_mm()) == before
    st.inventory['snapshot']['objects'][0]['track_id'] = 9
    c.post('/api/select', data={'text': 'oggetto 9'})
    assert st.scans._session(None)['orientation_track_id'] == '9'
    replanned = c.post('/api/scan/next').json()
    assert replanned['selected']['track_id'] == 9
    assert replanned['confirmation'] is None and replanned['scan_session'] == st.scans.active
    assert 'Measure object_height_mm' in replanned['reason']


def fake_colmap(tmp_path, mode='mesh'):
    script = tmp_path / 'colmap-fixture'
    script.write_text('#!' + sys.executable + '\n' + r'''
import pathlib, sys, time
cmd, args = sys.argv[1], sys.argv[2:]
def val(flag): return pathlib.Path(args[args.index(flag) + 1])
if cmd == '-h':
    print('mesh_texturer' if MODE == 'texture' else 'mapper')
elif MODE == 'slow':
    time.sleep(20)
elif cmd == 'exhaustive_matcher' and MODE == 'fail':
    sys.exit(3)
elif cmd == 'feature_extractor':
    assert val('--ImageReader.mask_path').is_dir()
    assert args[args.index('--ImageReader.camera_model') + 1] == 'FULL_OPENCV'
elif cmd == 'mapper':
    (val('--output_path') / '0').mkdir()
elif cmd == 'model_converter':
    root = val('--output_path')
    (root / 'images.txt').write_text('1 1 0 0 0 0 0 0 1 v0001.jpg\n\n2 1 0 0 0 0 0 0 1 v0002.jpg\n\n3 1 0 0 0 0 0 0 1 v0003.jpg\n\n')
    (root / 'points3D.txt').write_text('1 0 0 0 255 0 0 0.5 1 0 2 0 3 0\n')
elif cmd in ('stereo_fusion', 'poisson_mesher'):
    faces = 0 if MODE == 'empty' else 1
    val('--output_path').write_text('ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nelement face ' + str(faces) + '\nproperty list uchar int vertex_indices\nend_header\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n')
elif cmd == 'mesh_texturer':
    root = val('--output_path'); root.mkdir()
    (root / 'object.obj').write_text('mtllib object.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n')
    (root / 'object.mtl').write_text('newmtl object\nmap_Kd texture.png\n')
    (root / 'texture.png').write_bytes(b'fixture texture')
'''.replace('MODE', repr(mode)))
    script.chmod(0o755)
    return script


@pytest.mark.parametrize('mode,success,textured', [('mesh', True, False), ('texture', True, True), ('fail', False, False), ('empty', False, False)])
def test_local_reconstruction_pipeline_reports_real_artifact_state(tmp_path, mode, success, textured):
    book, session = dataset(tmp_path)
    jobs = ReconstructionJobs(tmp_path / 'jobs', executable=str(fake_colmap(tmp_path, mode)))
    result = jobs.start(session)
    jobs.jobs[result['id']]['_thread'].join(timeout=15)
    result = jobs.get(result['id'])
    assert result['status'] == ('completed' if success else 'failed'), result
    assert result['reconstructed'] is success and result['textured_mesh'] is textured
    assert result['quality']['metric_scale_claimed'] is False
    assert jobs.artifact(result['id'], 'dataset.zip').exists()
    assert json.loads(jobs.artifact(result['id'], 'report.json').read_text())['status'] == result['status']
    if success:
        assert jobs.artifact(result['id'], 'mesh.ply').exists()
        assert result['quality']['registered_images'] == 3
        restored = ReconstructionJobs(tmp_path / 'jobs', executable=str(fake_colmap(tmp_path, mode)))
        assert restored.get(result['id'])['reconstructed']
        assert restored.artifact(result['id'], 'mesh.ply').exists()
    with pytest.raises(KeyError):
        jobs.artifact(result['id'], '../../.env')
    assert all(Path(v['image']).exists() for v in session['views'])


def test_reconstruction_cancellation_and_missing_dependency(tmp_path):
    _book, session = dataset(tmp_path)
    jobs = ReconstructionJobs(tmp_path / 'jobs', executable=str(fake_colmap(tmp_path, 'slow')))
    created = jobs.start(session)
    until = time.monotonic() + 3
    while jobs.get(created['id'])['stage'] == 'dataset' and time.monotonic() < until:
        time.sleep(.01)
    jobs.cancel(created['id'])
    jobs.jobs[created['id']]['_thread'].join(timeout=5)
    assert jobs.get(created['id'])['status'] == 'cancelled'
    assert not jobs.get(created['id'])['reconstructed']
    jobs.executable = None
    missing = jobs.start(session)
    jobs.jobs[missing['id']]['_thread'].join(timeout=5)
    result = jobs.get(missing['id'])
    assert result['status'] == 'failed' and 'not installed' in result['reason']


def test_dataset_groups_and_integrity_are_not_silently_merged(tmp_path):
    book, session = dataset(tmp_path)
    book.mark_orientation('flipped')
    book.capture(jpeg(5), meta(50), {'box': [5, 5, 75, 59]})
    with pytest.raises(ValueError, match='three accepted'):
        prepare_dataset(session, tmp_path / 'flipped', 'flipped')
    original = prepare_dataset(session, tmp_path / 'original', 'as_found')
    assert len(original) == 3 and all(v['orientation'] == 'as_found' for v in original)
    export = book.export_colmap()
    assert export['orientation'] == 'flipped' and export['posed'] == 1
    assert not (Path(export['folder']) / 'images/v0001.jpg').exists()
    Path(original[0]['image']).write_bytes(jpeg(6))
    with pytest.raises(ValueError, match='changed since acquisition'):
        prepare_dataset(session, tmp_path / 'tampered', 'as_found')


def test_operate_queries_reuse_the_driver_sdk_without_actuator_calls(tmp_path, monkeypatch):
    import threading
    from test_pipeline import _client
    from test_telemetry import _Sdk
    from trova.arm.xarm import XArmBackend
    from trova.arm import CartesianArm, Limits
    from trova.server import AgentHost
    app, _client_handle = _client(tmp_path, monkeypatch)
    st = app.state.trova
    raw = _Sdk()
    backend = XArmBackend.__new__(XArmBackend)
    backend.arm, backend.lock, backend.yaw = raw, threading.Lock(), 12.0
    backend.accel = 500.0
    st.arm = CartesianArm(backend, z_hover_mm=150, home_mm=None,
                          limits=Limits.parse('0,600,-400,400,0,500'), speed_mm_s=20)
    reading = st.robot_telemetry()
    assert reading['source'] == 'telemetry' and reading['error_code'] == 0
    host = AgentHost(st)
    result = host.inverse([100, 50, 150, 180, 0, 12], reading['joints'])
    assert result['ik_verified'] and not result['motion_sent']
    assert not raw.writes and not raw.closed


def test_reconstruction_endpoints_report_downloads_and_failures(tmp_path, monkeypatch):
    from test_pipeline import _client
    app, client = _client(tmp_path / 'app', monkeypatch)
    book, session = dataset(tmp_path / 'dataset')
    st = app.state.trova
    st.scans = book
    st.reconstruction = ReconstructionJobs(tmp_path / 'jobs', executable=str(fake_colmap(tmp_path)))
    assert client.get('/api/reconstruction/capabilities').json()['available']
    started = client.post('/api/reconstruction/start').json()
    st.reconstruction.jobs[started['id']]['_thread'].join(timeout=10)
    result = client.get('/api/reconstruction/' + started['id']).json()
    assert result['status'] == 'completed' and result['reconstructed']
    assert client.get('/api/reconstruction/' + started['id'] + '/artifacts/mesh.ply').content.startswith(b'ply')
    assert client.get('/api/reconstruction/' + started['id'] + '/artifacts/.env').status_code == 404
    assert client.get('/api/reconstruction/unknown').status_code == 404
    assert not client.post('/api/reconstruction/' + started['id'] + '/cancel').json()['cancel_requested']
