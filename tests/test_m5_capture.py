import io
import json
from pathlib import Path
import pytest
from trova.acquire import ScanBook
from trova.pi_cam import PiCam, CameraError
from trova.scan3d.streaming import LiveStream
from trova.scan3d.evidence import workspace
from trova.scan3d.service import OrbitStore


def test_recorded_mouse_import_never_fabricates_robot_poses(tmp_path):
    try: workspace()
    except FileNotFoundError: pytest.skip('Optional retained dataset absent')
    book=ScanBook(tmp_path)
    report=book.import_recording()
    assert report['n_views']==60 and report['n_accepted']==0
    assert report['n_reconstruction_accepted']==60 and report['reconstruction_only']
    session=book._session(None)
    assert all(v['camera_pose'] is None and v['pose_source']=='recorded_sfm' for v in session['views'])
    book.mark_orientation('flipped')
    assert book._session(None)['needs_reselection']
    assert all(v['orientation']=='as_found' for v in session['views'])
    assert ScanBook(tmp_path)._session(None)['reconstruction_only']


def test_original_multipart_parser_and_existing_picam_share_protocol(monkeypatch):
    ending=json.dumps({'finished':True}).encode()
    body=b'--trovascan\r\nContent-Type: application/json\r\nContent-Length: '+str(len(ending)).encode()+b'\r\n\r\n'+ending+b'\r\n'
    class Response(io.BytesIO):
        headers={'X-Scan-Id':'recorded'}
    stream=LiveStream(Response(body))
    assert list(stream)==[] and stream.end['finished']
    camera=PiCam()
    calls=[]
    monkeypatch.setattr(camera,'_json',lambda *a,**k:calls.append((a,k)) or {'scan_id':'x'})
    assert camera.scan_start(width=1640)['scan_id']=='x'
    assert calls[0][0][1]=='/scan/start'
    with pytest.raises(CameraError):camera.scan_file('../secrets','meta.json')
    with pytest.raises(CameraError):camera.scan_file('valid','private.txt')


def test_orbit_settings_are_operator_owned_and_persistent(tmp_path):
    store=OrbitStore(tmp_path/'orbit.json')
    with pytest.raises(ValueError):store.configure(store.public(),False)
    with pytest.raises(ValueError):store.settings.validate()
    settings=store.public()|{'object_height_mm':35,'object_radius_mm':140,'tcp_offset_mm':[0]*6}
    store.configure(settings,True)
    assert OrbitStore(store.path).public()==settings
    assert not store.settings.geometry_confirmed
