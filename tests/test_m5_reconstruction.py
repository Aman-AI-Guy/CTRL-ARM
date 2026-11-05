import json
import os
from pathlib import Path
import time
import pytest
from trova.acquire import ScanBook
from trova.reconstruction import ReconstructionJobs,prepare_dataset
from trova.scan3d.evidence import workspace
from trova.scan3d.executables import discover


def archive():
    try:return workspace()
    except FileNotFoundError:pytest.skip('Optional retained workspace absent')


def test_colmap_discovery_validates_executables_and_configuration(tmp_path,monkeypatch):
    monkeypatch.setenv('TROVA_COLMAP',str(tmp_path/'missing'))
    result=discover()
    assert not result['available'] and result['path'] is None
    from test_scanning_reconstruction import fake_colmap
    binary=fake_colmap(tmp_path)
    assert discover(binary)['available']
    bad=tmp_path/'bad';bad.write_text('not executable')
    assert not discover(bad)['available']


def test_retained_dataset_validates_hashes_and_never_needs_active_calibration(tmp_path):
    archive()
    book=ScanBook(tmp_path/'scans');book.import_recording()
    session=book._session(None)
    views=prepare_dataset(session,tmp_path/'dataset','as_found')
    assert len(views)==60 and len(list((tmp_path/'dataset/images').glob('*.jpg')))==60
    Path(views[0]['image']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):prepare_dataset(session,tmp_path/'broken','as_found')


def test_real_mouse_postprocessing_job_exports_new_geometry(tmp_path):
    archive();pytest.importorskip('open3d')
    book=ScanBook(tmp_path/'scans');book.import_recording()
    session=dict(book._session(None),postprocessing_replay=True)
    jobs=ReconstructionJobs(tmp_path/'jobs')
    created=jobs.start(session)
    jobs.jobs[created['id']]['_thread'].join(timeout=90)
    result=jobs.get(created['id'])
    assert result['status']=='completed',result
    benchmark=result['benchmark']
    assert not benchmark['full_reconstruction_rerun'] and benchmark['postprocessing_rerun']
    assert benchmark['registered_images']==60
    assert benchmark['scale']['scale']==pytest.approx(53.4097165,rel=.001)
    dimensions=benchmark['estimated_object_dimensions_mm']
    assert dimensions['L']==pytest.approx(96.3,abs=1)
    assert dimensions['W']==pytest.approx(55.7,abs=1)
    assert dimensions['H']==pytest.approx(37.4,abs=1)
    assert benchmark['mesh_bbox_extent_mm'][0]>dimensions['L']
    assert jobs.artifact(created['id'],'object.glb').stat().st_size>1000
    assert jobs.artifact(created['id'],'front.png').exists()
    assert not result['textured_mesh'] and not result['cad']
    assert ReconstructionJobs(tmp_path/'jobs').get(created['id'])['reconstructed']


def test_full_reconstruction_missing_colmap_is_not_a_replay_success(tmp_path):
    archive()
    book=ScanBook(tmp_path/'scans');book.import_recording()
    jobs=ReconstructionJobs(tmp_path/'jobs')
    jobs.executable=None
    created=jobs.start(book._session(None))
    jobs.jobs[created['id']]['_thread'].join(timeout=15)
    result=jobs.get(created['id'])
    assert result['status']=='failed' and not result['reconstructed']
    assert 'not installed' in result['reason']
