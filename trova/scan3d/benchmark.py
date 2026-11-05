"""Real mouse reconstruction benchmark; retained SfM/dense inputs are explicit."""
import json
from pathlib import Path
import time
import numpy as np
from .evidence import workspace,sha256
from .photogrammetry import read_model,marker_scale,build_mesh,Stages


def profile(root=None):
    return json.loads(Path(__file__).with_name('mouse_profile.json').read_text())


def replay(output,root=None,cancel=None):
    import open3d as o3d
    started=time.monotonic()
    root=workspace(root);output=Path(output).resolve()
    if output.is_relative_to(root):raise ValueError('Never write inside the retained workspace.')
    output.mkdir(parents=True,exist_ok=True)
    retained=root/'data/recon/20261105-011356-orbit-v2/full'
    images=root/'data/scans/20261105-011356-orbit/images'
    cfg=profile(root)
    model=read_model(retained/'colmap/sparse_txt/0')
    if cancel and cancel.is_set():raise InterruptedError('Reconstruction cancelled.')
    scale=marker_scale(images,model,cfg['aruco'])
    if not scale.get('scale'):raise ValueError('Measured marker scale could not be recovered.')
    if cancel and cancel.is_set():raise InterruptedError('Reconstruction cancelled.')
    o3d.utility.random.seed(0)
    stages=Stages()
    mesh=build_mesh(retained/'colmap/dense/fused.ply',scale['scale'],model,cfg['mesh_settings'],output,stages)
    loaded=o3d.io.read_triangle_mesh(str(output/'mesh/object.ply'))
    bbox=np.asarray(loaded.vertices).max(0)-np.asarray(loaded.vertices).min(0)
    if cancel and cancel.is_set():raise InterruptedError('Reconstruction cancelled.')
    result={'provenance':'recorded_postprocessing_replay','full_reconstruction_rerun':False,
            'sfm_rerun':False,'dense_rerun':False,'postprocessing_rerun':True,
            'registered_images':len(model['images']),'sparse_points':model['n_points'],
            'mean_reprojection_px':model['mean_reproj_px'],'scale':scale,'mesh':mesh,
            'mesh_bbox_extent_mm':bbox.tolist(),'estimated_object_dimensions_mm':mesh['size_mm'],
            'ruler_dimensions_reported_mm':[98,58,35], 'ruler_remeasured':False,
            'camera_poses_source':'retained_COLMAP_text_model_not_new_estimation',
            'source_dense_sha256':sha256(retained/'colmap/dense/fused.ply'),
            'source_report_sha256':sha256(retained/'report.json'),
            'textured_mesh':False,'vertex_colored_mesh':True,'cad':False,
            'surface_coverage_known':False,'runtime_s':time.monotonic()-started,'stages':stages.times}
    (output/'benchmark.json').write_text(json.dumps(result,indent=2))
    return result
