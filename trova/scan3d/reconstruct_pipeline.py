"""Imported sparse/dense/metric object pipeline executed by Trova reconstruction jobs."""
from pathlib import Path
import time
import numpy as np
from .photogrammetry import run_sparse,run_dense,marker_scale,build_mesh,Stages,ReconError
from .benchmark import profile


def reconstruct(output, images, command, cancel):
    output=Path(output)
    cfg=profile()
    cfg['_command']=command
    stages=Stages()
    model,model_dir=run_sparse(cfg,cfg['settings'],output,Path(images),output/'colmap.log',stages)
    if cancel.is_set():raise InterruptedError('Reconstruction cancelled.')
    scale=marker_scale(Path(images),model,cfg['aruco'])
    if not scale.get('scale'):raise ReconError('Measured ArUco scale was not recovered; no assumed camera distance is permitted.')
    fused=run_dense(cfg,cfg['settings'],output,Path(images),model_dir,output/'colmap.log',stages)
    if cancel.is_set():raise InterruptedError('Reconstruction cancelled.')
    import open3d as o3d
    o3d.utility.random.seed(0)
    mesh=build_mesh(fused,scale['scale'],model,cfg['mesh_settings'],output,stages)
    loaded=o3d.io.read_triangle_mesh(str(output/'mesh/object.ply'))
    bbox=np.asarray(loaded.vertices).max(0)-np.asarray(loaded.vertices).min(0)
    return {'provenance':'new_COLMAP_and_Open3D_run','full_reconstruction_rerun':True,
            'registered_images':len(model['images']),'sparse_points':model['n_points'],
            'mean_reprojection_px':model['mean_reproj_px'],'scale':scale,'mesh':mesh,
            'mesh_bbox_extent_mm':bbox.tolist(),'estimated_object_dimensions_mm':mesh['size_mm'],
            'textured_mesh':False,'cad':False,'stages':stages.times}
