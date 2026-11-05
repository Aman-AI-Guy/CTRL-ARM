"""Optional local COLMAP jobs. No network and no access to robot drivers.

Each orientation is reconstructed separately. A human flipping an object invalidates
its old frame; no cross-orientation registration or metric scale is assumed.
"""

import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
import uuid
import zipfile

import cv2
import numpy as np


def ply_counts(path):
    counts = {}
    with Path(path).open('rb') as stream:
        if stream.readline().strip() != b'ply':
            raise ValueError('The mesh is not a PLY file')
        for _ in range(100):
            line = stream.readline().decode('ascii', errors='replace').strip()
            found = re.fullmatch(r'element (vertex|face) (\d+)', line)
            if found:
                counts[found[1]] = int(found[2])
            if line == 'end_header':
                return counts
    raise ValueError('The PLY header is incomplete')


def prepare_dataset(session, destination, orientation):
    destination = Path(destination)
    images, masks = destination / 'images', destination / 'masks'
    images.mkdir(parents=True)
    masks.mkdir()
    if session.get('reconstruction_only'):
        accepted = [v for v in session['views'] if v.get('reconstruction_accepted') and v.get('orientation') == orientation]
        if len(accepted)<3: raise ValueError('At least three recorded views of the same orientation are required.')
        if len({v['sha256'] for v in accepted}) != len(accepted):
            raise ValueError('Recorded dataset contains duplicate images.')
        for view in accepted:
            data=Path(view['image']).read_bytes()
            if hashlib.sha256(data).hexdigest()!=view['sha256']:raise ValueError('A recorded image changed since import.')
            frame=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
            if frame is None:raise ValueError('Invalid recorded JPEG.')
            sharpness=float(cv2.Laplacian(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var())
            if sharpness<20:raise ValueError('Recorded JPEG is too blurry for reconstruction.')
            name=view['source_name']
            if Path(name).name!=name:raise ValueError('Invalid recorded image name.')
            (images/name).write_bytes(data)
        (destination/'captures.json').write_text(json.dumps(accepted,indent=2))
        return accepted
    accepted = [v for v in session['views'] if v.get('accepted') and v.get('orientation') == orientation]
    if len(accepted) < 3:
        raise ValueError('At least three accepted overlapping views of one object orientation are required.')
    if any(v.get('pose_source') != 'telemetry' or not v.get('calibration_valid') or not v.get('camera_pose') for v in accepted):
        raise ValueError('Each reconstruction image requires valid calibration and a telemetry-derived camera pose.')
    if len({v.get('calibration_version') for v in accepted}) != 1:
        raise ValueError('Calibration changed between captures; start a new acquisition group.')
    if len({v.get('sha256') for v in accepted}) < 3:
        raise ValueError('The dataset contains duplicate photographs; capture different viewpoints.')
    positions = [np.asarray(v['camera_pose'], float)[:3, 3] for v in accepted]
    if max(np.linalg.norm(a - b) for a in positions for b in positions) < 2:
        raise ValueError('Camera translation is insufficient for triangulation.')
    for view in accepted:
        data = Path(view['image']).read_bytes()
        if hashlib.sha256(data).hexdigest() != view['sha256']:
            raise ValueError('A dataset photograph changed since acquisition.')
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError('A dataset JPEG cannot be decoded.')
        box = (view.get('detection') or {}).get('box')
        if not box or len(box) != 4:
            raise ValueError('An object bounding box is required for every reconstruction image.')
        x0, y0, x1, y1 = (int(v) for v in box)
        height, width = frame.shape[:2]
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError('An object bounding box is outside its image.')
        mask = np.zeros((height, width), np.uint8)
        mask[y0:y1, x0:x1] = 255
        name = view['viewpoint_id'] + '.jpg'
        (images / name).write_bytes(data)
        cv2.imwrite(str(masks / (name + '.png')), mask)
    (destination / 'captures.json').write_text(json.dumps(accepted, indent=2))
    (destination / 'calibration.json').write_text(json.dumps(session.get('intrinsics'), indent=2))
    return accepted


class ReconstructionJobs:
    def __init__(self, root, executable=None, popen=None, on_done=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        from .scan3d.executables import discover
        self.discovery = discover(executable)
        self.executable = self.discovery['path']
        self.popen = popen or subprocess.Popen
        self.jobs = {}
        self.lock = threading.Lock()
        self.on_done = on_done
        for report in sorted(self.root.glob('*/report.json')):
            try:
                job = json.loads(report.read_text())
                if job.get('id') != report.parent.name:
                    continue
                job['_folder'] = report.parent
                job['_cancel'] = threading.Event()
                if job.get('status') in {'queued', 'running'}:
                    job.update(status='interrupted', reason='The previous process ended; start a new job.')
                self.jobs[job['id']] = job
            except (OSError, ValueError, TypeError):
                continue

    def capabilities(self):
        return {'available': bool(self.executable), 'backend': 'COLMAP',
                'reason': self.discovery['reason'] if self.executable else 'COLMAP is not installed or executable validation failed. Acquisition and dataset export are available.',
                'open3d_postprocessing': __import__('importlib.util',fromlist=['find_spec']).find_spec('open3d') is not None,
                'metric_scale_claimed': False, 'cross_orientation_registration': False,
                'texturing': 'attempted only if the installed COLMAP advertises mesh_texturer'}

    def start(self, session):
        with self.lock:
            if any(j['status'] in {'queued', 'running'} for j in self.jobs.values()):
                raise RuntimeError('A reconstruction job is already running.')
            key = uuid.uuid4().hex
            folder = self.root / key
            folder.mkdir()
            job = {'id': key, 'session_id': session['id'], 'orientation': session['orientation'],
                   'status': 'queued', 'stage': 'dataset', 'reconstructed': False, 'textured_mesh': False,
                   'cad': False, 'artifacts': {}, 'reason': '', 'started_at': time.time(),
                   'quality': {'surface_coverage_known': False, 'closed_surface_verified': False,
                               'metric_scale_claimed': False, 'object_isolation': 'bounding_box_masks_only',
                               'cross_orientation_registration': False},
                   '_folder': folder, '_cancel': threading.Event()}
            self.jobs[key] = job
            snapshot = copy.deepcopy(session)
            job['_thread'] = threading.Thread(target=self._run, args=(job, snapshot), daemon=True)
            job['_thread'].start()
            return self.public(job)

    def public(self, job):
        return copy.deepcopy({k: v for k, v in job.items() if not k.startswith('_')})

    def get(self, key):
        return self.public(self.jobs[key])

    def cancel(self, key):
        job = self.jobs[key]
        if job['status'] not in {'queued', 'running'}:
            return {'id': key, 'cancel_requested': False, 'reason': 'The reconstruction job has already finished.'}
        job['_cancel'].set()
        return {'id': key, 'cancel_requested': True}

    def _command(self, job, args, timeout=600):
        if job['_cancel'].is_set():
            raise InterruptedError('Reconstruction cancelled.')
        args = list(args)
        if self.discovery.get('legacy_sift_flags'):
            translations={'--FeatureExtraction.use_gpu':'--SiftExtraction.use_gpu',
                          '--FeatureExtraction.max_image_size':'--SiftExtraction.max_image_size',
                          '--FeatureMatching.use_gpu':'--SiftMatching.use_gpu'}
            args=[translations.get(str(value),value) for value in args]
        # Sparse CPU extraction is portable; dense PatchMatch still reports unsupported CUDA explicitly.
        for index,value in enumerate(args[:-1]):
            if str(value).endswith('.use_gpu'): args[index+1]=os.getenv('TROVA_COLMAP_USE_GPU','0')
        job['stage'] = args[0]
        with (job['_folder'] / 'colmap.log').open('ab') as log:
            process = self.popen([self.executable, *map(str, args)], stdout=log, stderr=log,
                                 env={**os.environ, 'QT_QPA_PLATFORM': 'offscreen'})
            deadline = time.monotonic() + timeout
            try:
                while process.poll() is None:
                    if job['_cancel'].wait(.1):
                        raise InterruptedError('Reconstruction cancelled.')
                    if time.monotonic() > deadline:
                        raise TimeoutError(f'{args[0]} exceeded {timeout} seconds.')
                if process.returncode:
                    raise RuntimeError(f'COLMAP {args[0]} failed with exit code {process.returncode}; see the job log.')
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)

    def _run(self, job, session):
        folder = job['_folder']
        try:
            job['status'] = 'running'
            views = prepare_dataset(session, folder / 'dataset', job['orientation'])
            job['quality']['input_images'] = len(views)
            if session.get('reconstruction_only'):
                if session.get('postprocessing_replay'):
                    from .scan3d.benchmark import replay
                    result = replay(folder,cancel=job['_cancel'])
                else:
                    if not self.executable:raise RuntimeError(self.capabilities()['reason'])
                    from .scan3d.reconstruct_pipeline import reconstruct
                    result = reconstruct(folder,folder/'dataset/images',lambda args:self._command(job,args,timeout=3600),job['_cancel'])
                if job['_cancel'].is_set():raise InterruptedError('Reconstruction cancelled.')
                job.update(status='completed',reason='Recorded dataset processed; see provenance and rerun flags.',
                           reconstructed=True,provenance=result['provenance'],benchmark=result,textured_mesh=False)
                job['quality'].update(registered_images=result['registered_images'],
                    metric_scale_claimed=True, metric_scale_source='triangulated measured 39 mm ArUco markers',
                    estimated_object_dimensions_mm=result['estimated_object_dimensions_mm'],
                    mesh_bbox_extent_mm=result['mesh_bbox_extent_mm'],
                    full_reconstruction_rerun=result['full_reconstruction_rerun'])
                for path in [*folder.glob('mesh/*'),*folder.glob('renders/*'),*folder.glob('benchmark.json')]:
                    if path.is_file():job['artifacts'][path.name]=str(path.relative_to(folder))
                return
            if not self.executable:
                raise RuntimeError(self.capabilities()['reason'])
            images, masks = folder / 'dataset/images', folder / 'dataset/masks'
            database, sparse, dense = folder / 'database.db', folder / 'sparse', folder / 'dense'
            sparse.mkdir()
            dense.mkdir()
            intrinsics = session.get('intrinsics')
            if not intrinsics or not intrinsics.get('k'):
                raise ValueError('Camera intrinsics are missing from this scan session.')
            k = intrinsics['k']
            distortion = list(intrinsics.get('dist') or [])
            distortion = (distortion + [0] * 8)[:8]
            parameters = [k[0][0], k[1][1], k[0][2], k[1][2], *distortion]
            self._command(job, ['feature_extractor', '--database_path', database, '--image_path', images,
                                '--ImageReader.single_camera', '1', '--ImageReader.camera_model', 'FULL_OPENCV',
                                '--ImageReader.camera_params', ','.join(map(str, parameters)),
                                '--ImageReader.mask_path', masks])
            self._command(job, ['exhaustive_matcher', '--database_path', database])
            self._command(job, ['mapper', '--database_path', database, '--image_path', images, '--output_path', sparse])
            model = sparse / '0'
            if not model.is_dir():
                raise RuntimeError('COLMAP could not register an overlapping image model.')
            quality = folder / 'sparse_quality'
            quality.mkdir()
            self._command(job, ['model_converter', '--input_path', model, '--output_path', quality, '--output_type', 'TXT'])
            lines = (quality / 'images.txt').read_text().splitlines()
            registered = sum(bool(re.match(r'^\d+\s', line)) and line.endswith('.jpg') for line in lines)
            job['quality']['registered_images'] = registered
            job['quality']['registration_fraction'] = registered / len(views)
            if registered < 3:
                raise RuntimeError('Fewer than three photographs registered. Capture more overlapping textured views.')
            errors = []
            for line in (quality / 'points3D.txt').read_text().splitlines():
                parts = line.split()
                if parts and not line.startswith('#') and len(parts) >= 8:
                    errors.append(float(parts[7]))
            if not errors or not np.isfinite(errors).all():
                raise RuntimeError('No valid triangulated points were produced.')
            job['quality']['sparse_points'] = len(errors)
            job['quality']['median_reprojection_px'] = float(np.median(errors))
            job['quality']['warnings'] = []
            if registered < len(views):
                job['quality']['warnings'].append('Some photographs did not register; geometry may be incomplete.')
            if job['quality']['median_reprojection_px'] > 2:
                job['quality']['warnings'].append('Sparse reprojection error exceeds 2 px; geometry may be unreliable.')
            self._command(job, ['image_undistorter', '--image_path', images, '--input_path', model,
                                '--output_path', dense, '--output_type', 'COLMAP', '--max_image_size', '2000'])
            self._command(job, ['patch_match_stereo', '--workspace_path', dense, '--workspace_format', 'COLMAP',
                                '--PatchMatchStereo.geom_consistency', 'true'])
            self._command(job, ['stereo_fusion', '--workspace_path', dense, '--workspace_format', 'COLMAP',
                                '--input_type', 'geometric', '--output_path', dense / 'fused.ply'])
            self._command(job, ['poisson_mesher', '--input_path', dense / 'fused.ply', '--output_path', dense / 'mesh.ply'])
            counts = ply_counts(dense / 'mesh.ply')
            if not counts.get('vertex') or not counts.get('face'):
                raise RuntimeError('Surface reconstruction did not produce a non-empty mesh.')
            job['quality'].update(mesh_vertices=counts['vertex'], mesh_faces=counts['face'])
            job['reconstructed'] = True
            job['artifacts']['mesh.ply'] = 'dense/mesh.ply'
            job['artifacts']['cloud.ply'] = 'dense/fused.ply'
            # Texturing is version-dependent; never substitute a coloured point cloud for a textured mesh.
            try:
                probe = subprocess.run([self.executable, '-h'], capture_output=True, text=True, timeout=10)
                has_texturer = 'mesh_texturer' in probe.stdout + probe.stderr
            except (OSError, subprocess.SubprocessError):
                has_texturer = False
            if has_texturer:
                try:
                    self._command(job, ['mesh_texturer', '--workspace_path', dense, '--input_path', dense / 'mesh.ply',
                                        '--output_path', dense / 'textured'])
                    objects = list((dense / 'textured').rglob('*.obj'))
                    materials = list((dense / 'textured').rglob('*.mtl'))
                    textures = [p for p in (dense / 'textured').rglob('*') if p.suffix.lower() in {'.jpg', '.png'}]
                    job['textured_mesh'] = bool(objects and materials and textures and all(p.stat().st_size for p in objects + materials + textures))
                    if job['textured_mesh']:
                        with zipfile.ZipFile(folder / 'textured.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
                            for path in (dense / 'textured').rglob('*'):
                                if path.is_file():
                                    archive.write(path, path.relative_to(dense / 'textured'))
                        job['artifacts']['textured.zip'] = 'textured.zip'
                except (RuntimeError, TimeoutError) as exc:
                    job['quality']['texture_error'] = str(exc)
            else:
                job['quality']['texture_error'] = 'Installed COLMAP has no mesh_texturer command.'
            job['status'] = 'completed'
            job['reason'] = 'Mesh generated. Surface completeness, metric scale and cross-orientation registration remain unverified.'
        except InterruptedError as exc:
            job['status'], job['reason'] = 'cancelled', str(exc)
        except Exception as exc:  # Worker failures must be visible to the operator.
            job['status'], job['reason'] = 'failed', str(exc)
        finally:
            job['finished_at'] = time.time()
            if (folder / 'colmap.log').exists():
                job['artifacts']['log.txt'] = 'colmap.log'
            with zipfile.ZipFile(folder / 'dataset.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
                for path in (folder / 'dataset').rglob('*'):
                    if path.is_file():
                        archive.write(path, path.relative_to(folder / 'dataset'))
            job['artifacts']['dataset.zip'] = 'dataset.zip'
            job['artifacts']['report.json'] = 'report.json'
            (folder / 'report.json').write_text(json.dumps(self.public(job), indent=2))
            if self.on_done:
                self.on_done(self.public(job))

    def artifact(self, key, name):
        job = self.jobs[key]
        relative = job['artifacts'].get(name)
        if not relative:
            raise KeyError('Unknown artifact')
        path = (job['_folder'] / relative).resolve()
        if not path.is_relative_to(job['_folder'].resolve()) or not path.is_file():
            raise KeyError('Artifact is unavailable')
        return path
