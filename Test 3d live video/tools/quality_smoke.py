"""Step 0: local recorded-image model smoke test, not metric M1 qualification.

Run from the scan workspace: .venv/bin/python tools/quality_smoke.py
Uses a manually inspected pixel box. No camera/arm/network client is imported.
COLMAP intrinsics and a projected robot box are required for the later full M1.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    import cv2
    import numpy as np
    import torch
    from scan3d.config import load_config

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--with-depth', action='store_true', help='Also smoke-test MoGe-2, sequentially after SAM')
    args = ap.parse_args()
    cfg = load_config()['quality']
    smoke = cfg['smoke']
    source = ROOT / smoke['image']
    image = cv2.imread(str(source))
    if image is None:
        raise ValueError(f'Image does not decode: {source}')
    h, w = image.shape[:2]
    box = np.asarray(smoke['pixel_box_xyxy'], dtype=np.float32)
    if box.shape != (4,) or not (0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h):
        raise ValueError('Smoke box must be inside the recorded image')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; run outside the restricted sandbox with GPU access')
    destination = ROOT / smoke['output_root'] / time.strftime('%Y%m%d-%H%M%S')
    destination.mkdir(parents=True, exist_ok=False)
    models = Path(cfg['models_dir'])
    report = {
        'stage': 'step_0_dependency_smoke', 'image': str(source.relative_to(ROOT)),
        'image_sha256': sha256(source), 'image_size': [w, h],
        'prompt_source': 'manually_inspected_pixel_box_not_metric_projection',
        'pixel_box_xyxy': box.tolist(), 'metric_validated': False,
        'camera_accessed': False, 'robot_accessed': False, 'frames_sent_to_grok': False,
        'python': platform.python_version(), 'cuda_runtime': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(),
        'gpu_total_mib': torch.cuda.get_device_properties(0).total_memory / 2**20,
        'versions': {name: importlib.metadata.version(name) for name in
                     ['torch', 'torchvision', 'SAM-2', 'moge', 'opencv-python',
                      'numpy', 'open3d', 'pymeshlab', 'trimesh', 'pyransac3d']},
    }
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    checkpoint = models / cfg['sam_checkpoint']
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    # Optional custom connected-component CUDA extension is not needed for this test.
    predictor = SAM2ImagePredictor(build_sam2(cfg['sam_config'], str(checkpoint),
                                            device='cuda', apply_postprocessing=False))
    torch.cuda.synchronize()
    load_s = time.perf_counter() - started
    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        predictor.set_image(rgb)
        masks, scores, _ = predictor.predict(box=box, multimask_output=True)
    torch.cuda.synchronize()
    inference_s = time.perf_counter() - started
    best = int(np.argmax(scores))
    mask = masks[best].astype(bool)
    if not mask.any():
        raise RuntimeError('SAM returned an empty mask')
    if not cv2.imwrite(str(destination / 'mask.png'), mask.astype(np.uint8) * 255):
        raise OSError('Could not write mask')
    overlay = image.copy()
    overlay[mask] = (0.45 * overlay[mask] + 0.55 * np.array([30, 210, 50])).astype(np.uint8)
    cv2.rectangle(overlay, tuple(box[:2].astype(int)), tuple(box[2:].astype(int)), (0, 180, 255), 2)
    comparison = np.concatenate([image, overlay], axis=1)
    cv2.imwrite(str(destination / 'before-after.jpg'), comparison)
    report['sam'] = {
        'checkpoint_sha256': sha256(checkpoint), 'load_s': load_s, 'inference_s': inference_s,
        'timing_kind': 'first_inference_no_warmup',
        'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
        'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20,
        'scores': scores.tolist(), 'chosen_candidate': best, 'mask_pixels': int(mask.sum()),
        'score_is_ground_truth_iou': False,
        'mask_outside_prompt_pixels': int(mask.sum() - mask[int(box[1]):int(box[3]), int(box[0]):int(box[2])].sum()),
        'custom_cuda_postprocessing': False,
    }
    del predictor
    gc.collect()
    torch.cuda.empty_cache()
    # Preserve the completed segmentation evidence if the optional depth test fails.
    (destination / 'smoke.json').write_text(json.dumps(report, indent=2) + '\n')
    if args.with_depth:
        from moge.model.v2 import MoGeModel

        checkpoint = models / cfg['moge_checkpoint']
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        model = MoGeModel.from_pretrained(str(checkpoint)).to('cuda').eval()
        torch.cuda.synchronize()
        load_s = time.perf_counter() - started
        tensor = torch.tensor(rgb, dtype=torch.float32, device='cuda').permute(2, 0, 1) / 255
        started = time.perf_counter()
        with torch.inference_mode():
            prediction = model.infer(tensor, resolution_level=smoke['moge_resolution_level'], use_fp16=True)
        torch.cuda.synchronize()
        inference_s = time.perf_counter() - started
        depth = prediction['depth'].detach().float().cpu().numpy()
        valid = prediction['mask'].detach().cpu().numpy().astype(bool) & np.isfinite(depth) & (depth > 0)
        if not valid.any():
            raise RuntimeError('MoGe returned no valid depth')
        np.savez_compressed(destination / 'moge-unaligned-depth.npz', depth=depth, valid=valid)
        lo, hi = np.percentile(depth[valid], [2, 98])
        preview = np.clip((np.where(valid, depth, hi) - lo) / max(hi - lo, 1e-9), 0, 1)
        cv2.imwrite(str(destination / 'moge-depth.png'), (preview * 255).astype(np.uint8))
        report['moge'] = {
            'checkpoint_sha256': sha256(checkpoint), 'load_s': load_s, 'inference_s': inference_s,
            'timing_kind': 'first_inference_no_warmup',
            'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
            'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20,
            'valid_pixels': int(valid.sum()), 'aligned_to_mvs': False,
            'normal_output_shape': list(prediction['normal'].shape) if 'normal' in prediction else None,
            'depth_is_qualified_robot_mm': False,
        }
    (destination / 'smoke.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'output': str(destination), **report}, indent=2))


if __name__ == '__main__':
    main()
