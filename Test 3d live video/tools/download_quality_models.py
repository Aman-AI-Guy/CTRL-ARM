"""Download the approved Step 0 checkpoints outside the workspace; resume interrupted transfers."""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

MODELS = {
    'sam2.1_hiera_base_plus.pt': (
        'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt',
        'a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5'),
    'moge-2-vitl-normal.pt': (
        'https://huggingface.co/Ruicheng/moge-2-vitl-normal/resolve/main/model.pt',
        '280741fd09bc3f403ccff9967784c2a391b52d2c0742ae3efdb21d9f90cc1a01'),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--models-dir', type=Path, required=True)
    args = ap.parse_args()
    args.models_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, (url, expected) in MODELS.items():
        target = args.models_dir / name
        started = time.perf_counter()
        candidate = target if target.exists() else target.with_suffix('.part')
        if not target.exists():
            subprocess.run(['curl', '--fail', '--location', '--retry', '3', '--continue-at', '-',
                            '--output', str(candidate), url], check=True)
        with candidate.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != expected:
            raise ValueError(f'Checkpoint checksum mismatch: {candidate}; preserved for inspection')
        if candidate != target:
            candidate.rename(target)
        manifest[name] = {'url': url, 'bytes': target.stat().st_size, 'sha256': digest,
                          'elapsed_s': round(time.perf_counter() - started, 3)}
        print(name, json.dumps(manifest[name]), flush=True)
    (args.models_dir / 'trova-quality-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
