"""Hand-held 'orbit' while the arm (and so the camera) stays still: the object and the scale cubes sit on a board,
someone tilts and turns the board under the camera. For COLMAP that is the same as a camera orbit, as long as
the static table is masked out (its features would say 'the camera never moved').

    python -m scan3d.handheld capture --move 45 --clean 6 --note "mouse on board"
        beeps on this PC: 1 long = recording, start turning the board
                          2 short = take EVERYTHING out of the picture (board, hands), keep the table empty
                          3 short = done
    python -m scan3d.handheld masks --scan <id>      # clean plate = the empty-table frames at the end
    python -m scan3d.reconstruct --scan <id> --masks data/scans/<id>/masks --mode preview
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .config import ROOT, add_config_args, config_from_args


def beep(n: int, ms: int = 180, freq: int = 1400) -> None:
    try:
        import winsound
        for _ in range(n):
            winsound.Beep(freq, ms)
            time.sleep(0.08)
    except Exception:  # noqa: BLE001  (no sound device: the printed line is enough)
        print("\a" * n, end="", flush=True)


def cmd_capture(a, cfg) -> int:
    total = a.move + a.clean
    cmd = [sys.executable, "-u", "-m", "scan3d.capture", "--duration", str(total), "--note", a.note,
           "--label", "hand", "--set", f"capture.max_s={int(total + 30)}"]
    for s in a.overrides:
        cmd += ["--set", s]
    print("$", " ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    def schedule():
        print(">>> GO: turn / tilt the board <<<", flush=True)
        beep(1, 700, 1000)
        time.sleep(max(0.0, a.move - 5))
        for k in range(5, 0, -1):
            print(f"  {k}…", flush=True)
            time.sleep(1)
        print(">>> TAKE EVERYTHING OUT OF THE PICTURE (board + hands) <<<", flush=True)
        beep(2)
        time.sleep(a.clean)
        print(">>> DONE <<<", flush=True)
        beep(3)

    started = False
    for line in p.stdout:
        print(line, end="", flush=True)
        if not started and " recording (" in line:
            started = True
            threading.Thread(target=schedule, daemon=True).start()
    return p.wait()


def load_small(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    img = cv2.imread(str(path))
    if size:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img


def cmd_masks(a, cfg) -> int:
    """Mask = pixels that differ from the empty table (clean plate). The clean plate is the median of the last
    frames that look alike; frames without enough foreground (empty table, start/end) are moved aside."""
    scan = ROOT / cfg["paths"]["data_dir"] / "scans" / a.scan
    imgs = sorted((scan / "images").glob("*.jpg"))
    if not imgs:
        raise SystemExit(f"no images in {scan / 'images'}")
    small = (320, 240)
    thumbs = [cv2.GaussianBlur(load_small(p, small), (5, 5), 0).astype(np.float32) for p in imgs]
    # clean plate: walk back from the last frame while the frames stay like it
    last, k = thumbs[-1], len(thumbs) - 1
    while k > 0 and np.mean(np.abs(thumbs[k - 1] - last)) < a.still:
        k -= 1
    clean_idx = list(range(k, len(thumbs)))
    if len(clean_idx) < 2:
        raise SystemExit("no empty-table frames at the end (2 short beeps = take everything out)")
    full0 = cv2.imread(str(imgs[0]))
    h, w = full0.shape[:2]
    clean = np.median(np.stack([cv2.imread(str(imgs[i])).astype(np.float32) for i in clean_idx]), axis=0)
    clean = cv2.GaussianBlur(clean, (0, 0), 3)
    out = scan / "masks"
    over = scan / "mask_preview"
    for d in (out, over):
        d.mkdir(exist_ok=True)
        for f in d.iterdir():
            f.unlink()
    kept, dropped = [], []
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    for i, p in enumerate(imgs):
        img = cv2.imread(str(p)).astype(np.float32)
        blur = cv2.GaussianBlur(img, (0, 0), 3)
        # brightness may drift a little (shadows of people): match the medians first
        diff = np.abs(blur - clean * (np.median(blur) / max(1.0, np.median(clean)))).max(axis=2)
        m = (diff > a.thresh).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ker)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker, iterations=2)
        n, lab, st, _ = cv2.connectedComponentsWithStats(m)
        keep = np.zeros_like(m)
        for j in range(1, n):
            if st[j, cv2.CC_STAT_AREA] > a.min_area * w * h:
                keep[lab == j] = 1
        keep = cv2.dilate(keep, ker, iterations=2)
        frac = float(keep.mean())
        if i in clean_idx or frac < a.min_fg:
            dropped.append((p.name, round(frac, 3)))
            continue
        kept.append((p.name, round(frac, 3)))
        cv2.imwrite(str(out / (p.name + ".png")), keep * 255)
        if i % a.preview_every == 0:
            vis = img.astype(np.uint8).copy()
            vis[keep == 0] = (vis[keep == 0] * 0.3).astype(np.uint8)
            cv2.imwrite(str(over / p.name), cv2.resize(vis, (w // 2, h // 2)))
    sel = scan / "images_masked"
    sel.mkdir(exist_ok=True)
    for f in sel.iterdir():
        f.unlink()
    for name, _ in kept:
        src = scan / "images" / name
        try:
            (sel / name).hardlink_to(src)
        except OSError:
            (sel / name).write_bytes(src.read_bytes())
    print(f"{len(imgs)} images: clean plate from the last {len(clean_idx)}, kept {len(kept)}, dropped {len(dropped)}")
    if kept:
        fr = [f for _, f in kept]
        print(f"foreground fraction of kept images: min {min(fr):.2f}, median {np.median(fr):.2f}, max {max(fr):.2f}")
    print(f"-> {sel} (images), {out} (masks), {over} (previews)")
    print(f"next: python -m scan3d.reconstruct {sel} --masks {out} --name {a.scan} --mode preview")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m scan3d.handheld", description=__doc__.split("\n\n")[0])
    add_config_args(ap)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("capture")
    s.add_argument("--move", type=float, default=45, help="seconds of board motion")
    s.add_argument("--clean", type=float, default=6, help="seconds of empty table at the end (clean plate)")
    s.add_argument("--note", default="hand-held board orbit")
    s = sub.add_parser("masks")
    s.add_argument("--scan", required=True)
    s.add_argument("--still", type=float, default=6.0, help="mean abs grey difference for 'same as the clean plate'")
    s.add_argument("--thresh", type=float, default=28.0, help="pixel difference (0-255) that counts as foreground")
    s.add_argument("--min-area", type=float, default=0.002, help="smallest blob kept, fraction of the image")
    s.add_argument("--min-fg", type=float, default=0.08, help="drop images with less foreground than this")
    s.add_argument("--preview-every", type=int, default=10)
    a = ap.parse_args(argv)
    cfg = config_from_args(a)
    return {"capture": cmd_capture, "masks": cmd_masks}[a.cmd](a, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
