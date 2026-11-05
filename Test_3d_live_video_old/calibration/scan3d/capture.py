"""Record one scan for the test bench: start, tell you when to move the arm, time it, stop, collect the frames.

    python -m scan3d.capture --note "mug, lamp on"            # Enter = arm starts, Enter = arm stops
    python -m scan3d.capture --duration 60 --note "T1 static"  # no motion: record 60 s, no Enter
    python -m scan3d.capture --full                            # also download every frame (15 fps backup)
    python -m scan3d.capture --set capture.fps=10 --set capture.bin_s=0.5

This program NEVER moves the arm: it prints "MOVE THE ARM NOW" and teammates move it.
Output in data/scans/<scan_id>/:
    images/          the sharpest frame of every bin (live push + backfill), named <frame index>.jpg
    received.csv     per image: source (live/backfill), capture and arrival time on this PC, latency
    pi/              meta.json, frames.csv, bins.csv, camera_metadata.json from the Pi
    full/            with --full: every recorded frame (frames/*.jpg) for offline re-selection
    capture.json     settings, locked exposure, Pi summary, receiver report, timings
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import tarfile
import threading
import time
from pathlib import Path

from .config import ROOT, add_config_args, config_from_args
from .receiver import Receiver

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from trova_cam import CameraError, TrovaCam  # noqa: E402

PI_FILES = ("meta.json", "frames.csv", "bins.csv", "camera_metadata.json")


def scan_params(cfg: dict, label: str) -> dict:
    p = {k: v for k, v in cfg["capture"].items() if not k.startswith("_")}
    if label:
        p["label"] = label
    return p


def fetch_pi_files(cam: TrovaCam, sid: str, dest: Path) -> dict[int, float]:
    """Copy the Pi's small scan files; return frame index -> Pi wall time from frames.csv."""
    dest.mkdir(parents=True, exist_ok=True)
    walls = {}
    for name in PI_FILES:
        try:
            data = cam.scan_file(sid, name)
        except CameraError as e:
            print(f"  (could not fetch {name}: {e})")
            continue
        (dest / name).write_bytes(data)
        if name == "frames.csv":
            for row in csv.DictReader(io.StringIO(data.decode())):
                walls[int(row["index"])] = float(row["wall"])
    return walls


def download_full(cam: TrovaCam, sid: str, dest: Path) -> dict:
    dest.mkdir(parents=True, exist_ok=True)
    tar_path = dest / f"{sid}.tar"
    t0 = time.time()
    last = [0.0]

    def progress(n):
        if time.time() - last[0] > 5:
            last[0] = time.time()
            print(f"  full backup: {n / 1e6:.0f} MB, {n * 8 / (time.time() - t0) / 1e6:.1f} Mbit/s")
    n = cam.scan_tar(sid, tar_path, progress)
    secs = time.time() - t0
    with tarfile.open(tar_path) as tar:
        tar.extractall(dest, filter="data")
    os.remove(tar_path)
    frames = list((dest / sid / "frames").glob("*.jpg"))
    return {"mbytes": round(n / 1e6, 1), "seconds": round(secs, 1), "mbit_s": round(n * 8 / secs / 1e6, 2),
            "frames": len(frames), "dir": f"{dest.name}/{sid}"}          # relative to the scan folder


def progress_printer(rx: Receiver, every: float, stop: threading.Event):
    while not stop.wait(every):
        s = rx.live_stats()
        lat = s.get("latency_s", {}).get("p50")
        print(f"  … {s['frames']} frames received" + (f", lag {lat:.1f} s" if lat is not None else ""), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_config_args(ap)
    ap.add_argument("--note", default="", help="free text saved with the scan (object, light, distance, ...)")
    ap.add_argument("--label", default="", help="appended to the scan id (letters, digits, - _)")
    ap.add_argument("--duration", type=float, help="static capture: record this many seconds, no Enter presses")
    ap.add_argument("--full", action="store_true", help="also download every frame (slow: ~5x the scan length)")
    ap.add_argument("--delete-on-pi", action="store_true", help="delete the scan on the Pi when done")
    ap.add_argument("--host", help="Pi address (default pi.host in config)")
    args = ap.parse_args()
    cfg = config_from_args(args)
    rcfg, ccfg = cfg["receiver"], cfg["capture_cli"]
    cam = TrovaCam(args.host or cfg["pi"]["host"], cfg["pi"]["port"], timeout=rcfg["http_timeout_s"])

    try:
        st = cam.status()
    except CameraError as e:
        sys.exit(f"camera not reachable: {e}")
    if st["protection"] != "normal":
        sys.exit(f"camera protection is '{st['protection']}' ({st['temp']} C): wait until the Pi cools down")
    if st.get("scan") and st["scan"].get("state") in ("recording", "stopping", "metering"):
        sys.exit(f"a scan is already running on the Pi: {st['scan']}")
    offset, rtt = cam.clock_offset()
    params = scan_params(cfg, args.label)

    print(f"starting scan on {cam.base}: {params}")
    t_cmd = time.time()
    try:
        info = cam.scan_start(**params)
    except CameraError as e:
        sys.exit(f"scan did not start: {e}")
    t_started = time.time()
    sid = info["scan_id"]
    out = ROOT / cfg["paths"]["data_dir"] / "scans" / sid
    out.mkdir(parents=True, exist_ok=True)
    rx = Receiver(cam, sid, out, offset, rcfg["reconnect_delay_s"], rcfg["http_timeout_s"]).start()
    lock = info["locked"]
    print(f"scan {sid} recording ({t_started - t_cmd:.1f} s to start). Locked: shutter {lock['shutter_us']} us, "
          f"gain {lock['gain']}, white balance {lock['awb_gains']}")
    for w in info["warnings"]:
        print("  warning:", w)
    stop_progress = threading.Event()
    threading.Thread(target=progress_printer, args=(rx, ccfg["progress_every_s"], stop_progress), daemon=True).start()

    aborted = None
    t_motion_start = t_motion_end = None
    try:
        if args.duration:
            print(f"\nSTATIC capture: keep the arm still for {args.duration:g} s")
            t_motion_start = time.time()
            time.sleep(args.duration)
            t_motion_end = time.time()
        else:
            print("\n>>> MOVE THE ARM NOW <<<   (teammates move it; this program never moves the arm)")
            input("Press Enter when the arm STARTS moving... ")
            t_motion_start = time.time()
            input("Press Enter when the arm STOPS... ")
            t_motion_end = time.time()
        time.sleep(ccfg["tail_s"])
    except (KeyboardInterrupt, EOFError):
        aborted = "interrupted by the user"
        print("\ninterrupted: stopping the scan")
    stop_progress.set()

    t_stop_req = time.time()
    try:
        summary = cam.scan_stop()
    except CameraError as e:
        summary = {"error": str(e)}
        print(f"stop failed: {e}")
    t_stopped = time.time()
    print("collecting the last frames…")
    report = rx.finish(rcfg["drain_timeout_s"])
    walls = fetch_pi_files(cam, sid, out / "pi")
    for r in rx.received():              # capture time for every image, backfilled ones too
        if r.capture_pc is None and r.index in walls:
            r.capture_pc = walls[r.index] - offset
    rx.write_csv()

    full = None
    if args.full and "error" not in summary:
        print("downloading the full backup (every frame)…")
        try:
            full = download_full(cam, sid, out / "full")
        except (CameraError, OSError, tarfile.TarError) as e:
            full = {"error": str(e)}
    if args.delete_on_pi and (not args.full or (full and "error" not in full)):
        try:
            cam.scan_delete(sid)
        except CameraError as e:
            print(f"could not delete on the Pi: {e}")

    rec = rx.received()
    in_motion = [r for r in rec if r.capture_pc is not None and t_motion_start is not None
                 and t_motion_start <= r.capture_pc <= (t_motion_end or t_stop_req)]
    last_live = report["last_live_recv_pc"]
    timings = {
        "command_pc": t_cmd, "recording_pc": t_started, "start_latency_s": round(t_started - t_cmd, 2),
        "motion_start_pc": t_motion_start, "motion_end_pc": t_motion_end,
        "motion_s": round(t_motion_end - t_motion_start, 2) if t_motion_end else None,
        "stop_request_pc": t_stop_req, "stop_s": round(t_stopped - t_stop_req, 2),
        "last_live_after_motion_end_s": round(last_live - t_motion_end, 2) if last_live and t_motion_end else None,
        "all_frames_after_motion_end_s": round(report["done_pc"] - t_motion_end, 2) if t_motion_end else None,
        "images_in_motion_window": len(in_motion),
    }
    result = {"scan_id": sid, "note": args.note, "static": bool(args.duration), "aborted": aborted,
              "host": cam.base, "clock_offset_s": round(offset, 4), "clock_rtt_s": round(rtt, 4),
              "params": params, "start": info, "pi_summary": summary, "receiver": report, "timings": timings,
              "full": full, "config": cfg}
    (out / "capture.json").write_text(json.dumps(result, indent=1))

    s = summary
    print(f"\n=== scan {sid} {'(ABORTED: ' + aborted + ')' if aborted else ''}")
    if "error" not in s:
        print(f"Pi:       {s['frames']} frames in {s['duration_s']} s = {s['real_fps']} fps, dropped {s['dropped']}, "
              f"{s['mean_kb']} KB/frame, {s['bins']} bins, temp {s['temp_start']} -> {s['temp_max']} C, "
              f"rpicam CPU {s['cpu_mean'].get('rpicam', '?')} %")
    lv = report["live"]
    print(f"Received: {len(rec)} images ({report['received_live']} live, {report['backfilled']} backfilled, "
          f"{report['missing_after_backfill']} missing), live {lv.get('fps', '?')} img/s, {lv.get('mbit_s', '?')} Mbit/s, "
          f"lag p50 {lv.get('latency_s', {}).get('p50', '?')} s / max {lv.get('latency_s', {}).get('max', '?')} s, "
          f"{report['reconnects']} reconnects")
    print(f"Timing:   start {timings['start_latency_s']} s, motion {timings['motion_s']} s, "
          f"last live image {timings['last_live_after_motion_end_s']} s and all images "
          f"{timings['all_frames_after_motion_end_s']} s after the arm stopped")
    if full:
        print(f"Full:     {full}")
    print(f"Saved:    {out}")


if __name__ == "__main__":
    main()
