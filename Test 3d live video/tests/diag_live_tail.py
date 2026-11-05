"""Diagnose the delay of the last live frames after /scan/stop: was the Pi late sending them, or the network late
delivering them? Uses X-Sent-Mono-Ns (Pi monotonic clock when the part was sent). Camera static, no arm motion.

    .venv\\Scripts\\python tests\\diag_live_tail.py [--seconds 8] [--set k=v ...]
"""
import argparse
import os
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from trova_cam import TrovaCam  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="192.168.5.160")
    ap.add_argument("--seconds", type=float, default=8)
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args()
    cam = TrovaCam(a.host)
    params = {"label": "diag", "max_s": 60}
    for kv in a.set:
        k, v = kv.split("=", 1)
        params[k] = v
    # map Pi monotonic -> PC wall: pc = pi_mono + k
    best = None
    for _ in range(5):
        t0 = time.time()
        c = cam.scan_clock()
        t1 = time.time()
        if best is None or t1 - t0 < best[0]:
            best = (t1 - t0, (t0 + t1) / 2 - c["mono_ns"] / 1e9)
    k = best[1]
    t0 = time.time()
    info = cam.scan_start(**params)
    print(f"scan_start took {time.time() - t0:.2f} s, Pi says start_latency {info.get('start_latency_s')} s")
    sid = info["scan_id"]
    rows, end = [], {}

    def reader():
        with cam.scan_live(scan_id=sid, timeout=40) as live:
            for f in live:
                rows.append((f.index, f.bin, f.mono_ns / 1e9 + k, f.sent_mono_ns / 1e9 + k, f.recv_wall, f.nbytes))
            end["end"] = live.end
            end["t"] = time.time()
    th = threading.Thread(target=reader)
    th.start()
    time.sleep(a.seconds)
    t_stop = time.time()
    summ = cam.scan_stop()
    t_stopped = time.time()
    print({k: summ[k] for k in ("frames", "duration_s", "real_fps", "bins", "bins_sent", "dropped")})
    th.join(60)
    print(f"stop request at 0.00 s, stop returned at {t_stopped - t_stop:+.2f} s, end part at {end.get('t', 0) - t_stop:+.2f} s")
    print("index bin   captured   sent      received   (seconds relative to the stop request, PC clock)")
    print(f"{len(rows)} live parts; first:")
    for r in rows[:3]:
        print(f"{r[0]:5d} {r[1]:3d}  {r[2] - t_stop:+8.2f} {r[3] - t_stop:+8.2f} {r[4] - t_stop:+9.2f}   {r[5] // 1024} KB")
    print("last:")
    for r in rows[-8:]:
        print(f"{r[0]:5d} {r[1]:3d}  {r[2] - t_stop:+8.2f} {r[3] - t_stop:+8.2f} {r[4] - t_stop:+9.2f}   {r[5] // 1024} KB")
    cam.scan_delete(sid)


if __name__ == "__main__":
    main()
