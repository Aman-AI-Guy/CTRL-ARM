"""Smoke test of the scan mode on the real Pi (no arm motion): start, live push, viewer cap, stop, files, tar.

    .venv\\Scripts\\python tests\\pi_smoke.py [--host 192.168.5.160] [--seconds 15] [--tar] [--keep] [--set k=v ...]

Prints the numbers and writes them to data/smoke/<scan_id>.json. The scan is deleted from the Pi unless --keep.
"""
import argparse
import json
import os
import statistics
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tests.test_stream_scan import read_parts  # noqa: E402


def call(base, method, path, body=None, timeout=60):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(base + path, data=data, method=method,
                               headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def clock_offset(base, n=5):
    """Pi wall clock minus PC wall clock, from the fastest of n round trips."""
    best = None
    for _ in range(n):
        t0 = time.time()
        _, d = call(base, "GET", "/scan/clock")
        t1 = time.time()
        if best is None or t1 - t0 < best[0]:
            best = (t1 - t0, d["wall"] - (t0 + t1) / 2)
    return best[1], best[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="192.168.5.160")
    ap.add_argument("--seconds", type=float, default=15)
    ap.add_argument("--tar", action="store_true")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--set", action="append", default=[], help="scan parameter, e.g. fps=10")
    a = ap.parse_args()
    base = f"http://{a.host}:8080"
    params = {"max_s": a.seconds + 30, "label": "smoke"}
    for kv in a.set:
        k, v = kv.split("=", 1)
        params[k] = v

    off, rtt = clock_offset(base)
    print(f"clock offset Pi-PC {off * 1000:+.0f} ms (rtt {rtt * 1000:.0f} ms)")
    t = time.time()
    code, info = call(base, "POST", "/scan/start", params)
    print(f"start: HTTP {code} in {time.time() - t:.2f} s")
    if code != 200:
        sys.exit(json.dumps(info, indent=1))
    sid = info["scan_id"]
    print(json.dumps({k: info[k] for k in ("scan_id", "profile", "locked", "warnings")}, indent=1))
    print("auto:", {k: info["auto"].get(k) for k in ("ExposureTime", "AnalogueGain", "DigitalGain", "ColourGains", "Lux")})

    live = {"parts": [], "recv": []}

    def reader():
        """Read the live stream part by part, timing each arrival."""
        with urllib.request.urlopen(base + "/scan/live?since=-1", timeout=30) as r:
            while True:
                line = r.readline()
                if not line:
                    break
                line = line.strip()
                if line == b"--trovascan--":
                    break
                if line != b"--trovascan":
                    continue
                h = {}
                while True:
                    x = r.readline().strip()
                    if not x:
                        break
                    k, v = x.decode().split(":", 1)
                    h[k.strip().lower()] = v.strip()
                body = r.read(int(h["content-length"]))
                r.readline()
                live["parts"].append((h, len(body)))
                live["recv"].append(time.time())

    t_live = time.time()
    th = threading.Thread(target=reader)
    th.start()
    time.sleep(2)
    with urllib.request.urlopen(base + "/stream.mjpg", timeout=10) as r:
        t0 = time.monotonic()
        n_view = len(read_parts(r, b"frame", limit_s=3.0))
        viewer_fps = n_view / (time.monotonic() - t0)
    _, st = call(base, "GET", "/status.json")
    print("status during scan:", st["scan"], "temp", st["temp"])
    time.sleep(max(0.0, a.seconds - (time.time() - t_live)))
    t = time.time()
    code, summary = call(base, "POST", "/scan/stop")
    stop_s = time.time() - t
    th.join(30)
    t_end = time.time()

    frames = [(h, n, tr) for (h, n), tr in zip(live["parts"], live["recv"]) if h["content-type"] == "image/jpeg"]
    lat = [tr - (float(h["x-wall"]) - off) for h, n, tr in frames]
    nbytes = sum(n for _, n, _ in frames)
    dur = (frames[-1][2] - frames[0][2]) if len(frames) > 1 else 0
    res = {
        "scan_id": sid, "params": params, "start_latency_s": info.get("start_latency_s"), "stop_s": round(stop_s, 2),
        "locked": info["locked"], "auto": info["auto"], "warnings": info["warnings"], "summary": summary,
        "live": {"frames": len(frames), "mbit_s": round(nbytes * 8 / dur / 1e6, 2) if dur else None,
                 "delivered_fps": round((len(frames) - 1) / dur, 2) if dur else None,
                 "kb_mean": round(nbytes / max(1, len(frames)) / 1024, 1),
                 "latency_s": {"p50": round(statistics.median(lat), 3), "max": round(max(lat), 3)} if lat else None,
                 "missing_bins": summary["bins"] - len(frames),
                 "last_after_stop_s": round(t_end - t - stop_s, 2)},
        "viewer_fps_during_scan": round(viewer_fps, 2),
    }
    _, st = call(base, "GET", "/status.json")
    res["status_after"] = {k: st[k] for k in ("width", "height", "framerate", "quality", "camera")}
    if a.tar:
        t = time.time()
        out = os.path.join(ROOT, "data", "smoke", f"{sid}.tar")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with urllib.request.urlopen(base + f"/scan/{sid}.tar", timeout=60) as r, open(out, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        size = os.path.getsize(out)
        with tarfile.open(out) as tf:
            n_jpg = len([m for m in tf.getnames() if m.endswith(".jpg")])
        res["tar"] = {"mbytes": round(size / 1e6, 1), "seconds": round(time.time() - t, 1),
                      "mbit_s": round(size * 8 / (time.time() - t) / 1e6, 2), "jpg": n_jpg}
        os.remove(out)
    if not a.keep:
        res["deleted"] = call(base, "DELETE", f"/scan/{sid}")[1]
    os.makedirs(os.path.join(ROOT, "data", "smoke"), exist_ok=True)
    with open(os.path.join(ROOT, "data", "smoke", f"{sid}.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("auto",)}, indent=1))


if __name__ == "__main__":
    main()
