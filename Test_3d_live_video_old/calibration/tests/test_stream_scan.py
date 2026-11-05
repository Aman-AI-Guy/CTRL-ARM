"""Tests for the scan mode of trova-stream/stream.py, on a PC, with tests/fake_rpicam_vid.py as the camera.

    .venv\\Scripts\\python -m unittest tests.test_stream_scan -v

Unit tests check the pieces (default command unchanged, exposure lock, parameters, metadata parser, bins,
dropped frames). The integration test runs the real server on a free port and walks the whole scan API,
including the default API recognition depends on, the 2 fps viewer cap, max_s, disk and heat refusals.
"""
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STREAM_DIR = os.path.join(ROOT, "trova-stream")
FAKE = os.path.join(ROOT, "tests", "fake_rpicam_vid.py")
sys.path.insert(0, STREAM_DIR)
import stream  # noqa: E402


# ---------------------------------------------------------------- helpers
def read_parts(resp, boundary, limit_s=None):
    """Parse a multipart HTTP body part by part: list of (headers dict, body bytes)."""
    parts, t0 = [], time.monotonic()
    open_b, close_b = b"--" + boundary, b"--" + boundary + b"--"
    while limit_s is None or time.monotonic() - t0 < limit_s:
        line = resp.readline()
        if not line:
            break
        line = line.strip()
        if line == close_b:
            break
        if line != open_b:
            continue
        headers = {}
        while True:
            h = resp.readline().strip()
            if not h:
                break
            k, v = h.decode().split(":", 1)
            headers[k.strip().lower()] = v.strip()
        body = resp.read(int(headers["content-length"]))
        resp.readline()
        parts.append((headers, body))
    return parts


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Client:
    def __init__(self, port):
        self.base = f"http://127.0.0.1:{port}"

    def req(self, method, path, body=None, timeout=30):
        data = json.dumps(body).encode() if body is not None else None
        r = urllib.request.Request(self.base + path, data=data, method=method,
                                   headers={"Content-Type": "application/json"} if data else {})
        try:
            with urllib.request.urlopen(r, timeout=timeout) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def json(self, method, path, body=None, timeout=30):
        code, raw = self.req(method, path, body, timeout)
        return code, json.loads(raw)


# ---------------------------------------------------------------- unit tests
class UnitTests(unittest.TestCase):
    def test_default_command_unchanged(self):
        os.environ.pop("TROVA_RPICAM_VID", None)
        normal, warnings = stream.clamp_settings(SimpleNamespace(width=1600, height=900, framerate=10, quality=50))
        self.assertEqual(warnings, [])
        self.assertEqual(stream.build_cmd(normal, 180, []), [
            "rpicam-vid", "-v", "0", "-n", "-t", "0", "--flush", "--codec", "mjpeg", "-q", "50",
            "--width", "1600", "--height", "900", "--framerate", "10", "--mode", "1640:1232:10:P",
            "--rotation", "180", "-o", "-"])

    def test_defaults_and_limits_unchanged(self):
        self.assertEqual((stream.WIDTH, stream.HEIGHT, stream.FRAMERATE, stream.QUALITY), (1600, 900, 10, 50))
        self.assertEqual((stream.MAX_QUALITY, stream.FULL_RES_MAX_FPS, stream.COMFORT_QUALITY), (85, 20, 70))

    def test_scan_profile_warns_about_quality(self):
        prof, w = stream.clamp_settings(SimpleNamespace(width=1640, height=1232, framerate=15, quality=85))
        self.assertEqual(prof, stream.Profile(1640, 1232, 15, 85))
        self.assertTrue(any("quality 85" in x for x in w))
        prof, w = stream.clamp_settings(SimpleNamespace(width=1920, height=1080, framerate=30, quality=95))
        self.assertEqual((prof.framerate, prof.quality), (20, 85))     # hardware limits still clamp

    def test_compute_lock(self):
        auto = {"ExposureTime": 30000, "AnalogueGain": 2.75, "DigitalGain": 1.0, "ColourGains": [1.27, 1.96]}
        lock = stream.compute_lock(auto, 8000, 16)
        self.assertEqual(lock["shutter_us"], 8000)
        self.assertAlmostEqual(lock["gain"], 30000 * 2.75 / 8000, places=3)
        self.assertTrue(lock["shutter_capped"])
        self.assertFalse(lock["gain_capped"])
        self.assertEqual(lock["awb_gains"], [1.27, 1.96])
        self.assertAlmostEqual(lock["brightness_vs_auto"], 1.0)
        lock = stream.compute_lock(dict(auto, ExposureTime=5000), 8000, 16)      # under the cap: unchanged
        self.assertEqual((lock["shutter_us"], lock["gain"], lock["shutter_capped"]), (5000, 2.75, False))
        lock = stream.compute_lock(auto, 2000, 16)                               # gain capped: darker
        self.assertEqual(lock["gain"], 16)
        self.assertTrue(lock["gain_capped"])
        self.assertLess(lock["brightness_vs_auto"], 1)

    def test_parse_scan_params(self):
        p = stream.parse_scan_params({"fps": "10", "bin_s": "0.5", "label": "cup_1"})
        self.assertEqual((p["fps"], p["bin_s"], p["quality"], p["label"]), (10, 0.5, 85, "cup_1"))
        for bad in ({"nope": 1}, {"bin_s": 0}, {"max_s": 9999}, {"fps": "x"}, {"label": "../etc"}):
            with self.assertRaises(ValueError):
                stream.parse_scan_params(bad)

    def test_meta_tail_parses_partial_writes(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.json")
            objs = [{"ExposureTime": 100 * i, "AnalogueGain": 2.0, "DigitalGain": 1.0, "ColourGains": [1.2, 1.9],
                     "FocusFoM": i} for i in range(1, 8)]
            text = "[\n" + ",\n".join(json.dumps(o, indent=4) for o in objs)
            tail = stream.MetaTail(path)
            with open(path, "w") as f:
                f.write(text[:len(text) // 2])
            tail.poll()
            first = tail.count
            with open(path, "a") as f:
                f.write(text[len(text) // 2:] + "\n]\n")
            tail.stop()
            self.assertLess(first, 7)
            self.assertEqual(tail.count, 7)
            s = tail.settled()
            self.assertEqual(s["ExposureTime"], 500)           # median of the last 5
            self.assertEqual(s["ColourGains"], [1.2, 1.9])

    def _recorder(self, d, **params):
        p = stream.parse_scan_params(params)
        return stream.ScanRecorder("t", d, p, stream.Profile(1640, 1232, 15, 85), {}, {}, [], [], 1000, 50.0)

    def test_bins_pick_largest_frame(self):
        with tempfile.TemporaryDirectory() as d:
            rec = self._recorder(d, bin_s=0.3, skip_frames=1)
            sizes = [9, 5, 7, 3, 8, 2, 6, 4, 1, 9]           # frame 0 is skipped (gain ramp)
            for k, n in enumerate(sizes):
                rec.add(b"x" * n * 100, k * 100_000_000, 1000.0 + k * 0.1)   # 10 fps
            self.assertTrue(rec.begin_stop())
            rec.finish(False, "test")
            # recorded indices 0..8 = sizes[1:], 3 frames per 0.3 s bin
            self.assertEqual([e["index"] for e in rec.live], [1, 3, 8])
            self.assertEqual(rec.n, 9)
            self.assertEqual(rec.saved, 9)
            with open(os.path.join(d, "frames.csv")) as f:
                self.assertEqual(len(f.read().strip().splitlines()), 10)
            with open(os.path.join(d, "bins.csv")) as f:
                self.assertEqual(len(f.read().strip().splitlines()), 4)
            with open(os.path.join(d, "meta.json")) as f:
                self.assertEqual(json.load(f)["summary"]["state"], "done")

    def test_full_queue_drops_instead_of_blocking(self):
        class SlowRecorder(stream.ScanRecorder):
            def _write_loop(self):
                time.sleep(0.5)
                super()._write_loop()
        with tempfile.TemporaryDirectory() as d:
            p = stream.parse_scan_params({"queue_frames": 4, "skip_frames": 0})
            rec = SlowRecorder("t", d, p, stream.Profile(1640, 1232, 15, 85), {}, {}, [], [], 1000, 50.0)
            t = time.monotonic()
            for k in range(20):
                rec.add(b"y" * 1000, k * 66_000_000, 0.0)
            self.assertLess(time.monotonic() - t, 0.3)     # never waited for the disk
            rec.begin_stop()
            rec.finish(False, "test")
            self.assertEqual(rec.saved + len(rec.dropped), 20)
            self.assertGreaterEqual(len(rec.dropped), 10)
            self.assertEqual(rec.summary()["dropped"], len(rec.dropped))


# ---------------------------------------------------------------- integration test
class ServerTests(unittest.TestCase):
    @classmethod
    def start_server(cls, *extra_args):
        port = free_port()
        log = open(os.path.join(cls.tmp, f"server-{port}.log"), "w", encoding="utf-8")
        env = dict(os.environ, TROVA_RPICAM_VID=json.dumps([sys.executable, FAKE]), TROVA_TEMP_FILE=cls.temp_file,
                   PYTHONIOENCODING="utf-8")
        proc = subprocess.Popen([sys.executable, "-u", os.path.join(STREAM_DIR, "stream.py"), "--port", str(port),
                                 "--scans-dir", cls.scans, "--idle-timeout", "5", *extra_args],
                                env=env, stdout=log, stderr=log)
        log.close()                 # the child keeps its own handle
        c = Client(port)
        for _ in range(50):
            try:
                c.req("GET", "/status.json", timeout=2)
                break
            except OSError:
                time.sleep(0.1)
        return proc, c

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="trova-stream-test-")
        cls.scans = os.path.join(cls.tmp, "scans")
        cls.temp_file = os.path.join(cls.tmp, "temp")
        with open(cls.temp_file, "w") as f:
            f.write("50000")
        cls.proc, cls.c = cls.start_server()

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()
        cls.proc.wait(10)
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_all(self):
        c = self.c
        # --- the default API is unchanged
        code, st = c.json("GET", "/status.json")
        self.assertEqual(code, 200)
        for k in ("camera", "viewers", "fps", "width", "height", "framerate", "quality", "rotate", "temp",
                  "protection", "message", "warnings"):
            self.assertIn(k, st)
        self.assertEqual((st["width"], st["height"], st["framerate"], st["quality"]), (1600, 900, 10, 50))
        self.assertIsNone(st["scan"])
        code, jpg = c.req("GET", "/snapshot.jpg")
        self.assertEqual(code, 200)
        self.assertTrue(jpg.startswith(b"\xff\xd8"))
        code, page = c.req("GET", "/")
        self.assertEqual(code, 200)

        # --- start
        code, info = c.json("POST", "/scan/start", {"max_s": 30, "ae_settle_s": 0.5, "label": "test"})
        self.assertEqual(code, 200, info)
        sid = info["scan_id"]
        self.assertTrue(sid.endswith("-test"))
        self.assertEqual(info["locked"]["shutter_us"], 8000)
        self.assertAlmostEqual(info["locked"]["gain"], 30000 * 2.75 / 8000, places=2)
        self.assertEqual(info["locked"]["awb_gains"], [1.27, 1.96])
        self.assertTrue(any("quality 85" in w for w in info["warnings"]))
        self.assertEqual(c.json("POST", "/scan/start", {})[0], 409)                 # one scan at a time
        code, st = c.json("GET", "/status.json")
        self.assertEqual((st["framerate"], st["quality"], st["scan"]["state"]), (15, 85, "recording"))

        # --- live push in the background, a normal viewer at the same time
        live = {}

        def reader():
            with urllib.request.urlopen(c.base + "/scan/live?since=-1", timeout=30) as r:
                live["parts"] = read_parts(r, b"trovascan")
        t = threading.Thread(target=reader)
        t.start()
        with urllib.request.urlopen(c.base + "/stream.mjpg", timeout=10) as r:
            t0 = time.monotonic()
            viewer = read_parts(r, b"frame", limit_s=3.0)
            viewer_fps = len(viewer) / (time.monotonic() - t0)
        self.assertLessEqual(viewer_fps, 2.6)                                          # capped during a scan
        time.sleep(1.0)

        # --- stop
        code, summary = c.json("POST", "/scan/stop")
        self.assertEqual(code, 200, summary)
        t.join(15)
        self.assertEqual(summary["state"], "done")
        self.assertGreaterEqual(summary["frames"], 40)
        self.assertEqual(summary["dropped"], 0)
        self.assertEqual(summary["saved"], summary["frames"])
        self.assertGreater(summary["real_fps"], 12)
        self.assertGreaterEqual(summary["bins"], 10)
        parts = live["parts"]
        frames = [p for p in parts if p[0]["content-type"] == "image/jpeg"]
        end = json.loads(parts[-1][1])
        self.assertTrue(end["end"])
        self.assertEqual(len(frames), summary["bins"])                                # every bin delivered
        self.assertEqual(summary["bins_sent"], summary["bins"])
        idx = [int(h["x-frame-index"]) for h, _ in frames]
        self.assertEqual(idx, sorted(idx))
        self.assertEqual([int(h["x-bin"]) for h, _ in frames], sorted(int(h["x-bin"]) for h, _ in frames))
        for h, body in frames:
            self.assertEqual(int(h["x-bytes"]), len(body))
            self.assertTrue(body.startswith(b"\xff\xd8"))

        # --- resume after a "WiFi drop": only frames after the given index, then the end part
        mid = idx[len(idx) // 2]
        with urllib.request.urlopen(c.base + f"/scan/live?since={mid}", timeout=10) as r:
            again = [p for p in read_parts(r, b"trovascan") if p[0]["content-type"] == "image/jpeg"]
        self.assertEqual([int(h["x-frame-index"]) for h, _ in again], [i for i in idx if i > mid])

        # --- back to the default profile
        code, st = c.json("GET", "/status.json")
        self.assertEqual((st["width"], st["height"], st["framerate"], st["quality"]), (1600, 900, 10, 50))
        self.assertEqual(st["scan"]["state"], "done")
        self.assertEqual(c.json("POST", "/scan/stop")[1]["state"], "done")         # stop again = same summary

        # --- files on the Pi
        code, lst = c.json("GET", "/scan/list")
        self.assertIn(sid, [s["id"] for s in lst["scans"]])
        code, jpg = c.req("GET", f"/scan/{sid}/frame/0")
        self.assertEqual(code, 200)
        self.assertTrue(jpg.startswith(b"\xff\xd8"))
        code, meta = c.json("GET", f"/scan/{sid}/meta.json")
        self.assertEqual(meta["locked"]["shutter_us"], 8000)
        self.assertGreater(meta["summary"]["camera_metadata"]["frames"], 0)
        code, csv = c.req("GET", f"/scan/{sid}/frames.csv")
        self.assertEqual(len(csv.decode().strip().splitlines()) - 1, summary["frames"])
        code, bins = c.req("GET", f"/scan/{sid}/bins.csv")
        self.assertEqual(len(bins.decode().strip().splitlines()) - 1, summary["bins"])
        self.assertEqual(c.req("GET", f"/scan/{sid}/../../etc/passwd")[0], 404)
        code, raw = c.req("GET", f"/scan/{sid}.tar", timeout=60)
        self.assertEqual(code, 200)
        with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
            names = tar.getnames()
        self.assertEqual(len([n for n in names if n.endswith(".jpg")]), summary["saved"])
        self.assertIn(f"{sid}/meta.json", names)

        # --- delete
        self.assertEqual(c.json("DELETE", f"/scan/{sid}")[0], 200)
        self.assertNotIn(sid, [s["id"] for s in c.json("GET", "/scan/list")[1]["scans"]])
        self.assertEqual(c.req("GET", f"/scan/{sid}/frame/0")[0], 404)

        # --- max_s stops the scan by itself
        code, info = c.json("POST", "/scan/start", {"max_s": 2, "ae_settle_s": 0.3})
        self.assertEqual(code, 200, info)
        time.sleep(4.5)
        st = c.json("GET", "/status.json")[1]
        self.assertEqual(st["scan"]["state"], "done")
        self.assertIn("max_s", st["scan"]["reason"])
        c.json("DELETE", f"/scan/{info['scan_id']}")

        # --- heat: protection change aborts the scan, and a new scan is refused
        code, info = c.json("POST", "/scan/start", {"max_s": 60, "ae_settle_s": 0.3})
        self.assertEqual(code, 200, info)
        with open(self.temp_file, "w") as f:
            f.write("200000")
        for _ in range(30):
            st = c.json("GET", "/status.json")[1]
            if st["scan"]["state"] != "recording":
                break
            time.sleep(0.5)
        self.assertEqual(st["scan"]["state"], "aborted")
        self.assertIn("protection", st["scan"]["reason"])
        meta = c.json("GET", f"/scan/{info['scan_id']}/meta.json")[1]
        self.assertTrue(meta["summary"]["protection_events"])
        self.assertEqual(c.json("POST", "/scan/start", {})[0], 503)
        with open(self.temp_file, "w") as f:
            f.write("50000")

    def test_low_disk_refused(self):
        proc, c = self.start_server("--scan-min-free-mb", str(10 ** 9))
        try:
            code, body = c.json("POST", "/scan/start", {})
            self.assertEqual(code, 507, body)
            self.assertIn("free", body["error"])
        finally:
            proc.terminate()
            proc.wait(10)


if __name__ == "__main__":
    unittest.main()
