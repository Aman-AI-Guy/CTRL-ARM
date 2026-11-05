"""Phase 2 tests on a PC: trova_cam scan methods, the receiver (reconnect, backfill) and the capture CLI,
against the real stream.py server with tests/fake_rpicam_vid.py as the camera.

    .venv\\Scripts\\python -m unittest tests.test_capture -v
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scan3d.receiver import Receiver  # noqa: E402
from tests.test_stream_scan import FAKE, STREAM_DIR, free_port  # noqa: E402
from trova_cam import CameraError, TrovaCam  # noqa: E402


class DroppingCam(TrovaCam):
    """The first live connection breaks after 3 frames, like a WiFi drop."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.calls = 0

    def scan_live(self, since=-1, scan_id=None, timeout=20):
        self.calls += 1
        live = super().scan_live(since, scan_id, timeout)
        if self.calls > 1:
            return live
        inner_next = live.__next__
        count = [0]

        def broken_next():
            if count[0] == 3:
                live.close()
                raise CameraError("simulated WiFi drop")
            count[0] += 1
            return inner_next()

        class Wrapper:
            def __enter__(self_):
                return self_

            def __exit__(self_, *e):
                live.close()

            def __iter__(self_):
                return self_

            def __next__(self_):
                return broken_next()

            end = property(lambda self_: live.end)

            def close(self_):
                live.close()
        return Wrapper()


class NoLiveCam(TrovaCam):
    def scan_live(self, *a, **kw):
        raise CameraError("/scan/live: HTTP 404 no such scan in memory")


class CaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="trova-capture-test-")
        cls.port = free_port()
        temp = os.path.join(cls.tmp, "temp")
        with open(temp, "w") as f:
            f.write("50000")
        env = dict(os.environ, TROVA_RPICAM_VID=json.dumps([sys.executable, FAKE]), TROVA_TEMP_FILE=temp,
                   PYTHONIOENCODING="utf-8")
        log = open(os.path.join(cls.tmp, "server.log"), "w", encoding="utf-8")
        cls.proc = subprocess.Popen([sys.executable, "-u", os.path.join(STREAM_DIR, "stream.py"), "--port",
                                     str(cls.port), "--scans-dir", os.path.join(cls.tmp, "pi-scans"),
                                     "--idle-timeout", "5"], env=env, stdout=log, stderr=log)
        log.close()
        cls.cam = TrovaCam("127.0.0.1", cls.port)
        for _ in range(50):
            try:
                cls.cam.status()
                break
            except CameraError:
                time.sleep(0.1)

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()
        cls.proc.wait(10)
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_trova_cam_errors_and_clock(self):
        offset, rtt = self.cam.clock_offset()
        self.assertLess(abs(offset), 0.5)
        with self.assertRaises(CameraError) as e:
            self.cam.scan_frame("nope", 0)
        self.assertIn("404", str(e.exception))
        with self.assertRaises(CameraError) as e:
            self.cam.scan_start(bogus=1)
        self.assertIn("unknown scan parameter", str(e.exception))
        with self.assertRaises(CameraError):
            TrovaCam("127.0.0.1", free_port(), timeout=1).status()

    def test_receiver_reconnects_without_duplicates(self):
        cam = DroppingCam("127.0.0.1", self.port)
        info = cam.scan_start(ae_settle_s=0.3, max_s=30)
        out = os.path.join(self.tmp, "rx1")
        rx = Receiver(cam, info["scan_id"], out, 0.0, reconnect_delay_s=0.2, log=lambda *_: None).start()
        time.sleep(4)
        summary = cam.scan_stop()
        rep = rx.finish(20)
        self.assertEqual(rx.reconnects, 1)
        self.assertEqual(rep["missing_after_backfill"], 0)
        self.assertEqual(rep["bins_on_pi"], summary["bins"])
        self.assertEqual(rep["received_live"] + rep["backfilled"], summary["bins"])
        self.assertEqual(len(os.listdir(os.path.join(out, "images"))), summary["bins"])
        self.assertIsNotNone(rep["end"])
        rx.write_csv()
        cam.scan_delete(info["scan_id"])

    def test_backfill_when_live_is_unavailable(self):
        info = self.cam.scan_start(ae_settle_s=0.3, max_s=30)
        time.sleep(2)
        summary = self.cam.scan_stop()
        out = os.path.join(self.tmp, "rx2")
        rx = Receiver(NoLiveCam("127.0.0.1", self.port), info["scan_id"], out, 0.0, log=lambda *_: None).start()
        rep = rx.finish(5)
        self.assertEqual(rep["received_live"], 0)
        self.assertEqual(rep["backfilled"], summary["bins"])
        self.assertEqual(rep["missing_after_backfill"], 0)
        self.cam.scan_delete(info["scan_id"])

    def test_capture_cli_static_full(self):
        data = os.path.join(self.tmp, "data")
        r = subprocess.run([sys.executable, "-m", "scan3d.capture", "--host", "127.0.0.1", "--set",
                            f"pi.port={self.port}", "--set", f"paths.data_dir={json.dumps(data)}", "--set",
                            "capture.ae_settle_s=0.5", "--duration", "3", "--full", "--delete-on-pi",
                            "--note", "unit test"], cwd=ROOT, capture_output=True, text=True, timeout=120,
                           env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        scans = os.listdir(os.path.join(data, "scans"))
        self.assertEqual(len(scans), 1)
        out = os.path.join(data, "scans", scans[0])
        with open(os.path.join(out, "capture.json")) as f:
            cap = json.load(f)
        s, rep = cap["pi_summary"], cap["receiver"]
        self.assertEqual(cap["note"], "unit test")
        self.assertTrue(cap["static"])
        self.assertEqual(rep["missing_after_backfill"], 0)
        self.assertEqual(len(os.listdir(os.path.join(out, "images"))), s["bins"])
        self.assertEqual(cap["full"]["frames"], s["saved"])
        for name in ("meta.json", "frames.csv", "bins.csv", "camera_metadata.json"):
            self.assertTrue(os.path.isfile(os.path.join(out, "pi", name)), name)
        self.assertGreater(cap["timings"]["images_in_motion_window"], 0)
        self.assertNotIn(scans[0], [x["id"] for x in self.cam.scan_list()])        # --delete-on-pi
        with open(os.path.join(out, "received.csv")) as f:
            rows = f.read().strip().splitlines()
        self.assertEqual(len(rows) - 1, s["bins"])
        self.assertNotIn(",,", rows[1])                                            # capture time for every image


if __name__ == "__main__":
    unittest.main()
