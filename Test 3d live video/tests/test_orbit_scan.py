"""scan3d.orbit_scan: how the located object reaches the dry run, and arm.json for the scan (no arm, no camera)."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scan3d import locate as locate_mod  # noqa: E402
from scan3d import orbit_scan as osc  # noqa: E402

LOC = {"center_mm": [-0.5, -477.9, -13.5], "boxes": [[-168, -448, -104, -384, 12.0]], "calib": "camera_robot.json"}


class OrbitScan(unittest.TestCase):
    def run_main(self, argv):
        seen = {}

        def fake_dry_run(a, dist, loc=None):
            seen["args"] = osc.orbit_args(a, dist, loc)
            return None                                       # stop before any motion

        with mock.patch.object(locate_mod, "locate", return_value=LOC) as loc, \
                mock.patch.object(osc, "dry_run", side_effect=fake_dry_run), \
                contextlib.redirect_stdout(io.StringIO()):
            code = osc.main(argv)
        return code, loc, seen["args"]

    def test_find_disk_reaches_locate_and_the_target(self):
        code, loc, args = self.run_main(["--ip", "fake", "--find", "disk", "--diameter", "44", "--obj-height", "55"])
        self.assertEqual(code, 2)
        self.assertEqual(loc.call_args.args[1], 55.0)
        self.assertEqual(loc.call_args.kwargs, {"find": "disk", "diameter_mm": 44.0, "remembered_cubes": True})
        self.assertIn("--target=-0.5,-477.9,-13.5", args)
        self.assertIn("--box=-168,-448,-104,-384,12", args)

    def test_default_is_the_red_blob(self):
        _, loc, _ = self.run_main(["--ip", "fake"])
        self.assertEqual(loc.call_args.kwargs, {"find": "red", "diameter_mm": None, "remembered_cubes": True})

    def test_no_cubes_drops_the_remembered_boxes(self):
        _, loc, _ = self.run_main(["--ip", "fake", "--no-cubes"])
        self.assertFalse(loc.call_args.kwargs["remembered_cubes"])

    def test_locate_says_when_the_camera_does_not_look_down(self):
        import cv2
        import numpy as np

        jpeg = cv2.imencode(".jpg", np.full((900, 1600, 3), 120, np.uint8))[1].tobytes()
        with mock.patch.object(locate_mod, "read_still", return_value=(np.zeros(5), jpeg)), \
                self.assertRaisesRegex(RuntimeError, "does not look down"):
            locate_mod.locate("fake", 55.0, find="disk", diameter_mm=44.0, say=lambda *_: None)

    def test_attach_arm_writes_arm_json(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            log = d / "log.json"
            log.write_text(json.dumps({"q0": [0] * 5, "start_pc": 100.0, "dwell_s": [0.6],
                                       "views": [{"i": 0, "t_arrive": 2.0, "q": [1] * 5, "t": 2.6}]}))
            with contextlib.redirect_stdout(io.StringIO()):
                p = osc.attach_arm(d, log, str(ROOT / "calib" / "camera_robot.json"))
                self.assertIsNone(osc.attach_arm(d, d / "missing.json", str(ROOT / "calib" / "camera_robot.json")))
            arm = json.loads(p.read_text())
        self.assertEqual(arm["start_source"], "orbit_here")
        self.assertIn("T_flange_camera", arm["calibration"])

    def test_find_disk_needs_a_diameter(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            osc.main(["--ip", "fake", "--find", "disk"])


if __name__ == "__main__":
    unittest.main()
