"""scan3d.hover_check: red-object finder and the hover-still analysis, on synthetic pictures (no arm, no camera)."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scan3d import hover_check as hc  # noqa: E402

Q_HOVER = [-92.77, 47.77, -51.14, -88.52, 45.06]          # the real hover of 2026-11-05 03:11
TARGET = np.array([-14.2, -473.6, 0.9])


def picture(centre_px, axes=(250, 380)):
    img = np.full((900, 1600, 3), 120, np.uint8)
    cv2.ellipse(img, (int(round(centre_px[0])), int(round(centre_px[1]))), axes, 0, 0, 360, (20, 20, 220), -1)
    return img


class HoverCheck(unittest.TestCase):
    def test_red_blob_centre_and_cut(self):
        b = hc.red_blob(picture((800, 450)))
        np.testing.assert_allclose(b["pixel"], [800, 450], atol=0.5)
        self.assertEqual(b["cut"], [])
        self.assertEqual(hc.red_blob(picture((800, 300)))["cut"], ["top"])
        self.assertIsNone(hc.red_blob(np.full((900, 1600, 3), 120, np.uint8)))

    def test_white_disk_ignores_the_dimmer_rim(self):
        img = np.full((900, 1600, 3), 110, np.uint8)
        cv2.circle(img, (790, 540), 90, (185, 185, 185), -1)           # the cup's rim further down, dimmer, offset
        cv2.circle(img, (812, 536), 75, (240, 240, 240), -1)           # its upturned bottom, the expected size
        d = hc.find_object.__globals__["white_disk"](img, (0.75 * 77, 1.3 * 77))
        np.testing.assert_allclose(d["pixel"], [812, 536], atol=1.0)
        self.assertAlmostEqual(d["radius_px"], 75, delta=2)
        label = np.full((900, 1600, 3), 110, np.uint8)
        cv2.rectangle(label, (700, 450), (810, 560), (240, 240, 240), -1)   # a square label: not a disk
        self.assertIsNone(hc.find_object.__globals__["white_disk"](label, (50, 100)))

    def test_analyse_reports_the_offset_in_mm(self):
        cal = hc.Calibration.load(hc.CALIB)
        seen = cal.robot_to_pixel(TARGET + [5.0, -3.0, 0.0], Q_HOVER)     # object 5 mm +x, 3 mm -y off the target
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            cv2.imwrite(str(d / "000.jpg"), picture(seen))
            (d / "hover_target.json").write_text(json.dumps({"target_mm": TARGET.tolist(), "height_mm": 35.0}))
            rows = [{"view": "start", "file": "start.jpg", "q": Q_HOVER}, {"view": 0, "file": "000.jpg", "q": Q_HOVER}]
            (d / "snapshots.json").write_text(json.dumps({"snapshots": {"rows": rows}}))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(hc.analyse(d, cal), 0)
            out = json.loads((d / "hover_check.json").read_text())
        np.testing.assert_allclose(out["offset_mm"], [5.0, -3.0], atol=0.5)
        self.assertEqual(out["cut"], [])


if __name__ == "__main__":
    unittest.main()
