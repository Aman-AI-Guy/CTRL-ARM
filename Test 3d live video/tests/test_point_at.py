"""scan3d.point_at + scan3d.grok_find: Grok's box, the can's position in robot mm, the two-stage plan and the stage
checks. No network, no arm, no camera: Grok is mocked, the planner runs offline from typed joints (--q0)."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scan3d import grok_find, hover_check, point_at  # noqa: E402
from scan3d.locate import CALIB, Calibration  # noqa: E402

Q_START = [-93.544184, -0.2407, -43.002889, -43.494429, 48.565851]     # start pose of the 2026-11-05 cup hover
CAN_XY = (-0.5, -478.0)                                                 # where that cup stood
CAN_H = 134.0
Q_HOVER = [-90.1, 23.9, -41.6, -72.7, 43.9]                           # planned stage 2: lens 150 mm above the can


def grok_reply(result: dict) -> mock.MagicMock:
    r = mock.MagicMock()
    r.read.return_value = json.dumps({"output": [{"type": "message", "content": [
        {"type": "output_text", "text": json.dumps(result)}]}]}).encode()
    r.__enter__.return_value = r
    return r


def answer(box, found=True, question=None):
    return {"found": found, "label": "red bull can", "box_2d": box, "confidence": 0.9, "reason": "blue-silver can",
            "clarifying_question": question}


class GrokFind(unittest.TestCase):
    def test_request_and_box_in_pixels(self):
        with mock.patch.object(grok_find.urllib.request, "urlopen",
                               return_value=grok_reply(answer([400, 500, 600, 700]))) as up:
            r = grok_find.find_box(b"\xff\xd8jpeg", "red bull can", (1600, 900), key="k", model="grok-test")
        req = up.call_args[0][0]
        body = json.loads(req.data)
        self.assertEqual(req.full_url, grok_find.API_URL)
        self.assertEqual(req.get_header("Authorization"), "Bearer k")
        self.assertEqual(body["model"], "grok-test")
        self.assertEqual(body["text"]["format"]["schema"], grok_find.FIND_SCHEMA)
        content = body["input"][0]["content"]
        self.assertTrue(content[0]["image_url"].startswith("data:image/jpeg;base64,"))
        self.assertIn('"red bull can"', content[1]["text"])
        np.testing.assert_allclose(r["box_px"], [640, 450, 960, 630])
        np.testing.assert_allclose(r["pixel"], [800, 540])

    def test_not_found_question_and_no_key(self):
        for result, words in ((answer(None, found=False), "did not find"),
                              (answer([1, 2, 3, 4], question="the left or the right can?"), "left or the right")):
            with mock.patch.object(grok_find.urllib.request, "urlopen", return_value=grok_reply(result)):
                with self.assertRaisesRegex(grok_find.GrokFindError, words):
                    grok_find.find_box(b"x", "red bull can", (1600, 900), key="k")
        with tempfile.TemporaryDirectory() as d, mock.patch.dict("os.environ", {"XAI_API_KEY": ""}), \
                mock.patch.object(grok_find, "ENV_FILE", Path(d) / ".env"):
            with self.assertRaisesRegex(grok_find.GrokFindError, "--click"):
                grok_find.find_box(b"x", "red bull can", (1600, 900))

    def test_env_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / ".env"
            p.write_text("XAI_API_KEY= \nTROVA_MODEL=grok-4.7          # comment\n")
            with mock.patch.dict("os.environ", {"XAI_API_KEY": "", "TROVA_MODEL": ""}):
                self.assertEqual(grok_find.env_value("XAI_API_KEY", p), "")
                self.assertEqual(grok_find.env_value("TROVA_MODEL", p), "grok-4.7")


class PointAt(unittest.TestCase):
    def setUp(self):
        self.cal = Calibration.load(CALIB)

    def can_pixels(self, xy=CAN_XY, q=Q_START):
        z = self.cal.table_z_at(*xy)
        return {k: self.cal.robot_to_pixel([xy[0], xy[1], z + h], q) for k, h in
                (("bottom", 0.0), ("middle", CAN_H / 2), ("top", CAN_H))}

    def test_box_centre_gives_the_can_axis(self):
        px = self.can_pixels()
        top, mid = point_at.can_target(self.cal, px["middle"], Q_START, CAN_H)
        np.testing.assert_allclose(top[:2], CAN_XY, atol=1.0)
        self.assertAlmostEqual(top[2], self.cal.table_z_at(*top[:2]) + CAN_H, places=6)
        self.assertAlmostEqual(mid[2], self.cal.table_z_at(*mid[:2]) + CAN_H / 2, places=6)

    def test_two_stage_command(self):
        a = SimpleNamespace(ip="10.0.0.5", q0=None, above=300, height=134, radius=60, speed=10, dwell=2)
        cmd = [str(c) for c in hover_check.orbit_cmd(a, "c.json", [1, 2, 3], [], Path("r"), "u", go=False,
                                                       dists=[300, 150])]
        self.assertEqual(cmd[cmd.index("--tilts") + 1], "0,0")
        self.assertEqual(cmd[cmd.index("--orbit-dist") + 1], "300,150")
        self.assertNotIn("--go", cmd)
        self.assertNotIn("--pause", cmd)
        go = [str(c) for c in hover_check.orbit_cmd(a, "c.json", [1, 2, 3], [], Path("r"), "u", go=True,
                                                      dists=[300, 150], pause=True)]
        self.assertEqual(go[-2:], ["--pause", "--go"])
        one = [str(c) for c in hover_check.orbit_cmd(a, "c.json", [1, 2, 3], [], Path("r"), "u", go=False)]
        self.assertEqual((one[one.index("--tilts") + 1], one[one.index("--orbit-dist") + 1]), ("0", "300"))
        a.ip = None
        a.q0 = Q_START
        off = [str(c) for c in hover_check.orbit_cmd(a, "c.json", [1, 2, 3], [], Path("r"), "u", go=False)]
        self.assertTrue(off[3].startswith("--q0=-93.5442"))
        self.assertNotIn("--ip", off)

    def test_offline_dry_run_plans_both_stages(self):
        px = self.can_pixels()["middle"]
        img = np.full((900, 1600, 3), 120, np.uint8)
        with tempfile.TemporaryDirectory() as d:
            still = Path(d) / "can.jpg"
            cv2.imwrite(str(still), img)
            argv = ["red bull can", "--still", str(still), "--q0=" + ",".join(map(str, Q_START)),
                    "--pixel", f"{px[0]:.2f},{px[1]:.2f}", "--runs", d]
            with contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(point_at.main(argv), 0)
            run_dir = next(Path(d).glob("*_point"))
            info = json.loads((run_dir / "point_target.json").read_text())
            plan = json.loads((run_dir / "plan.json").read_text())
            self.assertTrue((run_dir / "before_annotated.jpg").exists())
        self.assertIn("nothing moved", out.getvalue())
        np.testing.assert_allclose(info["target_mm"][:2], CAN_XY, atol=1.0)
        self.assertEqual(len(info["stages"]), 2)
        self.assertEqual([s["name"] for s in info["stages"]], ["AIM", "APPROACH"])
        for s, want in zip(info["stages"], (300, 150)):
            self.assertIsNotNone(s["q"])
            self.assertAlmostEqual(s["lens_above_top_mm"], want, delta=2)
            self.assertLess(s["lens_off_axis_mm"], 2)
            self.assertGreater(min(g for g, _ in s["min_gap_on_the_way_mm"].values()), 0)
        self.assertEqual(plan["reachable"], 2)

    def test_offline_refuses_go_and_a_too_close_approach(self):
        with contextlib.redirect_stderr(io.StringIO()):
            for extra in (["--go"], ["--near", "100"], ["--near", "300"]):
                with self.assertRaises(SystemExit):
                    point_at.main(["--still", "x.jpg", "--q0=" + ",".join(map(str, Q_START)), *extra])

    def test_stage_checks_with_grok(self):
        target = np.array([CAN_XY[0], CAN_XY[1], self.cal.table_z + CAN_H])
        seen = self.cal.robot_to_pixel(target + [4.0, -2.0, 0.0], Q_HOVER)
        box = [seen[0] - 60, seen[1] - 60, seen[0] + 60, seen[1] + 60]
        norm = [round(box[0] / 1.6), round(box[1] / 0.9), round(box[2] / 1.6), round(box[3] / 0.9)]
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            cv2.imwrite(str(d / "000.jpg"), np.full((900, 1600, 3), 120, np.uint8))
            rows = [{"view": "start", "file": "start.jpg", "q": Q_START}, {"view": 0, "file": "000.jpg", "q": Q_HOVER}]
            (d / "snapshots.json").write_text(json.dumps({"snapshots": {"rows": rows}}))
            info = {"target_mm": target.tolist(), "find": "grok", "prompt": "red bull can", "height_mm": CAN_H}
            with mock.patch.dict("os.environ", {"XAI_API_KEY": "k"}), \
                    mock.patch.object(grok_find.urllib.request, "urlopen", return_value=grok_reply(answer(norm))), \
                    contextlib.redirect_stdout(io.StringIO()):
                out = point_at.check_stages(d, self.cal, info)
            self.assertTrue((d / "stage1_check.jpg").exists())
            saved = json.loads((d / "stage_check.json").read_text())
        self.assertEqual(len(out), 1)
        self.assertEqual(saved[0]["stage"], 1)
        np.testing.assert_allclose(out[0]["offset_mm"], [4.0, -2.0], atol=1.0)


if __name__ == "__main__":
    unittest.main()
