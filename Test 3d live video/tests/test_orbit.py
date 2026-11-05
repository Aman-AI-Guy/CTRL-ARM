"""scan3d.orbit without the arm: kinematics, IK, fit on synthetic taught poses, plan, run on the fake arm.

    .venv\\Scripts\\python -m unittest tests.test_orbit -v
"""
import io
import json
import math
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scan3d import orbit  # noqa: E402
from scan3d.config import load_config  # noqa: E402


def quiet(fn, *a, **k):
    with redirect_stdout(io.StringIO()) as out:
        rc = fn(*a, **k)
    return rc, out.getvalue()


class Kinematics(unittest.TestCase):
    def test_zero_pose_matches_controller(self):
        f = orbit.fk([0, 0, 0, 0, 0])
        np.testing.assert_allclose(f[:3, 3], [207, 0, 112], atol=1e-6)      # xArm 5 at all-zero joints
        np.testing.assert_allclose(f[:3, 2], [0, 0, -1], atol=1e-9)         # flange pointing down
        roll, pitch, yaw = orbit.matrix_rpy(f[:3, :3])
        self.assertAlmostEqual(abs(roll), 180, places=6)
        self.assertAlmostEqual(pitch, 0, places=6)

    def test_pitch_joints_are_parallel(self):
        # q2 + q3 + q4 = 0 keeps the flange pointing down, whatever the individual angles
        for q in ([10, 20, -50, 30, 0], [-30, -10, -80, 90, 45]):
            np.testing.assert_allclose(orbit.fk(q)[:3, 2], [0, 0, -1], atol=1e-9)

    def test_rpy_round_trip(self):
        r = orbit.rpy_matrix(170, -20, 35)
        np.testing.assert_allclose(orbit.matrix_rpy(r), (170, -20, 35), atol=1e-9)

    def test_ik_round_trip(self):
        cam = orbit.Camera(np.array([0, 0, 60.0]), orbit.axis_from_angles(0, 20))
        rng = np.random.default_rng(3)
        for _ in range(10):
            q = np.array([rng.uniform(-60, 60), rng.uniform(-10, 40), rng.uniform(-90, -30), rng.uniform(0, 60),
                          rng.uniform(-150, 150)])
            p, v = cam.ray(q)
            sol = orbit.solve_view(cam, p, v, q + rng.normal(0, 8, 5))
            self.assertIsNotNone(sol)
            p2, v2 = cam.ray(sol)
            self.assertLess(np.linalg.norm(p2 - p), 1.0)
            self.assertLess(math.degrees(math.acos(min(1.0, float(v2 @ v)))), 0.5)


class Pipeline(unittest.TestCase):
    """simulate -> fit -> plan -> run on the fake arm, through the CLI."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.files = ["--teach", os.path.join(cls.tmp, "teach.json"), "--model", os.path.join(cls.tmp, "model.json"),
                     "--plan", os.path.join(cls.tmp, "plan.json")]

    def cli(self, *args):
        return quiet(orbit.main, [*self.files, *args])

    def load(self, name):
        with open(os.path.join(self.tmp, name), encoding="utf-8") as f:
            return json.load(f)

    def test_1_fit_finds_object_and_camera_axis(self):
        self.assertEqual(self.cli("simulate", "--object", "420,0", "--axis-el", "25")[0], 0)
        self.assertEqual(self.cli("fit")[0], 0)
        fit = self.load("model.json")["fit"]
        self.assertLess(fit["vs_truth"]["target_err_mm"], 5)
        self.assertLess(fit["vs_truth"]["axis_err_deg"], 1.5)
        self.assertLess(fit["rms_miss_mm"], 10)

    def test_2_fit_without_touch_pose(self):
        files = [f.replace("teach.json", "teach_rays.json") if f.endswith("teach.json") else f for f in self.files]
        files = [f.replace("model.json", "model_rays.json") if f.endswith("model.json") else f for f in files]
        quiet(orbit.main, [*files, "simulate", "--object", "380,80", "--axis-el", "15", "--no-touch", "--noise-deg", "0.1"])
        quiet(orbit.main, [*files, "fit"])
        fit = self.load("model_rays.json")["fit"]
        self.assertEqual(fit["target_from"], "rays")
        self.assertLess(fit["vs_truth"]["target_err_mm"], 15)
        self.assertLess(fit["vs_truth"]["axis_err_deg"], 3)

    def test_3_plan_full_orbit_collision_free(self):
        rc, out = self.cli("plan", "--elevations", "auto")
        self.assertEqual(rc, 0, out)
        plan = self.load("plan.json")
        self.assertEqual(plan["summary"]["reachable"], 24, out)
        model, o = plan["model"], orbit.orbit_cfg(load_config())
        cam, sc = orbit.camera_from_model(model), orbit.scene_from_model(model, o)
        qs = [np.array(p["q"]) for p in plan["poses"] if p["q"] is not None]
        for a, b in zip(qs, qs[1:]):
            self.assertEqual(orbit.path_collisions(a, b, cam, sc), [])
            self.assertLessEqual(np.abs(b - a).max(), 60)
        for p in plan["poses"]:         # every view really looks at the object centre
            c, d = np.array(p["camera_mm"]), np.array(p["camera_dir"])
            to_t = sc.target - c
            miss = np.linalg.norm(to_t - d * (d @ to_t))
            self.assertLess(miss, 3.0)
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "plan.png")))

    def test_4_run_on_fake_arm(self):
        plan = self.load("plan.json")
        fake = orbit.FakeArm(q=plan["poses"][0]["q"])
        orig = orbit.connect_arm
        orbit.connect_arm = lambda ip, say=print: fake
        try:
            rc, out = self.cli("run", "--ip", "fake", "--yes", "--dwell", "0")
        finally:
            orbit.connect_arm = orig
        self.assertEqual(rc, 0, out)
        self.assertEqual(len(fake.moves), plan["summary"]["reachable"])
        np.testing.assert_allclose(fake.moves[-1], [p for p in plan["poses"] if p["q"]][-1]["q"], atol=1e-6)

    def test_5_run_refuses_a_colliding_approach(self):
        plan = self.load("plan.json")
        far = np.array(plan["poses"][12]["q"])          # the opposite side of the object: the straight move crosses it
        fake = orbit.FakeArm(q=far)
        model, o = plan["model"], orbit.orbit_cfg(load_config())
        hits = orbit.path_collisions(far, plan["poses"][0]["q"], orbit.camera_from_model(model),
                                     orbit.scene_from_model(model, o))
        orig = orbit.connect_arm
        orbit.connect_arm = lambda ip, say=print: fake
        try:
            rc, out = self.cli("run", "--ip", "fake", "--yes", "--dwell", "0")
        finally:
            orbit.connect_arm = orig
        if hits:
            self.assertEqual(rc, 2, out)
            self.assertEqual(fake.moves, [])
        else:
            self.assertEqual(rc, 0, out)


if __name__ == "__main__":
    unittest.main()
