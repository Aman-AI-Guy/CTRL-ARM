"""The kit on its own (no robot, no other project files): kinematics, CAD shapes, plans, the fake arm.

    python -m unittest discover -s tests -v        # from the kit folder; ~20 s
"""

import builtins
import contextlib
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
sys.path.insert(0, str(KIT / "model"))
sys.path.insert(0, str(KIT / "tests"))

import build_robot_model as brm  # noqa: E402
import orbit_here as oh  # noqa: E402
from fake_arm import FakeArm, matrix_rpy  # noqa: E402
from robot_geometry import SELF_PAIRS, RobotGeometry  # noqa: E402

GEO = RobotGeometry.load()
FREE = dict(table_z=-1e9, table_margin=0, obj_xy=(1e4, 1e4), obj_radius=0, obj_top=0, obj_margin=0)


def axis(az_deg, el_deg=0.0):
    a, e = math.radians(az_deg), math.radians(el_deg)
    return np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])


def start_pose(ox, oy, axis_az=90.0, aim_error_deg=0.0, cad=True):
    """Joints with the camera 432 mm up the view line from the object's mid height (ox, oy, 13): straight down, or
    off by aim_error_deg as when aimed by eye. cad=True: the CAD camera (perpendicular to the flange, turned axis_az
    on it); False: the old guess (offset 0,0,60, axis az/-10)."""
    if cad:
        ax = axis(axis_az)
        cam = oh.Camera(GEO.camera(ax)[0], ax)
    else:
        cam = oh.Camera([0, 0, 60.0], axis(axis_az, -10))
    e = math.radians(aim_error_deg)
    view = np.array([math.sin(e), 0.0, -math.cos(e)])
    pos = np.array([ox, oy, 13.0]) - view * 432.0
    turn = GEO.head_turn(cam.axis)
    for s in oh.generic_seeds(pos):
        q = oh.solve_view(cam, pos, view, s, scale=300, tol_mm=0.05, tol_deg=0.01)
        if q is not None and q[1] >= -60 and not GEO.collisions(oh.fk_frames(q), turn, **FREE):
            return q
    raise AssertionError("no start pose")


def args(q0, *extra):
    return [f"--q0={','.join(f'{v:.3f}' for v in q0)}", "--start-az", "180", "--direction", "1", *extra]


def plan_for(q0, *extra):
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "plan.json"
        with contextlib.redirect_stdout(io.StringIO()) as log:
            rc = oh.main(args(q0, "--out", str(out), "--log", str(Path(d) / "log.json"), *extra))
        return rc, json.loads(out.read_text()), log.getvalue()


class Kinematics(unittest.TestCase):
    def test_zero_pose_matches_controller(self):
        f = oh.fk([0, 0, 0, 0, 0])
        np.testing.assert_allclose(f[:3, 3], [207, 0, 112], atol=1e-6)      # xArm 5 at all-zero joints
        np.testing.assert_allclose(f[:3, 2], [0, 0, -1], atol=1e-9)         # flange pointing down

    def test_pitch_joints_are_parallel(self):
        for q in ([10, 20, -50, 30, 0], [-30, -10, -80, 90, 45]):        # q2 + q3 + q4 = 0: flange still down
            np.testing.assert_allclose(oh.fk(q)[:3, 2], [0, 0, -1], atol=1e-9)

    def test_rpy_round_trip(self):
        np.testing.assert_allclose(matrix_rpy(oh.rpy_matrix(170, -20, 35)), (170, -20, 35), atol=1e-9)


class Geometry(unittest.TestCase):
    def test_spheres_cover_the_real_meshes(self):
        shapes = [(n, brm.load_stl(brm.MESHES / f"{n}.stl") * 1000.0, GEO.shapes[n]) for n in GEO.data["links"]]
        shapes.append(("head", brm.cad_to_head(brm.load_stl(brm.HEAD_STL)), GEO.shapes["head"]))
        for name, tris, (c, r) in shapes:
            v = np.unique(np.round(tris.reshape(-1, 3), 1), axis=0)
            gap = (np.linalg.norm(v[:, None, :] - c[None], axis=2) - r[None]).min(1)
            self.assertLess(gap.max(), 0.5, f"{name}: a mesh vertex sticks out of the spheres by {gap.max():.1f} mm")

    def test_head_from_cad(self):
        np.testing.assert_allclose(GEO.cam_center, [42.0, 0.0, 40.07], atol=0.01)
        np.testing.assert_allclose(GEO.cam_axis, [1, 0, 0])
        off, dev = GEO.camera([0, -1, 0])                 # head turned -90 deg on the flange
        np.testing.assert_allclose(off, [0, -42.0, 40.07], atol=1e-6)
        self.assertLess(dev, 1e-6)
        _, dev = GEO.camera(axis(0, -12))                 # measured axis 12 deg off the CAD one
        self.assertAlmostEqual(dev, 12.0, places=6)
        self.assertEqual(GEO.turn_angle(axis(97)), 97.0)
        self.assertEqual(GEO.turn_angle(axis(97), snap_deg=45), 90.0)

    def test_head_against_forearm_is_caught(self):
        turn = GEO.head_turn([1, 0, 0])
        for j4 in (-60, 0, 60, 120):
            self.assertEqual(GEO.collisions(oh.fk_frames([0, 20, -70, j4, 0]), turn, **FREE), [], j4)
        self.assertIn("self head/link3", GEO.collisions(oh.fk_frames([0, 20, -70, 180, 0]), turn, **FREE))

    def test_no_self_hits_on_ordinary_poses(self):
        turn = GEO.head_turn([1, 0, 0])
        for q in ([0, -20, -60, 80, 0], [30, 10, -60, 50, 90], [-40, 30, -90, 60, -120]):
            fr = oh.fk_frames(q)
            for a, b in SELF_PAIRS:
                self.assertGreater(GEO.clearance(fr, turn, a, b), 5.0, (q, a, b))


class Planning(unittest.TestCase):
    def test_plan_aims_at_object_cad(self):
        rc, plan, log = plan_for(start_pose(400, 100))
        self.assertEqual(rc, 0, log)
        self.assertEqual(plan["collision_model"], "cad")
        tgt = np.array(plan["target_mm"])
        np.testing.assert_allclose(tgt, [400, 100, 13], atol=1.0)          # camera 432 mm up the line, mid height
        self.assertAlmostEqual(plan["table_z_mm"], -4.5, delta=1.0)        # mouse 35 mm tall
        np.testing.assert_allclose(plan["camera_offset_flange"], GEO.camera(plan["camera_axis_flange"])[0], atol=1e-6)
        cam = oh.Camera(plan["camera_offset_flange"], plan["camera_axis_flange"])
        sc = oh.Scene(tgt[:2], 140, -4.5 + 35, -4.5, table_margin=15, obj_margin=20, geometry=GEO,
                      head_turn=GEO.head_turn(plan["camera_axis_flange"]))
        ok = [p for p in plan["poses"] if p["q"]]
        self.assertEqual(len(ok), 18, log)
        prev = None
        for p in ok:
            q = np.array(p["q"])
            self.assertTrue(np.all(q >= oh.LIMITS[:, 0] - 1e-6) and np.all(q <= oh.LIMITS[:, 1] + 1e-6))
            self.assertEqual(oh.collisions(q, cam, sc), [])
            if prev is not None:
                self.assertEqual(oh.path_collisions(prev, q, cam, sc), [])
            prev = q
            c, v = cam.ray(q)
            d = tgt - c
            self.assertAlmostEqual(float(np.linalg.norm(d)), 350, delta=2)
            self.assertLess(math.degrees(math.acos(float(v @ d) / np.linalg.norm(d))), 1.0)
            self.assertAlmostEqual(p["el"], 50.0)                            # default ring: 40 deg from vertical

    def test_aim_by_eye_error_is_absorbed_by_cad(self):
        """Start pose aimed 8 deg off straight down, object on the crosshair: the CAD axis follows the real view line
        (object found), the old 'straight down' assumption misses it by about 432 * sin(8 deg) = 60 mm."""
        q0 = start_pose(420, -60, aim_error_deg=8.0)
        rc, plan, log = plan_for(q0, "--max-views", "1")
        self.assertEqual(rc, 0, log)
        self.assertLess(np.linalg.norm(np.array(plan["target_mm"]) - [420, -60, 13]), 2.0, log)
        self.assertIn("8.0 deg away from straight down", log)
        _, plan_old, _ = plan_for(q0, "--max-views", "1", "--axis-from", "start")
        self.assertGreater(np.linalg.norm(np.array(plan_old["target_mm"]) - [420, -60, 13]), 40)

    def test_no_cad_keeps_the_old_model(self):
        rc, plan, log = plan_for(start_pose(400, 100, axis_az=0, cad=False), "--no-cad")
        self.assertEqual(rc, 0, log)
        self.assertEqual(plan["collision_model"], "capsules")
        np.testing.assert_allclose(plan["camera_offset_flange"], [0, 0, 60])
        np.testing.assert_allclose(plan["target_mm"], [400, 100, 13], atol=1.0)

    def test_keep_out_box(self):
        q, ax = np.array([0, 20, -70, 60, 0.0]), [1.0, 0, 0]
        cam = oh.Camera(GEO.camera(ax)[0], ax)
        sc = oh.Scene((1e4, 1e4), 0, 0, -1e9, table_margin=0, obj_margin=20, geometry=GEO, head_turn=GEO.head_turn(ax))
        self.assertEqual(oh.collisions(q, cam, sc), [])
        w = GEO.world(oh.fk_frames(q), sc.head_turn)
        c, r = (np.concatenate(v) for v in zip(*(w[n] for n in oh.MOVING)))
        k = int(np.argmin(c[:, 2] - r))
        low, xy = c[k, 2] - r[k], c[k, :2]

        def under(gap):                                    # 20 x 20 mm box whose top is `gap` below the arm
            return np.array([xy[0] - 10, xy[1] - 10, low - gap - 50]), np.array([xy[0] + 10, xy[1] + 10, low - gap])
        sc.boxes = [under(10)]
        self.assertEqual(oh.collisions(q, cam, sc), ["box 0"])
        sc.boxes = [under(30)]
        self.assertEqual(oh.collisions(q, cam, sc), [])
        rc, plan, log = plan_for(start_pose(400, 100), "--max-views", "1", "--box=-300,-300,-250,-250,60")
        self.assertEqual(rc, 0, log)
        self.assertIn("keep-out box 0    x -300..-250  y -300..-250  up to z 60", log)

    def test_start_pose_with_flange_down_is_refused(self):
        with self.assertRaises(SystemExit):
            plan_for(np.array([0, 0, -60, 60, 0.0]))       # flange straight down: the camera looks sideways


class FakeArmRun(unittest.TestCase):
    def run_main(self, fake, answer, *extra):
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(oh, "connect", lambda ip: fake), \
                mock.patch.object(builtins, "input", lambda *_: answer), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            code = oh.main(["--ip", "fake", "--go", "--dwell", "0", "--start-az", "0", "--out", str(Path(d) / "p.json"),
                            "--log", str(Path(d) / "l.json"), *extra])
            log = json.loads((Path(d) / "l.json").read_text()) if (Path(d) / "l.json").exists() else None
        return code, out.getvalue(), log

    def test_go_drives_fake_arm_and_returns(self):
        q0 = start_pose(380, -80)
        fake = FakeArm(q0)
        code, out, log = self.run_main(fake, "y", "--max-views", "3", "--direction", "-1")
        self.assertEqual(code, 0, out)
        self.assertIn("kinematics OK", out)
        self.assertEqual(len(fake.moves), 4)                                # 3 views + back to the start
        np.testing.assert_allclose(fake.moves[-1], q0, atol=1e-6)
        self.assertEqual(len(log["views"]), 3)
        self.assertIn("start_pc", log)
        for v, q in zip(log["views"], fake.moves):                          # rest window + joints read there
            self.assertLessEqual(v["t_arrive"], v["t"] + 1e-3)
            np.testing.assert_allclose(v["q_read"], q, atol=1e-3)

    def test_go_refuses_without_yes(self):
        fake = FakeArm(start_pose(400, 0))
        code, out, _ = self.run_main(fake, "", "--direction", "1")
        self.assertEqual(code, 1, out)
        self.assertEqual(fake.moves, [])


if __name__ == "__main__":
    unittest.main()
