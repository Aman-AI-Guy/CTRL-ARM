"""scan3d.armpose + its use in scan3d.reconstruct: scale and robot frame from the arm's rest poses (no COLMAP run)."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scan3d import armpose  # noqa: E402
from scan3d import reconstruct as rc  # noqa: E402

CAL = armpose.Calibration.load(ROOT / "calib" / "camera_robot.json")
Q_CUP = [[-118.44, 17.97, -45.99, -43.04, 79.54], [-113.67, 24.67, -62.35, -24.38, 73.32],   # the 05:04 cup orbit
         [-107.86, 32.44, -78.31, -10.3, 65.9], [-101.43, 39.38, -91.78, -0.17, 57.88],
         [-94.68, 43.98, -100.5, 5.76, 49.56], [-87.8, 44.75, -101.96, 6.71, 41.14],
         [-80.99, 41.38, -95.6, 2.47, 32.76], [-74.42, 35.08, -83.5, -6.25, 24.61]]


def rotation(seed):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def colmap_images(qs, s0, R0, t0, noise_mm=0.0, seed=0):
    """What COLMAP would give for cameras at these joints, in a frame where robot = s0 R0 x + t0."""
    rng = np.random.default_rng(seed)
    images = {}
    for k, q in enumerate(qs):
        T = CAL.camera_pose(q)
        c = R0.T @ (T[:3, 3] + rng.normal(0, noise_mm, 3) - t0) / s0
        Rw = (R0.T @ T[:3, :3]).T                                     # COLMAP R: world -> camera
        images[f"{k:06d}.jpg"] = {"R": Rw, "t": -Rw @ c, "center": c}
    return images


class ArmPose(unittest.TestCase):
    def test_umeyama_recovers_a_similarity(self):
        A = np.random.default_rng(1).normal(size=(10, 3))
        R0, t0 = rotation(2), np.array([10.0, -400, 30])
        s, R, t = armpose.umeyama(A, (53.2 * (R0 @ A.T)).T + t0)
        self.assertAlmostEqual(s, 53.2, places=6)
        np.testing.assert_allclose(R, R0, atol=1e-9)
        np.testing.assert_allclose(t, t0, atol=1e-6)

    def test_rest_frames_from_new_and_old_logs(self):
        arm = {"start_pc": 1000.0, "dwell_s": [0.6], "views": [
            {"i": 0, "t_arrive": 2.0, "t": 2.62, "q": [0] * 5, "q_read": [1] * 5},     # new log: arrival + joints read
            {"i": 1, "t": 5.0, "q": [2] * 5}]}                                         # old log: last 0.6 s
        times = [1001.95, 1002.05, 1002.15, 1002.3, 1002.45, 1002.7, 1004.3, 1004.55, 1004.8, 1005.1]
        with tempfile.TemporaryDirectory() as d:
            rows = "index,bin,nbytes,source,capture_pc,recv_pc,latency_s\n" + "".join(
                f"{k},{k},1,live,{t},{t + 0.3},0.3\n" for k, t in enumerate(times)) + "99,99,1,backfill,,,\n"
            (Path(d) / "received.csv").write_text(rows)
            fr = armpose.rest_frames(d, arm, margin_s=0.1, per_view=2)
        self.assertEqual([(f["name"], f["view"]) for f in fr],
                         [("000003.jpg", 0), ("000004.jpg", 0), ("000008.jpg", 1), ("000007.jpg", 1)])
        self.assertEqual(fr[0]["q"], [1.0] * 5)                                        # q_read wins over q
        self.assertEqual(fr[2]["q"], [2.0] * 5)

    def test_write_falls_back_to_the_capture_start(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "capture.json").write_text(json.dumps({"timings": {"motion_start_pc": 123.5}}))
            (d / "log.json").write_text(json.dumps({"q0": [0] * 5, "views": [{"i": 0, "t": 3.0, "q": [0] * 5}]}))
            armpose.write(d, d / "log.json", ROOT / "calib" / "camera_robot.json", dwell_s=0.6)
            arm = armpose.load(d)
        self.assertEqual((arm["start_pc"], arm["start_source"], arm["dwell_s"]),
                         (123.5, "capture.json motion_start_pc", [0.6]))

    def test_align_scale_frame_and_a_bad_frame(self):
        R0, t0 = rotation(3), np.array([-20.0, -470, 150])
        images = colmap_images(Q_CUP, 53.5, R0, t0, noise_mm=0.5)
        images["000005.jpg"]["center"] = images["000005.jpg"]["center"] + 30.0 / 53.5     # COLMAP got one wrong
        frames = [{"name": f"{k:06d}.jpg", "view": k, "q": q} for k, q in enumerate(Q_CUP)]
        frames.append({"name": "999999.jpg", "view": 9, "q": Q_CUP[0]})                   # not registered
        al = armpose.align(images, frames, CAL)
        self.assertTrue(al["ok"], al)
        self.assertAlmostEqual(al["scale"] / 53.5, 1.0, delta=0.003)
        self.assertEqual(al["dropped"], ["000005.jpg"])
        self.assertEqual((al["registered"], al["used"]), (8, 7))
        self.assertLess(al["residual_mm"]["median"], 1.0)
        self.assertLess(al["rotation_deg"]["max"], 0.5)
        T = np.array(al["T_colmap_to_robot"])
        c = images["000002.jpg"]["center"]
        np.testing.assert_allclose(T[:3, :3] @ c + T[:3, 3], CAL.camera_pose(Q_CUP[2])[:3, 3], atol=2.0)

    def test_align_refuses_too_few_views(self):
        images = colmap_images(Q_CUP[:3], 50.0, np.eye(3), np.zeros(3))
        frames = [{"name": f"{k:06d}.jpg", "view": k, "q": q} for k, q in enumerate(Q_CUP[:3])] * 2
        al = armpose.align(images, frames, CAL)
        self.assertFalse(al["ok"])
        self.assertIn("from 3 views", al["reason"])

    def test_select_images_keeps_the_rest_frames(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "src"
            src.mkdir()
            for k in range(20):
                (src / f"{k:06d}.jpg").write_bytes(b"x")
            keep = ["000001.jpg", "000002.jpg", "000013.jpg"]
            names = rc.select_images(src, Path(d) / "dst", 6, keep=keep)
        self.assertEqual(len(names), 6)
        self.assertTrue(set(keep) <= set(names), names)

    def test_robot_block_maps_the_mesh_into_the_robot_frame(self):
        s, R0, t0 = 53.5, rotation(4), np.array([0.0, -474, -41])
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = s * R0, t0
        M = np.eye(4)                                       # some scaled-COLMAP -> mesh rigid transform
        M[:3, :3], M[:3, 3] = rotation(5), [3.0, -7, 12]
        rb = rc.robot_block({"T_colmap_to_robot": T.tolist(), "scale": s}, M, 55.0, CAL.data)
        x_c = np.array([0.1, -0.3, 0.2])                    # a point in COLMAP units
        x_m = M[:3, :3] @ (s * x_c) + M[:3, 3]
        Tm = np.array(rb["T_mesh_to_robot"])
        np.testing.assert_allclose(Tm[:3, :3] @ x_m + Tm[:3, 3], s * R0 @ x_c + t0, atol=1e-3)
        base = np.linalg.inv(M) @ [0, 0, 0, 1.0]
        np.testing.assert_allclose(rb["object_base_mm"], R0 @ base[:3] + t0, atol=0.06)
        self.assertEqual(rb["table_z_mm"]["calibration"], round(CAL.table_z_at(*rb["object_base_mm"][:2]), 1))


if __name__ == "__main__":
    unittest.main()
