"""Unit tests for scan3d/reconstruct.py pieces that do not need COLMAP (geometry, scale maths, GLB, renders).
The end-to-end check is the synthetic scan: tests/make_synthetic_scan.py + python -m scan3d.reconstruct.

    .venv\\Scripts\\python -m unittest tests.test_reconstruct -v
"""
import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scan3d import reconstruct as rc  # noqa: E402


def look_at(c, target):
    f = (target - c) / np.linalg.norm(target - c)
    r = np.cross(f, [0, 0, 1.0])
    r /= np.linalg.norm(r)
    R = np.stack([r, np.cross(f, r), f])
    return R, -R @ c


class GeometryTests(unittest.TestCase):
    def test_orbit_center(self):
        target = np.array([12.0, -5.0, 3.0])
        cams, axes = [], []
        for az in np.linspace(-2, 2, 9):
            c = target + 300 * np.array([np.cos(az) * 0.7, np.sin(az) * 0.7, 0.7])
            R, _ = look_at(c, target)
            cams.append(c)
            axes.append(R.T @ [0, 0, 1.0])
        np.testing.assert_allclose(rc.orbit_center(np.array(cams), np.array(axes)), target, atol=1e-6)

    def test_rotation_to_z(self):
        for n in ([0, 0, 1.0], [0, 0, -1.0], [1, 2, 3.0], [-0.2, 0.9, 0.1]):
            R = rc.rotation_to_z(np.array(n))
            np.testing.assert_allclose(R @ (np.array(n) / np.linalg.norm(n)), [0, 0, 1], atol=1e-9)
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)

    def test_quaternion(self):
        R = rc.quat_to_R([np.cos(0.3), 0, 0, np.sin(0.3)])           # 0.6 rad about z
        np.testing.assert_allclose(R[:2, :2], [[np.cos(0.6), -np.sin(0.6)], [np.sin(0.6), np.cos(0.6)]], atol=1e-9)

    def test_triangulate_and_distortion(self):
        """Corners of a 39 mm square seen by 3 cameras with OPENCV distortion are triangulated back exactly."""
        K, dist = rc.intrinsics("OPENCV", [1350, 1360, 820, 616, -0.08, 0.02, 0.001, -0.001])
        sq = np.array([[0, 0, 0], [39, 0, 0], [39, 39, 0], [0, 39, 0]], float)
        views = []
        for az in (-0.5, 0.0, 0.6):
            c = np.array([200 * np.cos(az), 200 * np.sin(az), 250.0])
            R, t = look_at(c, np.array([20, 20, 0.0]))
            px, _ = cv2.projectPoints(sq, cv2.Rodrigues(R)[0], t, K, dist)
            und = cv2.undistortPoints(px, K, dist).reshape(-1, 2)
            views.append((np.column_stack([R, t]), und))
        X = [rc.triangulate([(P, u[k]) for P, u in views])[0] for k in range(4)]
        np.testing.assert_allclose(X, sq, atol=1e-3)


class ExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import open3d as o3d
        m = o3d.geometry.TriangleMesh.create_box(80, 50, 35)
        m.translate([-40, -25, 0])
        m.paint_uniform_color([0.8, 0.2, 0.2])
        m.compute_vertex_normals()
        m.compute_triangle_normals()
        cls.mesh = m

    def test_glb_is_valid_and_in_metres_y_up(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.glb"
            rc.write_glb(p, self.mesh)
            data = p.read_bytes()
        magic, version, length = struct.unpack("<III", data[:12])
        self.assertEqual((magic, version, length), (0x46546C67, 2, len(data)))
        jlen, jtype = struct.unpack("<II", data[12:20])
        self.assertEqual(jtype, 0x4E4F534A)
        gltf = json.loads(data[20:20 + jlen])
        pos = gltf["accessors"][0]
        self.assertEqual(pos["count"], len(self.mesh.vertices))
        np.testing.assert_allclose(pos["max"], [0.04, 0.035, 0.025], atol=1e-6)   # x, up (=z), -y in metres
        np.testing.assert_allclose(pos["min"], [-0.04, 0.0, -0.025], atol=1e-6)
        self.assertEqual(gltf["accessors"][3]["count"], 3 * len(self.mesh.triangles))
        blen, btype = struct.unpack("<II", data[20 + jlen:28 + jlen])
        self.assertEqual(btype, 0x004E4942)
        self.assertEqual(blen, gltf["buffers"][0]["byteLength"])

    def test_renders(self):
        with tempfile.TemporaryDirectory() as d:
            paths = rc.render_views(self.mesh, Path(d) / "renders", {"L": 80, "W": 50, "H": 35}, 400)
            self.assertEqual(len(paths), 4)
            img = cv2.imread(str(Path(d) / "renders" / "front.png"))
            self.assertEqual(img.shape, (400, 400, 3))
            self.assertGreater((img[:, :, 2].astype(int) - img[:, :, 1] > 60).sum(), 1000)     # red box drawn


if __name__ == "__main__":
    unittest.main()
