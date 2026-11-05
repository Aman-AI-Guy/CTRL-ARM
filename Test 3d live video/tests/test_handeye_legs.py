"""scan3d.handeye with the cubes moved between legs, a tilted table and ruler answers (synthetic, no arm)."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scan3d import handeye as he  # noqa: E402

A, B, C = 0.012, -0.015, -60.0                       # table z = A x + B y + C: ~1.1 deg tilt, -60 under the base


def top(x, y):
    return (x, y, A * x + B * y + C + 53.0)


LEGS = [{2: top(-110, -460), 4: top(110, -455)},
        {2: top(-60, -380), 4: top(150, -540)}]


class Legs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.images, cls.obs, cls.truth = [], [], []
        for li, mk in enumerate(LEGS):
            centre = np.mean([v for v in mk.values()], axis=0) - [0, 0, 30]
            im, ob, size, s, truth = he.synthetic(seed=li, markers=mk, centre=centre)
            off = len(cls.images)
            cls.images += [dict(r, leg=li) for r in im]
            cls.obs += [(k + off, (li, m), c) for k, m, c in ob]
            cls.truth.append(truth)
        cls.size, cls.s = size, s

    def test_fit_plane(self):
        pts = [top(-100, -400), top(120, -420), top(0, -560), top(60, -380)]
        p = he.fit_plane([(x, y, z - 53) for x, y, z in pts])
        np.testing.assert_allclose([p["a"], p["b"], p["c"]], [A, B, C], atol=1e-9)
        self.assertEqual(he.fit_plane([(0, -400, -34), (100, -400, -36)])["a"], 0.0)     # 2 points: level

    def test_two_legs_give_camera_markers_and_plane(self):
        prob, x, _ = he.solve_all(self.images, self.obs, self.size, self.s, say=lambda *_: None)
        K, dist, T_fc, T_bm = prob.unpack(x)
        t = self.truth[0]
        self.assertLess(np.linalg.norm(T_fc[:3, 3] - t["T_fc"][:3, 3]), 2.0)
        geo = he.marker_geometry(prob, x)
        for li, mk in enumerate(LEGS):
            for m, c in mk.items():
                self.assertLess(np.linalg.norm(geo[(li, m)][0].mean(0) - c), 2.0, (li, m))
        feet = [P.mean(0) - nv * 53.0 for (P, nv) in geo.values() if nv[2] > 0.95]
        p = he.fit_plane(feet)
        self.assertAlmostEqual(p["tilt_deg"], np.degrees(np.arctan(np.hypot(A, B))), delta=0.15)

    def test_ruler_fact_pulls_the_marker_height(self):
        key, z_true = (0, 2), LEGS[0][2][2]
        prob, x, _ = he.solve_all(self.images, self.obs, self.size, self.s, say=lambda *_: None,
                                  ruler=[(key, z_true + 4.0, 0.05)])
        self.assertAlmostEqual(float(prob.unpack(x)[3][key][2, 3]), z_true + 4.0, delta=0.5)


if __name__ == "__main__":
    unittest.main()
