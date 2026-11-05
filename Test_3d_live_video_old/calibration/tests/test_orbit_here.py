"""The arm kit (arm_pc/) from the project: all its own tests (arm_pc/tests/test_kit.py), plus the check that its
kinematics and shapes are the ones scan3d.orbit uses.

    .venv\\Scripts\\python -m unittest tests.test_orbit_here -v
"""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "arm_pc" / "tests"))
sys.path.insert(0, str(ROOT))

from test_kit import FakeArmRun, Geometry, Kinematics, Planning  # noqa: E402,F401  (run here too)

import orbit_here as oh  # noqa: E402
from scan3d import orbit as ob  # noqa: E402


class SameAsScan3d(unittest.TestCase):
    def test_fk_matches_scan3d(self):
        rng = np.random.default_rng(0)
        for _ in range(10):
            q = rng.uniform(-120, 120, 5)
            np.testing.assert_allclose(oh.fk(q), ob.fk(q), atol=1e-9)

    def test_scan3d_uses_the_kit_geometry(self):
        o = ob.orbit_cfg({"orbit": {"use_cad_geometry": True}})
        geo = ob.cad_geometry(o)
        self.assertEqual(Path(sys.modules[type(geo).__module__].__file__).resolve().parent, ROOT / "arm_pc")


if __name__ == "__main__":
    unittest.main()
