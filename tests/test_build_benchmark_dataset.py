from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from build_benchmark_dataset import geometry_hash  # noqa: E402


class GeometryHashTests(unittest.TestCase):
    def test_geometry_hash_changes_for_finite_displacement(self):
        from ase import Atoms

        first = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[8, 8, 8], pbc=True)
        second = first.copy()
        second.positions[1, 2] += 1e-4
        self.assertNotEqual(geometry_hash(first), geometry_hash(second))

    def test_geometry_hash_ignores_sub_microangstrom_noise(self):
        from ase import Atoms

        first = Atoms("H", positions=[[0.1234561, 0, 0]], cell=[8, 8, 8], pbc=True)
        second = first.copy()
        second.positions[0, 0] += 1e-8
        self.assertEqual(geometry_hash(first), geometry_hash(second))


if __name__ == "__main__":
    unittest.main()
