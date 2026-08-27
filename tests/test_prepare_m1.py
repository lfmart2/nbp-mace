import sys
import unittest
from pathlib import Path

from ase import Atoms

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from prepare_m1 import composition_matrix, select_splits  # noqa: E402


class M1SplitTests(unittest.TestCase):
    def test_split_rules_do_not_overlap(self):
        frames = []
        for group, count in (("qe_clean_relax", 9), ("qe_h_relax_a", 39), ("qe_h_relax_b", 36)):
            for index in range(count):
                atoms = Atoms("H")
                atoms.info.update(source_group=group, source_frame=index)
                frames.append(atoms)
        splits = select_splits(frames)
        identities = {
            name: {(a.info["source_group"], a.info["source_frame"]) for a in items}
            for name, items in splits.items()
        }
        names = list(identities)
        for i, first in enumerate(names):
            for second in names[i + 1 :]:
                self.assertFalse(identities[first] & identities[second])
        self.assertEqual({name: len(items) for name, items in splits.items()}, {"train": 32, "valid": 10, "test_internal": 6, "test_overlap": 36})

    def test_composition_matrix(self):
        matrix = composition_matrix([Atoms("HNbP"), Atoms("NbP")], [1, 15, 41])
        self.assertEqual(matrix.tolist(), [[1.0, 1.0, 1.0], [0.0, 1.0, 1.0]])


if __name__ == "__main__":
    unittest.main()
