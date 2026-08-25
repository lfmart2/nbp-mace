import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "src" / "audit_scf.py"
SPEC = importlib.util.spec_from_file_location("audit_scf", MODULE_PATH)
audit_scf = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(audit_scf)


class AuditScfTests(unittest.TestCase):
    def test_parse_poscar_direct_and_species_order(self):
        text = """test
1.0
2 0 0
0 2 0
0 0 10
H Nb
1 1
Selective dynamics
Direct
0.5 0.5 0.8 F F F
0.0 0.0 0.7 T T T
"""
        parsed = audit_scf.parse_poscar_text(text)
        self.assertEqual(parsed["species"], ["H", "Nb"])
        self.assertEqual(parsed["positions_A"][0], [1.0, 1.0, 8.0])

    def test_parse_outcar_uses_final_energy_and_force_block(self):
        text = """
 number of ions     NIONS = 2
 ISPIN  = 2
 LNONCOLLINEAR = F
 LSORBIT = F
 ENCUT = 400.0 eV
 EDIFF = 0.1E-06
 ISMEAR = 1; SIGMA = 0.20
 free energy TOTEN = -1.0 eV
 energy without entropy = -0.9 energy(sigma->0) = -0.8
 TOTAL-FORCE (eV/Angst)
 -------------------------------------------------------------------
 0 0 0  0.1 0.2 0.3
 0 0 1 -0.1 -0.2 -0.3
 -------------------------------------------------------------------
 free energy TOTEN = -2.0 eV
 energy without entropy = -1.9 energy(sigma->0) = -1.8
 General timing and accounting informations for this job:
"""
        parsed = audit_scf.parse_outcar_text(text)
        self.assertTrue(parsed["completed"])
        self.assertEqual(parsed["energy_sigma0_eV"], -1.8)
        self.assertEqual(parsed["force_count"], 2)
        self.assertAlmostEqual(parsed["max_force_eV_per_A"], 0.3741657387)


if __name__ == "__main__":
    unittest.main()

