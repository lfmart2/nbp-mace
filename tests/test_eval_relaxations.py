from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from eval_relaxations import metric  # noqa: E402


class MetricTests(unittest.TestCase):
    def test_metric_known_values(self):
        result = metric(np.asarray([-1.0, 1.0]))
        self.assertEqual(result, {"mae": 1.0, "rmse": 1.0, "max_abs": 1.0})


if __name__ == "__main__":
    unittest.main()
