import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.utils import EarlyStopping


class EarlyStoppingTests(unittest.TestCase):
    def test_improving_scores_continue_training(self):
        for mode, scores in [("max", [0.5, 0.6, 0.7]), ("min", [0.7, 0.6, 0.5])]:
            with self.subTest(mode=mode):
                stop = EarlyStopping(patience=2, mode=mode)
                self.assertEqual([stop(score) for score in scores], [False, False, False])

    def test_plateau_stops_at_patience(self):
        for mode in ["max", "min"]:
            with self.subTest(mode=mode):
                stop = EarlyStopping(patience=2, mode=mode)
                self.assertEqual([stop(0.5) for _ in range(3)], [False, False, True])

    def test_improvement_resets_patience(self):
        stop = EarlyStopping(patience=2)
        scores = [0.5, 0.4, 0.6, 0.55, 0.55]
        self.assertEqual([stop(score) for score in scores], [False, False, False, False, True])

    def test_changes_below_min_delta_do_not_reset_patience(self):
        stop = EarlyStopping(patience=2, min_delta=0.01)
        self.assertEqual([stop(score) for score in [0.5, 0.504, 0.509]], [False, False, True])


if __name__ == "__main__":
    unittest.main()
