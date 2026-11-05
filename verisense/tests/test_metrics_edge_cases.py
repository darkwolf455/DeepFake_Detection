"""
Edge-case tests for metrics input validation guards.

Covers: empty inputs, 2-D inputs, mismatched lengths, invalid/fractional/NaN/Inf
labels, NaN/Inf predictions, out-of-range probabilities, compute_min_dcf p_target
bounds, compute_eer identical-score handling, compute_ece n_bins, and
compute_binary_metrics threshold parameter.

Does not reference ASVspoof or any dataset outside the four approved ones
(GenImage, FaceForensics++, Celeb-DF, FakeAVCeleb).

Run with:
    python -m unittest verisense/tests/test_metrics_edge_cases.py
"""

import unittest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common.metrics import (
    compute_binary_metrics,
    compute_ece,
    compute_eer,
    compute_min_dcf,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_Y_TRUE = np.array([0, 0, 1, 1])
_Y_PROB = np.array([0.1, 0.2, 0.8, 0.9])


# ---------------------------------------------------------------------------
# compute_binary_metrics
# ---------------------------------------------------------------------------


class TestBinaryMetricsValidation(unittest.TestCase):
    # --- empty / shape ---

    def test_empty_arrays_raise(self):
        with self.assertRaises(ValueError):
            compute_binary_metrics(np.array([]), np.array([]))

    def test_empty_y_true_raises(self):
        with self.assertRaises(ValueError):
            compute_binary_metrics(np.array([]), np.array([0.5]))

    def test_empty_y_prob_raises(self):
        with self.assertRaises(ValueError):
            compute_binary_metrics(np.array([0, 1]), np.array([]))

    def test_2d_y_true_raises(self):
        with self.assertRaises(ValueError):
            compute_binary_metrics(np.array([[0, 1], [0, 1]]), _Y_PROB)

    def test_2d_y_prob_raises(self):
        with self.assertRaises(ValueError):
            compute_binary_metrics(_Y_TRUE, np.array([[0.1, 0.9], [0.2, 0.8]]))

    def test_mismatched_lengths_raise(self):
        with self.assertRaises(ValueError):
            compute_binary_metrics(np.array([0, 1, 1]), np.array([0.1, 0.9]))

    # --- label content ---

    def test_string_labels_raise(self):
        with self.assertRaises((TypeError, ValueError)):
            compute_binary_metrics(np.array(["0", "1", "0", "1"]), _Y_PROB)

    def test_bool_labels_raise(self):
        # numpy bool_ dtype is not np.number; must be rejected like strings
        with self.assertRaises(TypeError):
            compute_binary_metrics(np.array([True, False, True, True]), _Y_PROB)

    def test_fractional_label_raises(self):
        # 0.5 must not silently truncate to 0
        with self.assertRaises(ValueError):
            compute_binary_metrics(
                np.array([0.0, 0.5, 1.0, 1.0]), np.array([0.1, 0.5, 0.8, 0.9])
            )

    def test_label_2_raises(self):
        with self.assertRaises(ValueError):
            compute_binary_metrics(
                np.array([0, 1, 2, 1]), np.array([0.1, 0.8, 0.9, 0.7])
            )

    def test_negative_label_raises(self):
        with self.assertRaises(ValueError):
            compute_binary_metrics(
                np.array([-1, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])
            )

    def test_nan_label_raises(self):
        with self.assertRaises(ValueError):
            compute_binary_metrics(
                np.array([0.0, float("nan"), 1.0, 1.0]),
                np.array([0.1, 0.5, 0.8, 0.9]),
            )

    def test_inf_label_raises(self):
        with self.assertRaises(ValueError):
            compute_binary_metrics(
                np.array([0.0, float("inf"), 1.0, 1.0]),
                np.array([0.1, 0.5, 0.8, 0.9]),
            )

    # --- prediction content ---

    def test_nan_prediction_raises(self):
        with self.assertRaises(ValueError):
            compute_binary_metrics(_Y_TRUE, np.array([0.1, float("nan"), 0.8, 0.9]))

    def test_pos_inf_prediction_raises(self):
        with self.assertRaises(ValueError):
            compute_binary_metrics(_Y_TRUE, np.array([0.1, float("inf"), 0.8, 0.9]))

    def test_neg_inf_prediction_raises(self):
        with self.assertRaises(ValueError):
            compute_binary_metrics(_Y_TRUE, np.array([0.1, float("-inf"), 0.8, 0.9]))

    def test_prob_below_zero_raises(self):
        with self.assertRaises(ValueError):
            compute_binary_metrics(_Y_TRUE, np.array([-0.01, 0.2, 0.8, 0.9]))

    def test_prob_above_one_raises(self):
        with self.assertRaises(ValueError):
            compute_binary_metrics(_Y_TRUE, np.array([0.1, 0.2, 0.8, 1.01]))

    # --- threshold parameter ---

    def test_nan_threshold_raises(self):
        with self.assertRaises(ValueError):
            compute_binary_metrics(_Y_TRUE, _Y_PROB, threshold=float("nan"))

    def test_inf_threshold_raises(self):
        with self.assertRaises(ValueError):
            compute_binary_metrics(_Y_TRUE, _Y_PROB, threshold=float("inf"))

    def test_string_threshold_raises(self):
        with self.assertRaises(TypeError):
            compute_binary_metrics(_Y_TRUE, _Y_PROB, threshold="0.5")

    def test_bool_threshold_raises(self):
        with self.assertRaises(TypeError):
            compute_binary_metrics(_Y_TRUE, _Y_PROB, threshold=True)

    def test_threshold_below_zero_raises(self):
        with self.assertRaises(ValueError):
            compute_binary_metrics(_Y_TRUE, _Y_PROB, threshold=-0.1)

    def test_threshold_above_one_raises(self):
        with self.assertRaises(ValueError):
            compute_binary_metrics(_Y_TRUE, _Y_PROB, threshold=1.1)

    def test_threshold_zero_accepted(self):
        # threshold=0.0 is a valid boundary
        m = compute_binary_metrics(_Y_TRUE, _Y_PROB, threshold=0.0)
        self.assertIn("auc", m)

    def test_threshold_one_accepted(self):
        # threshold=1.0 is a valid boundary
        m = compute_binary_metrics(_Y_TRUE, _Y_PROB, threshold=1.0)
        self.assertIn("auc", m)

    # --- normal operation is unchanged ---

    def test_perfect_predictions(self):
        m = compute_binary_metrics(_Y_TRUE, _Y_PROB)
        self.assertEqual(m["auc"], 1.0)
        self.assertEqual(m["accuracy"], 1.0)
        self.assertEqual(m["f1"], 1.0)
        self.assertEqual(m["tp"], 2)
        self.assertEqual(m["tn"], 2)

    def test_single_class_returns_sentinel(self):
        m = compute_binary_metrics(
            np.array([0, 0, 0, 0]), np.array([0.1, 0.2, 0.3, 0.4])
        )
        self.assertEqual(m["auc"], 0.5)
        self.assertEqual(m["average_precision"], 0.0)

    def test_exact_boundary_probs_accepted(self):
        # 0.0 and 1.0 are valid probability values
        m = compute_binary_metrics(np.array([0, 1]), np.array([0.0, 1.0]))
        self.assertEqual(m["auc"], 1.0)

    def test_integer_labels_as_floats_accepted(self):
        # 0.0 and 1.0 pass the exact-binary check (equal to 0 and 1)
        m = compute_binary_metrics(np.array([0.0, 0.0, 1.0, 1.0]), _Y_PROB)
        self.assertEqual(m["auc"], 1.0)


# ---------------------------------------------------------------------------
# compute_eer
# ---------------------------------------------------------------------------


class TestEERValidation(unittest.TestCase):
    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            compute_eer(np.array([]), np.array([]))

    def test_mismatched_lengths_raise(self):
        with self.assertRaises(ValueError):
            compute_eer(np.array([0, 1, 1]), np.array([0.1, 0.9]))

    def test_2d_y_true_raises(self):
        with self.assertRaises(ValueError):
            compute_eer(np.array([[0, 1], [0, 1]]), np.array([0.1, 0.9, 0.2, 0.8]))

    def test_string_labels_raise(self):
        with self.assertRaises((TypeError, ValueError)):
            compute_eer(np.array(["0", "1"]), np.array([0.1, 0.9]))

    def test_fractional_label_raises(self):
        with self.assertRaises(ValueError):
            compute_eer(np.array([0.0, 0.5, 1.0, 1.0]), np.array([0.1, 0.5, 0.8, 0.9]))

    def test_label_2_raises(self):
        with self.assertRaises(ValueError):
            compute_eer(np.array([0, 1, 2, 1]), np.array([0.1, 0.8, 0.9, 0.7]))

    def test_nan_label_raises(self):
        with self.assertRaises(ValueError):
            compute_eer(
                np.array([0.0, float("nan"), 1.0, 1.0]),
                np.array([0.1, 0.5, 0.8, 0.9]),
            )

    def test_nan_score_raises(self):
        with self.assertRaises(ValueError):
            compute_eer(_Y_TRUE, np.array([0.1, float("nan"), 0.8, 0.9]))

    def test_inf_score_raises(self):
        with self.assertRaises(ValueError):
            compute_eer(_Y_TRUE, np.array([0.1, float("inf"), 0.8, 0.9]))

    def test_identical_finite_scores_correct_result(self):
        # All scores equal: EER must be 0.5, threshold must be finite (not inf).
        eer, threshold = compute_eer(
            np.array([0, 0, 1, 1]), np.array([0.5, 0.5, 0.5, 0.5])
        )
        self.assertAlmostEqual(eer, 0.5, places=6)
        self.assertAlmostEqual(threshold, 0.5, places=6)
        self.assertTrue(
            np.isfinite(threshold),
            msg="EER threshold must be finite for identical scores, got %s" % threshold,
        )

    def test_single_class_returns_sentinel(self):
        eer, threshold = compute_eer(
            np.array([0, 0, 0, 0]), np.array([0.1, 0.2, 0.3, 0.4])
        )
        self.assertEqual(eer, 0.5)
        self.assertEqual(threshold, 0.5)

    def test_raw_logits_accepted(self):
        # EER does not restrict scores to [0, 1]
        eer, threshold = compute_eer(
            np.array([0, 0, 1, 1]), np.array([-2.0, -1.0, 1.0, 2.0])
        )
        self.assertGreaterEqual(eer, 0.0)
        self.assertLessEqual(eer, 1.0)

    def test_normal_operation_unaffected(self):
        eer, threshold = compute_eer(
            np.array([0] * 50 + [1] * 50),
            np.array([0.1] * 50 + [0.9] * 50),
        )
        self.assertLess(eer, 0.05)
        self.assertTrue(np.isfinite(threshold))


# ---------------------------------------------------------------------------
# compute_min_dcf
# ---------------------------------------------------------------------------


class TestMinDCFValidation(unittest.TestCase):
    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            compute_min_dcf(np.array([]), np.array([]))

    def test_mismatched_lengths_raise(self):
        with self.assertRaises(ValueError):
            compute_min_dcf(np.array([0, 1, 1]), np.array([0.1, 0.9]))

    def test_string_labels_raise(self):
        with self.assertRaises((TypeError, ValueError)):
            compute_min_dcf(np.array(["0", "1"]), np.array([0.1, 0.9]))

    def test_fractional_label_raises(self):
        with self.assertRaises(ValueError):
            compute_min_dcf(
                np.array([0.0, 0.5, 1.0, 1.0]), np.array([0.1, 0.5, 0.8, 0.9])
            )

    def test_label_2_raises(self):
        with self.assertRaises(ValueError):
            compute_min_dcf(np.array([0, 1, 2, 1]), np.array([0.1, 0.8, 0.9, 0.7]))

    def test_nan_score_raises(self):
        with self.assertRaises(ValueError):
            compute_min_dcf(_Y_TRUE, np.array([0.1, float("nan"), 0.8, 0.9]))

    def test_p_target_zero_raises(self):
        with self.assertRaises(ValueError):
            compute_min_dcf(_Y_TRUE, _Y_PROB, p_target=0.0)

    def test_p_target_one_raises(self):
        with self.assertRaises(ValueError):
            compute_min_dcf(_Y_TRUE, _Y_PROB, p_target=1.0)

    def test_p_target_negative_raises(self):
        with self.assertRaises(ValueError):
            compute_min_dcf(_Y_TRUE, _Y_PROB, p_target=-0.1)

    def test_p_target_above_one_raises(self):
        with self.assertRaises(ValueError):
            compute_min_dcf(_Y_TRUE, _Y_PROB, p_target=1.5)

    def test_p_target_string_raises(self):
        with self.assertRaises(TypeError):
            compute_min_dcf(_Y_TRUE, _Y_PROB, p_target="0.05")

    def test_p_target_bool_raises(self):
        with self.assertRaises(TypeError):
            compute_min_dcf(_Y_TRUE, _Y_PROB, p_target=True)

    def test_single_class_returns_one(self):
        dcf = compute_min_dcf(np.array([0, 0, 0, 0]), np.array([0.1, 0.2, 0.3, 0.4]))
        self.assertEqual(dcf, 1.0)

    def test_raw_logits_accepted(self):
        dcf = compute_min_dcf(np.array([0, 0, 1, 1]), np.array([-2.0, -1.0, 1.0, 2.0]))
        self.assertGreaterEqual(dcf, 0.0)

    def test_normal_operation_unaffected(self):
        dcf = compute_min_dcf(
            np.array([0] * 20 + [1] * 20),
            np.linspace(0.1, 0.9, 40),
        )
        self.assertGreaterEqual(dcf, 0.0)
        self.assertLessEqual(dcf, 1.0)


# ---------------------------------------------------------------------------
# compute_ece
# ---------------------------------------------------------------------------


class TestECEValidation(unittest.TestCase):
    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            compute_ece(np.array([]), np.array([]))

    def test_mismatched_lengths_raise(self):
        with self.assertRaises(ValueError):
            compute_ece(np.array([0, 1, 1]), np.array([0.1, 0.9]))

    def test_2d_y_prob_raises(self):
        with self.assertRaises(ValueError):
            compute_ece(_Y_TRUE, np.array([[0.1, 0.9], [0.2, 0.8]]))

    def test_string_labels_raise(self):
        with self.assertRaises((TypeError, ValueError)):
            compute_ece(np.array(["0", "1", "0", "1"]), _Y_PROB)

    def test_fractional_label_raises(self):
        with self.assertRaises(ValueError):
            compute_ece(np.array([0.0, 0.5, 1.0, 1.0]), np.array([0.1, 0.5, 0.8, 0.9]))

    def test_label_2_raises(self):
        with self.assertRaises(ValueError):
            compute_ece(np.array([0, 1, 2, 1]), np.array([0.1, 0.8, 0.9, 0.7]))

    def test_nan_prediction_raises(self):
        with self.assertRaises(ValueError):
            compute_ece(_Y_TRUE, np.array([0.1, float("nan"), 0.8, 0.9]))

    def test_inf_prediction_raises(self):
        with self.assertRaises(ValueError):
            compute_ece(_Y_TRUE, np.array([0.1, float("inf"), 0.8, 0.9]))

    def test_prob_below_zero_raises(self):
        with self.assertRaises(ValueError):
            compute_ece(_Y_TRUE, np.array([-0.01, 0.2, 0.8, 0.9]))

    def test_prob_above_one_raises(self):
        with self.assertRaises(ValueError):
            compute_ece(_Y_TRUE, np.array([0.1, 0.2, 0.8, 1.01]))

    def test_n_bins_zero_raises(self):
        with self.assertRaises(ValueError):
            compute_ece(_Y_TRUE, _Y_PROB, n_bins=0)

    def test_n_bins_negative_raises(self):
        with self.assertRaises(ValueError):
            compute_ece(_Y_TRUE, _Y_PROB, n_bins=-1)

    def test_n_bins_float_raises(self):
        with self.assertRaises(TypeError):
            compute_ece(_Y_TRUE, _Y_PROB, n_bins=10.5)

    def test_n_bins_string_raises(self):
        with self.assertRaises(TypeError):
            compute_ece(_Y_TRUE, _Y_PROB, n_bins="10")

    def test_n_bins_bool_raises(self):
        # bool is a subclass of int in Python, but True/False as bin counts is nonsensical
        with self.assertRaises(TypeError):
            compute_ece(_Y_TRUE, _Y_PROB, n_bins=True)

    def test_normal_operation_unaffected(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.25, 0.25, 0.75, 0.75])
        ece, details = compute_ece(y_true, y_prob, n_bins=4)
        # bin 0 ([0, 0.25]): conf=0.25, acc=0.0 -> |0.25-0|*0.5 = 0.125
        # bin 2 ((0.5, 0.75]): conf=0.75, acc=1.0 -> |0.75-1|*0.5 = 0.125
        # total ECE = 0.25
        self.assertAlmostEqual(ece, 0.25, places=6)
        self.assertEqual(len(details["bin_confidences"]), 4)

    def test_n_bins_one_accepted(self):
        ece, details = compute_ece(_Y_TRUE, _Y_PROB, n_bins=1)
        self.assertGreaterEqual(ece, 0.0)
        self.assertEqual(len(details["bin_counts"]), 1)


if __name__ == "__main__":
    unittest.main()
