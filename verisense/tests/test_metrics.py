"""
Unit tests for evaluation metrics: EER, minDCF, AUC, and ECE.
"""

import numpy as np
import pytest

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.common.metrics import (
    compute_binary_metrics,
    compute_ece,
    compute_eer,
    compute_min_dcf,
)


def test_binary_metrics_perfect():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.05, 0.1, 0.9, 0.95])
    m = compute_binary_metrics(y_true, y_prob)

    assert m["accuracy"] == 1.0
    assert m["auc"] == 1.0
    assert m["f1"] == 1.0
    assert m["tp"] == 2
    assert m["tn"] == 2


def test_eer_symmetric():
    # Symmetric separable predictions should yield EER near 0
    y_true = np.array([0] * 50 + [1] * 50)
    y_score = np.array([0.1] * 50 + [0.9] * 50)

    eer, threshold = compute_eer(y_true, y_score)
    assert eer < 0.05
    assert 0.1 <= threshold <= 0.9


def test_min_dcf_bounds():
    y_true = np.array([0] * 20 + [1] * 20)
    y_score = np.linspace(0.1, 0.9, 40)

    dcf = compute_min_dcf(y_true, y_score)
    assert 0.0 <= dcf <= 1.0


def test_ece_calibration():
    # With n_bins=4, bin boundaries are [0, 0.25, 0.5, 0.75, 1.0].
    # Bin 0 [0.0, 0.25]: both real samples (y_true=0) land here.
    #   conf=0.25, acc=0.0 → |0.25 - 0| * 0.5 = 0.125
    # Bin 2 (0.5, 0.75]: both fake samples (y_true=1) land here.
    #   conf=0.75, acc=1.0 → |0.75 - 1| * 0.5 = 0.125
    # ECE = 0.125 + 0.125 = 0.25
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.25, 0.25, 0.75, 0.75])

    ece, details = compute_ece(y_true, y_prob, n_bins=4)
    assert abs(ece - 0.25) < 1e-6
    assert len(details["bin_confidences"]) == 4
