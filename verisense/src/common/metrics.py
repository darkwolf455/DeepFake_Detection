"""
Evaluation metrics for deepfake detection across images, video sequences, and audio waveforms.
"""

from typing import Dict, Tuple
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def _validate_inputs(
    y_true,
    y_scores,
    *,
    require_prob_range: bool,
    scores_param_name: str = "y_prob",
) -> Tuple[np.ndarray, np.ndarray]:
    """Shared validator for all metric functions.

    Parameters
    ----------
    y_true:
        Binary ground-truth labels. Must be 1-D, numeric (integer or float),
        finite, and contain only exact 0 or 1 values. Boolean arrays and
        string arrays are rejected.
    y_scores:
        Predicted scores or probabilities. Must be 1-D and finite. When
        *require_prob_range* is True, values must lie within [0, 1].
    require_prob_range:
        Pass True for compute_binary_metrics / compute_ece (probabilities).
        Pass False for compute_eer / compute_min_dcf (raw logits accepted).
    scores_param_name:
        Name used in error messages when referring to y_scores.

    Returns
    -------
    (y_true_int, y_scores_float) — validated, cast arrays ready for sklearn.
    """
    # ---- y_true ----
    y_true = np.asarray(y_true)
    if not np.issubdtype(y_true.dtype, np.number):
        raise TypeError(
            "y_true must be numeric (integer or float), got dtype %s. "
            "String and boolean arrays are not supported." % y_true.dtype
        )
    if y_true.ndim != 1:
        raise ValueError(
            "y_true must be 1-dimensional, got shape %s." % (y_true.shape,)
        )
    if len(y_true) == 0:
        raise ValueError("y_true must not be empty.")
    if not np.all(np.isfinite(y_true)):
        raise ValueError("y_true contains NaN or infinite values.")
    not_binary = ~np.isin(y_true, [0, 1])
    if np.any(not_binary):
        bad = np.unique(y_true[not_binary]).tolist()
        raise ValueError(
            "y_true contains values outside {0, 1}: %s. "
            "Only exact binary labels (0 or 1) are supported." % bad
        )

    # ---- y_scores ----
    y_scores = np.asarray(y_scores, dtype=float)
    if y_scores.ndim != 1:
        raise ValueError(
            "%s must be 1-dimensional, got shape %s."
            % (scores_param_name, y_scores.shape)
        )
    if len(y_scores) == 0:
        raise ValueError("%s must not be empty." % scores_param_name)
    if len(y_true) != len(y_scores):
        raise ValueError(
            "y_true and %s must have the same length, got %d and %d."
            % (scores_param_name, len(y_true), len(y_scores))
        )
    if not np.all(np.isfinite(y_scores)):
        raise ValueError(
            "%s contains NaN or infinite values. "
            "Ensure model outputs are finite before computing metrics."
            % scores_param_name
        )
    if require_prob_range:
        lo, hi = float(np.min(y_scores)), float(np.max(y_scores))
        if lo < 0.0 or hi > 1.0:
            raise ValueError(
                "%s values must lie in [0, 1]; got min=%.6g, max=%.6g."
                % (scores_param_name, lo, hi)
            )

    return y_true.astype(int), y_scores


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Computes standard binary classification metrics."""
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise TypeError("threshold must be a float, got %s." % type(threshold).__name__)
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite, got %s." % threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "threshold must be in [0, 1] when y_prob is a probability, got %.6g."
            % threshold
        )
    y_true, y_prob = _validate_inputs(
        y_true, y_prob, require_prob_range=True, scores_param_name="y_prob"
    )
    y_pred = (y_prob >= threshold).astype(int)

    # Avoid zero division when single-class batches occur
    has_two_classes = len(np.unique(y_true)) > 1

    auc = roc_auc_score(y_true, y_prob) if has_two_classes else 0.5
    ap = average_precision_score(y_true, y_prob) if has_two_classes else 0.0
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    return {
        "accuracy": float(acc),
        "auc": float(auc),
        "average_precision": float(ap),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def compute_eer(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    """
    Computes Equal Error Rate (EER) and the operating threshold where FPR == FNR.

    Accepts raw scores or probabilities (not restricted to [0, 1]).
    """
    y_true, y_score = _validate_inputs(
        y_true, y_score, require_prob_range=False, scores_param_name="y_score"
    )

    if len(np.unique(y_true)) < 2:
        return 0.5, 0.5

    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)

    # sklearn always prepends np.inf as the first threshold value
    # (source: thresholds = np.r_[np.inf, thresholds]).
    # Remove it so brentq operates on a fully finite domain.
    finite_mask = np.isfinite(thresholds)
    thresholds = thresholds[finite_mask]
    tpr_f = tpr[finite_mask]
    fpr_f = fpr[finite_mask]
    fnr_f = 1.0 - tpr_f

    # When all scores are identical, sklearn produces exactly one finite threshold
    # after the inf sentinel is removed. brentq and interp1d both require at least
    # two points, so we use the direct fallback instead.
    if len(thresholds) < 2:
        idx = np.nanargmin(np.abs(fpr_f - fnr_f))
        return float((fpr_f[idx] + fnr_f[idx]) / 2.0), float(thresholds[idx])

    # EER is the point where fpr == fnr
    try:
        eer_threshold = brentq(
            lambda x: (
                1.0 - interp1d(thresholds, tpr_f)(x) - interp1d(thresholds, fpr_f)(x)
            ),
            min(thresholds),
            max(thresholds),
        )
        eer = float(interp1d(thresholds, fpr_f)(eer_threshold))
    except Exception:
        # Fallback to absolute difference minimum
        idx = np.nanargmin(np.abs(fpr_f - fnr_f))
        eer = float((fpr_f[idx] + fnr_f[idx]) / 2.0)
        eer_threshold = float(thresholds[idx])

    return float(eer), float(eer_threshold)


def compute_min_dcf(
    y_true: np.ndarray,
    y_score: np.ndarray,
    p_target: float = 0.05,
    c_miss: float = 1.0,
    c_fa: float = 10.0,
) -> float:
    """
    Computes Normalized Minimum Detection Cost Function (minDCF)
    used in binary deepfake and biometric detection evaluations.
    """
    if not isinstance(p_target, (int, float)) or isinstance(p_target, bool):
        raise TypeError("p_target must be a float, got %s." % type(p_target).__name__)
    if not (0.0 < p_target < 1.0):
        raise ValueError(
            "p_target must be strictly between 0 and 1, got %.6g. "
            "Values of 0 or 1 produce an undefined cost (division by zero)." % p_target
        )
    y_true, y_score = _validate_inputs(
        y_true, y_score, require_prob_range=False, scores_param_name="y_score"
    )

    if len(np.unique(y_true)) < 2:
        return 1.0

    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1.0 - tpr

    # Cost = C_miss * P_target * FNR + C_fa * (1 - P_target) * FPR
    cost = c_miss * p_target * fnr + c_fa * (1.0 - p_target) * fpr
    cost_default = min(c_miss * p_target, c_fa * (1.0 - p_target))
    min_dcf = np.min(cost) / cost_default

    return float(min_dcf)


def compute_ece(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> Tuple[float, Dict[str, list]]:
    """
    Computes Expected Calibration Error (ECE) and bin statistics for reliability diagrams.
    """
    if not isinstance(n_bins, int) or isinstance(n_bins, bool):
        raise TypeError(
            "n_bins must be a positive integer, got %s." % type(n_bins).__name__
        )
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1, got %d." % n_bins)
    y_true, y_prob = _validate_inputs(
        y_true, y_prob, require_prob_range=True, scores_param_name="y_prob"
    )

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_confs = []
    bin_accs = []
    bin_counts = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (
            (y_prob > bin_lower) & (y_prob <= bin_upper)
            if i > 0
            else (y_prob >= bin_lower) & (y_prob <= bin_upper)
        )
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            bin_confs.append(float(avg_confidence_in_bin))
            bin_accs.append(float(accuracy_in_bin))
            bin_counts.append(int(np.sum(in_bin)))
        else:
            bin_confs.append(float((bin_lower + bin_upper) / 2.0))
            bin_accs.append(0.0)
            bin_counts.append(0)

    return float(ece), {
        "bin_confidences": bin_confs,
        "bin_accuracies": bin_accs,
        "bin_counts": bin_counts,
    }
