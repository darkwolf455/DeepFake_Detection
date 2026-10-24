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


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Computes standard binary classification metrics."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
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
    Computes Equal Error Rate (EER) and operating threshold where FPR == FNR.
    Standard primary metric for ASVspoof biometric security.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    if len(np.unique(y_true)) < 2:
        return 0.5, 0.5

    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1.0 - tpr

    # EER is point where fpr == fnr
    try:
        eer_threshold = brentq(lambda x: 1.0 - interp1d(thresholds, tpr)(x) - interp1d(thresholds, fpr)(x), min(thresholds), max(thresholds))
        eer = float(interp1d(thresholds, fpr)(eer_threshold))
    except Exception:
        # Fallback to absolute difference minimum
        idx = np.nanargmin(np.abs(fpr - fnr))
        eer = float((fpr[idx] + fnr[idx]) / 2.0)
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
    used in ASVspoof 2019/2021 evaluations.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    if len(np.unique(y_true)) < 2:
        return 1.0

    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1.0 - tpr

    # Cost = C_miss * P_target * FNR + C_fa * (1 - P_target) * FPR
    cost = c_miss * p_target * fnr + c_fa * (1.0 - p_target) * fpr
    cost_default = min(c_miss * p_target, c_fa * (1.0 - p_target))
    min_dcf = np.min(cost) / cost_default

    return float(min_dcf)


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Tuple[float, Dict[str, list]]:
    """
    Computes Expected Calibration Error (ECE) and bin statistics for reliability diagrams.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_confs = []
    bin_accs = []
    bin_counts = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper) if i > 0 else (y_prob >= bin_lower) & (y_prob <= bin_upper)
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
