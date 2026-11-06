"""Unit tests for anomaly detection evaluation metrics and statistical routines."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure repo root is in python path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

import config
from evaluate import (
    compute_bootstrap_ci,
    compute_class_error_statistics,
    compute_confusion_at_threshold,
    compute_roc_metrics,
)


def test_perfectly_separated_auc():
    """Verify that completely separated classes yield an exact ROC AUC of 1.0."""
    y_true = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9], dtype=np.float64)

    auc, fpr, tpr, _ = compute_roc_metrics(y_true, y_score)
    assert auc == 1.0, f"Expected perfect AUC 1.0, got {auc}"
    assert len(fpr) == len(tpr)


def test_identical_distributions_auc():
    """Verify that uninformative random scores produce an AUC near 0.5."""
    rng = np.random.default_rng(config.SEED)
    y_true = np.array([0] * 500 + [1] * 500, dtype=np.int64)
    y_score = rng.normal(loc=0.0, scale=1.0, size=1000)

    auc, _, _, _ = compute_roc_metrics(y_true, y_score)
    # Loose statistical tolerance around 0.5
    assert 0.45 <= auc <= 0.55, f"Expected AUC near 0.5 for random noise, got {auc}"


def test_confusion_counts_at_known_threshold():
    """Verify that TP, FP, TN, FN match deterministic hand-calculated counts."""
    # 4 background (0) and 4 signal (1)
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    y_score = np.array([0.1, 0.3, 0.6, 0.8, 0.2, 0.4, 0.7, 0.9], dtype=np.float64)
    threshold = 0.5

    # Hand calculation at threshold 0.5:
    # Pred Positive (>= 0.5):
    #   bg at 0.6 -> FP
    #   bg at 0.8 -> FP
    #   sig at 0.7 -> TP
    #   sig at 0.9 -> TP
    # Pred Negative (< 0.5):
    #   bg at 0.1 -> TN
    #   bg at 0.3 -> TN
    #   sig at 0.2 -> FN
    #   sig at 0.4 -> FN
    # Expected: TP=2, FP=2, TN=2, FN=2, TPR=0.5, FPR=0.5
    confusion = compute_confusion_at_threshold(y_true, y_score, threshold=threshold)

    assert confusion["tp"] == 2
    assert confusion["fp"] == 2
    assert confusion["tn"] == 2
    assert confusion["fn"] == 2
    assert np.isclose(confusion["tpr"], 0.5)
    assert np.isclose(confusion["fpr"], 0.5)


def test_tpr_fpr_bounds():
    """Verify that TPR and FPR are strictly bounded in [0.0, 1.0]."""
    rng = np.random.default_rng(config.SEED)
    y_true = rng.integers(0, 2, size=100, endpoint=False)
    y_score = rng.random(size=100)

    for thresh in [-1.0, 0.0, 0.5, 1.0, 2.0]:
        confusion = compute_confusion_at_threshold(y_true, y_score, threshold=thresh)
        assert 0.0 <= confusion["tpr"] <= 1.0
        assert 0.0 <= confusion["fpr"] <= 1.0


def test_bootstrap_ci_bounds():
    """Verify that bootstrap confidence intervals are within [0.0, 1.0] and lower <= upper."""
    rng = np.random.default_rng(config.SEED)
    y_true = np.array([0] * 50 + [1] * 50, dtype=np.int64)
    y_score = rng.normal(loc=y_true * 0.5, scale=1.0, size=100)

    ci_lower, ci_upper = compute_bootstrap_ci(
        y_true, y_score, n_bootstraps=100, seed=config.SEED
    )

    assert 0.0 <= ci_lower <= ci_upper <= 1.0


def test_class_error_statistics():
    """Verify computation of class-specific error statistics."""
    y_true = np.array([0, 0, 1, 1], dtype=np.int64)
    y_score = np.array([1.0, 3.0, 10.0, 20.0], dtype=np.float64)

    stats = compute_class_error_statistics(y_true, y_score)

    assert np.isclose(stats["bg_mean"], 2.0)
    assert np.isclose(stats["bg_median"], 2.0)
    assert np.isclose(stats["sig_mean"], 15.0)
    assert np.isclose(stats["sig_median"], 15.0)
