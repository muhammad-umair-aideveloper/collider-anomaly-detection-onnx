"""Evaluation pipeline for collider anomaly detection autoencoder.

This module evaluates the trained autoencoder on the held-out test split,
computes ROC AUC with a 95% bootstrap confidence interval, measures classification
metrics at the 95th percentile validation threshold, produces ROC curve and error
histogram plots, and writes a reproducible metrics report.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

# Ensure repo root is in python path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

import config
from model import Autoencoder


def parse_args() -> argparse.Namespace:
    """Parse CLI options for evaluation script.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate anomaly detector on the held-out test split."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=config.PYTORCH_MODEL_PATH,
        help="Path to trained autoencoder checkpoint (.pt).",
    )
    parser.add_argument(
        "--threshold-path",
        type=Path,
        default=config.THRESHOLD_TXT_PATH,
        help="Path to decision threshold text file (.txt).",
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=config.TEST_SPLIT_PATH,
        help="Path to test dataset archive (.npz).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=config.RESULTS_DIR,
        help="Directory to save evaluation plots and metrics.md.",
    )
    parser.add_argument(
        "--n-bootstraps",
        type=int,
        default=1000,
        help="Number of bootstrap resamples for AUC confidence interval.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.SEED,
        help="Random seed for bootstrap resampling.",
    )
    return parser.parse_args()


def load_test_data(test_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load test features and true labels from compressed archive.

    Args:
        test_path: Path to test.npz.

    Returns:
        Tuple of (test_features, test_labels).
    """
    if not test_path.exists():
        raise FileNotFoundError(
            f"Test dataset not found at '{test_path}'. Run python/prepare_data.py first."
        )

    data = np.load(test_path)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    return X, y


def load_model_and_threshold(
    model_path: Path, threshold_path: Path
) -> Tuple[Autoencoder, float]:
    """Load trained autoencoder and decision threshold from disk.

    Args:
        model_path: Path to autoencoder.pt.
        threshold_path: Path to threshold.txt.

    Returns:
        Tuple of (model, threshold).
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at '{model_path}'. Run python/train.py first."
        )
    if not threshold_path.exists():
        raise FileNotFoundError(
            f"Threshold file not found at '{threshold_path}'. Run python/train.py first."
        )

    # Load threshold float
    with open(threshold_path, "r", encoding="utf-8") as f_thresh:
        threshold = float(f_thresh.read().strip())

    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = Autoencoder()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, threshold


def compute_reconstruction_errors(
    model: Autoencoder, test_X: np.ndarray
) -> np.ndarray:
    """Compute per-event reconstruction errors on test features.

    Args:
        model: Trained Autoencoder.
        test_X: Raw test features array of shape [N, 28].

    Returns:
        np.ndarray: Per-event reconstruction error array of shape [N].
    """
    model.eval()
    with torch.inference_mode():
        features_tensor = torch.from_numpy(test_X).float()
        errors = model.reconstruction_error(features_tensor).cpu().numpy()
    return errors.astype(np.float64)


def compute_roc_metrics(
    y_true: np.ndarray, y_score: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Compute ROC AUC score and ROC curve points.

    Args:
        y_true: Binary ground truth labels (0 = background, 1 = signal).
        y_score: Reconstruction error anomaly scores.

    Returns:
        Tuple of (auc_score, fpr_array, tpr_array, thresholds_array).
    """
    auc = float(roc_auc_score(y_true, y_score))
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    return auc, fpr, tpr, thresholds


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = config.SEED,
    confidence_level: float = 0.95,
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for ROC AUC score.

    Args:
        y_true: Binary ground truth labels.
        y_score: Reconstruction error scores.
        n_bootstraps: Number of bootstrap iterations.
        seed: Random seed for reproducibility.
        confidence_level: Desired coverage probability (e.g. 0.95 for 95% CI).

    Returns:
        Tuple of (ci_lower, ci_upper).
    """
    rng = np.random.default_rng(seed)
    n_samples = len(y_true)
    bootstrapped_aucs: List[float] = []

    for _ in range(n_bootstraps):
        indices = rng.integers(0, n_samples, size=n_samples)
        sample_y = y_true[indices]
        sample_score = y_score[indices]

        # Ensure both classes are present in resample
        if len(np.unique(sample_y)) < 2:
            continue

        sample_auc = float(roc_auc_score(sample_y, sample_score))
        bootstrapped_aucs.append(sample_auc)

    alpha = 1.0 - confidence_level
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    ci_lower = float(np.percentile(bootstrapped_aucs, lower_pct))
    ci_upper = float(np.percentile(bootstrapped_aucs, upper_pct))

    return ci_lower, ci_upper


def compute_confusion_at_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> Dict[str, Union[int, float]]:
    """Compute confusion matrix counts, TPR, and FPR at a specified threshold.

    An anomaly prediction is made when y_score >= threshold.

    Args:
        y_true: True binary labels (0 = background, 1 = signal).
        y_score: Predicted reconstruction errors.
        threshold: Decision threshold.

    Returns:
        Dict with keys: 'tp', 'fp', 'tn', 'fn', 'tpr', 'fpr'.
    """
    pred_positive = y_score >= threshold
    true_positive = y_true == config.LABEL_SIGNAL
    true_negative = y_true == config.LABEL_BACKGROUND

    tp = int(np.sum(pred_positive & true_positive))
    fp = int(np.sum(pred_positive & true_negative))
    tn = int(np.sum(~pred_positive & true_negative))
    fn = int(np.sum(~pred_positive & true_positive))

    tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "tpr": tpr,
        "fpr": fpr,
    }


def compute_class_error_statistics(
    y_true: np.ndarray, y_score: np.ndarray
) -> Dict[str, float]:
    """Compute central tendency statistics of reconstruction errors per class.

    Args:
        y_true: True binary labels.
        y_score: Reconstruction errors.

    Returns:
        Dict with background and signal mean and median errors.
    """
    bg_errors = y_score[y_true == config.LABEL_BACKGROUND]
    sig_errors = y_score[y_true == config.LABEL_SIGNAL]

    return {
        "bg_mean": float(np.mean(bg_errors)),
        "bg_median": float(np.median(bg_errors)),
        "sig_mean": float(np.mean(sig_errors)),
        "sig_median": float(np.median(sig_errors)),
    }


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_score: float,
    ci_lower: float,
    ci_upper: float,
    op_fpr: float,
    op_tpr: float,
    threshold: float,
    output_path: Path,
) -> None:
    """Plot ROC curve with baseline and operating point marker.

    Args:
        fpr: False positive rates across thresholds.
        tpr: True positive rates across thresholds.
        auc_score: ROC AUC metric.
        ci_lower: 95% CI lower bound.
        ci_upper: 95% CI upper bound.
        op_fpr: False positive rate at decision threshold.
        op_tpr: True positive rate at decision threshold.
        threshold: Chosen decision threshold.
        output_path: Save file destination.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 6))

    plt.plot(
        fpr,
        tpr,
        color="#1f77b4",
        lw=2.5,
        label=f"Autoencoder ROC (AUC = {auc_score:.4f} [{ci_lower:.4f}, {ci_upper:.4f}])",
    )
    plt.plot([0, 1], [0, 1], color="#7f7f7f", linestyle="--", lw=1.5, label="Random Guessing (AUC = 0.50)")

    # Plot operating point marker
    plt.scatter(
        [op_fpr],
        [op_tpr],
        color="#d62728",
        s=90,
        zorder=5,
        label=f"Threshold = {threshold:.4f} (TPR={op_tpr:.3f}, FPR={op_fpr:.3f})",
    )

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("False Positive Rate (Background Misidentified)", fontsize=11)
    plt.ylabel("True Positive Rate (Signal Flagged)", fontsize=11)
    plt.title("ROC Curve: Collider Event Anomaly Detection", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"ROC curve saved to: {output_path}")


def plot_error_histogram(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    output_path: Path,
) -> None:
    """Plot reconstruction error histograms for background vs signal on log scale.

    Args:
        y_true: True binary labels.
        y_score: Reconstruction error values.
        threshold: Decision threshold vertical marker.
        output_path: Save file destination.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bg_errors = y_score[y_true == config.LABEL_BACKGROUND]
    sig_errors = y_score[y_true == config.LABEL_SIGNAL]

    # Shared binning for consistent comparison
    combined_min = float(np.min(y_score))
    combined_max = float(np.percentile(y_score, 99.5))  # cut extreme outliers for legible plot
    bins = np.linspace(combined_min, combined_max, 60)

    plt.figure(figsize=(8, 5))
    plt.hist(
        bg_errors,
        bins=bins,
        alpha=0.6,
        color="#1f77b4",
        edgecolor="black",
        linewidth=0.5,
        label=f"Background (N={len(bg_errors)})",
    )
    plt.hist(
        sig_errors,
        bins=bins,
        alpha=0.6,
        color="#d62728",
        edgecolor="black",
        linewidth=0.5,
        label=f"Signal (N={len(sig_errors)})",
    )

    plt.axvline(
        threshold,
        color="black",
        linestyle="--",
        lw=2,
        label=f"Decision Threshold ({threshold:.4f})",
    )

    plt.yscale("log")
    plt.xlabel("Reconstruction Error (MSE)", fontsize=11)
    plt.ylabel("Event Count (log scale)", fontsize=11)
    plt.title("Reconstruction Error Distribution (Background vs Signal)", fontsize=12, fontweight="bold")
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Error histogram saved to: {output_path}")


def write_metrics_markdown(
    auc: float,
    ci_lower: float,
    ci_upper: float,
    confusion: Dict[str, Union[int, float]],
    class_stats: Dict[str, float],
    threshold: float,
    num_events: int,
    seed: int,
    output_path: Path,
) -> None:
    """Write evaluation metrics table to markdown file.

    Args:
        auc: ROC AUC score.
        ci_lower: Lower bound of 95% CI.
        ci_upper: Upper bound of 95% CI.
        confusion: Confusion matrix counts and rates.
        class_stats: Class-level error statistics.
        threshold: Chosen anomaly threshold.
        num_events: Total test events evaluated.
        seed: Random seed.
        output_path: Target path for metrics.md.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = f"""# Anomaly Detection Evaluation Metrics

Evaluation performed on held-out test split using best autoencoder checkpoint.

## Summary Metrics

| Metric | Value | Details / Specification |
| :--- | :--- | :--- |
| **ROC AUC** | **{auc:.4f}** | 95% Bootstrap CI: [{ci_lower:.4f}, {ci_upper:.4f}] (1000 resamples) |
| **Decision Threshold** | **{threshold:.6f}** | 95th percentile of validation background error |
| **Test Set Size** | **{num_events:,}** | 5,000 Background + 5,000 Signal events |
| **Random Seed** | **{seed}** | Fixed evaluation seed |

## Operating Point Performance (at Threshold = {threshold:.6f})

| Metric | Count / Rate | Definition |
| :--- | :--- | :--- |
| **True Positives (TP)** | {confusion['tp']:,} | Signal events correctly flagged as anomalies |
| **False Positives (FP)** | {confusion['fp']:,} | Background events incorrectly flagged as anomalies |
| **True Negatives (TN)** | {confusion['tn']:,} | Background events correctly passed as normal |
| **False Negatives (FN)** | {confusion['fn']:,} | Signal events missed (reconstructed with error < threshold) |
| **True Positive Rate (TPR)** | **{confusion['tpr']:.4f}** ({confusion['tpr'] * 100:.2f}%) | $\\text{{TP}} / (\\text{{TP}} + \\text{{FN}})$ |
| **False Positive Rate (FPR)** | **{confusion['fpr']:.4f}** ({confusion['fpr'] * 100:.2f}%) | $\\text{{FP}} / (\\text{{FP}} + \\text{{TN}})$ |

## Reconstruction Error Distributions (MSE)

| Event Class | Mean Error | Median Error | Event Count |
| :--- | :--- | :--- | :--- |
| **Background (Normal)** | {class_stats['bg_mean']:.6f} | {class_stats['bg_median']:.6f} | 5,000 |
| **Signal (Anomaly)** | {class_stats['sig_mean']:.6f} | {class_stats['sig_median']:.6f} | 5,000 |

*Generated automatically by `python/evaluate.py`.*
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Metrics table saved to: {output_path}")


def main() -> None:
    """Execute evaluation pipeline and generate all artifacts."""
    args = parse_args()

    # 1. Load inputs
    test_X, test_y = load_test_data(args.test_path)
    model, threshold = load_model_and_threshold(args.model_path, args.threshold_path)
    num_events = len(test_y)

    print("=" * 60)
    print("COLLIDER ANOMALY DETECTION: EVALUATION")
    print("=" * 60)
    print(f"Test Events Evaluated: {num_events}")
    print(f"Decision Threshold:    {threshold:.6f}")
    print(f"Bootstrap Seed:        {args.seed}")

    # 2. Compute reconstruction errors
    test_errors = compute_reconstruction_errors(model, test_X)

    # 3. Compute ROC AUC & Confidence Interval
    auc, fpr, tpr, _ = compute_roc_metrics(test_y, test_errors)
    ci_lower, ci_upper = compute_bootstrap_ci(
        test_y, test_errors, n_bootstraps=args.n_bootstraps, seed=args.seed
    )

    # 4. Compute confusion matrix and error statistics at threshold
    confusion = compute_confusion_at_threshold(test_y, test_errors, threshold)
    class_stats = compute_class_error_statistics(test_y, test_errors)

    # 5. Generate plots
    roc_plot_path = args.results_dir / "roc_curve.png"
    hist_plot_path = args.results_dir / "error_histogram.png"
    metrics_path = args.results_dir / "metrics.md"

    plot_roc_curve(
        fpr=fpr,
        tpr=tpr,
        auc_score=auc,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        op_fpr=confusion["fpr"],
        op_tpr=confusion["tpr"],
        threshold=threshold,
        output_path=roc_plot_path,
    )

    plot_error_histogram(
        y_true=test_y,
        y_score=test_errors,
        threshold=threshold,
        output_path=hist_plot_path,
    )

    # 6. Save markdown report
    write_metrics_markdown(
        auc=auc,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confusion=confusion,
        class_stats=class_stats,
        threshold=threshold,
        num_events=num_events,
        seed=args.seed,
        output_path=metrics_path,
    )

    # 7. Print summary
    print("=" * 60)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 60)
    print(f"ROC AUC:               {auc:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])")
    print(f"Operating Point TPR:   {confusion['tpr']:.4f} ({confusion['tp']} / 5000)")
    print(f"Operating Point FPR:   {confusion['fpr']:.4f} ({confusion['fp']} / 5000)")
    print(f"Confusion Matrix:      TP={confusion['tp']}, FP={confusion['fp']}, TN={confusion['tn']}, FN={confusion['fn']}")
    print(f"Background Mean Error: {class_stats['bg_mean']:.6f} (Median: {class_stats['bg_median']:.6f})")
    print(f"Signal Mean Error:     {class_stats['sig_mean']:.6f} (Median: {class_stats['sig_median']:.6f})")
    print("=" * 60)


if __name__ == "__main__":
    main()
