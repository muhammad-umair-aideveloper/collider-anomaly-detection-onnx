"""Pre-declared, leakage-free model selection experiment for collider anomaly detection.

This script evaluates pre-declared autoencoder architecture and feature-selection
variants (V0, V1, V2, V3) across multiple random seeds on the dedicated model-selection
set (select.npz). It derives mean and sample standard deviation AUC for each variant,
strictly adhering to a pre-registered decision rule:
- Select the variant with the highest mean AUC on select.npz.
- If the best variant's mean AUC is not greater than V0 by > 0.01, retain baseline V0.

CRITICAL:
- This script only accesses train.npz, val.npz, and select.npz.
- The held-out test split (test.npz) is NEVER loaded or evaluated in this script,
  guaranteeing zero data leakage.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

# Ensure python/ directory is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

import config
from model import PRE_DECLARED_VARIANTS, Autoencoder


class VariantResult(NamedTuple):
    """Holds selection evaluation metrics for a single model variant."""

    variant_id: str
    description: str
    feature_count: int
    bottleneck_dim: int
    seed_aucs: Dict[int, float]
    mean_auc: float
    std_auc: float
    delta_vs_v0: float


def parse_args() -> argparse.Namespace:
    """Parse CLI options for model selection experiment.

    Returns:
        argparse.Namespace: Parsed CLI options.
    """
    parser = argparse.ArgumentParser(
        description="Run pre-declared leakage-free model selection across variants and seeds."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.DEFAULT_EPOCHS,
        help=f"Number of training epochs per run (default: {config.DEFAULT_EPOCHS}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.DEFAULT_BATCH_SIZE,
        help=f"Mini-batch size (default: {config.DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=config.DEFAULT_LEARNING_RATE,
        help=f"Adam learning rate (default: {config.DEFAULT_LEARNING_RATE}).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="List of random seeds to evaluate (default: 0 1 2).",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.01,
        help="Minimum improvement margin over V0 required to select a variant (default: 0.01).",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=config.VARIANT_SELECTION_MD_PATH,
        help=f"Destination path for markdown report (default: {config.VARIANT_SELECTION_MD_PATH}).",
    )
    return parser.parse_args()


def load_selection_datasets(
    train_path: Path = config.TRAIN_SPLIT_PATH,
    val_path: Path = config.VAL_SPLIT_PATH,
    select_path: Path = config.SELECT_SPLIT_PATH,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load training, validation, and selection datasets.

    NOTE: The test split is deliberately omitted to prevent leakage.

    Args:
        train_path: Path to train.npz.
        val_path: Path to val.npz.
        select_path: Path to select.npz.

    Returns:
        Tuple of (train_X, val_X, select_X, select_y).
    """
    for path in (train_path, val_path, select_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Required dataset '{path}' does not exist. Run python/prepare_data.py first."
            )

    train_data = np.load(train_path)
    val_data = np.load(val_path)
    select_data = np.load(select_path)

    train_X = train_data["X"].astype(np.float32)
    train_y = train_data["y"].astype(np.int64)
    val_X = val_data["X"].astype(np.float32)
    val_y = val_data["y"].astype(np.int64)
    select_X = select_data["X"].astype(np.float32)
    select_y = select_data["y"].astype(np.int64)

    # Sanity checks
    if not np.all(train_y == config.LABEL_BACKGROUND):
        raise ValueError("Train split contains non-background events.")
    if not np.all(val_y == config.LABEL_BACKGROUND):
        raise ValueError("Validation split contains non-background events.")

    return train_X, val_X, select_X, select_y


def train_single_run(
    variant_id: str,
    train_X: np.ndarray,
    val_X: np.ndarray,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
) -> Autoencoder:
    """Train a single autoencoder variant run on CPU and return best checkpoint.

    Args:
        variant_id: Key identifying pre-declared variant (V0, V1, V2, V3).
        train_X: Training features array (background only).
        val_X: Validation features array (background only).
        seed: Random seed.
        epochs: Training epochs.
        batch_size: Mini-batch size.
        lr: Adam learning rate.

    Returns:
        Autoencoder: Best model checkpoint restored from lowest validation loss.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)

    device = torch.device("cpu")
    model = Autoencoder.from_variant(variant_id).to(device)
    model.set_standardization(train_X)

    train_tensor = torch.from_numpy(train_X).to(device)
    val_tensor = torch.from_numpy(val_X).to(device)

    train_loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=batch_size,
        shuffle=True,
    )

    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_state_dict = None

    for epoch in range(1, epochs + 1):
        model.train()
        for (batch_x,) in train_loader:
            optimizer.zero_grad()
            loss = model.reconstruction_error(batch_x).mean()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(model.reconstruction_error(val_tensor).mean().item())

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    assert best_state_dict is not None, "Failed to capture checkpoint."
    model.load_state_dict(best_state_dict)
    model.eval()
    return model


def evaluate_on_selection(
    model: Autoencoder,
    select_X: np.ndarray,
    select_y: np.ndarray,
) -> float:
    """Compute ROC AUC score on the selection set using reconstruction error.

    Args:
        model: Trained Autoencoder.
        select_X: Selection features array [N, 28].
        select_y: Selection true binary labels [N].

    Returns:
        float: ROC AUC score.
    """
    model.eval()
    with torch.no_grad():
        select_tensor = torch.from_numpy(select_X).float()
        errors = model.reconstruction_error(select_tensor).cpu().numpy()

    auc = float(roc_auc_score(select_y, errors))
    return auc


def run_variant_experiment(
    variant_ids: List[str],
    seeds: List[int],
    train_X: np.ndarray,
    val_X: np.ndarray,
    select_X: np.ndarray,
    select_y: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
) -> Dict[str, VariantResult]:
    """Execute training and selection evaluation across all variants and seeds.

    Args:
        variant_ids: List of variant identifiers.
        seeds: List of seeds to test.
        train_X: Training features.
        val_X: Validation features.
        select_X: Selection features.
        select_y: Selection labels.
        epochs: Epochs per run.
        batch_size: Batch size.
        lr: Learning rate.

    Returns:
        Dict[str, VariantResult]: Results keyed by variant ID.
    """
    raw_results: Dict[str, Dict[int, float]] = {}

    for var_id in variant_ids:
        raw_results[var_id] = {}
        var_spec = PRE_DECLARED_VARIANTS[var_id]
        print(f"\nEvaluating Variant '{var_id}': {var_spec['description']}")
        for s in seeds:
            t0 = time.perf_counter()
            model = train_single_run(
                variant_id=var_id,
                train_X=train_X,
                val_X=val_X,
                seed=s,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
            )
            auc = evaluate_on_selection(model, select_X, select_y)
            elapsed = time.perf_counter() - t0
            raw_results[var_id][s] = auc
            print(f"  -> Seed {s:2d} | Selection ROC AUC: {auc:.4f} ({elapsed:.2f}s)")

    # Baseline V0 mean AUC
    v0_mean_auc = float(np.mean(list(raw_results["V0"].values())))

    results: Dict[str, VariantResult] = {}
    for var_id in variant_ids:
        var_spec = PRE_DECLARED_VARIANTS[var_id]
        aucs = [raw_results[var_id][s] for s in seeds]
        mean_auc = float(np.mean(aucs))
        std_auc = float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0
        delta = mean_auc - v0_mean_auc

        feature_count = (
            len(var_spec["selected_features"])
            if var_spec["selected_features"] is not None
            else config.NUM_FEATURES
        )

        results[var_id] = VariantResult(
            variant_id=var_id,
            description=var_spec["description"],
            feature_count=feature_count,
            bottleneck_dim=var_spec["bottleneck_dim"],
            seed_aucs=raw_results[var_id],
            mean_auc=mean_auc,
            std_auc=std_auc,
            delta_vs_v0=delta,
        )

    return results


def select_best_variant(
    results: Dict[str, VariantResult], margin: float = 0.01
) -> Tuple[str, str]:
    """Apply pre-registered decision rule to choose winning variant.

    Rule:
    1. Identify variant with maximum mean AUC on select.npz.
    2. If max_mean_auc - v0_mean_auc > margin (0.01), select that variant.
    3. Otherwise, retain baseline V0.

    Args:
        results: Dictionary of VariantResult objects.
        margin: Required improvement threshold over V0.

    Returns:
        Tuple of (selected_variant_id, justification_string).
    """
    v0_mean = results["V0"].mean_auc

    # Find variant with highest mean AUC
    best_candidate_id = max(results.keys(), key=lambda k: results[k].mean_auc)
    best_candidate = results[best_candidate_id]
    delta = best_candidate.mean_auc - v0_mean

    if best_candidate_id == "V0":
        reason = (
            f"V0 has the highest mean AUC ({v0_mean:.4f}) on the selection set. "
            "Baseline V0 is retained."
        )
        return "V0", reason

    if delta > margin:
        reason = (
            f"Candidate '{best_candidate_id}' achieved mean AUC of {best_candidate.mean_auc:.4f}, "
            f"exceeding baseline V0 ({v0_mean:.4f}) by {delta:+.4f} (margin threshold: {margin:+.4f}). "
            f"Variant '{best_candidate_id}' is selected."
        )
        return best_candidate_id, reason
    else:
        reason = (
            f"Candidate '{best_candidate_id}' achieved highest mean AUC of {best_candidate.mean_auc:.4f} "
            f"(delta {delta:+.4f} vs V0 {v0_mean:.4f}), which does NOT satisfy the pre-registered "
            f"margin criterion (> {margin:+.4f}). Per decision rule, baseline V0 is retained."
        )
        return "V0", reason


def generate_markdown_report(
    results: Dict[str, VariantResult],
    selected_id: str,
    justification: str,
    seeds: List[int],
    margin: float,
) -> str:
    """Format experiment results as a clean Markdown table.

    Args:
        results: Dictionary of VariantResult objects.
        selected_id: Winning variant identifier.
        justification: Narrative explanation of the decision.
        seeds: Tested seed values.
        margin: Pre-registered threshold margin.

    Returns:
        str: Markdown document string.
    """
    seed_headers = " | ".join([f"Seed {s} AUC" for s in seeds])
    seed_dashes = " | ".join(["---:" for _ in seeds])

    lines = [
        "# Model Selection Experiment: Pre-Declared Architecture & Feature Variants",
        "",
        "## Experiment Overview",
        "",
        "- **Goal**: Rigorous, leakage-free evaluation of pre-declared autoencoder architecture and feature-subset variants.",
        "- **Dataset**: Evaluated on `data/processed/select.npz` (2,000 background + 2,000 signal events).",
        "- **Leakage Prevention**: The final test split (`data/processed/test.npz`) was **NOT** accessed during this selection.",
        "- **Training Protocol**: Trained exclusively on normal background events (`train.npz`), using validation loss (`val.npz`) for checkpoint selection, CPU only.",
        f"- **Pre-Registered Decision Rule**: Select candidate with highest mean AUC on `select.npz` only if `mean_auc - v0_mean_auc > {margin:.4f}`; otherwise keep baseline `V0`.",
        "",
        "## Results Table",
        "",
        f"| Variant | Features | Bottleneck | Architecture | {seed_headers} | Mean AUC | Std AUC | Delta vs V0 |",
        f"|:---|---:|---:|:---|{seed_dashes}|---:|---:|---:|",
    ]

    for var_id, res in results.items():
        var_spec = PRE_DECLARED_VARIANTS[var_id]
        arch_str = " -> ".join(map(str, var_spec["layers"]))
        seed_cols = " | ".join([f"{res.seed_aucs[s]:.4f}" for s in seeds])
        delta_str = f"{res.delta_vs_v0:+.4f}" if var_id != "V0" else "0.0000 (ref)"
        selected_flag = " **(Selected)**" if var_id == selected_id else ""
        lines.append(
            f"| `{var_id}`{selected_flag} | {res.feature_count} | {res.bottleneck_dim} | {arch_str} | {seed_cols} | **{res.mean_auc:.4f}** | {res.std_auc:.4f} | {delta_str} |"
        )

    lines.extend([
        "",
        "## Decision and Conclusion",
        "",
        f"- **Selected Variant**: `{selected_id}`",
        f"- **Decision Rule Margin**: `> {margin:.4f}` improvement over V0 required.",
        f"- **Justification**: {justification}",
        "",
        "## Scientific Notes",
        "",
        "1. In unsupervised anomaly detection on the HIGGS dataset, autoencoder reconstruction error is sensitive to feature semantics.",
        "2. Azimuthal angles (phi) are coordinate-system dependent and typically isotropic in collisions, offering no discrimination.",
        "3. High-level invariant masses (columns 21-27) carry reconstructed particle resonance physics.",
        "4. This selection was conducted completely out-of-sample relative to the final test split.",
    ])

    return "\n".join(lines) + "\n"


def main() -> None:
    """Execute complete model selection experiment pipeline."""
    args = parse_args()
    total_start = time.perf_counter()

    print("=" * 70)
    print("COLLIDER ANOMALY DETECTION: MODEL SELECTION EXPERIMENT")
    print("=" * 70)
    print(f"Pre-declared Variants: {list(PRE_DECLARED_VARIANTS.keys())}")
    print(f"Random Seeds:          {args.seeds}")
    print(f"Epochs per run:        {args.epochs}")
    print(f"Batch Size:            {args.batch_size}")
    print(f"Learning Rate:         {args.lr}")
    print(f"Decision Margin:       > {args.margin} over V0")
    print("=" * 70)

    # 1. Load data
    train_X, val_X, select_X, select_y = load_selection_datasets()
    print(f"Loaded Train Events:     {train_X.shape[0]} (Background only)")
    print(f"Loaded Val Events:       {val_X.shape[0]} (Background only)")
    print(f"Loaded Selection Events: {select_X.shape[0]} ({np.sum(select_y == 0)} bg, {np.sum(select_y == 1)} sig)")

    # 2. Run experiment
    variant_ids = ["V0", "V1", "V2", "V3"]
    results = run_variant_experiment(
        variant_ids=variant_ids,
        seeds=args.seeds,
        train_X=train_X,
        val_X=val_X,
        select_X=select_X,
        select_y=select_y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    # 3. Apply decision rule
    selected_id, justification = select_best_variant(results, margin=args.margin)

    # 4. Generate report
    report_md = generate_markdown_report(
        results=results,
        selected_id=selected_id,
        justification=justification,
        seeds=args.seeds,
        margin=args.margin,
    )

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_md, "w", encoding="utf-8") as f_out:
        f_out.write(report_md)

    total_time = time.perf_counter() - total_start
    print("\n" + "=" * 70)
    print(f"EXPERIMENT COMPLETED in {total_time:.2f} seconds")
    print(f"Report written to: {args.out_md}")
    print("=" * 70)
    print("\n" + report_md)


if __name__ == "__main__":
    main()
