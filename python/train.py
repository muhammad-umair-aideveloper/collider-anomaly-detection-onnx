"""Train autoencoder on normal background collision events and compute anomaly threshold.

This script executes the unsupervised training loop using background events exclusively,
computes and embeds standardization statistics from the training set into the model,
evaluates validation loss at each epoch, saves the best checkpoint, derives the 95th
percentile anomaly decision threshold, and generates training loss diagnostics.
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Ensure repo root is in python path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

import config
from model import Autoencoder


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for training.

    Returns:
        argparse.Namespace: Parsed CLI options.
    """
    parser = argparse.ArgumentParser(
        description="Train autoencoder on background events and determine threshold."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.DEFAULT_EPOCHS,
        help=f"Number of training epochs (default: {config.DEFAULT_EPOCHS}).",
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
        "--seed",
        type=int,
        default=config.SEED,
        help=f"Random seed for reproducibility (default: {config.SEED}).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Optional directory to save models and results (used for smoke tests).",
    )
    return parser.parse_args()


def load_training_and_val_data(
    train_path: Path = config.TRAIN_SPLIT_PATH,
    val_path: Path = config.VAL_SPLIT_PATH,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load background-only training and validation features from processed archives.

    CRITICAL: This function intentionally only accesses train.npz and val.npz.
    The test split is never referenced or loaded.

    Args:
        train_path: Path to train.npz.
        val_path: Path to val.npz.

    Returns:
        Tuple of (train_features, val_features).

    Raises:
        FileNotFoundError: If train or val archive does not exist.
        ValueError: If labels other than 0 (background) are found.
    """
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"Processed datasets not found at '{train_path}' or '{val_path}'. "
            "Please run python/prepare_data.py first."
        )

    train_data = np.load(train_path)
    val_data = np.load(val_path)

    train_X: np.ndarray = train_data["X"].astype(np.float32)
    train_y: np.ndarray = train_data["y"].astype(np.int64)
    val_X: np.ndarray = val_data["X"].astype(np.float32)
    val_y: np.ndarray = val_data["y"].astype(np.int64)

    # Strict check: Unsupervised background training requires pure label=0
    if not np.all(train_y == config.LABEL_BACKGROUND):
        raise ValueError("Data integrity violation: train split contains non-background events!")
    if not np.all(val_y == config.LABEL_BACKGROUND):
        raise ValueError("Data integrity violation: val split contains non-background events!")

    return train_X, val_X


def train_model(
    train_X: np.ndarray,
    val_X: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> Tuple[Autoencoder, Dict[str, List[float]], int, float]:
    """Execute autoencoder training loop and return best model checkpoint.

    Args:
        train_X: Training features array.
        val_X: Validation features array.
        epochs: Number of epochs.
        batch_size: Batch size.
        lr: Learning rate.
        seed: Random seed.

    Returns:
        Tuple of (best_model, loss_history, best_epoch, best_val_loss).
    """
    # Fix random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)

    device = torch.device("cpu")

    # Instantiate model and initialize internal standardization with training stats ONLY
    model = Autoencoder().to(device)
    model.set_standardization(train_X)

    train_tensor = torch.from_numpy(train_X).to(device)
    val_tensor = torch.from_numpy(val_X).to(device)

    train_dataset = TensorDataset(train_tensor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loss_history: List[float] = []
    val_loss_history: List[float] = []

    best_val_loss = float("inf")
    best_epoch = 1
    best_state_dict = None

    print("\nStarting Training Loop:")
    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        batch_losses = []
        for (batch_x,) in train_loader:
            optimizer.zero_grad()
            # Reconstruction error is mean squared error over features
            loss = model.reconstruction_error(batch_x).mean()
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        epoch_train_loss = float(np.mean(batch_losses))

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_loss = float(model.reconstruction_error(val_tensor).mean().item())

        train_loss_history.append(epoch_train_loss)
        val_loss_history.append(val_loss)

        # Track best model checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Print progress on first epoch, last epoch, and every 10 epochs
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"Epoch [{epoch:03d}/{epochs:03d}] | "
                f"Train Loss (MSE): {epoch_train_loss:.6f} | "
                f"Val Loss (MSE): {val_loss:.6f}"
            )

    # Restore best checkpoint
    assert best_state_dict is not None, "Failed to capture model checkpoint."
    model.load_state_dict(best_state_dict)
    model.eval()

    history = {
        "train_loss": train_loss_history,
        "val_loss": val_loss_history,
    }

    return model, history, best_epoch, best_val_loss


def compute_anomaly_threshold(
    model: Autoencoder,
    val_X: np.ndarray,
    percentile: float = config.THRESHOLD_PERCENTILE,
) -> float:
    """Calculate anomaly decision threshold from validation background reconstruction error.

    Args:
        model: Trained Autoencoder.
        val_X: Validation features array.
        percentile: Threshold percentile (default 95.0).

    Returns:
        float: Computed threshold value.
    """
    model.eval()
    with torch.no_grad():
        val_tensor = torch.from_numpy(val_X).float()
        val_errors = model.reconstruction_error(val_tensor).cpu().numpy()

    threshold = float(np.percentile(val_errors, percentile))
    return threshold


def plot_loss_curve(
    history: Dict[str, List[float]],
    output_path: Path,
) -> None:
    """Plot and save training and validation loss curves.

    Args:
        history: Dictionary containing 'train_loss' and 'val_loss'.
        output_path: File path to save the plot.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss (MSE)", color="#1f77b4", lw=2)
    plt.plot(epochs, history["val_loss"], label="Validation Loss (MSE)", color="#ff7f0e", lw=2)

    plt.title("Autoencoder Training & Validation Loss Curve", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Reconstruction Error (MSE)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()

    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Loss curve saved to: {output_path}")


def main() -> None:
    """Execute complete autoencoder training and threshold determination workflow."""
    args = parse_args()
    start_time = time.perf_counter()

    # Determine destination paths
    if args.out_dir is not None:
        model_dir = args.out_dir
        results_dir = args.out_dir
    else:
        model_dir = config.MODELS_DIR
        results_dir = config.RESULTS_DIR

    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    pt_save_path = model_dir / "autoencoder.pt"
    json_threshold_path = model_dir / "threshold.json"
    txt_threshold_path = model_dir / "threshold.txt"
    loss_curve_path = results_dir / "loss_curve.png"

    # Header and environment diagnostics
    print("=" * 60)
    print("COLLIDER ANOMALY DETECTION: AUTOENCODER TRAINING")
    print("=" * 60)
    print(f"Python Version:       {platform.python_version()}")
    print(f"PyTorch Version:      {torch.__version__} (CPU)")
    print(f"Random Seed:          {args.seed}")
    print(f"Autoencoder Layers:   {config.AUTOENCODER_LAYERS}")
    print(f"Epochs:               {args.epochs}")
    print(f"Batch Size:           {args.batch_size}")
    print(f"Learning Rate:        {args.lr}")
    print(f"Threshold Percentile: {config.THRESHOLD_PERCENTILE}% (Validation Background)")
    print("=" * 60)

    # 1. Load background-only datasets
    train_X, val_X = load_training_and_val_data()
    print(f"Loaded train events:  {train_X.shape[0]} (Background only)")
    print(f"Loaded val events:    {val_X.shape[0]} (Background only)")

    # 2. Train model
    best_model, history, best_epoch, best_val_loss = train_model(
        train_X=train_X,
        val_X=val_X,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )

    elapsed_time = time.perf_counter() - start_time

    # 3. Compute threshold on validation set using best model
    threshold = compute_anomaly_threshold(
        best_model, val_X, percentile=config.THRESHOLD_PERCENTILE
    )

    # 4. Save model checkpoint
    torch.save(
        {
            "model_state_dict": best_model.state_dict(),
            "mean": best_model.mean.cpu(),
            "std": best_model.std.cpu(),
            "config": {
                "layers": config.AUTOENCODER_LAYERS,
                "num_features": config.NUM_FEATURES,
                "epsilon": config.STANDARDIZATION_EPSILON,
                "seed": args.seed,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "threshold": threshold,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
            },
        },
        pt_save_path,
    )
    print(f"Saved PyTorch model checkpoint: {pt_save_path}")

    # 5. Save threshold files
    threshold_meta = {
        "threshold": threshold,
        "percentile": config.THRESHOLD_PERCENTILE,
        "num_validation_events": int(val_X.shape[0]),
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }
    with open(json_threshold_path, "w", encoding="utf-8") as f_json:
        json.dump(threshold_meta, f_json, indent=2)
    print(f"Saved threshold metadata:       {json_threshold_path}")

    with open(txt_threshold_path, "w", encoding="utf-8") as f_txt:
        f_txt.write(f"{threshold:.9g}\n")
    print(f"Saved threshold float:          {txt_threshold_path}")

    # 6. Save loss curve
    plot_loss_curve(history, loss_curve_path)

    print("=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Total Wall-Clock Time:  {elapsed_time:.2f} seconds")
    print(f"Initial Train Loss:     {history['train_loss'][0]:.6f}")
    print(f"Initial Val Loss:       {history['val_loss'][0]:.6f}")
    print(f"Final Train Loss:       {history['train_loss'][-1]:.6f}")
    print(f"Final Val Loss:         {history['val_loss'][-1]:.6f}")
    print(f"Best Validation Epoch:  {best_epoch} (Val Loss: {best_val_loss:.6f})")
    print(f"Anomaly Threshold:      {threshold:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
