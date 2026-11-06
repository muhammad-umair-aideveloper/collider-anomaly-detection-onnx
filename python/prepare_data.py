"""Prepare and partition the HIGGS dataset into reproducible splits.

This script loads the raw OpenML HIGGS sample, validates counts and schema,
performs stratified sampling by label (15,000 background and 5,000 signal),
partitions events into train, validation, and test sets without overlap,
and exports compressed numpy archives (.npz) and CSV demo/benchmark files.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# Ensure repo root is in python path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

import config


def parse_args() -> argparse.Namespace:
    """Parse command line options.

    Returns:
        argparse.Namespace: Command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Partition HIGGS sample into train, validation, and test splits."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=config.HIGGS_RAW_PATH,
        help="Path to input raw HIGGS sample CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.PROCESSED_DATA_DIR,
        help="Directory to save processed .npz files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.SEED,
        help="Random seed for sampling and shuffling.",
    )
    return parser.parse_args()


def load_and_validate_raw_data(csv_path: Path) -> pd.DataFrame:
    """Load raw HIGGS CSV and validate schema constraints.

    Args:
        csv_path: Path to the raw CSV file.

    Returns:
        pd.DataFrame: Validated raw dataframe without duplicate rows.

    Raises:
        FileNotFoundError: If input file does not exist.
        ValueError: If schema or minimum event count validation fails.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Input dataset not found at '{csv_path}'. "
            "Please run python/download_data.py first."
        )

    print(f"Loading raw data from '{csv_path}'...")
    df = pd.read_csv(csv_path, header=None)

    # Validate column count (1 label + 28 features = 29 columns)
    expected_cols = config.NUM_FEATURES + 1
    if df.shape[1] != expected_cols:
        raise ValueError(
            f"Expected {expected_cols} columns, found {df.shape[1]} columns."
        )

    # Validate missing values
    num_nans = df.isna().sum().sum()
    if num_nans != 0:
        raise ValueError(f"Input data contains {num_nans} NaN values; expected 0.")

    # Validate labels
    labels = df.iloc[:, 0].astype(int)
    unique_labels = set(labels.unique())
    if not unique_labels.issubset({config.LABEL_BACKGROUND, config.LABEL_SIGNAL}):
        raise ValueError(
            f"Unexpected label values {unique_labels}. Expected only {{0, 1}}."
        )

    # Remove exact duplicate events to prevent cross-split data leakage
    num_duplicates = int(df.duplicated().sum())
    if num_duplicates > 0:
        print(f"Notice: Dropping {num_duplicates} duplicate event(s) from pool.")
        df = df.drop_duplicates().reset_index(drop=True)

    total_rows = len(df)
    final_labels = df.iloc[:, 0].astype(int)
    bg_count = int((final_labels == config.LABEL_BACKGROUND).sum())
    sig_count = int((final_labels == config.LABEL_SIGNAL).sum())

    print(f"Raw dataset loaded: {total_rows} unique rows, {df.shape[1]} columns.")
    print(f"Available background events (label=0): {bg_count}")
    print(f"Available signal events (label=1): {sig_count}")

    if bg_count < config.SAMPLE_BACKGROUND_COUNT:
        raise ValueError(
            f"Insufficient background events: found {bg_count}, "
            f"required at least {config.SAMPLE_BACKGROUND_COUNT}."
        )

    if sig_count < config.SAMPLE_SIGNAL_COUNT:
        raise ValueError(
            f"Insufficient signal events: found {sig_count}, "
            f"required at least {config.SAMPLE_SIGNAL_COUNT}."
        )

    return df


def sample_and_split(
    df: pd.DataFrame, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Sample events by label and partition into train, val, and test splits.

    Args:
        df: Raw validated HIGGS DataFrame without duplicates.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df, test_df).

    Raises:
        RuntimeError: If data leakage or overlap across splits is detected.
    """
    print(f"Sampling events by label using random seed: {seed}...")
    bg_pool = df[df.iloc[:, 0] == config.LABEL_BACKGROUND]
    sig_pool = df[df.iloc[:, 0] == config.LABEL_SIGNAL]

    # Sample without replacement by label
    bg_sampled = bg_pool.sample(
        n=config.SAMPLE_BACKGROUND_COUNT, random_state=seed, replace=False
    )
    sig_sampled = sig_pool.sample(
        n=config.SAMPLE_SIGNAL_COUNT, random_state=seed, replace=False
    )

    # Verify no index overlap between background and signal samples
    if not set(bg_sampled.index).isdisjoint(set(sig_sampled.index)):
        raise RuntimeError("Overlap detected between sampled background and signal pools.")

    # Partition background into train (8,000), val (2,000), test (5,000)
    train_bg = bg_sampled.iloc[: config.TRAIN_BACKGROUND_COUNT]
    val_bg = bg_sampled.iloc[
        config.TRAIN_BACKGROUND_COUNT : config.TRAIN_BACKGROUND_COUNT
        + config.VAL_BACKGROUND_COUNT
    ]
    test_bg = bg_sampled.iloc[
        config.TRAIN_BACKGROUND_COUNT + config.VAL_BACKGROUND_COUNT :
    ]

    # Verify partition sizes
    assert len(train_bg) == config.TRAIN_BACKGROUND_COUNT, "Train partition size mismatch"
    assert len(val_bg) == config.VAL_BACKGROUND_COUNT, "Val partition size mismatch"
    assert len(test_bg) == config.TEST_BACKGROUND_COUNT, "Test bg partition size mismatch"

    # Strict disjoint index validation
    idx_train = set(train_bg.index)
    idx_val = set(val_bg.index)
    idx_test_bg = set(test_bg.index)

    if not idx_train.isdisjoint(idx_val):
        raise RuntimeError("Data leakage: overlap found between train and validation splits!")
    if not idx_train.isdisjoint(idx_test_bg):
        raise RuntimeError("Data leakage: overlap found between train and test splits!")
    if not idx_val.isdisjoint(idx_test_bg):
        raise RuntimeError("Data leakage: overlap found between validation and test splits!")

    # Combine test split: 5,000 background + 5,000 signal, then shuffle
    test_combined = pd.concat([test_bg, sig_sampled], axis=0)
    test_df = test_combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    train_df = train_bg.reset_index(drop=True)
    val_df = val_bg.reset_index(drop=True)

    # Feature-level duplicate check across splits
    train_tuples = set(map(tuple, train_df.iloc[:, 1:].values))
    val_tuples = set(map(tuple, val_df.iloc[:, 1:].values))
    test_tuples = set(map(tuple, test_df.iloc[:, 1:].values))

    overlap_train_val = train_tuples.intersection(val_tuples)
    overlap_train_test = train_tuples.intersection(test_tuples)
    overlap_val_test = val_tuples.intersection(test_tuples)

    if overlap_train_val:
        raise RuntimeError(
            f"Data leakage detected: {len(overlap_train_val)} identical events between train and val!"
        )
    if overlap_train_test:
        raise RuntimeError(
            f"Data leakage detected: {len(overlap_train_test)} identical events between train and test!"
        )
    if overlap_val_test:
        raise RuntimeError(
            f"Data leakage detected: {len(overlap_val_test)} identical events between val and test!"
        )

    print("Splits successfully created with zero event overlap.")
    return train_df, val_df, test_df


def save_npz_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
) -> Dict[str, Path]:
    """Save datasets into compressed numpy archives (.npz).

    Args:
        train_df: Training set.
        val_df: Validation set.
        test_df: Test set.
        output_dir: Destination folder.

    Returns:
        Dict mapping split names to saved paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": output_dir / "train.npz",
        "val": output_dir / "val.npz",
        "test": output_dir / "test.npz",
    }

    for name, df, path in [
        ("train", train_df, paths["train"]),
        ("val", val_df, paths["val"]),
        ("test", test_df, paths["test"]),
    ]:
        features = df.iloc[:, 1:].values.astype(np.float32)
        labels = df.iloc[:, 0].values.astype(np.int64)
        np.savez_compressed(path, X=features, y=labels)

    return paths


def save_csv_events(
    test_df: pd.DataFrame,
    test_csv_path: Path,
    sample_csv_path: Path,
    seed: int,
) -> Tuple[Path, Path]:
    """Write test_events.csv and sample_events.csv.

    Args:
        test_df: Full test dataframe (10,000 events).
        test_csv_path: Target path for full test events.
        sample_csv_path: Target path for demo 1,000 sample events.
        seed: Random seed for sampling demo events.

    Returns:
        Tuple of (test_csv_path, sample_csv_path).
    """
    test_csv_path.parent.mkdir(parents=True, exist_ok=True)
    sample_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Save full test split (git-ignored)
    test_df.to_csv(test_csv_path, header=False, index=False, float_format="%.9g")

    # Sample exactly 500 background and 500 signal from test split for demo file
    test_bg = test_df[test_df.iloc[:, 0] == config.LABEL_BACKGROUND]
    test_sig = test_df[test_df.iloc[:, 0] == config.LABEL_SIGNAL]

    demo_bg = test_bg.sample(
        n=config.SAMPLE_EVENTS_BACKGROUND_COUNT, random_state=seed, replace=False
    )
    demo_sig = test_sig.sample(
        n=config.SAMPLE_EVENTS_SIGNAL_COUNT, random_state=seed, replace=False
    )

    demo_df = (
        pd.concat([demo_bg, demo_sig], axis=0)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )

    # Save demo sample events (committed)
    demo_df.to_csv(sample_csv_path, header=False, index=False, float_format="%.9g")

    return test_csv_path, sample_csv_path


def main() -> None:
    """Execute complete data preparation and verification workflow."""
    args = parse_args()
    raw_df = load_and_validate_raw_data(args.input_csv)

    train_df, val_df, test_df = sample_and_split(raw_df, seed=args.seed)

    npz_paths = save_npz_splits(train_df, val_df, test_df, args.output_dir)
    test_csv, sample_csv = save_csv_events(
        test_df,
        config.TEST_EVENTS_PATH,
        config.SAMPLE_EVENTS_PATH,
        seed=args.seed,
    )

    # Print summary output
    print("\n" + "=" * 50)
    print("DATA PREPARATION SUMMARY")
    print("=" * 50)
    print(f"Random Seed: {args.seed}")
    print(
        f"Train Split: shape={train_df.shape}, "
        f"background={(train_df.iloc[:, 0] == 0).sum()}, signal={(train_df.iloc[:, 0] == 1).sum()}"
    )
    print(
        f"Val Split:   shape={val_df.shape}, "
        f"background={(val_df.iloc[:, 0] == 0).sum()}, signal={(val_df.iloc[:, 0] == 1).sum()}"
    )
    print(
        f"Test Split:  shape={test_df.shape}, "
        f"background={(test_df.iloc[:, 0] == 0).sum()}, signal={(test_df.iloc[:, 0] == 1).sum()}"
    )
    print(
        f"Demo Sample: shape=(1000, 29), "
        f"background={config.SAMPLE_EVENTS_BACKGROUND_COUNT}, signal={config.SAMPLE_EVENTS_SIGNAL_COUNT}"
    )
    print("\nFiles Written:")
    print(f"  - {npz_paths['train']} (train.npz)")
    print(f"  - {npz_paths['val']} (val.npz)")
    print(f"  - {npz_paths['test']} (test.npz)")
    print(f"  - {test_csv} (full test CSV)")
    print(f"  - {sample_csv} (sample demo CSV)")
    print("=" * 50)


if __name__ == "__main__":
    main()
