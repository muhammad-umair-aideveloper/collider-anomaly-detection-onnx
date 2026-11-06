"""Unit tests for dataset splits, label purity, and sample event sanity."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure repo root is in python path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

import config


@pytest.fixture(scope="module")
def processed_data():
    """Load processed numpy split archives.

    Skips the tests if processed data files are not present.
    """
    if (
        not config.TRAIN_SPLIT_PATH.exists()
        or not config.VAL_SPLIT_PATH.exists()
        or not config.TEST_SPLIT_PATH.exists()
    ):
        pytest.skip(
            f"Processed data files not found in {config.PROCESSED_DATA_DIR}. "
            "Run python/prepare_data.py first."
        )

    train = np.load(config.TRAIN_SPLIT_PATH)
    val = np.load(config.VAL_SPLIT_PATH)
    test = np.load(config.TEST_SPLIT_PATH)

    return {
        "train_X": train["X"],
        "train_y": train["y"],
        "val_X": val["X"],
        "val_y": val["y"],
        "test_X": test["X"],
        "test_y": test["y"],
    }


def test_split_shapes_and_sizes(processed_data):
    """Verify that split dimensions exactly match the specifications."""
    assert processed_data["train_X"].shape == (
        config.TRAIN_BACKGROUND_COUNT,
        config.NUM_FEATURES,
    )
    assert processed_data["train_y"].shape == (config.TRAIN_BACKGROUND_COUNT,)

    assert processed_data["val_X"].shape == (
        config.VAL_BACKGROUND_COUNT,
        config.NUM_FEATURES,
    )
    assert processed_data["val_y"].shape == (config.VAL_BACKGROUND_COUNT,)

    expected_test_size = config.TEST_BACKGROUND_COUNT + config.TEST_SIGNAL_COUNT
    assert processed_data["test_X"].shape == (expected_test_size, config.NUM_FEATURES)
    assert processed_data["test_y"].shape == (expected_test_size,)


def test_label_purity_and_distribution(processed_data):
    """Verify that train/val contain only background and test contains balanced classes."""
    # Train set must contain exclusively background events (label 0)
    assert np.all(processed_data["train_y"] == config.LABEL_BACKGROUND)

    # Validation set must contain exclusively background events (label 0)
    assert np.all(processed_data["val_y"] == config.LABEL_BACKGROUND)

    # Test set must contain exactly 5,000 background and 5,000 signal events
    test_y = processed_data["test_y"]
    bg_count = int((test_y == config.LABEL_BACKGROUND).sum())
    sig_count = int((test_y == config.LABEL_SIGNAL).sum())

    assert bg_count == config.TEST_BACKGROUND_COUNT
    assert sig_count == config.TEST_SIGNAL_COUNT
    assert bg_count + sig_count == len(test_y)


def test_no_nan_or_inf(processed_data):
    """Verify that no feature or label array contains NaN or Inf values."""
    for key, arr in processed_data.items():
        assert np.all(np.isfinite(arr)), f"Non-finite values detected in {key}."


def test_no_overlap_between_splits(processed_data):
    """Ensure zero data leakage: no collision of feature vectors across splits."""
    train_rows = set(map(tuple, processed_data["train_X"]))
    val_rows = set(map(tuple, processed_data["val_X"]))
    test_rows = set(map(tuple, processed_data["test_X"]))

    # Compare set intersections
    overlap_train_val = train_rows.intersection(val_rows)
    overlap_train_test = train_rows.intersection(test_rows)
    overlap_val_test = val_rows.intersection(test_rows)

    assert (
        len(overlap_train_val) == 0
    ), f"Leakage: {len(overlap_train_val)} overlapping rows between train and val."
    assert (
        len(overlap_train_test) == 0
    ), f"Leakage: {len(overlap_train_test)} overlapping rows between train and test."
    assert (
        len(overlap_val_test) == 0
    ), f"Leakage: {len(overlap_val_test)} overlapping rows between val and test."


def test_sample_events_csv():
    """Verify schema and balance of the committed sample_events.csv."""
    if not config.SAMPLE_EVENTS_PATH.exists():
        pytest.skip(f"Sample events CSV not found at {config.SAMPLE_EVENTS_PATH}.")

    df = pd.read_csv(config.SAMPLE_EVENTS_PATH, header=None)

    assert df.shape == (config.SAMPLE_EVENTS_COUNT, config.NUM_FEATURES + 1)
    assert df.isna().sum().sum() == 0
    assert np.all(np.isfinite(df.values))

    labels = df.iloc[:, 0].astype(int)
    bg_count = int((labels == config.LABEL_BACKGROUND).sum())
    sig_count = int((labels == config.LABEL_SIGNAL).sum())

    assert bg_count == config.SAMPLE_EVENTS_BACKGROUND_COUNT
    assert sig_count == config.SAMPLE_EVENTS_SIGNAL_COUNT

