"""Download and prepare the OpenML HIGGS dataset sample.

This script fetches the UCI HIGGS dataset subset from OpenML (data_id 23512),
validates the schema, formats label 0 (background) and 1 (signal) as column 0
followed by the 28 numerical features, and saves it to data/higgs_sample.csv.
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.datasets import fetch_openml

# Ensure repo root is in python path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

import config

OPENML_DATA_ID: int = 23512


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the download script.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Download UCI HIGGS dataset from OpenML and save as CSV."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite data/higgs_sample.csv if it already exists.",
    )
    return parser.parse_args()


def download_openml_higgs(data_id: int = OPENML_DATA_ID) -> pd.DataFrame:
    """Fetch OpenML HIGGS dataset and construct validated DataFrame.

    Args:
        data_id: OpenML dataset ID.

    Returns:
        pd.DataFrame: Formatted DataFrame with label in column 0 and 28 features.

    Raises:
        RuntimeError: If download fails or schema validation fails.
    """
    print(f"Fetching HIGGS dataset from OpenML (data_id={data_id})...")
    try:
        # Use parser='liac-arff' for robust parsing of integer nominal attributes
        dataset = fetch_openml(data_id=data_id, as_frame=True, parser="liac-arff")
    except Exception as exc:
        raise RuntimeError(
            f"Network or download error while fetching OpenML dataset {data_id}: {exc}"
        ) from exc

    features: pd.DataFrame = dataset.data
    raw_labels = dataset.target

    # Convert nominal target '0'/'1' to integer labels
    labels = raw_labels.astype(int)

    # Validate row count and labels
    num_rows = len(labels)
    unique_labels = set(labels.unique())
    if not unique_labels.issubset({0, 1}):
        raise RuntimeError(
            f"Unexpected label values found: {unique_labels}. Expected only {{0, 1}}."
        )

    # Validate feature dimensions
    num_features = features.shape[1]
    if num_features != config.NUM_FEATURES:
        raise RuntimeError(
            f"Expected {config.NUM_FEATURES} features, got {num_features}."
        )

    # Assemble combined DataFrame with label in column 0
    combined = pd.concat([labels.rename("label"), features], axis=1)

    # Validate column count
    expected_cols = config.NUM_FEATURES + 1
    if combined.shape[1] != expected_cols:
        raise RuntimeError(
            f"Expected {expected_cols} columns, got {combined.shape[1]}."
        )

    # Validate missing values
    num_nan = combined.isna().sum().sum()
    if num_nan > 0:
        raise RuntimeError(f"Dataset contains {num_nan} NaN values; expected 0.")

    # Print summary statistics
    count_background = int((labels == 0).sum())
    count_signal = int((labels == 1).sum())

    print(f"OpenML data_id: {data_id}")
    print(f"Total rows fetched: {num_rows}")
    print(f"Total columns: {combined.shape[1]} (1 label + {num_features} features)")
    print(f"Background events (label=0): {count_background}")
    print(f"Signal events (label=1): {count_signal}")

    return combined


def save_higgs_sample(df: pd.DataFrame, output_path: Path, force: bool = False) -> None:
    """Save formatted HIGGS DataFrame to CSV.

    Args:
        df: DataFrame to save.
        output_path: Target CSV file path.
        force: Overwrite existing file if True.

    Raises:
        FileExistsError: If target file exists and force is False.
    """
    if output_path.exists() and not force:
        raise FileExistsError(
            f"File already exists at '{output_path}'. Use --force to overwrite."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {output_path} (float_format='%.9g')...")
    df.to_csv(output_path, header=False, index=False, float_format="%.9g")
    print(f"Successfully saved {len(df)} rows to {output_path}.")


def main() -> None:
    """Entry point for downloading and verifying HIGGS sample."""
    args = parse_args()
    target_path = config.HIGGS_RAW_PATH

    if target_path.exists() and not args.force:
        print(
            f"Error: Target file '{target_path}' already exists. Pass --force to overwrite.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        df = download_openml_higgs(OPENML_DATA_ID)
        save_higgs_sample(df, target_path, force=args.force)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
