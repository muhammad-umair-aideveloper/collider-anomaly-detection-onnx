"""Central configuration constants for collider anomaly detection pipeline.

This module defines all hyperparameters, dataset split specifications, model
architecture dimensions, ONNX export configurations, benchmark parameters, and
canonical file paths anchored to the repository root.
"""

from pathlib import Path
from typing import Tuple

# ==============================================================================
# Random Seed & Reproducibility
# ==============================================================================
SEED: int = 42

# ==============================================================================
# HIGGS Dataset & Feature Specifications
# ==============================================================================
NUM_FEATURES: int = 28
LABEL_BACKGROUND: int = 0
LABEL_SIGNAL: int = 1

# Stratified sampling targets from data/higgs_sample.csv (20,000 events total)
SAMPLE_BACKGROUND_COUNT: int = 15000
SAMPLE_SIGNAL_COUNT: int = 5000

# Partition split sizes (AGENTS.md Section 8.1)
TRAIN_BACKGROUND_COUNT: int = 8000
VAL_BACKGROUND_COUNT: int = 2000
TEST_BACKGROUND_COUNT: int = 5000
TEST_SIGNAL_COUNT: int = 5000

# Selection set split sizes (disjoint from train, val, test)
SELECT_BACKGROUND_COUNT: int = 2000
SELECT_SIGNAL_COUNT: int = 2000

# Demo sample events count for committed test sample
SAMPLE_EVENTS_COUNT: int = 1000
SAMPLE_EVENTS_BACKGROUND_COUNT: int = 500
SAMPLE_EVENTS_SIGNAL_COUNT: int = 500

# ==============================================================================
# Model Architecture & Hyperparameters
# ==============================================================================
AUTOENCODER_LAYERS: Tuple[int, ...] = (28, 16, 8, 16, 28)
STANDARDIZATION_EPSILON: float = 1e-7

# Training hyperparameters
DEFAULT_EPOCHS: int = 30
DEFAULT_BATCH_SIZE: int = 64
DEFAULT_LEARNING_RATE: float = 1e-3

# Anomaly detection decision threshold
# 95th percentile of validation reconstruction error on background events
THRESHOLD_PERCENTILE: float = 95.0

# ==============================================================================
# ONNX Export & Verification Settings
# ==============================================================================
ONNX_OPSET: int = 17
ONNX_INPUT_NAME: str = "features"
ONNX_OUTPUT_NAME: str = "error"
ONNX_BATCH_SIZE: int = 1

# Verification tolerances
VERIFY_NUM_EVENTS: int = 100
VERIFY_TOLERANCE: float = 1e-5

# ==============================================================================
# Benchmark Configuration
# ==============================================================================
BENCHMARK_WARMUP_RUNS: int = 100
BENCHMARK_TIMED_RUNS: int = 10000

# ==============================================================================
# Feature selection groups
PHI_FEATURE_INDICES: Tuple[int, ...] = (2, 4, 7, 11, 15, 19)
HIGH_LEVEL_FEATURE_INDICES: Tuple[int, ...] = (21, 22, 23, 24, 25, 26, 27)
NON_PHI_FEATURE_INDICES: Tuple[int, ...] = (
    0, 1, 3, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27
)

# Repository Paths (Anchored to Repo Root)
# ==============================================================================
REPO_ROOT: Path = Path(__file__).resolve().parent.parent

# Data directory and datasets
DATA_DIR: Path = REPO_ROOT / "data"
HIGGS_RAW_PATH: Path = DATA_DIR / "higgs_sample.csv"
SAMPLE_EVENTS_PATH: Path = DATA_DIR / "sample_events.csv"
TEST_EVENTS_PATH: Path = DATA_DIR / "test_events.csv"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
TRAIN_SPLIT_PATH: Path = PROCESSED_DATA_DIR / "train.npz"
VAL_SPLIT_PATH: Path = PROCESSED_DATA_DIR / "val.npz"
TEST_SPLIT_PATH: Path = PROCESSED_DATA_DIR / "test.npz"
SELECT_SPLIT_PATH: Path = PROCESSED_DATA_DIR / "select.npz"

# Model artifacts
MODELS_DIR: Path = REPO_ROOT / "models"
PYTORCH_MODEL_PATH: Path = MODELS_DIR / "autoencoder.pt"
ONNX_MODEL_PATH: Path = MODELS_DIR / "detector_model.onnx"
THRESHOLD_JSON_PATH: Path = MODELS_DIR / "threshold.json"
THRESHOLD_TXT_PATH: Path = MODELS_DIR / "threshold.txt"

# Evaluation and benchmark outputs
RESULTS_DIR: Path = REPO_ROOT / "results"
EXPERIMENTS_DIR: Path = RESULTS_DIR / "experiments"
VARIANT_SELECTION_MD_PATH: Path = EXPERIMENTS_DIR / "variant_selection.md"
LOSS_CURVE_PATH: Path = RESULTS_DIR / "loss_curve.png"
ROC_CURVE_PATH: Path = RESULTS_DIR / "roc_curve.png"
ERROR_HISTOGRAM_PATH: Path = RESULTS_DIR / "error_histogram.png"
METRICS_PATH: Path = RESULTS_DIR / "metrics.md"
CPP_ERRORS_PATH: Path = RESULTS_DIR / "cpp_errors.txt"
CPP_BENCHMARK_PATH: Path = RESULTS_DIR / "cpp_benchmark.json"
BENCHMARK_MD_PATH: Path = RESULTS_DIR / "benchmark.md"

