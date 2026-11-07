"""Unit tests for latency benchmark statistical calculations and parity checks."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from benchmark import (
    compute_headline_metrics,
    compute_run_statistics,
    compute_speedup,
    parse_args,
    run_ort_cpp_benchmark,
    verify_pytorch_ort_parity,
)


def test_compute_run_statistics_known_values() -> None:
    """Verify statistical aggregation (mean, median, p99, min, max) on known data."""
    # 100 evenly spaced numbers from 1.0 to 100.0
    data = [float(i) for i in range(1, 101)]
    stats = compute_run_statistics(data)

    arr = np.array(data, dtype=np.float64)
    expected_mean = float(np.mean(arr))
    expected_median = float(np.median(arr))
    expected_p99 = float(np.percentile(arr, 99.0, method="linear"))
    expected_min = float(np.min(arr))
    expected_max = float(np.max(arr))

    assert abs(stats["mean"] - expected_mean) < 1e-9
    assert abs(stats["median"] - expected_median) < 1e-9
    assert abs(stats["p99"] - expected_p99) < 1e-9
    assert abs(stats["min"] - expected_min) < 1e-9
    assert abs(stats["max"] - expected_max) < 1e-9


def test_compute_run_statistics_empty() -> None:
    """Verify handling of empty latency lists returns zeros gracefully."""
    stats = compute_run_statistics([])
    assert stats["mean"] == 0.0
    assert stats["median"] == 0.0
    assert stats["p99"] == 0.0
    assert stats["min"] == 0.0
    assert stats["max"] == 0.0


def test_compute_headline_metrics() -> None:
    """Verify aggregation across interleaved repetitions (median and spread)."""
    repetitions = [
        {"mean": 10.0, "median": 9.5, "p99": 15.0, "min": 8.0, "max": 20.0},
        {"mean": 11.0, "median": 10.0, "p99": 16.0, "min": 8.5, "max": 22.0},
        {"mean": 10.5, "median": 9.8, "p99": 15.5, "min": 8.2, "max": 21.0},
    ]

    headline = compute_headline_metrics(repetitions)

    # Median across [9.5, 10.0, 9.8] is 9.8
    assert abs(headline["median"]["median"] - 9.8) < 1e-9
    assert abs(headline["median"]["min"] - 9.5) < 1e-9
    assert abs(headline["median"]["max"] - 10.0) < 1e-9

    # Spread percentage = (10.0 - 9.5) / 9.8 * 100
    expected_spread = (10.0 - 9.5) / 9.8 * 100.0
    assert abs(headline["median"]["spread_pct"] - expected_spread) < 1e-6


def test_compute_speedup() -> None:
    """Verify relative speedup ratio and invalid input handling."""
    speedup = compute_speedup(baseline_latency=100.0, target_latency=25.0)
    assert speedup is not None
    assert abs(speedup - 4.0) < 1e-9

    assert compute_speedup(100.0, 0.0) is None
    assert compute_speedup(0.0, 50.0) is None
    assert compute_speedup(-10.0, 20.0) is None


def test_skipped_cpp_path_handling(tmp_path: Path) -> None:
    """Verify run_ort_cpp_benchmark handles missing executable by returning None."""
    non_existent_exe = tmp_path / "does_not_exist.exe"
    dummy_file = tmp_path / "dummy.txt"
    dummy_file.write_text("0.0\n", encoding="utf-8")

    result = run_ort_cpp_benchmark(
        cpp_exe=non_existent_exe,
        csv_path=dummy_file,
        model_path=dummy_file,
        threshold_path=dummy_file,
        json_path=tmp_path / "out.json",
        warmup_runs=1,
        timed_runs=1,
    )

    assert result is None


def test_verify_pytorch_ort_parity_success() -> None:
    """Verify pre-benchmark parity check passes when outputs match within tolerance."""
    dummy_wrapper = MagicMock()
    # Return 0.05 for any event
    dummy_wrapper.side_effect = lambda x: torch.tensor([0.05])

    dummy_session = MagicMock()
    input_node = MagicMock()
    input_node.name = "features"
    dummy_session.get_inputs.return_value = [input_node]
    # Return 0.050001 with output shape [1] matching detector_model.onnx (diff = 1e-6 < 1e-5)
    dummy_session.run.return_value = [np.array([0.050001], dtype=np.float32)]

    features_20 = np.zeros((20, 28), dtype=np.float32)

    # Should not raise any exception
    verify_pytorch_ort_parity(
        dummy_wrapper,
        dummy_session,
        features_20,
        tolerance=1e-5,
    )


def test_verify_pytorch_ort_parity_failure() -> None:
    """Verify pre-benchmark parity check raises RuntimeError when outputs exceed tolerance."""
    dummy_wrapper = MagicMock()
    dummy_wrapper.side_effect = lambda x: torch.tensor([0.05])

    dummy_session = MagicMock()
    input_node = MagicMock()
    input_node.name = "features"
    dummy_session.get_inputs.return_value = [input_node]
    # Return 0.06 with output shape [1] matching detector_model.onnx (diff = 0.01 >= 1e-5)
    dummy_session.run.return_value = [np.array([0.06], dtype=np.float32)]

    features_20 = np.zeros((20, 28), dtype=np.float32)

    with pytest.raises(RuntimeError) as exc_info:
        verify_pytorch_ort_parity(
            dummy_wrapper,
            dummy_session,
            features_20,
            tolerance=1e-5,
        )
    assert "Pre-benchmark parity check failed" in str(exc_info.value)


def test_parse_args_out_dir_routing(tmp_path: Path) -> None:
    """Verify that specifying --out-dir redirects default outputs into that directory."""
    custom_dir = tmp_path / "custom_run"
    args = parse_args(["--out-dir", str(custom_dir)])

    assert args.out_dir == custom_dir
    assert args.out_md == custom_dir / "benchmark.md"
    assert args.out_raw_json == custom_dir / "benchmark_raw.json"
    assert args.cpp_json == custom_dir / "cpp_benchmark.json"
    assert custom_dir.exists()

