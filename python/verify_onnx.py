"""Verify numerical parity between PyTorch and ONNX Runtime inference.

This script executes strict numerical verification between the trained PyTorch model
and the exported ONNX Runtime engine:
1. Compares reconstruction error outputs on the first 100 events of data/sample_events.csv.
   Pass criteria: maximum absolute difference < 1e-5.
2. Checks binary anomaly decisions (NORMAL vs ANOMALY) at the saved threshold.
3. Evaluates ROC AUC parity across all 4,000 events of data/processed/select.npz.
4. Optionally compares ONNX Runtime errors against C++ inference outputs if provided.
5. Writes comprehensive verification report to results/onnx_verification.md.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
import torch
from sklearn.metrics import roc_auc_score

# Ensure python/ directory is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

import config
from export_onnx import load_model_for_export
from model import Autoencoder


def parse_args() -> argparse.Namespace:
    """Parse command line options for verification script.

    Returns:
        argparse.Namespace: Parsed CLI options.
    """
    parser = argparse.ArgumentParser(
        description="Verify numerical parity between PyTorch and ONNX Runtime."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=config.PYTORCH_MODEL_PATH,
        help=f"Path to PyTorch checkpoint (default: {config.PYTORCH_MODEL_PATH}).",
    )
    parser.add_argument(
        "--onnx-path",
        type=Path,
        default=config.ONNX_MODEL_PATH,
        help=f"Path to exported ONNX model (default: {config.ONNX_MODEL_PATH}).",
    )
    parser.add_argument(
        "--sample-events",
        type=Path,
        default=config.SAMPLE_EVENTS_PATH,
        help=f"Path to sample events CSV (default: {config.SAMPLE_EVENTS_PATH}).",
    )
    parser.add_argument(
        "--select-path",
        type=Path,
        default=config.SELECT_SPLIT_PATH,
        help=f"Path to select.npz dataset (default: {config.SELECT_SPLIT_PATH}).",
    )
    parser.add_argument(
        "--threshold-path",
        type=Path,
        default=config.THRESHOLD_TXT_PATH,
        help=f"Path to decision threshold file (default: {config.THRESHOLD_TXT_PATH}).",
    )
    parser.add_argument(
        "--cpp-errors",
        type=Path,
        default=None,
        help="Optional path to C++ error output file (results/cpp_errors.txt).",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=config.RESULTS_DIR / "onnx_verification.md",
        help="Path for markdown verification report.",
    )
    parser.add_argument(
        "--num-events",
        type=int,
        default=config.VERIFY_NUM_EVENTS,
        help=f"Number of sample events to evaluate (default: {config.VERIFY_NUM_EVENTS}).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=config.VERIFY_TOLERANCE,
        help=f"Maximum acceptable absolute error tolerance (default: {config.VERIFY_TOLERANCE}).",
    )
    return parser.parse_args()


def load_sample_events(
    csv_path: Path, num_events: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """Load first N events from demo CSV.

    Args:
        csv_path: Path to sample_events.csv (column 0 = label, cols 1..28 = features).
        num_events: Number of rows to load.

    Returns:
        Tuple of (labels, features).
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Sample events file not found at '{csv_path}'.")

    # Load with numpy (no header, comma separated)
    data = np.loadtxt(csv_path, delimiter=",", max_rows=num_events, dtype=np.float32)
    labels = data[:, 0].astype(np.int64)
    features = data[:, 1:].astype(np.float32)

    if features.shape[1] != config.NUM_FEATURES:
        raise ValueError(
            f"Expected {config.NUM_FEATURES} features, got {features.shape[1]}."
        )

    return labels, features


def create_ort_session(onnx_path: Path) -> ort.InferenceSession:
    """Instantiate ONNX Runtime inference session with single-threaded CPU execution.

    Args:
        onnx_path: Path to ONNX model file.

    Returns:
        ort.InferenceSession: Initialized session.
    """
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model file not found at '{onnx_path}'.")

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    return session


def compute_pytorch_errors(
    model: Autoencoder, features: np.ndarray
) -> np.ndarray:
    """Compute per-event reconstruction error using PyTorch in inference mode.

    Args:
        model: Trained Autoencoder.
        features: Feature array of shape [N, 28].

    Returns:
        np.ndarray: Per-event errors array of shape [N].
    """
    model.eval()
    torch.set_num_threads(1)
    with torch.no_grad():
        tensor = torch.from_numpy(features).float()
        errors = model.reconstruction_error(tensor).cpu().numpy()
    return errors


def compute_ort_single_event_errors(
    session: ort.InferenceSession, features: np.ndarray
) -> np.ndarray:
    """Compute per-event reconstruction error using ONNX Runtime with batch size 1.

    Args:
        session: Active ORT InferenceSession.
        features: Feature array of shape [N, 28].

    Returns:
        np.ndarray: Per-event errors array of shape [N].
    """
    input_name = session.get_inputs()[0].name
    errors = np.zeros(len(features), dtype=np.float32)

    for i in range(len(features)):
        # Shape: [1, 28]
        event_input = features[i : i + 1]
        out = session.run(None, {input_name: event_input})
        errors[i] = float(out[0][0])

    return errors


def load_cpp_errors(cpp_path: Path, max_count: int = 100) -> Optional[np.ndarray]:
    """Load C++ error outputs from file if present.

    Args:
        cpp_path: Path to cpp_errors.txt.
        max_count: Maximum events to load.

    Returns:
        Optional[np.ndarray]: Array of errors or None if file missing.
    """
    if cpp_path is None or not cpp_path.exists():
        return None

    try:
        errors = np.loadtxt(cpp_path, dtype=np.float32, max_rows=max_count)
        return errors
    except Exception as e:
        print(f"Warning: Failed to parse C++ errors from '{cpp_path}': {e}")
        return None


def generate_verification_report(
    max_abs_diff: float,
    mean_abs_diff: float,
    tolerance: float,
    passed: bool,
    disagreements: int,
    num_sample_events: int,
    threshold: float,
    pytorch_auc: float,
    ort_auc: float,
    auc_diff: float,
    num_select_events: int,
    cpp_max_diff: Optional[float],
    cpp_mean_diff: Optional[float],
    cpp_available: bool,
) -> str:
    """Format verification metrics as a clean Markdown report.

    Returns:
        str: Markdown document content.
    """
    status_str = "**PASSED**" if passed else "**FAILED**"
    if cpp_available:
        if cpp_max_diff < tolerance:
            cpp_status_str = f"**PASSED** (Max abs diff vs ORT: {cpp_max_diff:.3e}, Mean: {cpp_mean_diff:.3e})"
        else:
            cpp_status_str = f"**FAILED** (Max abs diff vs ORT: {cpp_max_diff:.3e} exceeds tolerance {tolerance:.1e})"
    else:
        cpp_status_str = "*(Skipped: C++ benchmark output not yet generated)*"

    lines = [
        "# ONNX Export Numerical Parity Verification Report",
        "",
        "## Summary",
        "",
        f"- **Verification Status**: {status_str}",
        f"- **Tolerance Criterion**: Maximum Absolute Difference < {tolerance:.1e}",
        "",
        "## 1. Single-Event Sample Comparison (First 100 Events)",
        "",
        f"Evaluated on the first {num_sample_events} events of `data/sample_events.csv` with batch size 1.",
        "",
        "| Metric | Measured Value | Threshold / Target | Status |",
        "| :--- | :---: | :---: | :---: |",
        f"| **Maximum Absolute Difference** | **{max_abs_diff:.6e}** | < {tolerance:.1e} | {status_str} |",
        f"| **Mean Absolute Difference** | **{mean_abs_diff:.6e}** | — | — |",
        f"| **Decision Disagreements** | **{disagreements}** / {num_sample_events} | 0 disagreements | {'**MATCH**' if disagreements == 0 else '**DISAGREEMENT**'} |",
        f"| **Anomaly Threshold Used** | `{threshold:.6f}` | `models/threshold.txt` | Validated |",
        "",
        "## 2. Selection Set Full Evaluation (4,000 Events)",
        "",
        f"Evaluated across all {num_select_events:,} events of `data/processed/select.npz` (2,000 background + 2,000 signal) to verify metric consistency.",
        "",
        "| Metric | PyTorch Eager (CPU) | ONNX Runtime (CPU) | Absolute Difference |",
        "| :--- | :---: | :---: | :---: |",
        f"| **ROC AUC** | **{pytorch_auc:.6f}** | **{ort_auc:.6f}** | **{auc_diff:.6e}** |",
        "",
        "## 3. C++ Inference Cross-Comparison",
        "",
        f"- **Status**: {cpp_status_str}",
        "",
        "## Environment & Hardware",
        "",
        "- **ONNX Runtime Version**: `" + ort.__version__ + "`",
        "- **PyTorch Version**: `" + torch.__version__ + "`",
        "- **Execution Provider**: `CPUExecutionProvider` (single thread)",
        "- **Batch Size**: `1` (single event inference)",
        "",
        "*Generated automatically by `python/verify_onnx.py`.*",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    """Execute complete numerical verification workflow."""
    args = parse_args()
    start_time = time.perf_counter()

    print("=" * 65)
    print("COLLIDER ANOMALY DETECTION: ONNX PARITY VERIFICATION")
    print("=" * 65)
    print(f"PyTorch Checkpoint:  {args.model_path}")
    print(f"ONNX Model Path:     {args.onnx_path}")
    print(f"Sample Events CSV:   {args.sample_events}")
    print(f"Selection Dataset:   {args.select_path}")
    print(f"Tolerance Threshold: {args.tolerance:.1e}")
    print("=" * 65)

    # 1. Load models and threshold
    pytorch_model, variant = load_model_for_export(args.model_path)
    ort_session = create_ort_session(args.onnx_path)

    if not args.threshold_path.exists():
        raise FileNotFoundError(f"Threshold file not found at '{args.threshold_path}'.")
    with open(args.threshold_path, "r", encoding="utf-8") as f:
        threshold = float(f.read().strip())

    print(f"Active Variant:      {variant}")
    print(f"Anomaly Threshold:   {threshold:.6f}")

    # 2. Evaluate first 100 sample events
    labels_100, features_100 = load_sample_events(args.sample_events, num_events=args.num_events)
    print(f"\nEvaluating first {len(features_100)} sample events (single-event inference)...")

    pytorch_errors_100 = compute_pytorch_errors(pytorch_model, features_100)
    ort_errors_100 = compute_ort_single_event_errors(ort_session, features_100)

    diffs_100 = np.abs(pytorch_errors_100 - ort_errors_100)
    max_abs_diff = float(np.max(diffs_100))
    mean_abs_diff = float(np.mean(diffs_100))

    passed = bool(max_abs_diff < args.tolerance)

    pytorch_decisions = pytorch_errors_100 >= threshold
    ort_decisions = ort_errors_100 >= threshold
    disagreements = int(np.sum(pytorch_decisions != ort_decisions))

    print(f"  -> Maximum Absolute Difference: {max_abs_diff:.6e} (Tolerance: {args.tolerance:.1e})")
    print(f"  -> Mean Absolute Difference:    {mean_abs_diff:.6e}")
    print(f"  -> Parity Check:                {'PASSED' if passed else 'FAILED'}")
    print(f"  -> Decision Disagreements:      {disagreements} / {len(features_100)}")

    # 3. Evaluate selection dataset (4,000 events)
    if not args.select_path.exists():
        raise FileNotFoundError(f"Selection dataset not found at '{args.select_path}'.")

    select_data = np.load(args.select_path)
    select_X = select_data["X"].astype(np.float32)
    select_y = select_data["y"].astype(np.int64)

    print(f"\nEvaluating complete selection dataset ({len(select_X):,} events)...")
    pytorch_select_errors = compute_pytorch_errors(pytorch_model, select_X)
    ort_select_errors = compute_ort_single_event_errors(ort_session, select_X)

    pytorch_auc = float(roc_auc_score(select_y, pytorch_select_errors))
    ort_auc = float(roc_auc_score(select_y, ort_select_errors))
    auc_diff = float(abs(pytorch_auc - ort_auc))

    print(f"  -> PyTorch Selection ROC AUC:   {pytorch_auc:.6f}")
    print(f"  -> ONNX Runtime ROC AUC:        {ort_auc:.6f}")
    print(f"  -> AUC Absolute Difference:     {auc_diff:.6e}")

    # 4. Optional C++ errors cross-check
    cpp_errors_path = args.cpp_errors or (config.RESULTS_DIR / "cpp_errors.txt")
    cpp_errors = load_cpp_errors(cpp_errors_path, max_count=len(features_100))

    if cpp_errors is not None:
        cpp_count = min(len(cpp_errors), len(ort_errors_100))
        cpp_diffs = np.abs(ort_errors_100[:cpp_count] - cpp_errors[:cpp_count])
        cpp_max_diff = float(np.max(cpp_diffs))
        cpp_mean_diff = float(np.mean(cpp_diffs))
        cpp_available = True
        cpp_passed = bool(cpp_max_diff < args.tolerance)
        if not cpp_passed:
            passed = False
        print(f"\nC++ Errors Cross-Check ({cpp_count} events from '{cpp_errors_path}'):")
        print(f"  -> C++ vs ORT Max Abs Diff:    {cpp_max_diff:.6e} (Tolerance: {args.tolerance:.1e})")
        print(f"  -> C++ vs ORT Mean Abs Diff:   {cpp_mean_diff:.6e}")
        print(f"  -> C++ Parity Check:           {'PASSED' if cpp_passed else 'FAILED'}")
    else:
        cpp_max_diff = None
        cpp_mean_diff = None
        cpp_available = False
        print(f"\nC++ Errors Cross-Check: Skipped (no output found at '{cpp_errors_path}')")

    # 5. Write report
    report_content = generate_verification_report(
        max_abs_diff=max_abs_diff,
        mean_abs_diff=mean_abs_diff,
        tolerance=args.tolerance,
        passed=passed,
        disagreements=disagreements,
        num_sample_events=len(features_100),
        threshold=threshold,
        pytorch_auc=pytorch_auc,
        ort_auc=ort_auc,
        auc_diff=auc_diff,
        num_select_events=len(select_X),
        cpp_max_diff=cpp_max_diff,
        cpp_mean_diff=cpp_mean_diff,
        cpp_available=cpp_available,
    )

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(report_content)

    elapsed = time.perf_counter() - start_time
    print("\n" + "=" * 65)
    print(f"VERIFICATION COMPLETE in {elapsed:.2f} seconds")
    print(f"Report saved to: {args.out_md}")
    print("=" * 65)

    if not passed:
        print("ERROR: Numerical tolerance check failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
