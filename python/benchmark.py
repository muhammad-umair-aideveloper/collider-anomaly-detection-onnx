"""Three-way single-event inference latency benchmark.

This module measures single-event latency (batch size 1) across three execution paths:
1. PyTorch Eager (CPU, 1 thread, torch.inference_mode())
2. ONNX Runtime Python (CPU, 1 intra-op, 1 inter-op thread, ORT_ENABLE_ALL)
3. ONNX Runtime C++ (Release build, 1 intra-op, 1 inter-op thread)

Protocol:
- Same model (detector_model.onnx derived from V2 autoencoder).
- Same 1,000 events of data/sample_events.csv preloaded in memory.
- Pre-timing parity verification (first 20 events match within 1e-5).
- 100 untimed warmup runs followed by 10,000 timed runs per repetition.
- 3 full interleaved repetitions per path.
- Headline statistics are the MEDIAN across repetitions; spread (min/max) is reported.
- Only the pure inference call is timed using nanosecond steady clocks.
"""

import argparse
import datetime
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
import torch

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

import config
from export_onnx import AnomalyDetectorExportWrapper, load_model_for_export


def parse_args(args_list: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI options for latency benchmark.

    Args:
        args_list: Optional list of CLI arguments (for programmatic invocation / testing).

    Returns:
        argparse.Namespace: Parsed CLI options.
    """
    parser = argparse.ArgumentParser(
        description="Run three-way single-event inference latency benchmark."
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
        help=f"Path to ONNX model (default: {config.ONNX_MODEL_PATH}).",
    )
    parser.add_argument(
        "--sample-events",
        type=Path,
        default=config.SAMPLE_EVENTS_PATH,
        help=f"Path to sample events CSV (default: {config.SAMPLE_EVENTS_PATH}).",
    )
    parser.add_argument(
        "--cpp-exe",
        type=Path,
        default=REPO_ROOT / "build" / "Release" / "collider_inference.exe",
        help="Path to compiled C++ inference executable.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Target directory for benchmark outputs (default: results/).",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=config.BENCHMARK_MD_PATH,
        help=f"Destination markdown report path (default: {config.BENCHMARK_MD_PATH}).",
    )
    parser.add_argument(
        "--out-raw-json",
        type=Path,
        default=config.RESULTS_DIR / "benchmark_raw.json",
        help="Destination raw JSON report path.",
    )
    parser.add_argument(
        "--cpp-json",
        type=Path,
        default=config.CPP_BENCHMARK_PATH,
        help=f"Path for C++ benchmark output JSON (default: {config.CPP_BENCHMARK_PATH}).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=config.BENCHMARK_WARMUP_RUNS,
        help=f"Warmup runs per path (default: {config.BENCHMARK_WARMUP_RUNS}).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=config.BENCHMARK_TIMED_RUNS,
        help=f"Timed runs per path (default: {config.BENCHMARK_TIMED_RUNS}).",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="Number of interleaved benchmark repetitions (default: 3).",
    )
    args = parser.parse_args(args_list)

    # If --out-dir is specified, route standard output artifacts into it unless explicitly overridden
    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        if args.out_md == config.BENCHMARK_MD_PATH:
            args.out_md = args.out_dir / "benchmark.md"
        if args.out_raw_json == config.RESULTS_DIR / "benchmark_raw.json":
            args.out_raw_json = args.out_dir / "benchmark_raw.json"
        if args.cpp_json == config.CPP_BENCHMARK_PATH:
            args.cpp_json = args.out_dir / "cpp_benchmark.json"

    return args


def get_hardware_info() -> Dict[str, str]:
    """Retrieve detailed host processor and OS metadata using PowerShell CIM/WMI.

    Returns:
        Dict[str, str]: CPU name, core counts, and system attributes.
    """
    cpu_name = platform.processor() or "Unknown CPU"
    cores = "Unknown"
    threads = "Unknown"

    try:
        cmd = [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors | ConvertTo-Json",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(res.stdout)
        if isinstance(data, list) and len(data) > 0:
            data = data[0]
        if isinstance(data, dict):
            cpu_name = data.get("Name", cpu_name).strip()
            cores = str(data.get("NumberOfCores", cores))
            threads = str(data.get("NumberOfLogicalProcessors", threads))
    except Exception as e:
        print(f"Warning: Failed to query Win32_Processor CIM instance: {e}")

    os_info = f"{platform.system()} {platform.release()} (Build {platform.version()})"
    return {
        "cpu_name": cpu_name,
        "cores": cores,
        "threads": threads,
        "os": os_info,
    }


def compute_run_statistics(latencies_us: List[float]) -> Dict[str, float]:
    """Compute central tendency and dispersion metrics from a single timed run.

    Percentile definition: Uses standard numpy linear interpolation (method='linear').

    Args:
        latencies_us: List of per-run latencies in microseconds.

    Returns:
        Dict[str, float]: mean, median, p99, min, max in microseconds.
    """
    if not latencies_us:
        return {"mean": 0.0, "median": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0}

    arr = np.array(latencies_us, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p99": float(np.percentile(arr, 99.0, method="linear")),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def compute_headline_metrics(
    repetitions: List[Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """Derive headline median and spread [min, max] across multiple repetitions.

    Args:
        repetitions: List of per-repetition metric dictionaries.

    Returns:
        Dict[str, Dict[str, float]]: Headline stats per metric.
    """
    metrics = ["mean", "median", "p99", "min", "max"]
    result: Dict[str, Dict[str, float]] = {}

    for m in metrics:
        values = [rep[m] for rep in repetitions if m in rep]
        if values:
            result[m] = {
                "median": float(np.median(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "spread_pct": (
                    float((np.max(values) - np.min(values)) / np.median(values) * 100.0)
                    if np.median(values) > 0
                    else 0.0
                ),
            }
        else:
            result[m] = {"median": 0.0, "min": 0.0, "max": 0.0, "spread_pct": 0.0}

    return result


def compute_speedup(
    baseline_latency: float, target_latency: float
) -> Optional[float]:
    """Calculate relative speedup factor (baseline_latency / target_latency).

    Args:
        baseline_latency: Baseline latency in microseconds.
        target_latency: Target latency in microseconds.

    Returns:
        Optional[float]: Speedup ratio, or None if target_latency is zero or invalid.
    """
    if target_latency <= 0.0 or baseline_latency <= 0.0:
        return None
    return float(baseline_latency / target_latency)


def verify_pytorch_ort_parity(
    pytorch_wrapper: AnomalyDetectorExportWrapper,
    ort_session: ort.InferenceSession,
    features_20: np.ndarray,
    tolerance: float = 1e-5,
) -> None:
    """Verify numerical parity between PyTorch and ORT-Python on first 20 events.

    Args:
        pytorch_wrapper: Wrapped PyTorch model.
        ort_session: Active ONNX Runtime session.
        features_20: Feature array of shape [20, 28].
        tolerance: Maximum acceptable absolute difference.

    Raises:
        RuntimeError: If outputs differ by more than tolerance.
    """
    input_name = ort_session.get_inputs()[0].name
    torch.set_num_threads(1)

    with torch.inference_mode():
        for i in range(len(features_20)):
            event_np = features_20[i : i + 1]
            event_torch = torch.from_numpy(event_np).float()

            pt_err = float(pytorch_wrapper(event_torch).item())
            ort_out = ort_session.run(None, {input_name: event_np})[0]
            ort_err = float(np.asarray(ort_out).item())

            diff = abs(pt_err - ort_err)
            if diff >= tolerance:
                raise RuntimeError(
                    f"Pre-benchmark parity check failed at event {i}: "
                    f"PyTorch={pt_err:.7e}, ORT={ort_err:.7e}, diff={diff:.3e} >= {tolerance:.1e}"
                )


def run_pytorch_benchmark(
    wrapper: AnomalyDetectorExportWrapper,
    features: np.ndarray,
    warmup_runs: int,
    timed_runs: int,
) -> Dict[str, float]:
    """Execute timed single-event benchmark on PyTorch eager path.

    Args:
        wrapper: Wrapped PyTorch model.
        features: Event features array [N, 28].
        warmup_runs: Number of untimed warmup iterations.
        timed_runs: Number of timed iterations.

    Returns:
        Dict[str, float]: Latency statistics in microseconds.
    """
    torch.set_num_threads(1)
    wrapper.eval()

    num_events = len(features)
    # Pre-create inputs in memory
    tensors = [torch.from_numpy(features[i : i + 1]).float() for i in range(num_events)]

    with torch.inference_mode():
        # Warmup loop (untimed)
        for w in range(warmup_runs):
            _ = wrapper(tensors[w % num_events])

        # Timed loop (pure inference call only)
        latencies_us: List[float] = []
        for r in range(timed_runs):
            inp = tensors[r % num_events]
            t0 = time.perf_counter_ns()
            _ = wrapper(inp)
            t1 = time.perf_counter_ns()
            latencies_us.append((t1 - t0) / 1000.0)

    return compute_run_statistics(latencies_us)


def run_ort_python_benchmark(
    session: ort.InferenceSession,
    features: np.ndarray,
    warmup_runs: int,
    timed_runs: int,
) -> Dict[str, float]:
    """Execute timed single-event benchmark on ONNX Runtime Python path.

    Args:
        session: Active ORT InferenceSession.
        features: Event features array [N, 28].
        warmup_runs: Number of untimed warmup iterations.
        timed_runs: Number of timed iterations.

    Returns:
        Dict[str, float]: Latency statistics in microseconds.
    """
    input_name = session.get_inputs()[0].name
    num_events = len(features)
    # Pre-create inputs in memory
    inputs = [features[i : i + 1] for i in range(num_events)]

    # Warmup loop (untimed)
    for w in range(warmup_runs):
        _ = session.run(None, {input_name: inputs[w % num_events]})

    # Timed loop (pure inference call only)
    latencies_us: List[float] = []
    for r in range(timed_runs):
        inp = inputs[r % num_events]
        t0 = time.perf_counter_ns()
        _ = session.run(None, {input_name: inp})
        t1 = time.perf_counter_ns()
        latencies_us.append((t1 - t0) / 1000.0)

    return compute_run_statistics(latencies_us)


def run_ort_cpp_benchmark(
    cpp_exe: Path,
    csv_path: Path,
    model_path: Path,
    threshold_path: Path,
    json_path: Path,
    warmup_runs: int,
    timed_runs: int,
) -> Optional[Tuple[Dict[str, float], Dict[str, Any]]]:
    """Execute timed single-event benchmark on ONNX Runtime C++ executable via subprocess.

    Args:
        cpp_exe: Path to collider_inference.exe.
        csv_path: Path to sample events CSV.
        model_path: Path to detector_model.onnx.
        threshold_path: Path to threshold.txt.
        json_path: Target path for C++ benchmark output JSON.
        warmup_runs: Warmup runs.
        timed_runs: Timed runs.

    Returns:
        Optional[Tuple[Dict[str, float], Dict[str, Any]]]: Latency statistics and metadata,
        or None if executable not found.
    """
    if not cpp_exe.exists():
        print(f"Warning: C++ executable not found at '{cpp_exe}'. Skipping C++ path.")
        return None

    cmd = [
        str(cpp_exe),
        "--model", str(model_path),
        "--csv", str(csv_path),
        "--threshold-file", str(threshold_path),
        "--bench",
        "--warmup", str(warmup_runs),
        "--runs", str(timed_runs),
        "--bench-json", str(json_path),
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error running C++ benchmark executable:\n{res.stderr}")
        return None

    if not json_path.exists():
        print(f"Error: C++ benchmark JSON not generated at '{json_path}'.")
        return None

    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    lat_dict = meta.get("latency_us", {})
    stats = {
        "mean": float(lat_dict.get("mean", 0.0)),
        "median": float(lat_dict.get("median", 0.0)),
        "p99": float(lat_dict.get("p99", 0.0)),
        "min": float(lat_dict.get("min", 0.0)),
        "max": float(lat_dict.get("max", 0.0)),
    }
    return stats, meta


def format_markdown_report(
    hw_info: Dict[str, str],
    headline_results: Dict[str, Dict[str, Dict[str, float]]],
    raw_results: Dict[str, List[Dict[str, float]]],
    speedups: Dict[str, Optional[float]],
    reps: int,
    warmup: int,
    runs: int,
    c_meta: Optional[Dict[str, Any]],
) -> str:
    """Format benchmark results into Markdown artifact adhering strictly to project guidelines.

    Args:
        hw_info: Hardware and system metadata.
        headline_results: Median and spread statistics per path.
        raw_results: Per-repetition raw metrics per path.
        speedups: Relative speedup metrics.
        reps: Number of repetitions.
        warmup: Warmup iterations.
        runs: Timed iterations.
        c_meta: Optional C++ engine metadata.

    Returns:
        str: Markdown document string.
    """
    c_compiler = c_meta.get("compiler", "Unknown") if c_meta else "Unknown"
    c_version = c_meta.get("version", "Unknown") if c_meta else "Unknown"
    c_build = c_meta.get("build_type", "Unknown") if c_meta else "Unknown"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def fmt_num(val: float) -> str:
        return f"{val:.3g}"

    def fmt_speedup(val: Optional[float]) -> str:
        return f"**{val:.2f}x**" if val is not None else "N/A"

    lines = [
        "# Single-Event Inference Latency Benchmark",
        "",
        "Rigorous single-event (batch size 1) latency benchmarking comparing PyTorch Eager,",
        "ONNX Runtime Python, and ONNX Runtime C++ on identical hardware, inputs, and models.",
        "",
        "## Machine & Environment Specification",
        "",
        "| Attribute | Value |",
        "| :--- | :--- |",
        f"| **Processor (CPU)** | {hw_info['cpu_name']} |",
        f"| **Cores / Threads** | {hw_info['cores']} physical cores / {hw_info['threads']} logical processors |",
        f"| **Operating System** | {hw_info['os']} |",
        f"| **Python Version** | {platform.python_version()} |",
        f"| **PyTorch Version** | {torch.__version__} (CPU, 1 thread) |",
        f"| **ONNX Runtime (Python)** | {ort.__version__} (CPU, intra=1, inter=1) |",
        f"| **ONNX Runtime (C++)** | {c_version} (MSVC {c_compiler}, Build: {c_build}) |",
        f"| **Benchmark Timestamp** | {timestamp} |",
        f"| **Benchmark Repetitions** | {reps} interleaved runs ({warmup} warmup + {runs:,} timed runs each) |",
        f"| **Timing Isolation** | Memory-resident pre-allocated inputs; pure inference call timed |",
        "",
        "## Headline Results (Median Across 3 Repetitions)",
        "",
        "| Inference Path | Mean ($\\mu$s) | Median ($\\mu$s) | p99 ($\\mu$s) | Min ($\\mu$s) | Max ($\\mu$s) | Relative Speedup vs PyTorch |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]

    paths = [
        ("PyTorch Eager (CPU)", "pytorch"),
        ("ONNX Runtime (Python)", "ort_py"),
        ("ONNX Runtime (C++)", "ort_cpp"),
    ]

    for label, key in paths:
        if key in headline_results:
            st = headline_results[key]
            speedup_val = speedups.get(f"pytorch_vs_{key}", 1.0 if key == "pytorch" else None)
            lines.append(
                f"| **{label}** | {fmt_num(st['mean']['median'])} | **{fmt_num(st['median']['median'])}** | "
                f"{fmt_num(st['p99']['median'])} | {fmt_num(st['min']['median'])} | {fmt_num(st['max']['median'])} | "
                f"{fmt_speedup(speedup_val)} |"
            )
        else:
            lines.append(f"| **{label}** | *Skipped* | *Skipped* | *Skipped* | *Skipped* | *Skipped* | N/A |")

    lines.extend([
        "",
        "## Relative Speedup Summary",
        "",
        f"- **PyTorch Eager $\\rightarrow$ ONNX Runtime Python**: {fmt_speedup(speedups.get('pytorch_vs_ort_py'))} faster",
        f"- **PyTorch Eager $\\rightarrow$ ONNX Runtime C++**: {fmt_speedup(speedups.get('pytorch_vs_ort_cpp'))} faster",
        f"- **ONNX Runtime Python $\\rightarrow$ ONNX Runtime C++**: {fmt_speedup(speedups.get('ort_py_vs_ort_cpp'))} faster",
        "",
        "## Detailed Per-Repetition Measurements",
        "",
        "Reports performance stability across repetitions to verify measurement reproducibility.",
        "",
        "| Path | Repetition | Mean ($\\mu$s) | Median ($\\mu$s) | p99 ($\\mu$s) | Min ($\\mu$s) | Max ($\\mu$s) |",
        "| :--- | :---: | ---: | ---: | ---: | ---: | ---: |",
    ])

    for label, key in paths:
        if key in raw_results:
            for rep_idx, rep in enumerate(raw_results[key], start=1):
                lines.append(
                    f"| {label} | Rep {rep_idx} | {fmt_num(rep['mean'])} | {fmt_num(rep['median'])} | "
                    f"{fmt_num(rep['p99'])} | {fmt_num(rep['min'])} | {fmt_num(rep['max'])} |"
                )

    lines.extend([
        "",
        "## How to Read These Results",
        "",
        "1. **Engine Equivalence**: ONNX Runtime Python and ONNX Runtime C++ invoke the same underlying",
        "   computational graph engine (`CPUExecutionProvider`). The measured latency difference between them",
        "   is hypothesized (unmeasured) to arise from Python C-API argument conversion and call overhead rather",
        "   than differences in mathematical kernel execution.",
        "2. **Framework Differences on Small Models**: On single-thread CPU execution, PyTorch eager execution",
        "   measured higher latency than ONNX Runtime. It is hypothesized (unmeasured) that dynamic framework",
        "   dispatch and per-call overhead contribute to this difference, but individual components were not",
        "   separately profiled or isolated.",
        "3. **Hardware Context**: Measurements were collected on single-thread CPU execution with preloaded memory-resident",
        "   inputs, isolating pure inference computation from disk I/O and data ingestion.",
        "4. **Causal Attribution**: The measured timings reflect the complete inference call on this specific hardware.",
        "   Any mechanical explanation for speed differences remains an untested hypothesis unless separately measured.",
        "",
        "*Generated automatically by `python/benchmark.py`.*",
    ])

    return "\n".join(lines) + "\n"


def main() -> None:
    """Execute complete three-way benchmark experiment."""
    args = parse_args()

    print("=" * 70)
    print("COLLIDER ANOMALY DETECTION: THREE-WAY LATENCY BENCHMARK")
    print("=" * 70)
    print("REMINDER: Ensure your laptop/machine is connected to AC power and")
    print("that no heavy background processes are running to ensure reproducible timings.")
    print("=" * 70)

    # 1. Hardware & environment audit
    hw_info = get_hardware_info()
    print(f"CPU:              {hw_info['cpu_name']} ({hw_info['cores']} cores, {hw_info['threads']} threads)")
    print(f"OS:               {hw_info['os']}")
    print(f"PyTorch:          {torch.__version__} (threads: 1)")
    print(f"ONNX Runtime Py:  {ort.__version__} (threads: intra=1, inter=1)")
    print(f"Warmup Runs:      {args.warmup}")
    print(f"Timed Runs:       {args.runs}")
    print(f"Repetitions:      {args.repetitions} (interleaved)")
    print("=" * 70)

    # 2. Load inputs and models
    if not args.sample_events.exists():
        raise FileNotFoundError(f"Sample events CSV not found at '{args.sample_events}'.")
    data = np.loadtxt(args.sample_events, delimiter=",", dtype=np.float32)
    features = data[:, 1:]
    print(f"Loaded {len(features)} sample events into memory.")

    # PyTorch model
    base_model, variant = load_model_for_export(args.model_path)
    pytorch_wrapper = AnomalyDetectorExportWrapper(base_model)
    pytorch_wrapper.eval()

    # ORT Python session
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    ort_session = ort.InferenceSession(
        str(args.onnx_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )

    # 3. Pre-benchmark parity verification (first 20 events)
    print("Verifying parity on first 20 events before benchmarking...")
    verify_pytorch_ort_parity(pytorch_wrapper, ort_session, features[:20], tolerance=1e-5)
    print("Parity verification: PASSED (PyTorch and ORT match within 1e-5).\n")

    # 4. Interleaved benchmark loop across repetitions
    raw_results: Dict[str, List[Dict[str, float]]] = {
        "pytorch": [],
        "ort_py": [],
        "ort_cpp": [],
    }
    c_meta: Optional[Dict[str, Any]] = None

    for rep in range(args.repetitions):
        print(f"--- Executing Interleaved Repetition {rep + 1}/{args.repetitions} ---")

        # 4a. PyTorch eager
        t_pt = time.perf_counter()
        pt_stats = run_pytorch_benchmark(
            pytorch_wrapper, features, args.warmup, args.runs
        )
        elapsed_pt = time.perf_counter() - t_pt
        raw_results["pytorch"].append(pt_stats)
        print(
            f"  [PyTorch Eager]    Median: {pt_stats['median']:6.2f} us | "
            f"Mean: {pt_stats['mean']:6.2f} us | p99: {pt_stats['p99']:6.2f} us ({elapsed_pt:.1f}s)"
        )

        # 4b. ONNX Runtime Python
        t_py = time.perf_counter()
        py_stats = run_ort_python_benchmark(
            ort_session, features, args.warmup, args.runs
        )
        elapsed_py = time.perf_counter() - t_py
        raw_results["ort_py"].append(py_stats)
        print(
            f"  [ORT Python]       Median: {py_stats['median']:6.2f} us | "
            f"Mean: {py_stats['mean']:6.2f} us | p99: {py_stats['p99']:6.2f} us ({elapsed_py:.1f}s)"
        )

        # 4c. ONNX Runtime C++
        t_cpp = time.perf_counter()
        cpp_res = run_ort_cpp_benchmark(
            cpp_exe=args.cpp_exe,
            csv_path=args.sample_events,
            model_path=args.onnx_path,
            threshold_path=config.THRESHOLD_TXT_PATH,
            json_path=args.cpp_json,
            warmup_runs=args.warmup,
            timed_runs=args.runs,
        )
        elapsed_cpp = time.perf_counter() - t_cpp
        if cpp_res is not None:
            cpp_stats, c_meta = cpp_res
            raw_results["ort_cpp"].append(cpp_stats)
            print(
                f"  [ORT C++]          Median: {cpp_stats['median']:6.2f} us | "
                f"Mean: {cpp_stats['mean']:6.2f} us | p99: {cpp_stats['p99']:6.2f} us ({elapsed_cpp:.1f}s)"
            )
        else:
            print("  [ORT C++]          Skipped (executable unavailable).")

    # Filter out empty paths if skipped
    valid_raw_results = {k: v for k, v in raw_results.items() if len(v) > 0}

    # 5. Compute headline statistics across repetitions
    headline_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for key, rep_list in valid_raw_results.items():
        headline_results[key] = compute_headline_metrics(rep_list)

    # Check stability (spread across repetitions vs median)
    for key, stats in headline_results.items():
        med_spread_pct = stats["median"]["spread_pct"]
        if med_spread_pct > 20.0:
            print(
                f"\nNOTE: Repetition spread for '{key}' median is {med_spread_pct:.1f}% (> 20%). "
                "Noise detected across runs."
            )

    # 6. Compute relative speedups
    pt_med = headline_results["pytorch"]["median"]["median"]
    py_med = headline_results["ort_py"]["median"]["median"]
    cpp_med = (
        headline_results["ort_cpp"]["median"]["median"]
        if "ort_cpp" in headline_results
        else None
    )

    speedups = {
        "pytorch_vs_ort_py": compute_speedup(pt_med, py_med),
        "pytorch_vs_ort_cpp": compute_speedup(pt_med, cpp_med) if cpp_med else None,
        "ort_py_vs_ort_cpp": compute_speedup(py_med, cpp_med) if cpp_med else None,
    }

    # 7. Generate markdown and raw JSON artifacts
    md_content = format_markdown_report(
        hw_info=hw_info,
        headline_results=headline_results,
        raw_results=valid_raw_results,
        speedups=speedups,
        reps=args.repetitions,
        warmup=args.warmup,
        runs=args.runs,
        c_meta=c_meta,
    )

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_md, "w", encoding="utf-8") as f_md:
        f_md.write(md_content)

    raw_json_data = {
        "hardware": hw_info,
        "parameters": {
            "warmup_runs": args.warmup,
            "timed_runs": args.runs,
            "repetitions": args.repetitions,
            "batch_size": 1,
            "threads": 1,
            "num_sample_events": len(features),
        },
        "c_cpp_meta": c_meta,
        "headline_metrics": headline_results,
        "speedups": speedups,
        "repetitions": valid_raw_results,
        "timestamp": datetime.datetime.now().isoformat(),
    }

    args.out_raw_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_raw_json, "w", encoding="utf-8") as f_json:
        json.dump(raw_json_data, f_json, indent=2)

    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY REPORT")
    print("=" * 70)
    print(md_content)
    print("=" * 70)
    print(f"Markdown report saved to: {args.out_md}")
    print(f"Raw JSON data saved to:   {args.out_raw_json}")
    print("=" * 70)


if __name__ == "__main__":
    main()
