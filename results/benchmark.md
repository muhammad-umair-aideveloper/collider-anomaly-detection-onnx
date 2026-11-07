# Single-Event Inference Latency Benchmark

Rigorous single-event (batch size 1) latency benchmarking comparing PyTorch Eager,
ONNX Runtime Python, and ONNX Runtime C++ on identical hardware, inputs, and models.

## Machine & Environment Specification

| Attribute | Value |
| :--- | :--- |
| **Processor (CPU)** | Intel(R) Core(TM) i5-6300U CPU @ 2.40GHz |
| **Cores / Threads** | 2 physical cores / 4 logical processors |
| **Operating System** | Windows 10 (Build 10.0.19045) |
| **Python Version** | 3.14.3 |
| **PyTorch Version** | 2.15.0+cpu (CPU, 1 thread) |
| **ONNX Runtime (Python)** | 1.31.1 (CPU, intra=1, inter=1) |
| **ONNX Runtime (C++)** | 1.20.1 (MSVC MSVC 1944, Build: Release) |
| **Benchmark Timestamp** | 2026-11-07 10:53:08 |
| **Benchmark Repetitions** | 3 interleaved runs (100 warmup + 10,000 timed runs each) |
| **Timing Isolation** | Memory-resident pre-allocated inputs; pure inference call timed |

## Headline Results (Median Across 3 Repetitions)

| Inference Path | Mean ($\mu$s) | Median ($\mu$s) | p99 ($\mu$s) | Min ($\mu$s) | Max ($\mu$s) | Relative Speedup vs PyTorch |
| :--- | ---: | ---: | ---: | ---: | ---: | :---: |
| **PyTorch Eager (CPU)** | 449 | **184** | 8.06e+03 | 162 | 8.32e+04 | **1.00x** |
| **ONNX Runtime (Python)** | 65.7 | **34.2** | 200 | 32.7 | 1.06e+04 | **5.38x** |
| **ONNX Runtime (C++)** | 37.5 | **20.8** | 93.5 | 19.8 | 9e+03 | **8.85x** |

## Relative Speedup Summary

- **PyTorch Eager $\rightarrow$ ONNX Runtime Python**: **5.38x** faster
- **PyTorch Eager $\rightarrow$ ONNX Runtime C++**: **8.85x** faster
- **ONNX Runtime Python $\rightarrow$ ONNX Runtime C++**: **1.64x** faster

## Detailed Per-Repetition Measurements

Reports performance stability across repetitions to verify measurement reproducibility.

| Path | Repetition | Mean ($\mu$s) | Median ($\mu$s) | p99 ($\mu$s) | Min ($\mu$s) | Max ($\mu$s) |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: |
| PyTorch Eager (CPU) | Rep 1 | 474 | 184 | 8.06e+03 | 163 | 1.5e+05 |
| PyTorch Eager (CPU) | Rep 2 | 428 | 183 | 8.15e+03 | 162 | 8.32e+04 |
| PyTorch Eager (CPU) | Rep 3 | 449 | 185 | 6.87e+03 | 157 | 4.32e+04 |
| ONNX Runtime (Python) | Rep 1 | 71.2 | 34.5 | 200 | 32.7 | 1.44e+04 |
| ONNX Runtime (Python) | Rep 2 | 65.7 | 34.2 | 157 | 32.8 | 1.06e+04 |
| ONNX Runtime (Python) | Rep 3 | 51.9 | 34.2 | 253 | 32 | 7.8e+03 |
| ONNX Runtime (C++) | Rep 1 | 40.9 | 20.8 | 76.4 | 20 | 1.31e+04 |
| ONNX Runtime (C++) | Rep 2 | 37.5 | 20.5 | 93.5 | 19.8 | 9e+03 |
| ONNX Runtime (C++) | Rep 3 | 33.4 | 21.2 | 132 | 19.8 | 6.17e+03 |

## How to Read These Results

1. **Engine Equivalence**: ONNX Runtime Python and ONNX Runtime C++ execute the exact same underlying
   computational graph engine (`CPUExecutionProvider`). The measured latency gap between them represents
   Python C-API binding, argument conversion, and dispatch call overhead, rather than mathematical kernel speed.
2. **Framework Overhead on Tiny Models**: For a tiny autoencoder (7 high-level inputs, 4 bottleneck dimensions),
   PyTorch eager mode incurs significant per-call dynamic graph dispatch, tensor allocation, and framework overhead.
   Exporting to a compiled static ONNX graph eliminates interpreter overhead and fuses operations efficiently.
3. **Hardware Context**: Measurements were collected on single-thread CPU execution with preloaded memory-resident
   inputs, isolating pure inference computation from disk I/O and data ingestion.
4. **Causal Interpretation**: Any speed difference beyond binding overhead is a hypothesis and depends on specific
   compiler flags, memory alignment, and CPU cache residency during execution.

*Generated automatically by `python/benchmark.py`.*
