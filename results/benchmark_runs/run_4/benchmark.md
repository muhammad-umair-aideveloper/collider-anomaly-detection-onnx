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
| **Benchmark Timestamp** | 2026-11-07 11:01:29 |
| **Benchmark Repetitions** | 3 interleaved runs (100 warmup + 10,000 timed runs each) |
| **Timing Isolation** | Memory-resident pre-allocated inputs; pure inference call timed |

## Headline Results (Median Across 3 Repetitions)

| Inference Path | Mean ($\mu$s) | Median ($\mu$s) | p99 ($\mu$s) | Min ($\mu$s) | Max ($\mu$s) | Relative Speedup vs PyTorch |
| :--- | ---: | ---: | ---: | ---: | ---: | :---: |
| **PyTorch Eager (CPU)** | 304 | **184** | 4.44e+03 | 159 | 1.45e+04 | **1.00x** |
| **ONNX Runtime (Python)** | 64.1 | **34.5** | 128 | 31.8 | 4.49e+04 | **5.32x** |
| **ONNX Runtime (C++)** | 32.2 | **20.9** | 62 | 19.6 | 8.41e+03 | **8.78x** |

## Relative Speedup Summary

- **PyTorch Eager $\rightarrow$ ONNX Runtime Python**: **5.32x** faster
- **PyTorch Eager $\rightarrow$ ONNX Runtime C++**: **8.78x** faster
- **ONNX Runtime Python $\rightarrow$ ONNX Runtime C++**: **1.65x** faster

## Detailed Per-Repetition Measurements

Reports performance stability across repetitions to verify measurement reproducibility.

| Path | Repetition | Mean ($\mu$s) | Median ($\mu$s) | p99 ($\mu$s) | Min ($\mu$s) | Max ($\mu$s) |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: |
| PyTorch Eager (CPU) | Rep 1 | 412 | 189 | 4.72e+03 | 159 | 1.03e+05 |
| PyTorch Eager (CPU) | Rep 2 | 304 | 184 | 4.34e+03 | 158 | 1.45e+04 |
| PyTorch Eager (CPU) | Rep 3 | 293 | 182 | 4.44e+03 | 159 | 1.31e+04 |
| ONNX Runtime (Python) | Rep 1 | 95.6 | 39.2 | 237 | 31.7 | 1.25e+05 |
| ONNX Runtime (Python) | Rep 2 | 50.1 | 33.9 | 95.6 | 31.8 | 9.81e+03 |
| ONNX Runtime (Python) | Rep 3 | 64.1 | 34.5 | 128 | 31.9 | 4.49e+04 |
| ONNX Runtime (C++) | Rep 1 | 30.7 | 20.9 | 55.1 | 19.6 | 8.09e+03 |
| ONNX Runtime (C++) | Rep 2 | 32.2 | 20.4 | 62 | 19.2 | 9.64e+03 |
| ONNX Runtime (C++) | Rep 3 | 33.9 | 21 | 73.4 | 19.8 | 8.41e+03 |

## How to Read These Results

1. **Engine Equivalence**: ONNX Runtime Python and ONNX Runtime C++ invoke the same underlying
   computational graph engine (`CPUExecutionProvider`). The measured latency difference between them
   is hypothesized (unmeasured) to arise from Python C-API argument conversion and call overhead rather
   than differences in mathematical kernel execution.
2. **Framework Differences on Small Models**: On single-thread CPU execution, PyTorch eager execution
   measured higher latency than ONNX Runtime. It is hypothesized (unmeasured) that dynamic framework
   dispatch and per-call overhead contribute to this difference, but individual components were not
   separately profiled or isolated.
3. **Hardware Context**: Measurements were collected on single-thread CPU execution with preloaded memory-resident
   inputs, isolating pure inference computation from disk I/O and data ingestion.
4. **Causal Attribution**: The measured timings reflect the complete inference call on this specific hardware.
   Any mechanical explanation for speed differences remains an untested hypothesis unless separately measured.

*Generated automatically by `python/benchmark.py`.*
