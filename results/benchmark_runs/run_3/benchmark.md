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
| **Benchmark Timestamp** | 2026-11-07 11:01:00 |
| **Benchmark Repetitions** | 3 interleaved runs (100 warmup + 10,000 timed runs each) |
| **Timing Isolation** | Memory-resident pre-allocated inputs; pure inference call timed |

## Headline Results (Median Across 3 Repetitions)

| Inference Path | Mean ($\mu$s) | Median ($\mu$s) | p99 ($\mu$s) | Min ($\mu$s) | Max ($\mu$s) | Relative Speedup vs PyTorch |
| :--- | ---: | ---: | ---: | ---: | ---: | :---: |
| **PyTorch Eager (CPU)** | 374 | **184** | 4.54e+03 | 156 | 6.7e+04 | **1.00x** |
| **ONNX Runtime (Python)** | 54 | **33.8** | 115 | 31.7 | 9.25e+03 | **5.46x** |
| **ONNX Runtime (C++)** | 30.7 | **20.8** | 62.7 | 19.4 | 6.92e+03 | **8.87x** |

## Relative Speedup Summary

- **PyTorch Eager $\rightarrow$ ONNX Runtime Python**: **5.46x** faster
- **PyTorch Eager $\rightarrow$ ONNX Runtime C++**: **8.87x** faster
- **ONNX Runtime Python $\rightarrow$ ONNX Runtime C++**: **1.62x** faster

## Detailed Per-Repetition Measurements

Reports performance stability across repetitions to verify measurement reproducibility.

| Path | Repetition | Mean ($\mu$s) | Median ($\mu$s) | p99 ($\mu$s) | Min ($\mu$s) | Max ($\mu$s) |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: |
| PyTorch Eager (CPU) | Rep 1 | 374 | 184 | 4.54e+03 | 157 | 6.7e+04 |
| PyTorch Eager (CPU) | Rep 2 | 566 | 197 | 7.02e+03 | 156 | 8.59e+04 |
| PyTorch Eager (CPU) | Rep 3 | 300 | 181 | 4.28e+03 | 156 | 3.89e+04 |
| ONNX Runtime (Python) | Rep 1 | 70.3 | 43.7 | 237 | 32.6 | 1.37e+04 |
| ONNX Runtime (Python) | Rep 2 | 52.8 | 33.7 | 115 | 31.7 | 7.97e+03 |
| ONNX Runtime (Python) | Rep 3 | 54 | 33.8 | 102 | 31.6 | 9.25e+03 |
| ONNX Runtime (C++) | Rep 1 | 33.3 | 21.7 | 95.5 | 20 | 7.03e+03 |
| ONNX Runtime (C++) | Rep 2 | 30.6 | 20.6 | 62.7 | 19.4 | 6.92e+03 |
| ONNX Runtime (C++) | Rep 3 | 30.7 | 20.8 | 61.8 | 19.1 | 5.73e+03 |

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
