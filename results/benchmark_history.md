# Benchmark History Across Repeated Runs

This document records the reproducibility of single-event inference latency across multiple independent executions of the three-way benchmark suite (`python/benchmark.py`).

Each run consists of 3 interleaved repetitions of 10,000 timed single-event evaluations (batch size 1, 100 untimed warmup runs) for each execution path on single-thread CPU execution.

## Median Latency Across Runs ($\mu$s)

| Run Identifier | Timestamp | PyTorch Eager (CPU) | ONNX Runtime (Python) | ONNX Runtime (C++) | Notes |
| :--- | :--- | ---: | ---: | ---: | :--- |
| **Run 2 (Headline)** | 2026-11-07 10:53:08 | 184.10 | 34.20 | 20.80 | Committed in `results/benchmark.md` |
| **Run 3** | 2026-11-07 11:01:00 | 184.50 | 33.80 | 20.80 | Independent repeat run |
| **Run 4** | 2026-11-07 11:01:29 | 183.60 | 34.50 | 20.90 | Independent repeat run |
| **Run 5** | 2026-11-07 11:01:59 | 181.90 | 35.00 | 20.70 | Independent repeat run |

## Cross-Run Dispersion Summary

| Execution Path | Minimum Median ($\mu$s) | Maximum Median ($\mu$s) | Cross-Run Median ($\mu$s) | Cross-Run Spread (%) |
| :--- | ---: | ---: | ---: | ---: |
| **PyTorch Eager (CPU)** | 181.90 | 184.50 | 183.85 | 1.41% |
| **ONNX Runtime (Python)** | 33.80 | 35.00 | 34.35 | 3.49% |
| **ONNX Runtime (C++)** | 20.70 | 20.90 | 20.80 | 0.96% |

*Note on Spread formula*: $\text{Spread} = \frac{\text{Maximum} - \text{Minimum}}{\text{Cross-Run Median}} \times 100\%$.

## Mandatory Transparency Note

A first run (run 1) showed a 70.6% spread between repetitions for ONNX Runtime Python. Its raw data was not retained, and the cause was not investigated or determined. The benchmark was repeated and the later runs are reported here.
