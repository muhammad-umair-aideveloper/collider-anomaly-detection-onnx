# ONNX Export Numerical Parity Verification Report

## Summary

- **Verification Status**: **PASSED**
- **Tolerance Criterion**: Maximum Absolute Difference < 1.0e-05

## 1. Single-Event Sample Comparison (First 100 Events)

Evaluated on the first 100 events of `data/sample_events.csv` with batch size 1.

| Metric | Measured Value | Threshold / Target | Status |
| :--- | :---: | :---: | :---: |
| **Maximum Absolute Difference** | **1.341105e-07** | < 1.0e-05 | **PASSED** |
| **Mean Absolute Difference** | **1.190114e-08** | — | — |
| **Decision Disagreements** | **0** / 100 | 0 disagreements | **MATCH** |
| **Anomaly Threshold Used** | `0.193124` | `models/threshold.txt` | Validated |

## 2. Selection Set Full Evaluation (4,000 Events)

Evaluated across all 4,000 events of `data/processed/select.npz` (2,000 background + 2,000 signal) to verify metric consistency.

| Metric | PyTorch Eager (CPU) | ONNX Runtime (CPU) | Absolute Difference |
| :--- | :---: | :---: | :---: |
| **ROC AUC** | **0.537105** | **0.537105** | **0.000000e+00** |

## 3. C++ Inference Cross-Comparison

- **Status**: Max abs diff vs ORT: 0.000e+00 (Mean: 0.000e+00)

## Environment & Hardware

- **ONNX Runtime Version**: `1.31.1`
- **PyTorch Version**: `2.15.0+cpu`
- **Execution Provider**: `CPUExecutionProvider` (single thread)
- **Batch Size**: `1` (single event inference)

*Generated automatically by `python/verify_onnx.py`.*
