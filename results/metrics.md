# Anomaly Detection Evaluation Metrics

Evaluation performed on held-out test split using best autoencoder checkpoint.

## Model Comparison

| Model | Architecture | ROC AUC | 95% Bootstrap CI | Decision Threshold | TPR (Sensitivity) | FPR (Fall-out) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline V0, measured before selection** | 28 -> 16 -> 8 -> 16 -> 28 | 0.5069 | [0.4960, 0.5188] | 0.944065 | 0.0510 | 0.0484 |
| **Selected Variant V2** | 7 -> 16 -> 4 -> 16 -> 7 (7 high-level invariant masses) | **0.5318** | [0.5209, 0.5431] | 0.193124 | **0.0452** | **0.0472** |

## Summary Metrics (Selected Variant V2)

| Metric | Value | Details / Specification |
| :--- | :--- | :--- |
| **Model Variant** | **V2** | Pre-declared selection winner (7 high-level invariant masses) |
| **ROC AUC** | **0.5318** | 95% Bootstrap CI: [0.5209, 0.5431] (1000 resamples) |
| **Decision Threshold** | **0.193124** | 95th percentile of validation background error |
| **Test Set Size** | **10,000** | 5,000 Background + 5,000 Signal events |
| **Random Seed** | **42** | Fixed evaluation seed |

## Operating Point Performance (at Threshold = 0.193124)

| Metric | Count / Rate | Definition |
| :--- | :--- | :--- |
| **True Positives (TP)** | 226 | Signal events correctly flagged as anomalies |
| **False Positives (FP)** | 236 | Background events incorrectly flagged as anomalies |
| **True Negatives (TN)** | 4,764 | Background events correctly passed as normal |
| **False Negatives (FN)** | 4,774 | Signal events missed (reconstructed with error < threshold) |
| **True Positive Rate (TPR)** | **0.0452** (4.52%) | $\text{TP} / (\text{TP} + \text{FN})$ |
| **False Positive Rate (FPR)** | **0.0472** (4.72%) | $\text{FP} / (\text{FP} + \text{TN})$ |

## Reconstruction Error Distributions (MSE)

| Event Class | Mean Error | Median Error | Event Count |
| :--- | :--- | :--- | :--- |
| **Background (Normal)** | 0.053396 | 0.020669 | 5,000 |
| **Signal (Anomaly)** | 0.056205 | 0.023719 | 5,000 |

*Generated automatically by `python/evaluate.py`.*
