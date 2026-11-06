# Anomaly Detection Evaluation Metrics

Evaluation performed on held-out test split using best autoencoder checkpoint.

## Summary Metrics

| Metric | Value | Details / Specification |
| :--- | :--- | :--- |
| **ROC AUC** | **0.5069** | 95% Bootstrap CI: [0.4960, 0.5188] (1000 resamples) |
| **Decision Threshold** | **0.944065** | 95th percentile of validation background error |
| **Test Set Size** | **10,000** | 5,000 Background + 5,000 Signal events |
| **Random Seed** | **42** | Fixed evaluation seed |

## Operating Point Performance (at Threshold = 0.944065)

| Metric | Count / Rate | Definition |
| :--- | :--- | :--- |
| **True Positives (TP)** | 255 | Signal events correctly flagged as anomalies |
| **False Positives (FP)** | 242 | Background events incorrectly flagged as anomalies |
| **True Negatives (TN)** | 4,758 | Background events correctly passed as normal |
| **False Negatives (FN)** | 4,745 | Signal events missed (reconstructed with error < threshold) |
| **True Positive Rate (TPR)** | **0.0510** (5.10%) | $\text{TP} / (\text{TP} + \text{FN})$ |
| **False Positive Rate (FPR)** | **0.0484** (4.84%) | $\text{FP} / (\text{FP} + \text{TN})$ |

## Reconstruction Error Distributions (MSE)

| Event Class | Mean Error | Median Error | Event Count |
| :--- | :--- | :--- | :--- |
| **Background (Normal)** | 0.485360 | 0.414346 | 5,000 |
| **Signal (Anomaly)** | 0.484336 | 0.417253 | 5,000 |

*Generated automatically by `python/evaluate.py`.*
