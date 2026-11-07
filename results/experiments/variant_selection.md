# Model Selection Experiment: Pre-Declared Architecture & Feature Variants

## Experiment Overview

- **Goal**: Rigorous, leakage-free evaluation of pre-declared autoencoder architecture and feature-subset variants.
- **Dataset**: Evaluated on `data/processed/select.npz` (2,000 background + 2,000 signal events).
- **Leakage Prevention**: The final test split (`data/processed/test.npz`) was **NOT** accessed during this selection.
- **Training Protocol**: Trained exclusively on normal background events (`train.npz`), using validation loss (`val.npz`) for checkpoint selection, CPU only.
- **Pre-Registered Decision Rule**: Select candidate with highest mean AUC on `select.npz` only if `mean_auc - v0_mean_auc > 0.0100`; otherwise keep baseline `V0`.

## Results Table

| Variant | Features | Bottleneck | Architecture | Seed 0 AUC | Seed 1 AUC | Seed 2 AUC | Mean AUC | Std AUC | Delta vs V0 |
|:---|---:|---:|:---|---: | ---: | ---:|---:|---:|---:|
| `V0` | 28 | 8 | 28 -> 16 -> 8 -> 16 -> 28 | 0.4814 | 0.4858 | 0.4750 | **0.4807** | 0.0054 | 0.0000 (ref) |
| `V1` | 22 | 8 | 22 -> 16 -> 8 -> 16 -> 22 | 0.4990 | 0.4934 | 0.4957 | **0.4960** | 0.0028 | +0.0153 |
| `V2` **(Selected)** | 7 | 4 | 7 -> 16 -> 4 -> 16 -> 7 | 0.5045 | 0.5137 | 0.5128 | **0.5103** | 0.0051 | +0.0296 |
| `V3` | 22 | 4 | 22 -> 16 -> 4 -> 16 -> 22 | 0.4959 | 0.4822 | 0.4896 | **0.4892** | 0.0069 | +0.0085 |

## Decision and Conclusion

- **Selected Variant**: `V2`
- **Decision Rule Margin**: `> 0.0100` improvement over V0 required.
- **Justification**: Candidate 'V2' achieved mean AUC of 0.5103, exceeding baseline V0 (0.4807) by +0.0296 (margin threshold: +0.0100). Variant 'V2' is selected.

## Scientific Notes

1. In unsupervised anomaly detection on the HIGGS dataset, autoencoder reconstruction error is sensitive to feature semantics.
2. Azimuthal angles (phi) are coordinate-system dependent and typically isotropic in collisions, offering no discrimination.
3. High-level invariant masses (columns 21-27) carry reconstructed particle resonance physics.
4. This selection was conducted completely out-of-sample relative to the final test split.
