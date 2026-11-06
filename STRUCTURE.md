# Project Structure: `collider-anomaly-detection-onnx`

An autoencoder trained on background collision events only. Events it
reconstructs badly are flagged as anomalies. The model is exported to ONNX and
run from C++ with ONNX Runtime, and single-event latency is benchmarked
honestly (PyTorch vs ONNX Runtime Python vs ONNX Runtime C++).

## Folder layout

```
collider-anomaly-detection-onnx/
├── README.md                 # Results, method, benchmark, how to run
├── AGENTS.md                 # Rules for the AI coding agent
├── STRUCTURE.md              # This file
├── requirements.txt          # Python dependencies
├── .gitignore
├── LICENSE                   # MIT
├── data/
│   ├── higgs_sample.csv      # First 200,000 HIGGS rows (YOU create it, git-ignored)
│   ├── sample_events.csv     # 1,000 demo test events (committed)
│   ├── test_events.csv       # Full test split for benchmarks (git-ignored)
│   └── processed/            # .npz splits (git-ignored)
├── python/
│   ├── config.py             # All constants: seed, sizes, paths, epochs, lr
│   ├── prepare_data.py       # Stratified sampling and train/val/test splits
│   ├── model.py              # Autoencoder with built-in standardization
│   ├── train.py              # Train on background, compute threshold
│   ├── evaluate.py           # ROC AUC, histograms, TPR/FPR at threshold
│   ├── export_onnx.py        # torch.onnx.export, batch size 1
│   ├── verify_onnx.py        # PyTorch vs ONNX vs C++ output check
│   └── benchmark.py          # Latency: PyTorch, ORT Python, ORT C++
├── cpp/
│   ├── CMakeLists.txt        # Release by default, -DONNXRUNTIME_DIR=...
│   └── inference.cpp         # ONNX Runtime C++ inference + --bench mode
├── models/
│   ├── autoencoder.pt        # PyTorch checkpoint
│   ├── detector_model.onnx   # Exported model (raw features in, error out)
│   ├── threshold.json        # 95th percentile of validation error
│   └── threshold.txt         # Same threshold as a single float (for C++)
├── tests/
│   ├── test_data.py          # Split sizes, label purity, no overlap
│   └── test_model.py         # Shapes, non-negative error, finite gradients
├── third_party/              # ONNX Runtime C++ package (git-ignored)
└── results/
    ├── loss_curve.png
    ├── roc_curve.png
    ├── error_histogram.png
    ├── metrics.md            # AUC, TPR/FPR at threshold (real numbers only)
    ├── cpp_errors.txt        # First 100 C++ errors, for comparison
    ├── cpp_benchmark.json    # C++ latency results
    └── benchmark.md          # Final latency table
```

## Pipeline

```
HIGGS csv ─► prepare_data.py ─► train.py ─► autoencoder.pt + threshold
                                    │
                                    ├─► evaluate.py ─► ROC AUC, plots
                                    │
                                    └─► export_onnx.py ─► detector_model.onnx
                                                │
                       ┌────────────────────────┴───────────────┐
                       ▼                                        ▼
                verify_onnx.py                            inference.cpp
              (PyTorch vs ONNX)                       (C++ outputs + timing)
                       └────────────────┬───────────────────────┘
                                        ▼
                                  benchmark.py ─► results/benchmark.md
```

## Data details

- Source: UCI HIGGS dataset, https://archive.ics.uci.edu/dataset/280/higgs
- Format: no header; column 0 = label (1 = signal, 0 = background); columns
  1 to 28 = features.
- Splits (fixed seed, sampled **by label**):
  - train: 8,000 background
  - validation: 2,000 background
  - test: 5,000 background + 5,000 signal
- The full HIGGS file is several GB. Do not commit it.

## Prerequisites (Windows)

1. **Python 3.10+** and Git.
2. **C++ compiler:** Visual Studio Build Tools with the "Desktop development
   with C++" workload (MinGW is an optional alternative).
3. **CMake** (check with `cmake --version`).
4. **ONNX Runtime C++ package:** download the prebuilt Windows x64 zip
   (`onnxruntime-win-x64-<version>.zip`) from the ONNX Runtime GitHub
   releases and extract it to a path without spaces, for example
   `third_party\onnxruntime`.
5. **HIGGS sample:** download `HIGGS.csv.gz` from UCI, then create the sample
   without unzipping the whole file (PowerShell, from the repo root):

```powershell
python -c "import gzip, itertools; f = gzip.open('HIGGS.csv.gz', 'rt'); open('data/higgs_sample.csv', 'w').writelines(itertools.islice(f, 200000))"
```

## How to run (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python -m pytest -v
python python/prepare_data.py
python python/train.py
python python/evaluate.py
python python/export_onnx.py
python python/verify_onnx.py

cmake -S cpp -B build -DONNXRUNTIME_DIR="third_party/onnxruntime"
cmake --build build --config Release
# then run the C++ executable (exact command is documented in the README)

python python/benchmark.py
```

## Rules for this structure

- Do not add files or folders outside this layout without a reason.
- Keep paths relative to the repo root (`pathlib.Path`, no hardcoded `C:\...`).
- Commit small results and the final ONNX model; never commit raw datasets,
  `build/`, or the ONNX Runtime package.
- Every change to any file gets its own conventional commit (see `AGENTS.md`).
