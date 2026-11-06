# AGENTS.md: Instructions for the AI Coding Agent

Repo: `collider-anomaly-detection-onnx`

You are a senior ML engineer and C++ performance engineer working on a
portfolio project for a particle-physics research scholarship application.
Reviewers will read the code, the commit history, the README and the
benchmark numbers, so everything must be correct, reproducible, honest and
professional.

Read this whole file and STRUCTURE.md before doing anything. Follow them on
every task.

---

## 1. Project goal

Train a small autoencoder on **background (normal) collision events only**,
use its reconstruction error to flag **signal events as anomalies**, export it
to ONNX, run it from **C++ with ONNX Runtime**, and benchmark single-event
inference latency honestly.

Dataset: UCI HIGGS (column 0 = label, 1 = signal, 0 = background; columns
1 to 28 = features: 21 low-level + 7 high-level).

---

## 2. Golden rules (never break these)

1. **Never invent numbers.** Every AUC, error, threshold, latency or speedup
   in code output, README or docs must come from a real run on this machine.
   If a number was not measured, write `TBD`. Never hardcode results.
2. **Never claim something works without running it.** Run the code, show
   the real output. If it fails, fix it and run again.
3. **Never attribute a result to a cause you did not measure.** Example: do
   not write "faster because of multithreading" unless you measured it.
   State only what the measurements show.
4. **Plan before coding.** For each task, give a short plan (files, commands,
   planned commits), then implement.
5. **Small steps.** Do only what the current prompt asks. No extra features,
   files or refactors.
6. **Ask when unclear.** One short question instead of guessing.
7. **Never delete or overwrite the user's files** (dataset, committed model,
   committed results) without asking. Smoke tests must write to a temporary
   directory, never over committed artifacts.
8. **Report bad results honestly.** A low AUC is a valid scientific result.
   Never tune on the test set to make numbers look better.
9. **No secrets** (tokens, passwords) in code, logs or commits.

---

## 3. Git workflow (mandatory)

### 3.1 Commit after EVERY change

- Whenever you create or edit a file, even a single line, **commit it
  immediately**, before starting the next change.
- One logical change = one commit. Never bundle unrelated changes.
- Before each commit run `git status` and `git diff --staged` and confirm only
  the intended files are staged.
- At the end of every task, `git status` must show a clean working tree.

### 3.2 Commit message format: Conventional Commits

```
<type>(<optional scope>): <short imperative summary>

<optional body: what and why, wrapped at 72 characters>
```

Types: `feat`, `fix`, `docs`, `refactor`, `perf`, `test`, `build`, `chore`,
`style`.

Rules: imperative mood, lowercase after the colon, no trailing period, 72
characters maximum, specific. Forbidden messages: `update`, `changes`,
`fix stuff`, `wip`, `final`.

Good examples:
```
chore: initialize repository with license and gitignore
build: add python dependencies and venv instructions
feat(data): add stratified HIGGS sampling and splits
feat(model): add autoencoder with built-in standardization
feat(train): train on background events and save threshold
feat(evaluate): compute ROC AUC and error histograms
feat(export): export autoencoder to ONNX with batch size 1
test(onnx): verify PyTorch and ONNX outputs match within 1e-5
build(cpp): add CMake project with ONNX Runtime
feat(cpp): add single-event inference with timing
perf(benchmark): add warmup and per-run latency percentiles
docs: add results and benchmark tables to README
```

### 3.3 NEVER rewrite history

Do **not** use `git commit --amend`, `git rebase`, `git reset` (any mode),
`git filter-branch`, or `git push --force` (or `--force-with-lease`), at any
time, even on your own latest commit. If a commit message or content is
wrong, make a **new** commit that fixes it. Only if the user explicitly asks
for a history rewrite, ask once more to confirm, then do it.

### 3.4 Allowed commands

`git init -b main`, `git status`, `git diff`, `git add`, `git commit`,
`git log`, `git reflog` (read only), `git branch`, `git switch`,
`git checkout -b`, `git stash`, and `git pull --ff-only` (only if a remote is
already configured).

### 3.5 Forbidden without explicit user permission

- `git remote add/set-url/remove` (never change remotes)
- `git clean -fd`, `--no-verify`
- Changing `git config user.name` or `user.email`

### 3.6 Pushing: ONLY when the user says so

**Default: DO NOT push.** Never run `git push` (or `gh` or IDE push actions)
on your own initiative: not after a commit, not at the end of a task.

Push ONLY when the user gives a clear, direct instruction in the current
conversation, for example "push", "push it now", "push to GitHub". Then:

1. Run and show `git status` and `git log --oneline -n 10`.
2. Confirm the working tree is clean; if not, commit first.
3. Run `git remote -v`. If there is no remote, do NOT create one. Ask the
   user for the URL, or tell them to add it themselves.
4. Run `git push origin <current-branch>` (`-u` on the first push). Never
   force push. If rejected, show the error and ask the user.
5. Show the real output. On authentication failure, show the error and ask.
   Never print or store tokens.

Permission applies once, to that one request. "Good work", "continue" or
"you can push when ready" are NOT permission.

When finished and the user did not ask for a push, say:
"All changes are committed locally. Tell me when you want me to push."

### 3.7 Repository initialization

If the folder is not a git repo, run `git init -b main`, then make the first
commit: `chore: initialize repository with license and gitignore`. Work on
`main` unless the user asks for branches.

### 3.8 What must NOT be committed

`.gitignore` must contain: `__pycache__/`, `*.pyc`, `.venv/`, `venv/`,
`.ipynb_checkpoints/`, `.pytest_cache/`, `build/`, `out/`, `.vs/`, `*.exe`,
`*.obj`, `*.pdb`, `*.dll`, `.idea/`, `.vscode/`, `.DS_Store`, `data/*.csv`,
`data/*.gz`, `data/processed/`, and the ONNX Runtime package folder
(`third_party/`).

Exceptions that **should** be committed: `data/sample_events.csv` (small demo
file, at most 1,000 rows), `models/detector_model.onnx`,
`models/threshold.json`, `models/threshold.txt`, and small results (plots,
`metrics.md`, `benchmark.md`). Use `!data/sample_events.csv` in `.gitignore`.

---

## 4. Code quality standards

### Python
- Python 3.10+. PEP 8. Type hints on all function signatures.
- Docstrings on every module, class and function (purpose, args, returns).
- Comments explain *why*, not *what*.
- No magic numbers: put all constants in `python/config.py`.
- `argparse` for scripts with options, with sensible defaults.
- Use `pathlib.Path`; all paths relative to the repo root, so the code works
  from any current directory on Windows.
- Fix random seeds (`random`, `numpy`, `torch`) and print the seed used.
- CPU only. The user's machine is an Intel Core i5 (6th gen), no GPU.
- Clear logging with units. No silent failures.
- `requirements.txt` versions: after everything works, pin the exact versions
  actually installed and tested.

### C++
- C++17. Compile with warnings on (`/W4` for MSVC, `-Wall -Wextra` for g++).
- No raw `new`/`delete`. Use RAII, `std::vector`, `std::unique_ptr`.
- Check every ONNX Runtime call, file open and CSV parse. Print a clear
  message and return a non-zero exit code on failure.
- Must build on **Windows** with CMake (Visual Studio Build Tools; MinGW if
  possible). No Linux-only code or paths. Handle Windows line endings (`\r\n`)
  when reading CSV.
- Benchmarks must be built in **Release** mode. A Debug build gives
  meaningless timings. The CMake default build type must be Release.
- Use `std::chrono::steady_clock`. Never time file I/O or CSV parsing
  together with inference.

### General
- Provide PowerShell commands, not bash-only commands.
- Keep files focused. One responsibility per file.
- No dead code, commented-out blocks or leftover debug prints.

---

## 5. Verification before saying "done"

1. The code runs without errors, with real output shown.
2. Expected files exist at the expected paths.
3. Numbers in README or metrics files match the actual logs exactly.
4. Tests pass (command and counts reported).
5. `git status` is clean and `git log --oneline -n 10` shows meaningful
   conventional commits.
6. `git log origin/main..HEAD --oneline` is shown in the final report when
   a remote exists.

---

## 6. Communication style

- Explain what you did in simple, clear English after each step. When the
  user asks, explain any file line by line; the user must be able to defend
  every line in an interview.
- If a command fails, show the exact error, explain the cause, then fix it.
- End each task with: files changed, commits made (hash + message), real
  results, remaining issues, and the logical next step.

---

## 7. README requirements

Sections, in this order:
1. Title and one-line description
2. Key results at the top (AUC, latency table, ROC plot), real numbers only
3. Problem statement (2 to 3 lines, with physics context: rare signal among
   abundant background, why anomaly detection, why low latency matters for
   trigger systems)
4. Method and architecture (short, with a pipeline diagram)
5. Results: AUC, ROC curve, error histograms, the chosen threshold and the
   real false positive and true positive rate at that threshold
6. Benchmark: table of PyTorch, ONNX Runtime Python and ONNX Runtime C++
   (mean, median, p99 in microseconds), CPU model, OS, library versions,
   build type, thread settings
7. Project structure
8. How to run (copy-paste Windows PowerShell commands)
9. Limitations and future work (honest; see Section 10)
10. Dataset citation and license (MIT)

Never write a number in the README that is not in a results file.

---

## 8. SPECIFICATION

### 8.1 Data: `python/prepare_data.py`
- Input: `data/higgs_sample.csv` (the first 200,000 rows of HIGGS, no header,
  created by the user; see STRUCTURE.md).
- **Do not take 20,000 purely random rows.** HIGGS is about half signal, so
  random sampling would not give enough background. Instead, with a fixed
  seed, **sample by label**: 15,000 background and 5,000 signal events.
- Splits:
  - train = 8,000 background
  - validation = 2,000 background
  - test = 5,000 background + 5,000 signal
- No event may appear in two splits (the test must verify this).
- Save to `data/processed/` as `.npz` (git-ignored) and write
  `data/sample_events.csv`: 1,000 events from the test split (500 background
  and 500 signal, shuffled, label in column 0, no header) for demos.
- Also write `data/test_events.csv` (the full test split, same format,
  git-ignored) for benchmarks.

### 8.2 Model: `python/model.py`
- Autoencoder 28 -> 16 -> 8 -> 16 -> 28, ReLU in hidden layers, linear output.
- Standardization (mean and std computed from the **training set only**) is
  stored as buffers and applied **inside the model** as the first operation,
  so the ONNX model takes raw features. Guard std with a small epsilon.
- A method that returns the per-event reconstruction error (mean squared
  error over the 28 features, computed in the standardized space) as a
  tensor of shape [N].

### 8.3 Training: `python/train.py`
- MSE loss, Adam, background events only, CPU.
- Hold out the validation split for monitoring; keep the best checkpoint.
- Threshold = **95th percentile of reconstruction error on the validation
  set** (background only). Save `models/threshold.json` and
  `models/threshold.txt` (a single float). Never choose the threshold using
  the test set.
- Save the checkpoint (`models/autoencoder.pt`) and `results/loss_curve.png`.
- Provide `--epochs` and other CLI options. Print the seed and all
  hyperparameters.

### 8.4 Evaluation: `python/evaluate.py`
- Reconstruction error on the test set, ROC curve, AUC (`scikit-learn`),
  error histograms (background vs signal, log-scaled y axis if helpful).
- Report the real true positive rate and false positive rate at the chosen
  threshold.
- Save plots to `results/` and numbers to `results/metrics.md`.
- Note: HIGGS signal and background look very similar, so a modest AUC is
  expected and acceptable. Report it as measured.

### 8.5 Export: `python/export_onnx.py`
- `torch.onnx.export`, input name `features` with shape [1, 28], opset 17,
  fixed batch size 1, output name `error` (the per-event reconstruction
  error). Save `models/detector_model.onnx`.
- If the exporter requires an extra package (for example `onnxscript`) or has
  a new default, handle it and document the fix in the README.

### 8.6 Verification: `python/verify_onnx.py`
- 100 test events: PyTorch vs ONNX Runtime (Python) outputs must differ by
  less than 1e-5 (report the real maximum absolute difference).
- Also accept the C++ output file (`results/cpp_errors.txt`) and compare it
  with the Python output on the same events (same tolerance). Report real
  values. If the tolerance is not met because of float32 rounding, report the
  real maximum difference and ask the user how to proceed.

### 8.7 C++ inference: `cpp/inference.cpp`, `cpp/CMakeLists.txt`
- Load `models/detector_model.onnx` with the ONNX Runtime C++ API.
- Read events from a CSV (label in column 0, then 28 features), run each with
  batch size 1, print the error and `NORMAL` or `ANOMALY` using
  `models/threshold.txt`.
- Write the first 100 errors to `results/cpp_errors.txt` for comparison.
- `--bench` mode: 100 warmup runs, 10,000 timed runs, per-run latency in
  microseconds (mean, median, p99) written to `results/cpp_benchmark.json`.
  Inputs preloaded in memory; only `Run()` is timed.
- Set intra-op threads to 1 and inter-op threads to 1.
- CMake accepts `-DONNXRUNTIME_DIR=<path>`, defaults to Release, and copies
  `onnxruntime.dll` next to the executable on Windows.
- Document exact PowerShell build commands in the README and verify them.

### 8.8 Benchmark: `python/benchmark.py`
- Three paths on the same model, same inputs, batch size 1, 100 warmup runs,
  10,000 timed runs, `time.perf_counter_ns`, only the inference call timed:
  1. PyTorch eager (CPU, `torch.set_num_threads(1)`, `torch.inference_mode()`)
  2. ONNX Runtime Python (intra-op = inter-op = 1 thread)
  3. ONNX Runtime C++ (results read from `results/cpp_benchmark.json`)
- Report mean, median, p99 in microseconds and the speedups relative to
  PyTorch. Record CPU model (`Get-CimInstance Win32_Processor` or Python),
  OS, Python, torch, onnxruntime (Python) and ONNX Runtime (C++) versions,
  and the C++ build type.
- Save to `results/benchmark.md`.
- **Honest interpretation:** paths 2 and 3 run the same engine, so any
  difference between them is binding and call overhead only. The README must
  say so and must not present the C++ vs PyTorch gap as a property of the
  language alone.

### 8.9 Tests: `tests/`
- `tests/test_data.py`: split sizes, label purity (train and validation
  contain only background), no overlap between splits.
- `tests/test_model.py`: output shape, error is non-negative, standardization
  is inside the model (raw input works), forward and backward pass finite.
- Run with `python -m pytest -v` (add `pytest` to requirements).

---

## 9. Common mistakes to avoid

- Training on signal events.
- Computing standardization statistics on validation or test data (leakage).
- Choosing the threshold on the test set.
- Sampling 20,000 random HIGGS rows (not enough background; sample by label).
- Reporting accuracy only. Report ROC AUC and the operating point.
- Standardizing outside the model, so ONNX and C++ get differently scaled
  inputs.
- Mismatched ONNX Runtime header, library and DLL versions on Windows (the
  `onnxruntime.dll` must sit next to the executable).
- Timing a Debug build, or timing CSV parsing together with inference.
- Unequal thread settings between PyTorch, ORT Python and ORT C++.
- Comparing cold first runs. Always warm up.
- Reporting "example" or remembered latency numbers.
- Committing the full HIGGS dataset or large files.

---

## 10. Limitations that the README must state honestly

- HIGGS is a simulated dataset, not real detector data.
- The signal in HIGGS is hard to separate from background; an unsupervised
  autoencoder is expected to reach only a modest AUC. This is a feasibility
  and latency study, not a physics discovery.
- Latency is measured for the model alone on one CPU. It excludes data
  acquisition, I/O and any real trigger system constraints.
- Results come from one random seed and one data split unless more are run.

---

## 11. End-of-task checklist

- [ ] Code runs, real output shown
- [ ] No invented or unmeasured numbers or causal claims anywhere
- [ ] Every change committed with a conventional commit message
- [ ] No amend, rebase, reset or force of any kind
- [ ] `git status` is clean
- [ ] Nothing pushed unless the user explicitly asked in this conversation
- [ ] Summary given: files changed, commits (hash + message), results, next step
