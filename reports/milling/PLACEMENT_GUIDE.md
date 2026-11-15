# Where the Milling contribution belongs in the team repository

Copy/merge the contents of this folder into the repository root. This is an
integration overlay; it does not replace any existing team folders or files.

| This package path | Destination in the team repository | Purpose |
|---|---|---|
| `src/milling_pipeline.py` | `src/milling_pipeline.py` | Complete runnable Milling pipeline: data loading, waveform feature engineering, measured-VB target handling, case split, RF, XGBoost, LSTM, asymmetric objectives, metrics and EDA. |
| `run_milling.py` | `run_milling.py` | Small root-level launcher. |
| `requirements-milling.txt` | `requirements-milling.txt` | Milling-specific dependencies, matching the repo's per-component `requirements-bearings.txt` pattern. |
| `data/raw/milling/README.md` | `data/raw/milling/README.md` | Documents where the raw `mill.mat` belongs. The actual 73 MB data file is deliberately not included. |
| `models/milling/*` | `models/milling/*` | Trained RF, XGBoost and LSTM artifacts plus preprocessing metadata. |
| `reports/milling/*` | `reports/milling/*` | Metrics, predictions, case audit, configuration, results summary and team-sync note. |
| `reports/figures/milling/*` | `reports/figures/milling/*` | EDA and model comparison figures. |
| `notebooks/milling/README.md` | `notebooks/milling/README.md` | Place for optional Milling notebooks; training logic remains in `src/`. |
| `tests/test_milling_pipeline.py` | `tests/test_milling_pipeline.py` | Basic tests for waveform features and asymmetric loss. |
| `GITIGNORE_MILLING_SNIPPET.txt` | Merge entries into root `.gitignore` | Excludes raw data and regenerable run outputs. Do not overwrite the team's `.gitignore`. |

## Why the training code is one file for now

The supplied Stage 2 implementation is a self-contained end-to-end script. Keep
it together at `src/milling_pipeline.py` for the first integration rather than
splitting functions manually and accidentally breaking imports. The functions
are clearly separated and can later be extracted into the existing
`src/data/`, `src/features/`, `src/models/`, and `src/evaluation/` packages in a
separate refactor/PR.

## Run from repository root

```bash
python -m pip install -r requirements-milling.txt
python run_milling.py --data data/raw/milling/mill.mat --output reports/milling/run
```

Equivalent module command:

```bash
python -m src.milling_pipeline --data data/raw/milling/mill.mat --output reports/milling/run
```

The script writes freshly trained model files under
`reports/milling/run/models/`. To update the versioned checkpoints in
`models/milling/` after reviewing the run, copy those files there. Keep the
metrics and configuration from the same run together.

## Tests

```bash
python -m pytest -q tests/test_milling_pipeline.py
```

## Git staging suggestion

After merging, inspect `git status`, then stage only the Milling paths. Example:

```bash
git add src/milling_pipeline.py run_milling.py requirements-milling.txt \
  data/raw/milling/README.md models/milling notebooks/milling \
  reports/milling reports/figures/milling tests/test_milling_pipeline.py
```

Do not stage `mill.mat` unless the team has explicitly approved committing large
raw datasets or Git LFS storage.
