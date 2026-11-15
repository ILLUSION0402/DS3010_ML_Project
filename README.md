# Predictive Maintenance for Industrial Equipment

DS3010 Introduction to Machine Learning, course project.

The project predicts how close a machine part is to failure, using sensor data
recorded while it degrades: its **Remaining Useful Life (RUL)**, or a health or
wear measure that leads to it.
Each team member takes one kind of equipment and one public dataset, and builds
the full pipeline for it: data preparation, models, evaluation and a report.

## Components

| Component | Dataset | What is predicted | Owner | Where to look |
|---|---|---|---|---|
| Turbofan engine | NASA C-MAPSS, subset FD001 | RUL in flight cycles | Vedant Singh | [reports/turbofan](reports/turbofan/Turbofan_Component_Report.md), code in `src/models/turbofan/` |
| Rolling-element bearing | NASA IMS / Rexnord bearings | Health index from 1 (healthy) to 0 (failed) | Samyuktha K | [reports/bearings](reports/bearings/Bearings_Component_Report.md), experiment results in `reports/bearings/experiments/`, code in `notebooks/bearings/`, `src/models/bearings/` |
| Lithium-ion battery | NASA PCoE Battery dataset | RUL in charge cycles | Ch. Srinivasa Saketh | [reports/battery](reports/battery/Battery_Component_Report.md), code in `src/data/battery/`, `src/models/battery/` |
| Milling tool | NASA PCoE milling dataset | Tool flank wear VB in mm | S Mithil Reddy | [reports/milling](reports/milling/TEAM_SYNC_SUMMARY.md), code in `src/milling_pipeline.py` (run with `run_milling.py`) |

Each component's report lists its pipeline, its recorded results and its
limitations, and explains how to reproduce it. The milling component predicts
tool wear directly: it does not convert wear to RUL yet, because the dataset
gives no failure threshold for VB (see its team-sync summary).

**The results are not ranked against each other.** The four components measure
different machines in different units (flight cycles, charge cycles, a 0-1
health index, millimetres of tool wear), with different test-set sizes. The PHM08 score is
a sum over test points, so it grows with the number of test points. The numbers
are presented side by side, one component at a time.

## Repository layout

```text
.
├── data/                     raw, interim and processed data (not in Git)
│   ├── raw/<component>/      original downloaded datasets
│   ├── interim/
│   └── processed/<component>/
├── models/<component>/       trained models (not in Git)
├── notebooks/<component>/    notebooks (bearings pipeline and experiments)
├── src/
│   ├── api/main.py           FastAPI service (placeholder)
│   ├── data/                 data preparation (battery, early turbofan EDA)
│   ├── models/<component>/   training, evaluation and inference code
│   └── milling_pipeline.py   complete milling pipeline in one file
├── reports/
│   ├── <component>/          component report, result CSVs, figures/
│   ├── figures/milling/      milling figures
│   ├── Summary.md            early FD001 exploratory analysis
│   └── data_schema.md        shared processed-data schema
├── tests/                    pytest tests
├── run_milling.py            entry point for the milling pipeline
├── requirements.txt          shared dependencies
└── requirements-<component>.txt   extra dependencies for one component
```

## Setup

Python 3.11 or newer is recommended. From the repository root:

```bash
pip install -r requirements.txt
```

Some components need extra packages on top of the shared ones:

```bash
pip install -r requirements-turbofan.txt
```

```bash
pip install -r requirements-bearings.txt
```

```bash
pip install -r requirements-milling.txt
```

Then follow the "Reproduce" section of the component's report. Each report
says which files to download and where to put them under `data/raw/`.

## Data policy

Raw datasets and trained model files are not committed to Git; `data/` and
`models/` are ignored (see `.gitignore`). The one exception is
`models/milling/`: its trained models are small (under 2 MB) and are kept in
Git on purpose, so the milling results can be checked without retraining. Each component keeps its own
subfolder (`data/raw/turbofan/`, `models/bearings/`, ...), so components never
overwrite each other's files. Result CSVs and figures that the reports refer to
are committed under `reports/<component>/`.

## API

The FastAPI service is a placeholder for now. It only has a health check and
does not load a trained model yet.

```bash
uvicorn src.api.main:app --reload
```

Then open `http://127.0.0.1:8000/docs` or `http://127.0.0.1:8000/health`.

## Tests

```bash
pytest
```

`tests/test_api.py` needs FastAPI from `requirements.txt`;
`tests/test_milling_pipeline.py` checks the milling feature code and its
asymmetric loss.
