# Predictive Maintenance for Industrial Equipment

DS3010 Introduction to Machine Learning, course project.

The project predicts **Remaining Useful Life (RUL)**: how long a machine part
can keep running before it fails, using sensor data recorded while it degrades.
Each team member takes one kind of equipment and one public dataset, and builds
the full pipeline for it: data preparation, models, evaluation and a report.

## Components

| Component | Dataset | What is predicted | Owner | Where to look |
|---|---|---|---|---|
| Turbofan engine | NASA C-MAPSS, subset FD001 | RUL in flight cycles | Vedant Singh | [reports/turbofan](reports/turbofan/Turbofan_Component_Report.md), code in `src/models/turbofan/` |
| Rolling-element bearing | NASA IMS / Rexnord bearings | Health index from 1 (healthy) to 0 (failed) | Samyuktha K | [reports/bearings](reports/bearings/Bearings_Component_Report.md), code in `notebooks/bearings/`, `src/models/bearings/` |
| Lithium-ion battery | NASA PCoE Battery dataset | RUL in charge cycles | Ch. Srinivasa Saketh | [reports/battery](reports/battery/Battery_Component_Report.md), code in `src/data/battery/`, `src/models/battery/` |
| Milling tool | NASA milling dataset | RUL in cutting runs | S Mithil Reddy | not uploaded yet |

Each component's report lists its pipeline, its recorded results and its
limitations, and explains how to reproduce it.

**The results are not ranked against each other.** The four components measure
different machines in different units (flight cycles, charge cycles, a 0-1
health index, cutting runs), with different test-set sizes. The PHM08 score is
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
├── notebooks/bearings/       bearings pipeline notebook
├── src/
│   ├── api/main.py           FastAPI service (placeholder)
│   ├── data/                 data preparation (battery, early turbofan EDA)
│   └── models/<component>/   training, evaluation and inference code
├── reports/
│   ├── <component>/          component report, result CSVs, figures/
│   ├── Summary.md            early FD001 exploratory analysis
│   └── data_schema.md        shared processed-data schema
├── tests/                    pytest tests
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

Then follow the "Reproduce" section of the component's report. Each report
says which files to download and where to put them under `data/raw/`.

## Data policy

Raw datasets and trained model files are not committed to Git; `data/` and
`models/` are ignored (see `.gitignore`). Each component keeps its own
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
