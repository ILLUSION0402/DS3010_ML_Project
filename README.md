# Predictive Maintenance for Industrial Equipment

## Project structure

This repository is the shared project workspace for the predictive-maintenance project.

Current team dataset plan:
- Primary: NASA C-MAPSS FD001
- Backup options: NASA Bearing / FEMTO Bearing, if retained by the team

The exact dataset structure, sensor meanings, units, and official train/test split should be documented by Person 1.

## Repository layout

```text
predictive-maintenance/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── notebooks/
├── src/
│   ├── api/
│   │   └── main.py
│   ├── data/
│   ├── features/
│   ├── models/
│   └── evaluation/
├── models/
├── reports/
├── tests/
├── requirements.txt
├── .gitignore
└── README.md
```

## Shared processed-data schema

The team should treat the processed dataset as one row per asset at one operating cycle/time point.

### Core columns

| Column | Type | Description |
|---|---|---|
| `dataset` | string | Dataset identifier, e.g. `cmapss_fd001` |
| `asset_id` | integer/string | Engine or equipment identifier |
| `cycle` | integer | Time/cycle index for the asset |
| `split` | string | Dataset split: `train`, `validation`, or `test` |
| `op_setting_01` | float | Operating condition 1, when available |
| `op_setting_02` | float | Operating condition 2, when available |
| `op_setting_03` | float | Operating condition 3, when available |
| `sensor_01` ... `sensor_21` | float | Standardized sensor feature columns for C-MAPSS |
| `rul` | float/null | Remaining Useful Life target; populated only where a ground-truth target is available |

Notes:
- This is a schema decision only. No preprocessing, scaling, imputation, feature engineering, or RUL generation belongs here.
- For datasets with fewer/different signals, the team should preserve the same metadata fields and document the dataset-specific mapping rather than silently inventing values.
- Raw data must remain untouched in `data/raw/`.
- Processed files should preferably use Parquet (`.parquet`) for the shared analytical format.

## Data policy

Do not commit large raw datasets or trained model binaries to Git unless the team explicitly decides to use Git LFS. Put raw datasets locally in `data/raw/` and document their source and expected filenames in the dataset README.

## API

The FastAPI service is intentionally only a placeholder at this stage.

Run locally:

```bash
uvicorn src.api.main:app --reload
```

Then open:
- Swagger UI: `http://127.0.0.1:8000/docs`
- Health endpoint: `http://127.0.0.1:8000/health`

The API does not load a trained model yet.
