# Milling notebooks

The reproducible EDA and training pipeline currently lives in
`src/milling_pipeline.py`; it generates the figures in `reports/figures/milling/`
and the metrics/predictions in `reports/milling/`.

Add exploratory or presentation notebooks here if needed. Avoid duplicating
training logic in notebooks; import reusable functions from
`src.milling_pipeline` instead.
