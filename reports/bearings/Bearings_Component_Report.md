# IMS bearings: health-index prediction

Owner: Samyuktha K. Dataset: NASA IMS/Rexnord bearing experiments.
See the dataset README for rig details, channel mapping and attribution.

## Pipeline

The final ExtraTrees model uses 26 vibration features plus 26 slopes.
Its configuration is 500 trees, min_samples_leaf=50, max_features=0.3,
random_state=0. Each bearing uses operating hours 40-70 as its baseline.
Predictions use rolling-median/EMA smoothing and a cumulative minimum.
The constructed target is 1 until detected onset, then declines to 0
at the run endpoint. Final v2 uses automatically detected onset.

## Recorded results

The uploaded notebook recorded mean RMSE 0.120, MAE 0.053 and
late-stage RMSE 0.418. Evaluation holds out entire test experiments:
three groups containing four failed-bearing trajectories. Late-stage
RMSE uses reference target < 0.5. PHM08 is not calculated.

final_v2_heldout_recorded.csv and final_v2_heldout.png contain historical
final-v2 results. Experiment CSVs use manual reference labels;
final_heldout.png is the earlier manual-onset version, and
test_predictions.png is an earlier baseline. These artifacts are copied
from the original work; this preparation script does not rerun training.

Model selection reused the held-out groups, so these are exploratory
results. The baseline uses future observations for early retrospective
predictions. Normalisation checks positivity across the supplied run.
Near-failure overestimation remains substantial. Fault-frequency evidence
and remaining-hours ranges are heuristic, not validated diagnoses or forecasts.

## Reproduce

Install requirements-bearings.txt from the repository root.
Set BEARINGS_DATA_ROOT to the existing bearings directory, then run
notebooks/bearings/bearings_pipeline.ipynb in order.

The first run rebuilds data/interim/bearings/features_raw.parquet from
raw snapshots. The old pickle is not reused because it is incompatible
with the repository's pandas 2.x dependency. Models save in models/bearings.
Raw data, the cache and models remain ignored by Git. Existing model
bundles may require retraining if scikit-learn versions differ.
