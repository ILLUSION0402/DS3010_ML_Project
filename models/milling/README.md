# Milling trained model artifacts

These are the saved Stage 2 artifacts produced by `src/milling_pipeline.py`:

- `random_forest_measured_vb.joblib`
- `xgboost_symmetric.json`
- `xgboost_asymmetric.json`
- `lstm_symmetric.pt`
- `lstm_asymmetric.pt`
- `preprocessing.joblib`
- `xgboost_target_scalers.json`

The preprocessing artifact stores feature metadata/scaling information needed
for consistent inference. Keep these artifacts versioned together with the
pipeline revision that produced them. Re-run the training command after changing
feature extraction, model architecture, split, or loss configuration.
