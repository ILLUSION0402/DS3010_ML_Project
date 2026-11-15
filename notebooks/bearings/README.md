# Bearing Health Prediction — Experiments

This directory contains the Jupyter notebooks for the NASA IMS bearing predictive-maintenance component.

## Baseline

**01_baseline/feature_extraction_and_baseline.ipynb**

Extracts statistical, spectral, and bearing fault-frequency features from raw vibration data. Implements the original Extra Trees health-index prediction model.

## Additional Experiments

**02_experiments/lstm_asymmetric_loss.ipynb**

Investigates temporal sequence modeling using PyTorch LSTM and an asymmetric loss function.

**02_experiments/extratrees_optimization.ipynb**

Experiments with rolling temporal features, late-stage sample weighting, and Extra Trees hyperparameter tuning.

**02_experiments/early_warning_feature_ablation.ipynb**

Evaluates early failure warnings across bearings, compares feature groups through ablation, and investigates difficult failure cases.

## Model Inference

The baseline inference implementation is located in `src/models/bearings/bearing_predictor2.py`.

Experimental Python implementations are located in `src/models/bearings/experiments/`.

## Results

Evaluation CSVs and figures are stored under `reports/bearings/`. Historical superseded experiments are preserved under the corresponding archive directories.