# Milling — team sync summary

## Completed in Stage 2

- Added a PyTorch sequence-to-one LSTM using the last five cuts in each case.
- Added symmetric XGBoost and a custom asymmetric pseudo-Huber XGBoost objective.
- Trained symmetric and asymmetric LSTM variants to isolate the effect of the loss.
- Kept missing VB labels missing; train/evaluate only on actually measured VB.
- Retained Case 6: its sole row is a genuine measured run (run 1, VB=0.0 mm); it is in the training set but provides no wear trajectory.
- Added training-range-normalized RMSE and wear-underprediction metrics.
- Saved checkpoints, per-case metrics, predictions, feature importance, and EDA plots.

## Current best model

**5-cut LSTM with symmetric loss:** RMSE 0.2331 mm, MAE 0.1624 mm, NRMSE 0.3151 on 80 measured test labels.

## What the asymmetric-loss experiment says

- XGBoost asymmetric: RMSE 0.2842 mm versus 0.2915 mm for symmetric XGBoost; modest improvement.
- LSTM asymmetric: RMSE 0.2824 mm versus 0.2331 mm for symmetric LSTM; worse on this holdout.
- Therefore asymmetric loss has been implemented and evaluated, but it is not a universal winner for Milling.

## RUL / cap decision

Milling predicts flank wear `VB` in mm. The dataset README does not provide a numeric end-of-life VB threshold, so no RUL countdown or RUL cap is defensible yet. Do not clip VB at a guessed threshold. Obtain an agreed `VB_EOL` threshold before adding a derived remaining-cuts target.

## Limitation to call out

Training VB max = 0.74 mm; test includes Case 13 at 1.53 mm and Case 14 at 1.14 mm. The held-out replicate includes wear values beyond the training range, causing high-wear extrapolation error.
