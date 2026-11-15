# C-MAPSS turbofan: remaining useful life prediction

Owner: Vedant Singh. Dataset: NASA C-MAPSS, subset FD001 (simulated engines,
one operating condition, one fault mode: HPC degradation). 100 training engines
run to failure; 100 test engines are cut off early, with the true RUL at the
cut-off given in RUL_FD001.txt. See CMAPSS_readme.txt for the dataset notes.
Every design choice, with the reasoning and the numbers behind it, is in
decisions.md (Decisions 1-17).

## Pipeline

Steps 1-6 load the data, drop 7 near-constant sensors (unique-value count <= 5,
computed on train only; this catches sensor_6, which a variance check misses),
build RUL labels capped at 125 cycles, drop op_setting_3 and standardise with
train-only statistics. 16 columns remain (2 op settings, 14 sensors).

Tree models get rolling features per engine: mean, std, min and max over a
window, plus the drift from that engine's first reading (16 -> 96 features).
The window (5 cycles) was chosen by 5-fold cross-validation grouped by engine.
CV scores each validation engine at 5 random cut points (cycle >= 31) to mimic
the truncated test engines. The LSTM reads the raw 16 columns as a sequence;
its settings (70-cycle sequences, dropout 0, learning rate 0.0005) were also
chosen by the same grouped CV. The test set was never used to choose anything.

Scoring uses the last cycle of each test engine only (100 predictions): RMSE,
MAE and the PHM08 score with d = predicted - true, exp(-d/13) - 1 when early
and exp(d/10) - 1 when late. Late predictions cost more. PHM08 is a sum over
engines, so it depends on the number of test points.

## Recorded results

100 test engines, mean ± std over 5 random seeds (seed_robustness.csv):

| Model | RMSE | MAE | PHM08 | Within ±10 cycles |
|---|---|---|---|---|
| Random Forest, raw 16 features (seed 42) | 17.19 | 12.13 | 875 | - |
| Random Forest, rolling window 20 | 14.27 ± 0.13 | 9.91 | 365 ± 11 | 69 |
| Random Forest, rolling window 5 | 13.52 ± 0.15 | 9.60 | 322 ± 5 | 66 |
| XGBoost, rolling window 5 | 13.47 ± 0.20 | 9.97 | 298 ± 11 | 64 |
| XGBoost, tuned (lambda 10) | 13.52 ± 0.10 | 10.10 | 298 ± 6 | 63 |
| XGBoost, asymmetric loss (late x5) | 14.31 ± 0.23 | 10.28 | 316 ± 20 | 64 |
| LSTM, untuned (30-cycle sequences) | 15.44 ± 0.61 | 11.11 | 521 ± 120 | - |
| LSTM, tuned (70-cycle sequences) | 14.07 ± 0.27 | 10.35 | 358 ± 19 | 61 |

Rolling features are the main gain (RMSE 17.2 -> 13.5). Random Forest and
XGBoost tie on RMSE; XGBoost has the best PHM08 (fewer costly late misses).
Tuning helped the LSTM but not XGBoost. The asymmetric loss made the fewest
late predictions but did not improve PHM08. SHAP shows the models rely on a
group of temperature, pressure and speed sensors (3, 4, 9 and 11) that drift
together as the HPC degrades; Gini importance overstates the top feature by
about 2x. Earlier single-seed numbers (random_state=42) were slightly
flattering, which is why the table uses seed averages.

Limitations: FD001 only (FD002-FD004 untested), simulated data, point
predictions without uncertainty, RF and XGBoost mostly default settings, and
the step-5 scaling uses all training engines, so CV folds share it.
The ONNX export of the tuned LSTM matches PyTorch to 0.00003 cycles but is not
faster at this model size.

## Reproduce

Install requirements-turbofan.txt from the repository root. Put
train_FD001.txt, test_FD001.txt and RUL_FD001.txt in data/raw/turbofan/, then
run the scripts in src/models/turbofan/ in this order:

    step1, step3, step4, step5, step6        data preparation and checks
    step8, step9, make_charts                baseline vs rolling features
    step10, step11                           grouped CV, XGBoost, asymmetric loss
    step12, step13                           untuned LSTM, SHAP
    step16, step17, step14                   LSTM tuning (slow), tuned LSTM, ONNX export
    step18, step19, step20, step15           XGBoost tuning, seed check, charts

Step 16 takes about 80 minutes on a laptop CPU. It skips settings already in
reports/turbofan/lstm_tuning_cv.csv, so delete that file to re-run the search.
Processed data goes to data/processed/turbofan/ and models to models/turbofan/;
both stay ignored by Git. Results, run logs and figures are in reports/turbofan/.
Developed with pandas 3.0, scikit-learn 1.8, xgboost 3.4 and PyTorch 2.12 (CPU).
Not yet tested against the repository's pandas<3 pin.
