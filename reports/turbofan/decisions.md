# Decisions Log — FD001 Turbofan Pipeline

Why each choice was made, not just what was done.

Note: nothing in this pipeline removes any ROWS. Only columns are dropped. Row counts stay at 20,631 (train) and 13,096 (test) from raw files through to the final normalised CSVs.

---

## Decision 1: Start with FD001, not FD002/003/004

**What:** Working with the FD001 subset only.

**Why:** FD001 is the simplest of the four — one operating condition (sea level), one fault mode (HPC degradation). FD002 and FD004 have six operating conditions, which means normalisation has to be done per-condition rather than globally. That is real extra complexity we did not need while the pipeline was still being built.

**Effect:** faster iteration. **Cost:** results only speak to the easiest subset. We have not tested whether the approach holds on FD002–FD004.

---

## Decision 2: Drop 7 sensor columns, chosen by unique-value count

**What:** Removed `sensor_1, 5, 6, 10, 16, 18, 19`. Kept 14 sensors plus 2 operational settings.

**Why:** A sensor that never meaningfully changes cannot help distinguish a healthy engine from a degrading one.

**Why unique-value count instead of variance:** `sensor_6` is the reason. It has a non-zero standard deviation (0.001389), so a check like `if std < 1e-5` keeps it. But it only ever takes two values across all 20,631 training rows — 21.60 and 21.61. That is rounding noise, not signal. Counting distinct values catches it; a variance threshold does not.

| Sensor | Unique values | Std | Verdict |
|---|---|---|---|
| sensor_1 | 1 | 0.0 | drop |
| sensor_5 | 1 | 0.0 | drop |
| sensor_6 | 2 | 0.001389 | drop — the one variance misses |
| sensor_10 | 1 | 0.0 | drop |
| sensor_16 | 1 | 0.0 | drop |
| sensor_18 | 1 | 0.0 | drop |
| sensor_19 | 1 | 0.0 | drop |
| all others | 13–6,403 | non-trivial | keep |

**Threshold:** 5 or fewer unique values. Deliberately loose — the real sensors jump from 2 distinct values straight to 53, so there is a wide gap and no risk of catching a useful one.

**Leakage note:** the drop list is computed from training data only, then applied to test.

---

## Decision 3: Cap the RUL label at 125 cycles

**What:** Both train and test RUL labels are clipped so no value exceeds 125.

**Why:** An engine 300 cycles from failure and one 250 cycles from failure look identical in the sensor readings — both are simply healthy. The degradation signal only emerges in roughly the last 50 cycles. An uncapped label would ask the model to output a number that is not recoverable from its inputs, then penalise it for getting that impossible number wrong.

**Why 125 specifically:** the shortest-lived training engine fails at cycle 128. A cap of 125 sits just below that, so it applies to every engine without cutting into any engine's actual failure history, and each one retains a flat healthy stretch before the countdown begins.

This is the standard "piecewise-linear RUL" convention used across almost all published FD001 work.

**How train and test differ:**
- Train: `RUL = engine's own last recorded cycle − current cycle`, since every training engine's last row is its actual failure.
- Test: `RUL = value from RUL_FD001.txt + (engine's last recorded cycle in test − current cycle)`. Test engines are cut off before failure, so their last row is NOT the failure point — `RUL_FD001.txt` supplies how many cycles were left at that cutoff.

**Result:** train RUL ranges 0–125, test RUL ranges 7–125.

---

## Decision 4: Drop op_setting_3, then normalise on train-only statistics

**What:** `op_setting_3` dropped (1 unique value). Remaining features standardised using mean/std from training data only, applied unchanged to test.

**Why op_setting_3 had to go first:** this was caught as a bug, not planned. The first attempt normalised all three operational settings without checking them the way sensors were checked. `op_setting_3` has a standard deviation of exactly 0 (FD001 has one operating condition, so throttle never changes) — dividing by zero produced `NaN` for every row in that column, silently corrupting it. Checking unique-value counts caught it. `op_setting_1` (158 unique values) and `op_setting_2` (13) have real variation and were kept.

**Why train-only statistics:** computing mean/std from train+test combined would leak information about the test distribution into preprocessing, before any model sees the data.

**How to show it worked:** after scaling, train features have mean ≈ 0 and std = 1 exactly. Test features are close to but not exactly 0 and 1 — that asymmetry is the proof the scaler was not refit on test.

**Result:** 19 columns — `unit_number`, `cycle`, `op_setting_1`, `op_setting_2`, 14 sensors, `RUL`. No NaNs in either file.

---

## Decision 5: Evaluate on the last cycle of each test engine only

**What:** The model trains on all 20,631 training rows but is scored on 100 predictions — one per test engine, at its final recorded cycle.

**Why:** This is the actual prognostics task: given everything observed so far, how many cycles are left right now? `RUL_FD001.txt` supplies ground truth for exactly this moment, one value per engine.

Scoring every intermediate test row instead would inflate results, because early-life rows sit at the capped RUL of 125 and are trivially easy to predict — the model would look better without being better at the part that matters.

**Effect:** this makes our numbers look worse than they would under a per-row protocol. That is correct, and worth saying if results are compared against published figures that use a different protocol.

---

## Decision 6: Baseline first, then feature engineering

**What:** Two Random Forests trained and compared — one on the normalised sensor values as-is, one with rolling-window features added.

**Why:** Running the simple version first gives a reference point. Without it there is no way to show that feature engineering helped, or by how much.

**Results:**

| Metric | Baseline | With feature engineering | Change |
|---|---|---|---|
| RMSE | 17.193 | 14.041 | 18.3% better |
| MAE | 12.128 | 9.783 | 19.3% better |
| PHM08 score | 874.62 | 351.29 | 59.8% better |

Feature count went from 16 to 96.

---

## Decision 7: Rolling-window features, window size 20

**What:** For each of the 16 remaining sensor/setting columns, five derived features: rolling mean, standard deviation, min and max over a 20-cycle window, plus a delta against that engine's first recorded value.

**Why rolling features at all:** a single row tells the model what a sensor reads right now, but degradation is a trend — the useful signal is how a reading has been drifting. Rolling statistics give that temporal context without needing a sequence model.

**Why window = 20:** large enough to smooth the sensor noise this dataset is known to contain, small enough to still react when an engine starts degrading. A much larger window would blur the signal; a much smaller one would mostly pass through noise. **This was chosen by judgement and not tuned** — tuning it is a legitimate next step. (Done later in Decision 9: grouped CV prefers window 5.)

**Why the delta feature:** the readme states engines start with different amounts of initial wear. Measuring change from each engine's own starting value normalises that away, so the model sees how far this engine has drifted from its own baseline rather than an absolute reading polluted by manufacturing variation.

**No leakage across engines:** all rolling calculations are grouped by `unit_number`, so a window never pulls values from a different engine. `min_periods=1` means early cycles use whatever history exists rather than producing NaN.

**What the model found most useful:** `sensor_3_mean` dominates at 52% of total importance, followed by `sensor_2_mean` (10%) and `sensor_11_delta` (10%). All of the top 20 features are engineered, and engineered features hold 99.2% of the total importance. (An earlier draft said "three of the top four", which understated it.) This is Gini importance; see Decision 12 for the SHAP check. `sensor_3` is the HPC outlet temperature, and FD001's fault mode is HPC degradation — the model found the failing component without being given any physics.

---

## Decision 8: The PHM08 scoring function, and a correction

**What:** Alongside RMSE and MAE we report the PHM08 challenge score, which penalises errors exponentially and asymmetrically. With `d = predicted RUL − true RUL`:

```
d < 0  (early prediction):  exp(-d / 13) - 1
d >= 0 (late prediction):   exp( d / 10) - 1
```

**Why:** RMSE and MAE treat all errors equally, but maintenance does not. Telling a team an engine has 40 cycles left when it has 10 risks an in-service failure; telling them 10 when it has 40 just means an early service. The smaller divisor on the late branch makes that side steeper. At an error of 50 cycles: late scores 147, early scores 46 — roughly 3× the penalty for the same magnitude.

**The correction.** An earlier version of this code had the two constants swapped — 13 on the late branch and 10 on the early branch — which made the metric punish early predictions harder, the opposite of its purpose. Every PHM08 number produced before this fix was wrong. RMSE, MAE, the predictions, the feature importances and all the intermediate CSVs were unaffected, because the scoring function is only used after prediction.

| | Before fix (wrong) | Corrected |
|---|---|---|
| Baseline PHM08 | 521.59 | **874.62** |
| With features | 369.19 | **351.29** |
| Improvement | 29.2% | **59.8%** |

**Why the constants are easy to get wrong.** The Saxena et al. paper's text states "a1 = 10, and a2 = 13", which read literally produces the backwards behaviour. But the same page states that "late predictions were more heavily penalized than early predictions", and Figure 9 plots the score curve reaching approximately 150 on the late side and approximately 45 on the early side. The figure and the stated intent agree with each other and contradict the printed constants. It is a known typo; essentially all implementations in the literature use the version above.

**Effect of the fix on the story:** it strengthens it. PHM08 now improves by 60% against 18% on RMSE, which says the rolling features disproportionately removed the dangerous late errors rather than just lowering average error.

---

# Phase 2 (steps 10-15)

All scripts live in `src/models/turbofan/`. From step 10 on they import shared code from `common.py` (the ONE copy of the loading, rolling-feature and PHM08 code). Data is read from `data/raw/turbofan/` and `data/processed/turbofan/`, models go to `models/turbofan/`, and results go to `reports/turbofan/` (charts in `figures/`, run logs in `logs/`). Paths are worked out from each script's location, so they run from any folder. Run order: see the component report.

---

## Decision 9: 5-fold cross-validation grouped by engine (step 10)

**What:** `GroupKFold(5)` over the 100 training engines (20 per fold, every row of an engine in the same fold). Compared Random Forest and XGBoost at rolling windows 5, 10, 20, 30, 40.

**Why grouped:** neighbouring rows of one engine are nearly identical. A row-level split would let the model "see the answer" from the engine's neighbouring cycles.

**How a validation engine is scored:** training engines run to failure, test engines do not. Scoring every validation row would include the easy capped early-life rows (the same inflation Decision 5 avoids). So each training engine gets 5 random cut cycles (cycle >= 31, the length of the shortest test engine; seed 42) and only those rows are scored. All features look backwards only, so the row at cycle t has exactly the features the engine would have if it had been cut off at t. 20 engines x 5 cuts = 100 points per fold, the same count as the test set.

**Results (mean ± std over folds):**

| Window | RF RMSE | XGB RMSE | RF PHM08 | XGB PHM08 |
|---|---|---|---|---|
| 5 | **15.06 ± 1.45** | **14.28 ± 1.01** | **529** | **431** |
| 10 | 15.40 ± 1.75 | 14.75 ± 1.54 | 543 | 465 |
| 20 | 16.39 ± 1.96 | 15.29 ± 2.49 | 650 | 544 |
| 30 | 16.25 ± 2.28 | 15.50 ± 2.05 | 685 | 580 |
| 40 | 16.09 ± 2.34 | 15.11 ± 2.17 | 598 | 527 |

**Reading it honestly:** the differences (~1 cycle) are smaller than the fold-to-fold spread. The stronger evidence is the paired comparison: RF at window 5 beats window 20 in 5 of 5 folds; XGBoost in 4 of 5. XGBoost beats RF at every window on average.

**Caveats:** step-5 scaling used all 100 training engines, so validation engines slightly influenced the mean/std (a shift and scale only; tiny but not zero). The window grid stops at 5; smaller windows were not tried.

---

## Decision 10: XGBoost as a second baseline, window chosen by CV (step 11)

**What:** XGBoost (500 trees, learning rate 0.05, depth 6, subsample 0.8, colsample 0.8 - common defaults, **not tuned**) and the RF, both at the CV-chosen window 5. The test set was only used after the window was fixed.

**Results on the 100 test engines:**

| Model | RMSE | MAE | PHM08 | Late preds | Late by 30+ | Worst late |
|---|---|---|---|---|---|---|
| RF, window 20 (step 9) | 14.041 | 9.783 | 351.29 | 60 | 3 | 37.8 |
| RF, window 5 | 13.302 | **9.247** | 318.76 | 58 | 4 | 38.0 |
| XGBoost, window 5 | **13.138** | 9.701 | **279.34** | 52 | 2 | 35.8 |

The RF window-20 row reproduces step 9 exactly, which checks that `common.py` matches the original code. The CV choice of window 5 also held up on the test set.

---

## Decision 11: Asymmetric loss (step 11) - a negative result

**What:** XGBoost trained with a custom loss: `w * d^2` with `w = LATE_WEIGHT` when the prediction is late (d > 0) and 1 when early. Gradient `2wd`, hessian `2w`.

**Why not PHM08 itself as the loss:** its exponential gradients explode while the model is still bad (an error of 100 gives exp(10) ≈ 22,000) and boosting becomes unstable. Weighted squared error pushes in the same direction and stays well behaved.

**Choosing the weight:** grouped CV (same folds and cut points as Decision 9), lowest mean CV PHM08. The first grid [1, 1.5, 2, 3, 5] picked 5, the edge, so it was extended to 8 and 12 to confirm a real minimum:

| Weight | 1 | 1.5 | 2 | 3 | **5** | 8 | 12 |
|---|---|---|---|---|---|---|---|
| CV PHM08 | 436 | 432 | 414 | 403 | **374** | 392 | 398 |

**Test result:** RMSE 14.614, MAE 10.245, PHM08 **333.11**, worse than plain XGBoost (279.34) even though CV predicted an improvement. It did what it was designed to do on direction: late predictions 52 -> 47, late by 30+ 2 -> 1. But shifting everything earlier added early errors, which PHM08 also charges for, and with only 100 engines a few big misses swing the sum a lot. **We report it as a negative result rather than tuning against the test set until it looks good.**

---

## Decision 12: LSTM in PyTorch (step 12)

**What:** 2-layer LSTM (hidden 64, dropout 0.2) -> 32 -> 1, reading the last 30 cycles of the 16 normalised columns. **No engineered features**, so the LSTM has to learn the trend itself; otherwise we could not tell what it adds. Label divided by 125 for training. Adam, lr 0.001, batch 256.

**Training per seed:** hold out 20 training engines, train on 80, early-stop on validation RMSE (patience 10, max 60 epochs), then retrain on all 100 engines for that many epochs so it sees the same data as the RF. Five seeds (0-4), because one neural-network run can be lucky.

**Results (100 test engines):**

| | RMSE | MAE | PHM08 |
|---|---|---|---|
| Mean ± std over 5 seeds | 15.44 ± 0.61 | 11.11 ± 0.22 | 521 ± 120 |
| Best / worst seed | 14.64 / 16.16 | 10.80 / 11.41 | 394 / 675 |
| Average of the 5 models' predictions | 14.37 | 10.20 | 414.81 |

**Reading it honestly:** the LSTM does not beat the tree models with hand-built rolling features. Best epochs were early (6-17), so it overfits quickly. Its settings (sequence length, size, learning rate) were not tuned here; Decision 15 tunes them. Published FD001 LSTMs report roughly 13-16 RMSE on this last-cycle protocol, so this is in the normal range, not a broken model. Runs on CPU in about 4 minutes for all 5 seeds.

---

## Decision 13: SHAP vs Gini importance (step 13)

**What:** exact TreeSHAP on the 100 test engines for the window-5 RF and the window-5 XGBoost. The additivity check passes (average prediction 86.81 + sum of SHAP = prediction, max error 0.0002 cycles).

**Findings:**
- **Gini exaggerates the top feature.** Window-5 RF: `sensor_4_mean` gets 0.62 of the Gini importance but 0.31 of the SHAP. Window-20 RF (the step-9 model): `sensor_3_mean` is 0.52 Gini vs 0.26 SHAP. XGBoost spreads credit much more evenly (its top feature has 0.11).
- **The top sensor depends on the window.** Window 20 -> `sensor_3` (HPC outlet temperature) on top; window 5 -> `sensor_4` (LPT outlet temperature). Sensors 2, 3, 4, 9 and 11 all drift together as the HPC degrades, and the forest leans on whichever one its splits happened to favour.
- Engineered features still dominate: 97.4% of the window-5 RF's SHAP.
- Worst late miss (window-5 RF): test engine 50, true 79, predicted 117 (38 cycles late). `reports/turbofan/figures/turbofan_shap_worst_engine.png` shows why.

**Effect on the story:** "the model found sensor_3, the HPC outlet temperature" is too strong. A safer claim: "the model relies on a group of temperature, pressure and speed sensors (3, 4, 9, 11) that all respond to HPC degradation".

---

## Decision 14: ONNX export (step 14)

**What:** the seed-0 LSTM exported to `models/turbofan/lstm_seed0.onnx` (274 KB, single file) and run with ONNX Runtime. (Step 14 now exports the tuned model instead; see Decision 15. The numbers below are for the untuned export.)

**Check:** largest difference from PyTorch on the 100 test engines is 0.00005 cycles; RMSE/MAE/PHM08 are identical (14.641 / 10.796 / 405.06).

**Speed:** no gain here (about 2 ms for 100 engines either way on CPU). The model is too small for it to matter. The point of ONNX is portability: the file runs without PyTorch or our code. TensorRT was not attempted.

Only the LSTM was exported. Tree models would need an extra converter package and the ~95 MB RF would not be a sensible deployment artifact anyway.

---

## Decision 15: Tuning the LSTM (steps 16-17)

**Why:** the untuned LSTM (Decision 12) stopped training after 6-17 epochs and lost to the trees. Tuning was done on training engines only, with the same 5 engine folds and 100 cut points per fold as Decision 9, so the CV numbers sit next to the tree models' (RF 15.06, XGBoost 14.28 at window 5).

**How (step 16):** inside each fold, 16 of the 80 training engines decide when to stop (early stopping), the model trains on the other 64, and it is scored at the cut points of the fold's 20 validation engines. Each fold uses the same seed in every setting, so settings are compared pairwise. Sequences longer than an engine's history are padded at the start by repeating the engine's first reading. Training includes padded sequences, so the model has seen padding before it meets a short test engine (7 test engines have fewer than 50 cycles).

**Round 1: one change at a time from the step-12 setup (CV RMSE; untuned 16.27):**

| Change | CV RMSE | Folds better |
|---|---|---|
| sequence 50 | 15.46 | 3/5 |
| dropout 0 | 15.82 | 3/5 |
| learning rate 0.0005 | 15.91 | 3/5 |
| sequence 40 / 1 layer / dropout 0.4 | 16.05 / 16.20 / 16.31 | 2/5 |
| batch 64 / lr 0.003 / hidden 128 / hidden 32 / sequence 20 | 16.70-17.78 | 0-1/5 |

**Rounds 2-3:** combine the helpful changes, and keep extending the sequence length while it sits at the edge of the grid (the same rule as the asymmetric weight):

| Setting | CV RMSE | CV PHM08 | Folds better |
|---|---|---|---|
| **sequence 70 + dropout 0 + lr 0.0005** | **14.21** | **373** | **5/5** |
| sequence 70 alone | 14.35 | 378 | 5/5 |
| sequence 110 + dropout 0 + lr 0.0005 | 14.55 | 457 | 4/5 |
| sequence 50 + dropout 0 + lr 0.0005 | 14.67 | 420 | 4/5 |
| sequence 90 + dropout 0 + lr 0.0005 | 14.81 | 432 | 5/5 |

Sequence length was by far the biggest lever, and 70 is a real best (90 and 110 are worse). A likely reason: the trees get each engine's whole history through the `_delta` features (change since the first reading), while a 30-cycle LSTM never sees an engine's starting point. The trees need short windows (5); the LSTM needs long ones (70).

**Test result (step 17):** both settings trained with identical code and the same 5 seeds, scored once on the 100 test engines:

| | RMSE | MAE | PHM08 |
|---|---|---|---|
| Untuned (same code as tuned) | 15.43 ± 1.33 | 11.00 ± 0.69 | 514 ± 232 |
| **Tuned, mean ± std of 5 seeds** | **14.07 ± 0.27** | **10.36 ± 0.20** | **358 ± 19** |
| Tuned, average of the 5 models' predictions | 13.149 | 9.789 | 295.07 |

Tuning lowered the error and made the LSTM far more consistent from seed to seed (the PHM08 spread fell from 232 to 19). A single tuned LSTM still trails XGBoost (13.138 / 9.701 / 279.34). Averaging 5 tuned LSTMs roughly matches XGBoost on RMSE, but that is an ensemble of 5 networks against one tree model. Within ±10 cycles: 59-63% of engines per seed (61% for the average), against 64% for XGBoost; within ±30: 95-97%.

**ONNX:** step 14 now exports the tuned seed-0 model (`models/turbofan/lstm_tuned_seed0.onnx`, 70-cycle input). It matches PyTorch to 0.00003 cycles; RMSE 13.661 on both.

**Still untuned:** network size beyond 32/64/128 units, weight decay, and the early-stopping rule. The sequence-length grid has a gap between 70 and 90.

---

## Decision 16: Tuning XGBoost (steps 18-19) - no real gain

**How:** same rules as the LSTM tuning: grouped CV on training engines (same folds and cut points as Decision 9), one change at a time, then combinations, and extending any winner at the edge of its range. The baseline row reproduces step 10's XGBoost exactly (CV RMSE 14.28 / MAE 10.30 / PHM08 430.95), which confirms the script measures the same thing.

**Round 1 (CV RMSE; baseline 14.28):** lambda 10 was best (14.01, better in 5/5 folds), then colsample 1.0 (14.15), depth 8 (14.20) and 1000 trees (14.21), each better in 4/5. Depth 3 and 4 and subsample 0.6 were worse. **Window 3** scored 14.22 (3/5 folds), no better than 5, so window 5 stays; that closes the "window grid stopped at 5" caveat from Decision 9.

**Round 2:** lambda 30 / 100 scored 14.12 / 14.32, so 10 is a real best. Depth 10 scored 14.82 (worse). Combinations did not stack: lambda 10 + colsample 1.0 + min_child 5 scored 14.07; adding depth 8 and 1000 trees scored 14.21-14.22. **Chosen: lambda 10 alone (CV 14.01).**

**Test (step 19, seed 42):** untuned 13.138 / 9.701 / 279.34; tuned 13.203 / 9.663 / 293.80; both 64/100 within ±10 and 95/100 within ±30. Over seeds 0-4 (Decision 17): 13.47 ± 0.20 vs 13.52 ± 0.10, PHM08 298 vs 298.

**Conclusion:** the CV gain (0.27 RMSE, about 2%) does not show up on the test set; the difference is within seed noise. XGBoost's defaults were already close to their best. This is a null result, and we report it as one. (Contrast with the LSTM, where tuning gained about 1.4 cycles on test.)

---

## Decision 17: Report seed-averaged test scores (step 20)

**Why:** every tree-model number so far came from one run with `random_state=42`. Random Forest and XGBoost use randomness, so the same model with a different seed scores differently. Checking this showed **seed 42 was a lucky seed for XGBoost**: 13.138 / PHM08 279 with seed 42, but 13.47 ± 0.20 / 298 ± 11 over seeds 0-4. Any "A beats B" claim smaller than the seed-to-seed spread is not a real difference.

**What:** step 20 retrains every tree model with seeds 0-4 and reports mean ± std on the 100 test engines. The tuned LSTM's 5 seeds come from step 17. Nothing here was used to pick a model or a setting.

**Headline results (100 test engines, mean ± std over 5 seeds):**

| Model | RMSE | MAE | PHM08 | Within ±10 | Within ±30 | Late |
|---|---|---|---|---|---|---|
| RF, window 20 | 14.27 ± 0.13 | 9.91 ± 0.12 | 365 ± 11 | 69 | 94 | 58 |
| RF, window 5 | 13.52 ± 0.15 | **9.60 ± 0.20** | 322 ± 5 | 66 | 94 | 59 |
| **XGBoost, window 5** | **13.47 ± 0.20** | 9.97 ± 0.10 | **298 ± 11** | 64 | 95 | 53 |
| XGBoost tuned (lambda 10) | 13.52 ± 0.10 | 10.10 ± 0.15 | 298 ± 6 | 63 | 95 | 53 |
| XGBoost asymmetric (weight 5) | 14.31 ± 0.23 | 10.28 ± 0.22 | 316 ± 20 | 64 | 94 | **45** |
| LSTM untuned (step 12) | 15.44 ± 0.61 | 11.11 ± 0.22 | 521 ± 120 | - | - | - |
| LSTM tuned | 14.07 ± 0.27 | 10.35 ± 0.20 | 358 ± 19 | 61 | 96 | 54 |

**What survives the seed check:**
- Window 5 beats window 20 for the RF: 13.52 vs 14.27, a gap of about 5× the spread. Real.
- RF and XGBoost (window 5) **tie on RMSE** (13.52 vs 13.47). XGBoost is clearly better on **PHM08** (298 vs 322), which means fewer costly late misses. RF is slightly better on MAE (9.60 vs 9.97).
- Asymmetric loss: fewest late predictions (45 vs 53), but PHM08 is still not better (316 ± 20 vs 298 ± 11). Single-seed numbers (333 vs 279) exaggerated the gap; the conclusion in Decision 11 stands in softened form.
- Tuned LSTM beats untuned LSTM clearly, but still trails the tree models with engineered features.

**Earlier single-seed numbers** in Decisions 6-15 are kept as the record of what was run. For slides, use this table.
