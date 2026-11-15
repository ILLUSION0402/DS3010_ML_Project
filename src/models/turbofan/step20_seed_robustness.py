# Step 20: how much do the test scores move if only the random seed changes?
#
# Steps 9 and 11 trained each tree model once, with random_state=42. Random
# Forest and XGBoost both use randomness (which rows and features each tree
# sees), so one run can be a little lucky or unlucky. Here every tree model is
# trained with 5 different seeds (0-4) and we report the mean and the spread.
# The LSTM already has this (steps 12 and 17); its tuned numbers are read in.
#
# Why it matters: XGBoost with seed 42 scored RMSE 13.138 / PHM08 279, but over
# seeds 0-4 it averages about 13.5 / 298. Seed 42 happened to be a good one.
# Claims like "model A beats model B" should only be made when the gap is
# bigger than this seed-to-seed spread.
#
# Nothing here is used to choose a model or a setting; it only measures noise.
# Runtime: about 5 minutes (mostly the Random Forests).

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from common import load_normalized, add_rolling_features, get_feature_columns, last_cycle_rows
from common import score_all, RESULTS_DIR

SEEDS = [0, 1, 2, 3, 4]


def asymmetric_objective(y_true, y_pred):
    # late errors cost 5x (the CV-chosen weight from step 11)
    d = y_pred - y_true
    weight = np.where(d > 0, 5.0, 1.0)
    return 2 * weight * d, 2 * weight


def make_model(name, seed, base_score):
    if name.startswith("RF"):
        return RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
    if name == "XGBoost tuned (lambda 10)":
        return XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8,
                            colsample_bytree=0.8, reg_lambda=10, random_state=seed, n_jobs=-1)
    if name == "XGBoost asymmetric":
        return XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8,
                            colsample_bytree=0.8, objective=asymmetric_objective,
                            base_score=base_score, random_state=seed, n_jobs=-1)
    return XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8,
                        colsample_bytree=0.8, random_state=seed, n_jobs=-1)


#          model name                    window
MODELS = [("RF window 20",               20),
          ("RF window 5",                5),
          ("XGBoost window 5",           5),
          ("XGBoost tuned (lambda 10)",  5),
          ("XGBoost asymmetric",         5)]

train_data, test_data = load_normalized()
rows = []
for name, window in MODELS:
    train_fe = add_rolling_features(train_data, window)
    last_rows = last_cycle_rows(add_rolling_features(test_data, window))
    feature_columns = get_feature_columns(train_fe)
    y_test = last_rows["RUL"].values
    for seed in SEEDS:
        model = make_model(name, seed, float(train_fe["RUL"].mean()))
        model.fit(train_fe[feature_columns], train_fe["RUL"])
        predictions = model.predict(last_rows[feature_columns])
        rmse, mae, score = score_all(y_test, predictions)
        errors = predictions - y_test
        rows.append([name, seed, rmse, mae, score,
                     int(np.sum(np.abs(errors) <= 10)), int(np.sum(np.abs(errors) <= 30)),
                     int(np.sum(errors > 0))])
        del model
    print("done:", name)

# tuned LSTM, 5 seeds, from step 17
lstm = pd.read_csv(os.path.join(RESULTS_DIR, "lstm_final_results.csv"))
lstm = lstm[lstm["setting"] != "baseline"]
lstm_predictions = pd.read_csv(os.path.join(RESULTS_DIR, "predictions_lstm_tuned.csv"))
for i in range(len(lstm)):
    seed = int(lstm["seed"].iloc[i])
    column = lstm["setting"].iloc[i] + " seed %d" % seed
    errors = lstm_predictions[column].values - lstm_predictions["true_RUL"].values
    rows.append(["LSTM tuned", seed, lstm["rmse"].iloc[i], lstm["mae"].iloc[i], lstm["phm08"].iloc[i],
                 int(np.sum(np.abs(errors) <= 10)), int(np.sum(np.abs(errors) <= 30)),
                 int(np.sum(errors > 0))])

table = pd.DataFrame(rows, columns=["model", "seed", "rmse", "mae", "phm08",
                                    "within_10", "within_30", "late_predictions"])
table.to_csv(os.path.join(RESULTS_DIR, "seed_robustness.csv"), index=False)

print()
print("=== 100 test engines, mean (std) over seeds 0-4 ===")
order = []
for name, window in MODELS:
    order.append(name)
order.append("LSTM tuned")
summary = table.groupby("model")[["rmse", "mae", "phm08", "within_10", "within_30", "late_predictions"]]
summary = summary.agg(["mean", "std"]).reindex(order)
print(summary.round(2).to_string())
