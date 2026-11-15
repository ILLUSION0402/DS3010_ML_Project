# Step 18: tune XGBoost with grouped cross-validation on the TRAINING engines.
#
# Same rules as the LSTM tuning (step 16):
#   - the test set is not touched
#   - same 5 engine folds and the same 100 cut points per fold as step 10
#   - change ONE setting at a time from the step-10/11 XGBoost, then combine
#     the changes that helped, and extend any winner sitting at the edge of
#     its range
#
# The "baseline" row must reproduce step 10's XGBoost at window 5
# (CV RMSE 14.28). That check proves this script measures the same thing.
#
# What each setting means, in plain words:
#   depth         how many questions each tree may ask in a row (deeper = more complex trees)
#   lr, trees     learning rate and number of trees; a smaller rate needs more trees
#   min_child     a leaf must hold at least this much data (bigger = smoother, less overfitting)
#   subsample     fraction of rows each tree sees
#   colsample     fraction of features each tree sees
#   lambda        penalty on large leaf values (bigger = more cautious)
#
# Results are saved after every setting; settings already in
# reports/turbofan/xgb_tuning_cv.csv are skipped when the script is re-run.

import os
import time
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from common import load_normalized, add_rolling_features, get_feature_columns
from common import make_cut_points, make_engine_folds, score_all, RESULTS_DIR

#            name                window depth lr     trees min_child subsample colsample lambda
SETTINGS = [
    ("baseline",                 5,     6,    0.05,  500,  1,        0.8,      0.8,      1),
    ("depth 3",                  5,     3,    0.05,  500,  1,        0.8,      0.8,      1),
    ("depth 4",                  5,     4,    0.05,  500,  1,        0.8,      0.8,      1),
    ("depth 8",                  5,     8,    0.05,  500,  1,        0.8,      0.8,      1),
    ("lr 0.02 x1250",            5,     6,    0.02,  1250, 1,        0.8,      0.8,      1),
    ("lr 0.1 x250",              5,     6,    0.1,   250,  1,        0.8,      0.8,      1),
    ("trees 250",                5,     6,    0.05,  250,  1,        0.8,      0.8,      1),
    ("trees 1000",               5,     6,    0.05,  1000, 1,        0.8,      0.8,      1),
    ("min_child 5",              5,     6,    0.05,  500,  5,        0.8,      0.8,      1),
    ("min_child 20",             5,     6,    0.05,  500,  20,       0.8,      0.8,      1),
    ("subsample 0.6",            5,     6,    0.05,  500,  1,        0.6,      0.8,      1),
    ("subsample 1.0",            5,     6,    0.05,  500,  1,        1.0,      0.8,      1),
    ("colsample 0.5",            5,     6,    0.05,  500,  1,        0.8,      0.5,      1),
    ("colsample 1.0",            5,     6,    0.05,  500,  1,        0.8,      1.0,      1),
    ("lambda 10",                5,     6,    0.05,  500,  1,        0.8,      0.8,      10),
    ("window 3",                 3,     6,    0.05,  500,  1,        0.8,      0.8,      1),
    # ---- round 2: lambda 10 helped most (5/5 folds), then colsample 1.0, depth 8 and
    # 1000 trees (4/5 each). lambda 10 and depth 8 were at the edge of their ranges,
    # so extend them, and build the combinations up one piece at a time.
    ("lambda 30",                5,     6,    0.05,  500,  1,        0.8,      0.8,      30),
    ("lambda 100",               5,     6,    0.05,  500,  1,        0.8,      0.8,      100),
    ("depth 10",                 5,     10,   0.05,  500,  1,        0.8,      0.8,      1),
    ("l10+col1",                 5,     6,    0.05,  500,  1,        0.8,      1.0,      10),
    ("l10+col1+d8",              5,     8,    0.05,  500,  1,        0.8,      1.0,      10),
    ("l10+col1+d8+t1000",        5,     8,    0.05,  1000, 1,        0.8,      1.0,      10),
    ("l10+col1+mc5",             5,     6,    0.05,  500,  5,        0.8,      1.0,      10),
]

output_path = os.path.join(RESULTS_DIR, "xgb_tuning_cv.csv")

train_data, test_data = load_normalized()
cut_table = make_cut_points(train_data)
fold_engine_lists = make_engine_folds(train_data)

# rolling features for each window, built once and reused
features_by_window = {}


def get_window_data(window):
    if window not in features_by_window:
        data_fe = add_rolling_features(train_data, window)
        data_fe = data_fe.merge(cut_table, on=["unit_number", "cycle"], how="left")
        data_fe["is_cut"] = data_fe["is_cut"].fillna(False).astype(bool)
        features_by_window[window] = data_fe
    return features_by_window[window]


done_names = []
rows = []
if os.path.exists(output_path):
    previous = pd.read_csv(output_path)
    done_names = list(previous["setting"].unique())
    rows = previous.values.tolist()

for setting in SETTINGS:
    name, window, depth, learning_rate, trees, min_child, subsample, colsample, reg_lambda = setting
    if name in done_names:
        continue
    start_time = time.time()
    data_fe = get_window_data(window)
    feature_columns = get_feature_columns(data_fe)
    feature_columns.remove("is_cut")

    for fold in range(5):
        is_val = data_fe["unit_number"].isin(fold_engine_lists[fold])
        fit_part = data_fe[~is_val]
        score_part = data_fe[is_val & data_fe["is_cut"]]
        model = XGBRegressor(n_estimators=trees, learning_rate=learning_rate, max_depth=depth,
                             min_child_weight=min_child, subsample=subsample,
                             colsample_bytree=colsample, reg_lambda=reg_lambda,
                             random_state=42, n_jobs=-1)
        model.fit(fit_part[feature_columns], fit_part["RUL"])
        rmse, mae, score = score_all(score_part["RUL"], model.predict(score_part[feature_columns]))
        rows.append([name, window, depth, learning_rate, trees, min_child, subsample, colsample,
                     reg_lambda, fold + 1, rmse, mae, score])

    table = pd.DataFrame(rows, columns=["setting", "window", "depth", "learning_rate", "trees",
                                        "min_child", "subsample", "colsample", "reg_lambda",
                                        "fold", "rmse", "mae", "phm08"])
    table.to_csv(output_path, index=False)
    this = table[table["setting"] == name]
    print("%-28s RMSE %6.2f +- %4.2f   MAE %6.2f   PHM08 %7.1f   (%.0f s)"
          % (name, this["rmse"].mean(), this["rmse"].std(), this["mae"].mean(),
             this["phm08"].mean(), time.time() - start_time))

print()
print("=== ALL SETTINGS, sorted by mean CV RMSE (100 cut points x 5 folds) ===")
table = pd.read_csv(output_path)
summary = table.groupby("setting")[["rmse", "mae", "phm08"]].mean()
summary["rmse_std"] = table.groupby("setting")["rmse"].std()
baseline_rmse = table[table["setting"] == "baseline"].sort_values("fold")["rmse"].values
folds_better = []
for name in summary.index:
    this_rmse = table[table["setting"] == name].sort_values("fold")["rmse"].values
    folds_better.append(int(np.sum(this_rmse < baseline_rmse)))
summary["folds_better_than_baseline"] = folds_better
print(summary.sort_values("rmse").round(2).to_string())
