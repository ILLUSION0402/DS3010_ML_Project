# Step 10: 5-fold cross-validation grouped by engine, on the TRAINING set only.
#
# Why: step 9 gives one number from one train/test split. That number could be
# lucky or unlucky. Cross-validation repeats the experiment 5 times on
# different engines and reports the average and the spread.
#
# Why "grouped by engine": rows from the same engine are almost identical to
# their neighbours. If engine 7's cycle 100 is in training and cycle 101 is in
# validation, the model has basically seen the answer. GroupKFold puts every
# row of an engine in the same fold, so validation engines are truly unseen.
#
# How a validation engine is scored: training engines run all the way to
# failure, but the real test engines are cut off early. So for each validation
# engine we pick 5 random cut points (cycle >= 31, the shortest test engine)
# and score the prediction at those rows. Because every rolling feature only
# looks BACKWARDS, the features at row t are exactly what you would have if the
# engine had been cut off at t. 20 engines x 5 cuts = 100 points per fold,
# the same count as the test set, so PHM08 sums are on a similar scale.
#
# What it compares:
#   - Random Forest vs XGBoost
#   - rolling window sizes 5, 10, 20, 30, 40  (step 9 used 20 by judgement)
# The test set is NOT touched here. The window is chosen from CV results only.
#
# One honest caveat: the step-5 scaling used mean/std from all 100 training
# engines, so validation engines contributed a little to that scaling. It is
# only a shift and a scale, so the effect is tiny, but it is not zero.
#
# Runtime: about 10-15 minutes on a Ryzen 7 (most of it is the Random Forests).

import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from common import load_normalized, add_rolling_features, get_feature_columns, score_all, RESULTS_DIR
from common import make_cut_points, make_engine_folds

WINDOWS = [5, 10, 20, 30, 40]
CUTS_PER_ENGINE = 5
MIN_CUT_CYCLE = 31

train_data, test_data = load_normalized()

# ---- pick the cut points once, so every model and window is scored on the same rows ----
cut_table = make_cut_points(train_data, CUTS_PER_ENGINE, MIN_CUT_CYCLE)
print("Cut points chosen:", len(cut_table))

# ---- split the 100 engines into 5 groups of 20 ----
fold_engine_lists = make_engine_folds(train_data)
print("Engines per validation fold:", [len(x) for x in fold_engine_lists])

results = []
for window in WINDOWS:
    data_fe = add_rolling_features(train_data, window)
    data_fe = data_fe.merge(cut_table, on=["unit_number", "cycle"], how="left")
    data_fe["is_cut"] = data_fe["is_cut"].notna()   # True where a cut point was merged in
    feature_columns = get_feature_columns(data_fe)
    feature_columns.remove("is_cut")

    for fold in range(5):
        val_engines = fold_engine_lists[fold]
        is_val = data_fe["unit_number"].isin(val_engines)
        fit_part = data_fe[~is_val]
        score_part = data_fe[is_val & data_fe["is_cut"]]

        x_fit = fit_part[feature_columns]
        y_fit = fit_part["RUL"]
        x_score = score_part[feature_columns]
        y_score = score_part["RUL"]

        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(x_fit, y_fit)
        rmse, mae, score = score_all(y_score, rf.predict(x_score))
        results.append(["RandomForest", window, fold + 1, rmse, mae, score])
        del rf

        xgb = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                           subsample=0.8, colsample_bytree=0.8,
                           random_state=42, n_jobs=-1)
        xgb.fit(x_fit, y_fit)
        rmse_x, mae_x, score_x = score_all(y_score, xgb.predict(x_score))
        results.append(["XGBoost", window, fold + 1, rmse_x, mae_x, score_x])

        print("window %2d  fold %d   RF rmse %6.2f phm %7.1f   XGB rmse %6.2f phm %7.1f"
              % (window, fold + 1, rmse, score, rmse_x, score_x))

table = pd.DataFrame(results, columns=["model", "window", "fold", "rmse", "mae", "phm08"])
table.to_csv(os.path.join(RESULTS_DIR, "cv_results.csv"), index=False)

print()
print("=== MEAN (STD) OVER 5 FOLDS, 100 cut points per fold ===")
summary = table.groupby(["model", "window"])[["rmse", "mae", "phm08"]].agg(["mean", "std"])
print(summary.round(2).to_string())

best = table.groupby(["model", "window"])["rmse"].mean().reset_index()
for model_name in ["RandomForest", "XGBoost"]:
    rows = best[best["model"] == model_name]
    winner = rows.sort_values("rmse").iloc[0]
    print("Best window for", model_name, "by mean CV RMSE:", int(winner["window"]))
