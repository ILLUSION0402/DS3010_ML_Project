# Step 11: XGBoost as a second model, and an asymmetric loss that punishes
# late predictions more (the same idea as PHM08).
#
# Run step 10 first: the window size for each model is read from
# reports/turbofan/cv_results.csv (lowest mean CV RMSE). The test set is only used
# here, at the very end, once every choice has been made on training data.
#
# PART A - plain models on the 100 test engines:
#   - Random Forest, window 20       (reproduces step 9 exactly)
#   - Random Forest, CV-chosen window
#   - XGBoost, CV-chosen window
#
# PART B - asymmetric loss.
#   Normal squared error treats "10 cycles too late" and "10 cycles too early"
#   the same. Maintenance does not: late means the engine can fail in service.
#   So we change the loss to:
#       d = predicted - true
#       loss = LATE_WEIGHT * d^2   if d > 0  (late)
#       loss =               d^2   if d <= 0 (early)
#   XGBoost needs the first and second derivative of the loss (gradient and
#   hessian): grad = 2 * w * d, hess = 2 * w, where w is LATE_WEIGHT or 1.
#
#   Why not train on PHM08 directly? Its exponential penalty blows up while the
#   model is still bad (an error of 100 cycles gives exp(10) = 22,000), and the
#   boosting becomes unstable. Weighted squared error keeps the same direction
#   of push (late costs more) while staying well behaved.
#
#   LATE_WEIGHT is chosen by the same grouped CV as step 10, picking the value
#   with the lowest mean CV PHM08 - again, without looking at the test set.
#
# Runtime: about 5 minutes.

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from common import load_normalized, add_rolling_features, get_feature_columns, last_cycle_rows
from common import make_cut_points, make_engine_folds, score_all, print_scores, RESULTS_DIR, MODELS_DIR

# The first run used [1, 1.5, 2, 3, 5]; CV picked 5, the edge of the grid, with
# PHM08 still falling - so the grid was extended to make sure it is a real minimum.
LATE_WEIGHTS = [1, 1.5, 2, 3, 5, 8, 12]


def make_xgb(objective="reg:squarederror", base_score=None):
    return XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                        subsample=0.8, colsample_bytree=0.8,
                        objective=objective, base_score=base_score,
                        random_state=42, n_jobs=-1)


def make_asymmetric_objective(late_weight):
    # Returns a function XGBoost calls on every boosting round.
    def objective(y_true, y_pred):
        d = y_pred - y_true
        weight = np.where(d > 0, late_weight, 1.0)
        grad = 2 * weight * d
        hess = 2 * weight
        return grad, hess
    return objective


def late_summary(true_values, predicted_values):
    d = np.array(predicted_values) - np.array(true_values)
    late_count = int(np.sum(d > 0))
    big_late = int(np.sum(d >= 30))
    worst_late = float(np.max(d))
    return late_count, big_late, worst_late


# ---- window sizes chosen by step 10 ----
cv_table = pd.read_csv(os.path.join(RESULTS_DIR, "cv_results.csv"))
mean_rmse = cv_table.groupby(["model", "window"])["rmse"].mean().reset_index()
rf_rows = mean_rmse[mean_rmse["model"] == "RandomForest"].sort_values("rmse")
xgb_rows = mean_rmse[mean_rmse["model"] == "XGBoost"].sort_values("rmse")
rf_window = int(rf_rows.iloc[0]["window"])
xgb_window = int(xgb_rows.iloc[0]["window"])
print("CV-chosen window: Random Forest", rf_window, "| XGBoost", xgb_window)

train_data, test_data = load_normalized()

# =================== PART A ===================
print()
print("=== PART A: 100 test engines ===")
results = []
saved = {}

for model_name, window in [("RF window 20", 20), ("RF window %d" % rf_window, rf_window),
                           ("XGBoost window %d" % xgb_window, xgb_window)]:
    if model_name in saved:
        continue
    train_fe = add_rolling_features(train_data, window)
    test_fe = add_rolling_features(test_data, window)
    feature_columns = get_feature_columns(train_fe)
    last_rows = last_cycle_rows(test_fe)

    if model_name.startswith("RF"):
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        model = make_xgb()
    model.fit(train_fe[feature_columns], train_fe["RUL"])
    predictions = model.predict(last_rows[feature_columns])
    if model_name.startswith("XGBoost"):
        model.save_model(os.path.join(MODELS_DIR, "xgb_model.json"))

    y_test = last_rows["RUL"].values
    rmse, mae, score = print_scores(model_name, y_test, predictions)
    late_count, big_late, worst_late = late_summary(y_test, predictions)
    results.append([model_name, rmse, mae, score, late_count, big_late, worst_late])
    saved[model_name] = predictions
    engine_ids = last_rows["unit_number"].values
    true_rul = y_test
    del model

# =================== PART B ===================
print()
print("=== PART B: choosing LATE_WEIGHT by grouped CV (window %d) ===" % xgb_window)
cut_table = make_cut_points(train_data)
fold_engine_lists = make_engine_folds(train_data)
train_fe = add_rolling_features(train_data, xgb_window)
train_fe = train_fe.merge(cut_table, on=["unit_number", "cycle"], how="left")
train_fe["is_cut"] = train_fe["is_cut"].notna()   # True where a cut point was merged in
feature_columns = get_feature_columns(train_fe)
feature_columns.remove("is_cut")

weight_rows = []
for late_weight in LATE_WEIGHTS:
    for fold in range(5):
        is_val = train_fe["unit_number"].isin(fold_engine_lists[fold])
        fit_part = train_fe[~is_val]
        score_part = train_fe[is_val & train_fe["is_cut"]]
        model = make_xgb(make_asymmetric_objective(late_weight), float(fit_part["RUL"].mean()))
        model.fit(fit_part[feature_columns], fit_part["RUL"])
        rmse, mae, score = score_all(score_part["RUL"], model.predict(score_part[feature_columns]))
        weight_rows.append([late_weight, fold + 1, rmse, mae, score])
weight_table = pd.DataFrame(weight_rows, columns=["late_weight", "fold", "rmse", "mae", "phm08"])
weight_table.to_csv(os.path.join(RESULTS_DIR, "cv_late_weight.csv"), index=False)
weight_summary = weight_table.groupby("late_weight")[["rmse", "mae", "phm08"]].agg(["mean", "std"])
print(weight_summary.round(2).to_string())
best_weight = float(weight_table.groupby("late_weight")["phm08"].mean().idxmin())
print("Chosen LATE_WEIGHT (lowest mean CV PHM08):", best_weight)

# train on all 100 training engines with the chosen weight, then score on test
test_fe = add_rolling_features(test_data, xgb_window)
last_rows = last_cycle_rows(test_fe)
train_all = train_fe.drop(columns=["is_cut"])
model = make_xgb(make_asymmetric_objective(best_weight), float(train_all["RUL"].mean()))
model.fit(train_all[feature_columns], train_all["RUL"])
predictions = model.predict(last_rows[feature_columns])
model.save_model(os.path.join(MODELS_DIR, "xgb_asymmetric_model.json"))

print()
name = "XGBoost asym w=%g" % best_weight
rmse, mae, score = print_scores(name, last_rows["RUL"].values, predictions)
late_count, big_late, worst_late = late_summary(last_rows["RUL"].values, predictions)
results.append([name, rmse, mae, score, late_count, big_late, worst_late])
saved[name] = predictions

# =================== save everything ===================
print()
summary = pd.DataFrame(results, columns=["model", "rmse", "mae", "phm08",
                                         "late_predictions", "late_by_30_plus", "worst_late_miss"])
print(summary.round(3).to_string(index=False))
summary.to_csv(os.path.join(RESULTS_DIR, "test_results_tree_models.csv"), index=False)

out = pd.DataFrame({"engine_id": engine_ids, "true_RUL": true_rul})
for name in saved:
    out[name] = saved[name]
out.to_csv(os.path.join(RESULTS_DIR, "predictions_tree_models.csv"), index=False)
print("Saved test_results_tree_models.csv and predictions_tree_models.csv")
