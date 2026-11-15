# Step 19: the tuned XGBoost on the 100 test engines, next to the untuned one.
#
# The tuned setting is read from reports/turbofan/xgb_tuning_cv.csv (step 18): the
# setting with the lowest mean CV RMSE. Nothing here is chosen by test score.
# Both models are trained on all 100 training engines and scored once.
#
# The untuned row should reproduce step 11's XGBoost window 5 exactly
# (RMSE 13.138, PHM08 279.34).

import os
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from common import load_normalized, add_rolling_features, get_feature_columns, last_cycle_rows
from common import print_scores, RESULTS_DIR, MODELS_DIR

tuning = pd.read_csv(os.path.join(RESULTS_DIR, "xgb_tuning_cv.csv"))
mean_rmse = tuning.groupby("setting")["rmse"].mean().sort_values()
tuned_name = mean_rmse.index[0]
print("Lowest mean CV RMSE:", tuned_name, "(%.2f)" % mean_rmse.iloc[0])

train_data, test_data = load_normalized()
result_rows = []
prediction_table = None

for name in ["baseline", tuned_name]:
    first = tuning[tuning["setting"] == name].iloc[0]
    window = int(first["window"])
    train_fe = add_rolling_features(train_data, window)
    test_fe = add_rolling_features(test_data, window)
    feature_columns = get_feature_columns(train_fe)
    last_rows = last_cycle_rows(test_fe)
    y_test = last_rows["RUL"].values

    model = XGBRegressor(n_estimators=int(first["trees"]), learning_rate=float(first["learning_rate"]),
                         max_depth=int(first["depth"]), min_child_weight=float(first["min_child"]),
                         subsample=float(first["subsample"]), colsample_bytree=float(first["colsample"]),
                         reg_lambda=float(first["reg_lambda"]), random_state=42, n_jobs=-1)
    model.fit(train_fe[feature_columns], train_fe["RUL"])
    predictions = model.predict(last_rows[feature_columns])
    if name == tuned_name:
        model.save_model(os.path.join(MODELS_DIR, "xgb_tuned_model.json"))

    rmse, mae, score = print_scores("XGBoost " + name, y_test, predictions)
    errors = predictions - y_test
    within_10 = int(np.sum(np.abs(errors) <= 10))
    within_30 = int(np.sum(np.abs(errors) <= 30))
    late = int(np.sum(errors > 0))
    print("   within 10 cycles: %d/100   within 30: %d/100   late predictions: %d" % (within_10, within_30, late))
    result_rows.append([name, rmse, mae, score, within_10, within_30, late])

    if prediction_table is None:
        prediction_table = pd.DataFrame({"engine_id": last_rows["unit_number"].values, "true_RUL": y_test})
    prediction_table["XGBoost " + name] = predictions

results = pd.DataFrame(result_rows, columns=["setting", "rmse", "mae", "phm08",
                                             "within_10", "within_30", "late_predictions"])
results.to_csv(os.path.join(RESULTS_DIR, "xgb_final_results.csv"), index=False)
prediction_table.to_csv(os.path.join(RESULTS_DIR, "predictions_xgb_tuned.csv"), index=False)
print()
print(results.round(3).to_string(index=False))
