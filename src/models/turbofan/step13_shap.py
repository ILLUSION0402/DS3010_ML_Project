# Step 13: SHAP explanations, compared with the Gini importance from step 9.
#
# Gini importance (step 9) = how much a feature reduced error during TRAINING,
# summed over all splits. Two known weaknesses:
#   - it favours features with many distinct values
#   - when features are correlated (sensor_3_mean / _min / _max move together),
#     it can give most of the credit to whichever one the trees happened to pick
#
# SHAP answers a different question: for ONE prediction, how much did each
# feature push it up or down from the average prediction? The pushes add up
# exactly:  average prediction + sum of SHAP values = this engine's prediction.
# Averaging the size of the pushes over the 100 test engines gives a global
# ranking, and looking at one engine explains that single prediction.
#
# We explain the Random Forest (our headline model) on the 100 test engines,
# using the CV-chosen window from step 10. We also run XGBoost to see whether
# both models rely on the same signals.
#
# Outputs (in reports/turbofan/, charts in reports/turbofan/figures/):
#   shap_importance.csv              SHAP vs Gini, side by side, both models
#   turbofan_shap_summary.png        top 15 features, one dot per test engine
#   turbofan_shap_worst_engine.png   why the worst late miss was predicted late
#
# Runtime: a few minutes (Random Forest trees are deep, so exact SHAP is slow-ish).

import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from common import load_normalized, add_rolling_features, get_feature_columns, last_cycle_rows, RESULTS_DIR, MODELS_DIR, FIGURES_DIR

cv_table = pd.read_csv(os.path.join(RESULTS_DIR, "cv_results.csv"))
mean_rmse = cv_table.groupby(["model", "window"])["rmse"].mean().reset_index()
rf_window = int(mean_rmse[mean_rmse["model"] == "RandomForest"].sort_values("rmse").iloc[0]["window"])
xgb_window = int(mean_rmse[mean_rmse["model"] == "XGBoost"].sort_values("rmse").iloc[0]["window"])
print("Windows: RF", rf_window, "| XGBoost", xgb_window)

train_data, test_data = load_normalized()

# ================= Random Forest =================
train_fe = add_rolling_features(train_data, rf_window)
test_fe = add_rolling_features(test_data, rf_window)
feature_columns = get_feature_columns(train_fe)
last_rows = last_cycle_rows(test_fe)
x_test = last_rows[feature_columns]

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(train_fe[feature_columns], train_fe["RUL"])
rf_predictions = rf.predict(x_test)

start = time.time()
rf_explainer = shap.TreeExplainer(rf)
rf_shap = rf_explainer.shap_values(x_test)
print("RF SHAP took %.0f seconds" % (time.time() - start))

# additivity check: average + sum of pushes should equal the prediction
rebuilt = rf_explainer.expected_value + rf_shap.sum(axis=1)
print("Largest additivity error (cycles): %.6f" % np.max(np.abs(rebuilt - rf_predictions)))
print("Average prediction the pushes start from: %.2f cycles" % float(np.ravel(rf_explainer.expected_value)[0]))

rf_table = pd.DataFrame({
    "feature": feature_columns,
    "rf_mean_abs_shap": np.abs(rf_shap).mean(axis=0),
    "rf_gini": rf.feature_importances_,
})

# ================= XGBoost =================
train_fe_x = add_rolling_features(train_data, xgb_window)
test_fe_x = add_rolling_features(test_data, xgb_window)
feature_columns_x = get_feature_columns(train_fe_x)
x_test_x = last_cycle_rows(test_fe_x)[feature_columns_x]
xgb = XGBRegressor()
xgb.load_model(os.path.join(MODELS_DIR, "xgb_model.json"))
xgb_shap = shap.TreeExplainer(xgb).shap_values(x_test_x)
xgb_table = pd.DataFrame({
    "feature": feature_columns_x,
    "xgb_mean_abs_shap": np.abs(xgb_shap).mean(axis=0),
})

table = rf_table.merge(xgb_table, on="feature", how="outer")
table["rf_shap_share"] = table["rf_mean_abs_shap"] / table["rf_mean_abs_shap"].sum()
table["xgb_shap_share"] = table["xgb_mean_abs_shap"] / table["xgb_mean_abs_shap"].sum()
table["rank_rf_shap"] = table["rf_mean_abs_shap"].rank(ascending=False).astype(int)
table["rank_rf_gini"] = table["rf_gini"].rank(ascending=False).astype(int)
table["rank_xgb_shap"] = table["xgb_mean_abs_shap"].rank(ascending=False).astype(int)
table = table.sort_values("rank_rf_shap")
table.to_csv(os.path.join(RESULTS_DIR, "shap_importance.csv"), index=False)

print()
print("Top 15 by Random Forest SHAP (share = fraction of total |SHAP|):")
print(table[["feature", "rf_shap_share", "rank_rf_shap", "rf_gini", "rank_rf_gini",
             "xgb_shap_share", "rank_xgb_shap"]].head(15).round(3).to_string(index=False))

# group by the underlying sensor (sensor_3_mean, sensor_3_max ... -> sensor_3)
sensor_names = []
for name in table["feature"]:
    parts = name.split("_")
    sensor_names.append(parts[0] + "_" + parts[1])
table["sensor"] = sensor_names
by_sensor = table.groupby("sensor")[["rf_shap_share", "rf_gini", "xgb_shap_share"]].sum()
by_sensor = by_sensor.sort_values("rf_shap_share", ascending=False)
print()
print("Same thing, added up per original sensor (all 6 versions of each sensor together):")
print(by_sensor.head(8).round(3).to_string())

raw_names = get_feature_columns(train_data)
engineered_share = 0
for i in range(len(table)):
    if table["feature"].iloc[i] not in raw_names:
        engineered_share = engineered_share + table["rf_shap_share"].iloc[i]
print()
print("Share of RF SHAP on engineered features: %.1f%%" % (engineered_share * 100))

# ================= charts =================
plt.figure()
shap.summary_plot(rf_shap, x_test, max_display=15, show=False)
plt.title("Random Forest: what drives each test engine's prediction", loc="left", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "turbofan_shap_summary.png"), dpi=170)
plt.close("all")

errors = rf_predictions - last_rows["RUL"].values
worst = int(np.argmax(errors))
worst_engine = int(last_rows["unit_number"].values[worst])
print()
print("Worst late miss: test engine %d, true RUL %d, predicted %.1f (%.1f cycles late)"
      % (worst_engine, last_rows["RUL"].values[worst], rf_predictions[worst], errors[worst]))
explanation = shap.Explanation(values=rf_shap[worst],
                               base_values=float(np.ravel(rf_explainer.expected_value)[0]),
                               data=x_test.iloc[worst].values,
                               feature_names=feature_columns)
plt.figure()
shap.plots.waterfall(explanation, max_display=12, show=False)
plt.title("Test engine %d: predicted %.0f, true %d" % (worst_engine, rf_predictions[worst],
          last_rows["RUL"].values[worst]), loc="left", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "turbofan_shap_worst_engine.png"), dpi=170, bbox_inches="tight")
plt.close("all")
print("Charts saved.")
