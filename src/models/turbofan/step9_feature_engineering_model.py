# Step 9: same model, but with 20 cycles of rolling history per sensor.
#
# 16 feature columns become 96: mean, std, min and max over a 20-cycle window,
# plus each sensor's drift from that engine's own FIRST reading.
#
# Two details that matter:
#   - every rolling call is grouped by unit_number, so a window never pulls
#     values from a different engine
#   - the _delta feature cancels out the differing initial wear the readme
#     describes, so the model sees how far THIS engine has drifted
#
# Scoring is the same corrected PHM08 as step8 - see the note there.

import os
import pandas as pd
from common import PROCESSED_DIR, MODELS_DIR, RESULTS_DIR
import math
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

WINDOW = 20

train_data = pd.read_csv(os.path.join(PROCESSED_DIR, "train_FD001_normalized.csv"))
test_data = pd.read_csv(os.path.join(PROCESSED_DIR, "test_FD001_normalized.csv"))

sensor_columns = []
for col in train_data.columns:
    if col.startswith("sensor_") or col.startswith("op_setting_"):
        sensor_columns.append(col)

def add_rolling_features(data):
    data = data.sort_values(["unit_number", "cycle"]).copy()
    new_frames = [data]
    for col in sensor_columns:
        grouped = data.groupby("unit_number")[col]
        rolling_mean = grouped.rolling(WINDOW, min_periods=1).mean().reset_index(level=0, drop=True)
        rolling_std = grouped.rolling(WINDOW, min_periods=1).std().reset_index(level=0, drop=True)
        rolling_min = grouped.rolling(WINDOW, min_periods=1).min().reset_index(level=0, drop=True)
        rolling_max = grouped.rolling(WINDOW, min_periods=1).max().reset_index(level=0, drop=True)
        delta = data[col] - grouped.transform("first")
        frame = pd.DataFrame({
            col + "_mean": rolling_mean,
            col + "_std": rolling_std.fillna(0),
            col + "_min": rolling_min,
            col + "_max": rolling_max,
            col + "_delta": delta,
        })
        new_frames.append(frame)
    return pd.concat(new_frames, axis=1)

train_fe = add_rolling_features(train_data)
test_fe = add_rolling_features(test_data)

feature_columns = []
for col in train_fe.columns:
    if col not in ("unit_number", "cycle", "RUL"):
        feature_columns.append(col)

print("Feature count:", len(feature_columns))

x_train = train_fe[feature_columns]
y_train = train_fe["RUL"]

last_rows = test_fe.groupby("unit_number").tail(1)
x_test = last_rows[feature_columns]
y_test = last_rows["RUL"]

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(x_train, y_train)
predictions = model.predict(x_test)

rmse = mean_squared_error(y_test, predictions) ** 0.5
mae = mean_absolute_error(y_test, predictions)

score = 0
y_test_list = y_test.tolist()
for i in range(len(predictions)):
    d = predictions[i] - y_test_list[i]
    if d < 0:
        score = score + (math.exp(-d / 13) - 1)
    else:
        score = score + (math.exp(d / 10) - 1)

print()
print("=== WITH FEATURE ENGINEERING ===")
print("RMSE:", round(rmse, 3))
print("MAE:", round(mae, 3))
print("PHM08 score:", round(score, 2))

with open(os.path.join(MODELS_DIR, "rf_model_fe.pkl"), "wb") as f:
    pickle.dump(model, f)

out = pd.DataFrame({
    "engine_id": last_rows["unit_number"].values,
    "predicted_RUL": predictions,
    "true_RUL": y_test_list,
})
out.to_csv(os.path.join(RESULTS_DIR, "predictions_rf_fe.csv"), index=False)

importances = pd.Series(model.feature_importances_, index=feature_columns).sort_values(ascending=False)
print()
print("Top 10 features:")
print(importances.head(10))
importances.to_csv(os.path.join(RESULTS_DIR, "feature_importances.csv"))
