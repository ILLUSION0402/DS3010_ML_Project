import os
import pandas as pd
import numpy as np
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

PROCESSED_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed'))
feature_cols = ["v_mean", "v_min", "v_max", "c_mean", "c_min", "c_max", "t_mean", "t_min", "t_max", "capacity"]

train_data = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "train_battery.csv"))
test_data = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "test_battery.csv"))

feature_cols = [col for col in feature_cols if train_data[col].nunique() > 5]

train_data["target"] = train_data["target"].clip(upper=125)
test_data["target"] = test_data["target"].clip(upper=125)

for col in feature_cols:
    mean_val, std_val = train_data[col].mean(), train_data[col].std() or 1e-6
    train_data[col] = (train_data[col] - mean_val) / std_val
    test_data[col] = (test_data[col] - mean_val) / std_val

x_train, y_train = train_data[feature_cols], train_data["target"]
x_test, y_test = test_data[feature_cols], test_data["target"]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(x_train, y_train)
preds = model.predict(x_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))
mae = mean_absolute_error(y_test, preds)
score = sum(math.exp((preds[i] - y_test.iloc[i]) / (13 if (preds[i] - y_test.iloc[i]) >= 0 else -10)) - 1 for i in range(len(preds)))

print(f"RMSE: {rmse:.4f}\nMAE: {mae:.4f}\nPHM-style score: {score:.4f}")
