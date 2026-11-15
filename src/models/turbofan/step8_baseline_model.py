# Step 8: baseline Random Forest on the raw normalised sensors.
#
# NOTE ON SCORING - this is the part that was wrong in the earlier version.
# The PHM08 score penalises LATE predictions (saying an engine has more life
# left than it does) far harder than EARLY ones. With d = predicted - true:
#     d < 0  (early) -> exp(-d / 13) - 1
#     d >= 0 (late)  -> exp( d / 10) - 1
# The smaller divisor on the late branch is what makes it steeper. At an error
# of 50 cycles: late scores 147, early scores 46.
#
# The paper's text prints "a1 = 10, a2 = 13", which read literally gives the
# OPPOSITE behaviour. But the same page says late predictions are penalised
# more heavily, and Figure 9 plots the curve reaching ~150 late vs ~45 early.
# The figure and the stated intent agree; the printed constants are a known
# typo. The version below is what the figure shows and what the literature uses.

import os
import pandas as pd
from common import PROCESSED_DIR
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

train_data = pd.read_csv(os.path.join(PROCESSED_DIR, "train_FD001_normalized.csv"))
test_data = pd.read_csv(os.path.join(PROCESSED_DIR, "test_FD001_normalized.csv"))

feature_columns = []
for col in train_data.columns:
    if col not in ("unit_number", "cycle", "RUL"):
        feature_columns.append(col)

x_train = train_data[feature_columns]
y_train = train_data["RUL"]

# Score only the LAST cycle of each test engine - 100 predictions, not 13,096.
# That is what RUL_FD001.txt gives ground truth for, and it is the real
# question: given everything seen so far, how long does this engine have?
last_rows = test_data.groupby("unit_number").tail(1)
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

print("=== BASELINE (no feature engineering) ===")
print("RMSE:", round(rmse, 3))
print("MAE:", round(mae, 3))
print("PHM08 score:", round(score, 2))
