import os
import pandas as pd
from common import RAW_DIR, PROCESSED_DIR

columns = ["unit_number", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
for i in range(1, 22):
    columns.append("sensor_" + str(i))

train_data = pd.read_csv(os.path.join(RAW_DIR, "train_FD001.txt"), sep=r"\s+", header=None)
train_data = train_data.iloc[:, :len(columns)]
train_data.columns = columns

test_data = pd.read_csv(os.path.join(RAW_DIR, "test_FD001.txt"), sep=r"\s+", header=None)
test_data = test_data.iloc[:, :len(columns)]
test_data.columns = columns

print("Sensor | unique values | std")
sensor_columns = []
for col in columns:
    if col.startswith("sensor_"):
        sensor_columns.append(col)

for col in sensor_columns:
    print(col, "|", train_data[col].nunique(), "|", round(train_data[col].std(), 6))

flat_sensors = []
for col in sensor_columns:
    if train_data[col].nunique() <= 5:
        flat_sensors.append(col)

print()
print("Dropping:", flat_sensors)

train_data = train_data.drop(columns=flat_sensors)
test_data = test_data.drop(columns=flat_sensors)

train_data.to_csv(os.path.join(PROCESSED_DIR, "train_FD001_sensors_filtered.csv"), index=False)
test_data.to_csv(os.path.join(PROCESSED_DIR, "test_FD001_sensors_filtered.csv"), index=False)
print("Train rows:", len(train_data), "Test rows:", len(test_data))
