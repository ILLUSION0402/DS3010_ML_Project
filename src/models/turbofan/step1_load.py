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

rul_final = pd.read_csv(os.path.join(RAW_DIR, "RUL_FD001.txt"), header=None)

print("Train shape:", train_data.shape)
print("Test shape:", test_data.shape)
print("RUL file rows:", len(rul_final))
print("Train engines:", train_data.unit_number.nunique())
print("Test engines:", test_data.unit_number.nunique())

engine_lifespans = train_data.groupby("unit_number").cycle.max()
print("Shortest engine life:", engine_lifespans.min())
print("Longest engine life:", engine_lifespans.max())
