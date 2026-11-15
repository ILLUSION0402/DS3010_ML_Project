import os
import pandas as pd
from common import RAW_DIR, PROCESSED_DIR

RUL_CAP = 125

train_data = pd.read_csv(os.path.join(PROCESSED_DIR, "train_FD001_sensors_filtered.csv"))
test_data = pd.read_csv(os.path.join(PROCESSED_DIR, "test_FD001_sensors_filtered.csv"))
rul_final = pd.read_csv(os.path.join(RAW_DIR, "RUL_FD001.txt"), header=None)[0].tolist()

max_cycle_train = train_data.groupby("unit_number")["cycle"].transform("max")
train_data["RUL"] = max_cycle_train - train_data["cycle"]
train_data["RUL"] = train_data["RUL"].clip(upper=RUL_CAP)

max_cycle_test = test_data.groupby("unit_number")["cycle"].transform("max")
rul_lookup = {}
for i in range(len(rul_final)):
    rul_lookup[i + 1] = rul_final[i]
rul_at_end = test_data["unit_number"].map(rul_lookup)
test_data["RUL"] = rul_at_end + (max_cycle_test - test_data["cycle"])
test_data["RUL"] = test_data["RUL"].clip(upper=RUL_CAP)

train_data.to_csv(os.path.join(PROCESSED_DIR, "train_FD001_with_RUL.csv"), index=False)
test_data.to_csv(os.path.join(PROCESSED_DIR, "test_FD001_with_RUL.csv"), index=False)

print("Train RUL range:", train_data.RUL.min(), "-", train_data.RUL.max())
print("Test RUL range:", test_data.RUL.min(), "-", test_data.RUL.max())
print()
print("Engine 1, last 5 rows:")
print(train_data[train_data.unit_number == 1][["unit_number", "cycle", "RUL"]].tail())
