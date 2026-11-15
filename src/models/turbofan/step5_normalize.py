import os
import pandas as pd
from common import RAW_DIR, PROCESSED_DIR

train_data = pd.read_csv(os.path.join(PROCESSED_DIR, "train_FD001_with_RUL.csv"))
test_data = pd.read_csv(os.path.join(PROCESSED_DIR, "test_FD001_with_RUL.csv"))

for col in ["op_setting_1", "op_setting_2", "op_setting_3"]:
    print(col, "- unique values:", train_data[col].nunique(), "- std:", train_data[col].std())

train_data = train_data.drop(columns=["op_setting_3"])
test_data = test_data.drop(columns=["op_setting_3"])

feature_columns = []
for col in train_data.columns:
    if col not in ("unit_number", "cycle", "RUL"):
        feature_columns.append(col)

for col in feature_columns:
    mean_value = train_data[col].mean()
    std_value = train_data[col].std()
    train_data[col] = (train_data[col] - mean_value) / std_value
    test_data[col] = (test_data[col] - mean_value) / std_value

train_data.to_csv(os.path.join(PROCESSED_DIR, "train_FD001_normalized.csv"), index=False)
test_data.to_csv(os.path.join(PROCESSED_DIR, "test_FD001_normalized.csv"), index=False)

print()
print("Any NaN in train?", train_data.isna().any().any())
print("Any NaN in test?", test_data.isna().any().any())
print("Final columns:", list(train_data.columns))
