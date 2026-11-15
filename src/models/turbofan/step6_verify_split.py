import os
import pandas as pd
from common import PROCESSED_DIR

train_data = pd.read_csv(os.path.join(PROCESSED_DIR, "train_FD001_normalized.csv"))
test_data = pd.read_csv(os.path.join(PROCESSED_DIR, "test_FD001_normalized.csv"))

print("Train engines:", train_data["unit_number"].nunique())
print("Test engines:", test_data["unit_number"].nunique())
print("Train rows:", len(train_data))
print("Test rows:", len(test_data))

train_min_rul = train_data.groupby("unit_number")["RUL"].min()
count_reaching_zero = 0
for value in train_min_rul:
    if value == 0:
        count_reaching_zero = count_reaching_zero + 1
print("Train engines reaching RUL 0:", count_reaching_zero, "out of", len(train_min_rul))

test_min_rul = test_data.groupby("unit_number")["RUL"].min()
count_reaching_zero = 0
for value in test_min_rul:
    if value == 0:
        count_reaching_zero = count_reaching_zero + 1
print("Test engines reaching RUL 0:", count_reaching_zero, "out of", len(test_min_rul))
print("Lowest RUL in test:", test_min_rul.min())
