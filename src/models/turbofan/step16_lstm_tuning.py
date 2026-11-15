# Step 16: tune the LSTM with grouped cross-validation on the TRAINING engines.
#
# The test set is not touched here. Every setting is judged on the same
# 5 engine folds and the same 100 cut points per fold as step 10, so the
# numbers are comparable with the tree models' CV numbers.
#
# Inside each fold (80 training engines, 20 validation engines):
#   - 16 of the 80 are held back to decide when to stop training (early stopping)
#   - the model trains on the other 64
#   - it is scored at the 100 cut points of the fold's 20 validation engines
# The same seed is used for the same fold in every setting, so two settings
# differ only in the setting itself (a "paired" comparison).
#
# Search plan - change ONE thing at a time from step 12's setup, so each
# result has a clear explanation:
#   baseline : 30-cycle sequences, 64 units, 2 layers, dropout 0.2, lr 0.001, batch 256
#   then each row below differs from the baseline in one place only.
# After this first round, the changes that helped are combined and added at the
# bottom of the list (see the "round 2" rows), and the script is run again. It
# skips settings that are already in reports/turbofan/lstm_tuning_cv.csv.
#
# Runtime: roughly 30-45 minutes for round 1 on a Ryzen 7 CPU.

import os
import time
import numpy as np
import pandas as pd
from common import load_normalized, get_sensor_columns, make_cut_points, make_engine_folds
from common import score_all, RESULTS_DIR
from lstm_model import make_train_sequences, make_cut_sequences, fit_early_stopping, predict_cycles

#            name            seq  hidden layers dropout  lr     batch
SETTINGS = [
    ("baseline",             30,  64,    2,     0.2,     0.001, 256),
    ("seq 20",               20,  64,    2,     0.2,     0.001, 256),
    ("seq 40",               40,  64,    2,     0.2,     0.001, 256),
    ("seq 50",               50,  64,    2,     0.2,     0.001, 256),
    ("hidden 32",            30,  32,    2,     0.2,     0.001, 256),
    ("hidden 128",           30,  128,   2,     0.2,     0.001, 256),
    ("1 layer",              30,  64,    1,     0.2,     0.001, 256),
    ("dropout 0",            30,  64,    2,     0.0,     0.001, 256),
    ("dropout 0.4",          30,  64,    2,     0.4,     0.001, 256),
    ("lr 0.0005",            30,  64,    2,     0.2,     0.0005, 256),
    ("lr 0.003",             30,  64,    2,     0.2,     0.003, 256),
    ("batch 64",             30,  64,    2,     0.2,     0.001, 64),
    # ---- round 2: round 1's three helpful changes were seq 50, dropout 0 and
    # lr 0.0005. Combine them, and try seq 70 because 50 was the edge of the grid.
    ("seq 70",               70,  64,    2,     0.2,     0.001, 256),
    ("seq50+drop0",          50,  64,    2,     0.0,     0.001, 256),
    ("seq50+lr0.0005",       50,  64,    2,     0.2,     0.0005, 256),
    ("seq50+drop0+lr0.0005", 50,  64,    2,     0.0,     0.0005, 256),
    ("seq70+drop0+lr0.0005", 70,  64,    2,     0.0,     0.0005, 256),
    # ---- round 3: seq 70 was again the edge of the grid and still improving.
    ("seq90+drop0+lr0.0005", 90,  64,    2,     0.0,     0.0005, 256),
    ("seq110+drop0+lr0.0005", 110, 64,   2,     0.0,     0.0005, 256),
]

output_path = os.path.join(RESULTS_DIR, "lstm_tuning_cv.csv")

train_data, test_data = load_normalized()
feature_columns = get_sensor_columns(train_data)
cut_table = make_cut_points(train_data)
fold_engine_lists = make_engine_folds(train_data)

done_names = []
if os.path.exists(output_path):
    previous = pd.read_csv(output_path)
    done_names = list(previous["setting"].unique())
    rows = previous.values.tolist()
else:
    rows = []

for setting in SETTINGS:
    name, seq_len, hidden, layers, dropout, learning_rate, batch_size = setting
    if name in done_names:
        print("skipping", name, "(already done)")
        continue
    start_time = time.time()
    for fold in range(5):
        val_engines = fold_engine_lists[fold]
        fit_engines = []
        for engine in sorted(train_data["unit_number"].unique()):
            if engine not in val_engines:
                fit_engines.append(engine)
        stop_engines = list(np.random.RandomState(fold).choice(fit_engines, size=16, replace=False))

        fit_part = train_data[train_data["unit_number"].isin(fit_engines)
                              & ~train_data["unit_number"].isin(stop_engines)]
        stop_part = train_data[train_data["unit_number"].isin(stop_engines)]
        val_part = train_data[train_data["unit_number"].isin(val_engines)]
        val_cuts = cut_table[cut_table["unit_number"].isin(val_engines)]

        x_fit, y_fit = make_train_sequences(fit_part, feature_columns, seq_len)
        x_stop, y_stop = make_train_sequences(stop_part, feature_columns, seq_len)
        x_val, y_val = make_cut_sequences(val_part, feature_columns, val_cuts, seq_len)

        model, best_epoch = fit_early_stopping(x_fit, y_fit, x_stop, y_stop, len(feature_columns),
                                               fold, hidden, layers, dropout, learning_rate, batch_size)
        rmse, mae, score = score_all(y_val, predict_cycles(model, x_val))
        rows.append([name, seq_len, hidden, layers, dropout, learning_rate, batch_size,
                     fold + 1, best_epoch, rmse, mae, score])

    table = pd.DataFrame(rows, columns=["setting", "seq_len", "hidden", "layers", "dropout",
                                        "learning_rate", "batch_size", "fold", "best_epoch",
                                        "rmse", "mae", "phm08"])
    table.to_csv(output_path, index=False)    # save after every setting
    this = table[table["setting"] == name]
    print("%-22s RMSE %6.2f +- %4.2f   MAE %6.2f   PHM08 %7.1f   epochs %4.1f   (%.0f s)"
          % (name, this["rmse"].mean(), this["rmse"].std(), this["mae"].mean(),
             this["phm08"].mean(), this["best_epoch"].mean(), time.time() - start_time))

print()
print("=== ALL SETTINGS, sorted by mean CV RMSE (100 cut points x 5 folds) ===")
table = pd.read_csv(output_path)
summary = table.groupby("setting")[["rmse", "mae", "phm08", "best_epoch"]].mean()
summary["rmse_std"] = table.groupby("setting")["rmse"].std()
baseline_rmse = table[table["setting"] == "baseline"].sort_values("fold")["rmse"].values
folds_better = []
for name in summary.index:
    this_rmse = table[table["setting"] == name].sort_values("fold")["rmse"].values
    folds_better.append(int(np.sum(this_rmse < baseline_rmse)))
summary["folds_better_than_baseline"] = folds_better
print(summary.sort_values("rmse").round(2).to_string())
