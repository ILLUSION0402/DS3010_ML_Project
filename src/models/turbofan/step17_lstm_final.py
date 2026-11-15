# Step 17: the tuned LSTM on the 100 test engines, next to the untuned one.
#
# The tuned setting is read from reports/turbofan/lstm_tuning_cv.csv (step 16): the
# setting with the lowest mean CV RMSE. Nothing here is chosen by test score.
#
# For a fair before/after, BOTH settings are trained here with exactly the same
# code and the same 5 seeds:
#   1. hold out 20 training engines, train on 80, early-stop on the 20
#   2. retrain a fresh model on all 100 training engines for that many epochs
#   3. predict the 100 test engines
# (This is the step-12 recipe. The step-12 numbers themselves came from slightly
# different random-number handling, so expect the "baseline" here to be close
# to step 12's 15.44 but not identical.)
#
# Runtime: roughly 10-20 minutes.

import os
import time
import numpy as np
import pandas as pd
import torch
from common import load_normalized, get_sensor_columns, print_scores, RESULTS_DIR, MODELS_DIR
from lstm_model import make_train_sequences, make_last_sequences, fit_early_stopping
from lstm_model import fit_fixed_epochs, predict_cycles

SEEDS = [0, 1, 2, 3, 4]

tuning = pd.read_csv(os.path.join(RESULTS_DIR, "lstm_tuning_cv.csv"))
mean_rmse = tuning.groupby("setting")["rmse"].mean().sort_values()
tuned_name = mean_rmse.index[0]
print("Lowest mean CV RMSE:", tuned_name, "(%.2f)" % mean_rmse.iloc[0])

settings_to_run = ["baseline"]
if tuned_name != "baseline":
    settings_to_run.append(tuned_name)

train_data, test_data = load_normalized()
feature_columns = get_sensor_columns(train_data)
all_engines = sorted(train_data["unit_number"].unique())

result_rows = []
prediction_table = None

for name in settings_to_run:
    first = tuning[tuning["setting"] == name].iloc[0]
    seq_len = int(first["seq_len"])
    hidden = int(first["hidden"])
    layers = int(first["layers"])
    dropout = float(first["dropout"])
    learning_rate = float(first["learning_rate"])
    batch_size = int(first["batch_size"])
    print()
    print("=== %s: seq %d, hidden %d, layers %d, dropout %.1f, lr %g, batch %d ==="
          % (name, seq_len, hidden, layers, dropout, learning_rate, batch_size))

    x_test, y_test, test_engines = make_last_sequences(test_data, feature_columns, seq_len)
    x_all, y_all = make_train_sequences(train_data, feature_columns, seq_len)
    if prediction_table is None:
        prediction_table = pd.DataFrame({"engine_id": test_engines, "true_RUL": y_test})

    seed_predictions = []
    for seed in SEEDS:
        start_time = time.time()
        stop_engines = np.random.RandomState(seed).choice(all_engines, size=20, replace=False)
        fit_part = train_data[~train_data["unit_number"].isin(stop_engines)]
        stop_part = train_data[train_data["unit_number"].isin(stop_engines)]
        x_fit, y_fit = make_train_sequences(fit_part, feature_columns, seq_len)
        x_stop, y_stop = make_train_sequences(stop_part, feature_columns, seq_len)

        model, best_epoch = fit_early_stopping(x_fit, y_fit, x_stop, y_stop, len(feature_columns), seed,
                                               hidden, layers, dropout, learning_rate, batch_size)
        model = fit_fixed_epochs(x_all, y_all, len(feature_columns), seed, best_epoch,
                                 hidden, layers, dropout, learning_rate, batch_size)
        predictions = predict_cycles(model, x_test)
        rmse, mae, score = print_scores("%s seed %d (%d ep)" % (name, seed, best_epoch), y_test, predictions)
        result_rows.append([name, seed, best_epoch, rmse, mae, score])
        seed_predictions.append(predictions)
        prediction_table[name + " seed %d" % seed] = predictions
        if name == tuned_name and seed == 0:
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, "lstm_tuned_seed0.pt"))
        print("   (%.0f seconds)" % (time.time() - start_time))

    average = np.mean(np.array(seed_predictions), axis=0)
    prediction_table[name + " average"] = average
    print_scores(name + " average of 5", y_test, average)

results = pd.DataFrame(result_rows, columns=["setting", "seed", "epochs", "rmse", "mae", "phm08"])
results.to_csv(os.path.join(RESULTS_DIR, "lstm_final_results.csv"), index=False)
prediction_table.to_csv(os.path.join(RESULTS_DIR, "predictions_lstm_tuned.csv"), index=False)

print()
print("=== 100 test engines, mean (std) over 5 seeds ===")
summary = results.groupby("setting")[["rmse", "mae", "phm08"]].agg(["mean", "std"])
print(summary.round(3).to_string())
