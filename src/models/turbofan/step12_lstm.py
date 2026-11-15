# Step 12: LSTM in PyTorch.
#
# Why an LSTM: the Random Forest only sees trends because we hand-built rolling
# features for it (step 9). An LSTM reads the last 30 cycles in order and
# learns the trend itself. So it gets ONLY the 16 normalised columns, not the
# 96 engineered features - otherwise we could not tell what the LSTM itself adds.
#
# How training works, per seed:
#   1. Hold out 20 of the 100 training engines (whole engines, never rows).
#      Train on the other 80 and watch the error on the 20 after every epoch.
#      Remember the epoch where that error was lowest ("early stopping").
#   2. Train a fresh model on all 100 engines for exactly that many epochs.
#      This way the LSTM sees the same 100 engines the Random Forest saw.
#   3. Predict the last 30 cycles of each of the 100 test engines.
#
# Why 5 seeds: a neural network starts from random weights, so two runs give
# different answers. Reporting one lucky run would be overclaiming. We report
# every seed, the mean and spread, and the average of the 5 models' predictions.
#
# The label is divided by 125 for training (networks learn better with targets
# around 0-1) and multiplied back by 125 before scoring.
#
# Runs on CPU (FD001 is small). About 4-5 minutes for all 5 seeds on a Ryzen 7.

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from common import load_normalized, get_sensor_columns, print_scores, RESULTS_DIR, MODELS_DIR, RUL_CAP
from lstm_model import RULModel, make_train_sequences, make_last_sequences

SEEDS = [0, 1, 2, 3, 4]
MAX_EPOCHS = 60
PATIENCE = 10          # stop if validation has not improved for this many epochs
BATCH_SIZE = 256
LEARNING_RATE = 0.001

train_data, test_data = load_normalized()
feature_columns = get_sensor_columns(train_data)
print("LSTM input columns:", len(feature_columns))

x_test, y_test, test_engines = make_last_sequences(test_data, feature_columns)
x_test_tensor = torch.tensor(x_test)


def train_one_epoch(model, optimizer, x_train, y_train, generator):
    model.train()
    loss_function = nn.MSELoss()
    order = torch.randperm(len(x_train), generator=generator)
    for start in range(0, len(x_train), BATCH_SIZE):
        batch_index = order[start:start + BATCH_SIZE]
        x_batch = x_train[batch_index]
        y_batch = y_train[batch_index]
        optimizer.zero_grad()
        loss = loss_function(model(x_batch), y_batch)
        loss.backward()
        optimizer.step()


def predict_cycles(model, x):
    model.eval()
    with torch.no_grad():
        predictions = model(x).numpy() * RUL_CAP
    return np.clip(predictions, 0, RUL_CAP)


all_engines = sorted(train_data["unit_number"].unique())
seed_rows = []
seed_predictions = []

for seed in SEEDS:
    start_time = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    generator = torch.Generator().manual_seed(seed)

    # ---- 1. find the best number of epochs on 80 / 20 engines ----
    val_engines = np.random.RandomState(seed).choice(all_engines, size=20, replace=False)
    fit_part = train_data[~train_data["unit_number"].isin(val_engines)]
    val_part = train_data[train_data["unit_number"].isin(val_engines)]
    x_fit, y_fit = make_train_sequences(fit_part, feature_columns)
    x_val, y_val = make_train_sequences(val_part, feature_columns)
    x_fit = torch.tensor(x_fit)
    y_fit = torch.tensor(y_fit / RUL_CAP)
    x_val = torch.tensor(x_val)

    model = RULModel(len(feature_columns))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_val_rmse = 1e9
    best_epoch = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        train_one_epoch(model, optimizer, x_fit, y_fit, generator)
        val_rmse = np.sqrt(np.mean((predict_cycles(model, x_val) - y_val) ** 2))
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
        if epoch - best_epoch >= PATIENCE:
            break
    print("seed %d: best epoch %d (validation RMSE %.2f on all windows)" % (seed, best_epoch, best_val_rmse))

    # ---- 2. retrain on all 100 engines for best_epoch epochs ----
    torch.manual_seed(seed)
    x_all, y_all = make_train_sequences(train_data, feature_columns)
    x_all = torch.tensor(x_all)
    y_all = torch.tensor(y_all / RUL_CAP)
    model = RULModel(len(feature_columns))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    for epoch in range(best_epoch):
        train_one_epoch(model, optimizer, x_all, y_all, generator)

    # ---- 3. score on the 100 test engines ----
    predictions = predict_cycles(model, x_test_tensor)
    rmse, mae, score = print_scores("LSTM seed %d" % seed, y_test, predictions)
    seed_rows.append([seed, best_epoch, rmse, mae, score])
    seed_predictions.append(predictions)
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, "lstm_seed%d.pt" % seed))
    print("   (%.0f seconds)" % (time.time() - start_time))

seed_table = pd.DataFrame(seed_rows, columns=["seed", "epochs", "rmse", "mae", "phm08"])
seed_table.to_csv(os.path.join(RESULTS_DIR, "lstm_seed_results.csv"), index=False)

print()
print("=== LSTM over %d seeds (100 test engines) ===" % len(SEEDS))
for name in ["rmse", "mae", "phm08"]:
    print("%-6s mean %8.3f   std %7.3f   min %8.3f   max %8.3f"
          % (name, seed_table[name].mean(), seed_table[name].std(),
             seed_table[name].min(), seed_table[name].max()))

average_prediction = np.mean(np.array(seed_predictions), axis=0)
print_scores("LSTM average of 5 seeds", y_test, average_prediction)

out = pd.DataFrame({"engine_id": test_engines, "true_RUL": y_test, "predicted_RUL": average_prediction})
for i in range(len(SEEDS)):
    out["seed_%d" % SEEDS[i]] = seed_predictions[i]
out.to_csv(os.path.join(RESULTS_DIR, "predictions_lstm.csv"), index=False)
print("Saved predictions_lstm.csv")
