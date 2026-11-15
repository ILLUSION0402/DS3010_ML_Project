# Shared helpers for steps 10 onwards.
#
# Steps 1-9 each carry their own copy of the loading and scoring code. From
# step 10 on, every script imports it from here instead, so there is exactly
# ONE copy of the PHM08 formula. (The earlier constant swap is the reason:
# one copy can only be wrong in one place.)
#
# Paths are worked out from where this file lives, so the scripts can be run
# from any folder. Layout (same pattern as the bearings component):
#   src/models/turbofan/        <- these scripts
#   data/raw/turbofan/          <- train_FD001.txt, test_FD001.txt, RUL_FD001.txt (not in Git)
#   data/processed/turbofan/    <- CSVs written by steps 3-5 (not in Git)
#   models/turbofan/            <- trained models (not in Git)
#   reports/turbofan/           <- result CSVs, with figures/ and logs/ inside

import os
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(CODE_DIR)))   # repository root
RAW_DIR = os.path.join(ROOT_DIR, "data", "raw", "turbofan")
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed", "turbofan")
MODELS_DIR = os.path.join(ROOT_DIR, "models", "turbofan")
RESULTS_DIR = os.path.join(ROOT_DIR, "reports", "turbofan")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

for folder in [PROCESSED_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR]:
    os.makedirs(folder, exist_ok=True)

RUL_CAP = 125
WINDOW = 20


def load_normalized():
    # The output of step 5: 19 columns, train-only scaling, RUL capped at 125.
    train_data = pd.read_csv(os.path.join(PROCESSED_DIR, "train_FD001_normalized.csv"))
    test_data = pd.read_csv(os.path.join(PROCESSED_DIR, "test_FD001_normalized.csv"))
    return train_data, test_data


def get_sensor_columns(data):
    # The 16 columns that get rolling features: 2 op settings + 14 sensors.
    sensor_columns = []
    for col in data.columns:
        if col.startswith("sensor_") or col.startswith("op_setting_"):
            sensor_columns.append(col)
    return sensor_columns


def get_feature_columns(data):
    # Everything except the id, the cycle counter and the label.
    feature_columns = []
    for col in data.columns:
        if col not in ("unit_number", "cycle", "RUL"):
            feature_columns.append(col)
    return feature_columns


def add_rolling_features(data, window=WINDOW):
    # Same as step 9. Every feature at cycle t only looks at cycles <= t of the
    # SAME engine (rolling windows look backwards, delta uses the first reading),
    # so a row's features never use the future.
    data = data.sort_values(["unit_number", "cycle"]).copy()
    sensor_columns = get_sensor_columns(data)
    new_frames = [data]
    for col in sensor_columns:
        grouped = data.groupby("unit_number")[col]
        rolling_mean = grouped.rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
        rolling_std = grouped.rolling(window, min_periods=1).std().reset_index(level=0, drop=True)
        rolling_min = grouped.rolling(window, min_periods=1).min().reset_index(level=0, drop=True)
        rolling_max = grouped.rolling(window, min_periods=1).max().reset_index(level=0, drop=True)
        delta = data[col] - grouped.transform("first")
        frame = pd.DataFrame({
            col + "_mean": rolling_mean,
            col + "_std": rolling_std.fillna(0),
            col + "_min": rolling_min,
            col + "_max": rolling_max,
            col + "_delta": delta,
        })
        new_frames.append(frame)
    return pd.concat(new_frames, axis=1)


def last_cycle_rows(data):
    # One row per engine: its final recorded cycle. This is what we score on.
    return data.sort_values(["unit_number", "cycle"]).groupby("unit_number").tail(1)


def make_cut_points(train_data, cuts_per_engine=5, min_cycle=31, seed=42):
    # For cross-validation (see step 10): 5 random cut cycles per training engine.
    # Scoring a validation engine at these rows mimics a test engine that was
    # cut off before failure.
    rng = np.random.RandomState(seed)
    cut_rows = []
    for engine in sorted(train_data["unit_number"].unique()):
        engine_rows = train_data[train_data["unit_number"] == engine]
        possible_cycles = engine_rows[engine_rows["cycle"] >= min_cycle]["cycle"].values
        chosen = rng.choice(possible_cycles, size=cuts_per_engine, replace=False)
        for cycle in chosen:
            cut_rows.append((engine, cycle))
    cut_table = pd.DataFrame(cut_rows, columns=["unit_number", "cycle"])
    cut_table["is_cut"] = True
    return cut_table


def make_engine_folds(train_data, n_folds=5, seed=42):
    # A list of 5 lists, each holding the engine ids of one validation fold.
    engines = train_data["unit_number"].values
    splitter = GroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_engine_lists = []
    for train_index, val_index in splitter.split(train_data, groups=engines):
        fold_engine_lists.append(sorted(set(engines[val_index])))
    return fold_engine_lists


def phm08(true_values, predicted_values):
    # d = predicted - true.  Early (d < 0): exp(-d/13) - 1.  Late (d >= 0): exp(d/10) - 1.
    # Late is steeper (divide by 10), so over-estimating life costs more.
    # Sanity check: an error of +50 gives 147.4, an error of -50 gives 45.8.
    true_list = list(true_values)
    pred_list = list(predicted_values)
    total = 0
    for i in range(len(pred_list)):
        d = pred_list[i] - true_list[i]
        if d < 0:
            total = total + (math.exp(-d / 13) - 1)
        else:
            total = total + (math.exp(d / 10) - 1)
    return total


def score_all(true_values, predicted_values):
    # Returns rmse, mae, phm08 in that order.
    rmse = mean_squared_error(true_values, predicted_values) ** 0.5
    mae = mean_absolute_error(true_values, predicted_values)
    score = phm08(true_values, predicted_values)
    return rmse, mae, score


def print_scores(name, true_values, predicted_values):
    rmse, mae, score = score_all(true_values, predicted_values)
    print(name.ljust(28), "RMSE %7.3f   MAE %7.3f   PHM08 %9.2f" % (rmse, mae, score))
    return rmse, mae, score
