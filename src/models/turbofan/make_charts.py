import pandas as pd
import numpy as np
import math
# Needs matplotlib on top of pandas/scikit-learn:  pip install matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

import os
from common import RAW_DIR, FIGURES_DIR

UP = RAW_DIR + os.sep          # raw .txt files (data/raw/turbofan/)
OUT = FIGURES_DIR + os.sep     # charts (reports/turbofan/figures/)

BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

RUL_CAP = 125
WINDOW = 20

columns = ["unit_number", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
for i in range(1, 22):
    columns.append("sensor_" + str(i))


def load_raw(path):
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.iloc[:, :len(columns)]
    df.columns = columns
    return df


train_data = load_raw(UP + "train_FD001.txt")
test_data = load_raw(UP + "test_FD001.txt")
rul_final = pd.read_csv(UP + "RUL_FD001.txt", header=None)[0].tolist()

# keep a raw copy before normalisation, for the degradation-trend chart
raw_train = train_data.copy()

# drop near-constant sensors
sensor_columns = [c for c in columns if c.startswith("sensor_")]
flat_sensors = []
for col in sensor_columns:
    if train_data[col].nunique() <= 5:
        flat_sensors.append(col)
train_data = train_data.drop(columns=flat_sensors)
test_data = test_data.drop(columns=flat_sensors)

# RUL labels
max_cycle_train = train_data.groupby("unit_number")["cycle"].transform("max")
train_data["RUL"] = (max_cycle_train - train_data["cycle"]).clip(upper=RUL_CAP)

max_cycle_test = test_data.groupby("unit_number")["cycle"].transform("max")
rul_lookup = {}
for i in range(len(rul_final)):
    rul_lookup[i + 1] = rul_final[i]
rul_at_end = test_data["unit_number"].map(rul_lookup)
test_data["RUL"] = (rul_at_end + (max_cycle_test - test_data["cycle"])).clip(upper=RUL_CAP)

# drop constant op setting then normalise on train stats
train_data = train_data.drop(columns=["op_setting_3"])
test_data = test_data.drop(columns=["op_setting_3"])

feature_columns = [c for c in train_data.columns if c not in ("unit_number", "cycle", "RUL")]
for col in feature_columns:
    m = train_data[col].mean()
    s = train_data[col].std()
    train_data[col] = (train_data[col] - m) / s
    test_data[col] = (test_data[col] - m) / s


def phm08(true_list, pred_list):
    total = 0
    for i in range(len(pred_list)):
        d = pred_list[i] - true_list[i]
        if d < 0:
            total = total + (math.exp(-d / 13) - 1)
        else:
            total = total + (math.exp(d / 10) - 1)
    return total


# ---------------- baseline ----------------
x_train = train_data[feature_columns]
y_train = train_data["RUL"]
last_rows = test_data.groupby("unit_number").tail(1)
x_test = last_rows[feature_columns]
y_test = last_rows["RUL"]

base_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
base_model.fit(x_train, y_train)
base_pred = base_model.predict(x_test)

base_rmse = mean_squared_error(y_test, base_pred) ** 0.5
base_mae = mean_absolute_error(y_test, base_pred)
base_score = phm08(y_test.tolist(), base_pred)

# ---------------- feature engineered ----------------
sensor_cols_fe = [c for c in train_data.columns if c.startswith("sensor_") or c.startswith("op_setting_")]


def add_rolling(data):
    data = data.sort_values(["unit_number", "cycle"]).copy()
    frames = [data]
    for col in sensor_cols_fe:
        g = data.groupby("unit_number")[col]
        frames.append(pd.DataFrame({
            col + "_mean": g.rolling(WINDOW, min_periods=1).mean().reset_index(level=0, drop=True),
            col + "_std": g.rolling(WINDOW, min_periods=1).std().reset_index(level=0, drop=True).fillna(0),
            col + "_min": g.rolling(WINDOW, min_periods=1).min().reset_index(level=0, drop=True),
            col + "_max": g.rolling(WINDOW, min_periods=1).max().reset_index(level=0, drop=True),
            col + "_delta": data[col] - g.transform("first"),
        }))
    return pd.concat(frames, axis=1)


train_fe = add_rolling(train_data)
test_fe = add_rolling(test_data)
fe_cols = [c for c in train_fe.columns if c not in ("unit_number", "cycle", "RUL")]

last_fe = test_fe.groupby("unit_number").tail(1)
fe_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
fe_model.fit(train_fe[fe_cols], train_fe["RUL"])
fe_pred = fe_model.predict(last_fe[fe_cols])
fe_true = last_fe["RUL"]

fe_rmse = mean_squared_error(fe_true, fe_pred) ** 0.5
fe_mae = mean_absolute_error(fe_true, fe_pred)
fe_score = phm08(fe_true.tolist(), fe_pred)

print("BASELINE  RMSE %.3f  MAE %.3f  PHM %.2f" % (base_rmse, base_mae, base_score))
print("FEATENG   RMSE %.3f  MAE %.3f  PHM %.2f" % (fe_rmse, fe_mae, fe_score))


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=11)


# ============ CHART 1: true vs predicted RUL, sorted ============
order = np.argsort(fe_true.values)
true_sorted = fe_true.values[order]
pred_sorted = fe_pred[order]

fig, ax = plt.subplots(figsize=(10, 5.2), facecolor=SURFACE)
style(ax)
xs = np.arange(1, 101)
ax.plot(xs, true_sorted, color=BLUE, linewidth=2.2, label="True RUL", zorder=3)
ax.plot(xs, pred_sorted, color=ORANGE, linewidth=1.6, linestyle="--",
        marker="o", markersize=3.5, label="Predicted RUL", zorder=4)
ax.set_xlabel("Test engines (sorted by true RUL)", color=INK2, fontsize=12)
ax.set_ylabel("Remaining Useful Life (cycles)", color=INK2, fontsize=12)
ax.set_title("Turbofan: true vs predicted RUL on 100 held-out engines",
             color=INK, fontsize=14, pad=14, loc="left")
leg = ax.legend(frameon=False, fontsize=11, loc="upper left")
for t in leg.get_texts():
    t.set_color(INK2)
fig.tight_layout()
fig.savefig(OUT + "turbofan_true_vs_pred.png", dpi=170, facecolor=SURFACE)
plt.close(fig)

# ============ CHART 2: baseline vs FE ============
fig, axes = plt.subplots(1, 3, figsize=(11, 4.0), facecolor=SURFACE)
metrics = [
    ("RMSE (cycles)", base_rmse, fe_rmse),
    ("MAE (cycles)", base_mae, fe_mae),
    ("PHM08 score", base_score, fe_score),
]
for ax, (name, b, f) in zip(axes, metrics):
    style(ax)
    bars = ax.bar(["Raw\nsensors", "+ Rolling\nfeatures"], [b, f],
                  color=[MUTED, ORANGE], width=0.56, zorder=3)
    drop = (b - f) / b * 100
    ax.set_title(name, color=INK, fontsize=13, pad=22)
    ax.text(0.5, 1.035, "%.0f%% lower" % drop, transform=ax.transAxes,
            ha="center", color="#006300", fontsize=12, fontweight="bold")
    top = max(b, f)
    ax.set_ylim(0, top * 1.22)
    for rect, val in zip(bars, [b, f]):
        ax.text(rect.get_x() + rect.get_width() / 2, val + top * 0.035,
                ("%.2f" % val) if val < 100 else ("%.0f" % val),
                ha="center", color=INK, fontsize=12, fontweight="bold")
fig.suptitle("Adding 20 cycles of history improves every metric",
             color=INK, fontsize=14, x=0.015, ha="left", y=0.995)
fig.tight_layout(rect=[0, 0.06, 1, 0.93])
fig.savefig(OUT + "turbofan_baseline_vs_fe.png", dpi=170, facecolor=SURFACE)
plt.close(fig)

# ============ CHART 3: feature importance ============
imp = pd.Series(fe_model.feature_importances_, index=fe_cols).sort_values(ascending=False).head(10)
imp = imp.iloc[::-1]
fig, ax = plt.subplots(figsize=(9.5, 5.0), facecolor=SURFACE)
style(ax)
cols = [ORANGE if ("_mean" in n or "_delta" in n or "_std" in n or "_min" in n or "_max" in n) else BLUE
        for n in imp.index]
ax.barh(range(len(imp)), imp.values, color=cols, height=0.62, zorder=3)
ax.set_yticks(range(len(imp)))
ax.set_yticklabels(imp.index, fontsize=11, color=INK2)
ax.set_xlabel("Importance", color=INK2, fontsize=12)
ax.set_title("Engineered trend features dominate what the model uses",
             color=INK, fontsize=14, pad=14, loc="left")
for i, v in enumerate(imp.values):
    ax.text(v + 0.008, i, "%.2f" % v, va="center", color=INK2, fontsize=10.5)
ax.set_xlim(0, max(imp.values) * 1.16)
fig.tight_layout()
fig.savefig(OUT + "turbofan_feature_importance.png", dpi=170, facecolor=SURFACE)
plt.close(fig)

# ============ CHART 4: sensor degradation trend ============
fig, ax = plt.subplots(figsize=(10, 5.0), facecolor=SURFACE)
style(ax)
for eng in range(1, 101):
    sub = raw_train[raw_train.unit_number == eng]
    life = sub.cycle.max()
    ax.plot(sub.cycle - life, sub.sensor_3.rolling(10, min_periods=1).mean(),
            color=BLUE, alpha=0.13, linewidth=0.9, zorder=2)
for eng in [1, 24, 69]:
    sub = raw_train[raw_train.unit_number == eng]
    life = sub.cycle.max()
    ax.plot(sub.cycle - life, sub.sensor_3.rolling(10, min_periods=1).mean(),
            color=ORANGE, linewidth=2.0, zorder=4)
ax.set_xlabel("Cycles before failure", color=INK2, fontsize=12)
ax.set_ylabel("sensor_3  (HPC outlet temperature, °R)", color=INK2, fontsize=12)
ax.set_title("Every engine's HPC temperature climbs as failure approaches",
             color=INK, fontsize=14, pad=14, loc="left")
ax.text(0.015, 0.95, "100 training engines · 3 highlighted",
        transform=ax.transAxes, color=MUTED, fontsize=11, va="top")
ax.set_xlim(-300, 2)
fig.tight_layout()
fig.savefig(OUT + "turbofan_degradation.png", dpi=170, facecolor=SURFACE)
plt.close(fig)

print("charts written")
