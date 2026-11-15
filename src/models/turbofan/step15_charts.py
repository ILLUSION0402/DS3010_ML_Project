# Step 15: charts for the new results. Same colours and style as make_charts.py.
#
# Run after steps 10, 11 and 12. Writes charts into reports/turbofan/figures/:
#   turbofan_cv_window.png      CV error vs window size, RF and XGBoost
#   turbofan_model_compare.png  all models on the 100 test engines
#   turbofan_engine_tracks.png  predicted vs true RUL over each cycle, 4 test engines

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from common import load_normalized, add_rolling_features, get_feature_columns, RESULTS_DIR, FIGURES_DIR

BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
LIGHT_BAR = "#c3c2b7"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(LIGHT_BAR)
    ax.tick_params(colors=MUTED, labelsize=10.5)


# ================= CHART 1: CV error vs window =================
cv_table = pd.read_csv(os.path.join(RESULTS_DIR, "cv_results.csv"))
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
for ax, metric, title in [(axes[0], "rmse", "CV RMSE (cycles)"), (axes[1], "phm08", "CV PHM08 score")]:
    style(ax)
    for model_name, colour, label in [("RandomForest", BLUE, "Random Forest"), ("XGBoost", ORANGE, "XGBoost")]:
        rows = cv_table[cv_table["model"] == model_name]
        means = rows.groupby("window")[metric].mean()
        stds = rows.groupby("window")[metric].std()
        ax.errorbar(means.index, means.values, yerr=stds.values, color=colour, linewidth=2,
                    marker="o", markersize=7, capsize=4, elinewidth=1, label=label, zorder=3)
        ax.text(means.index[-1] + 1, means.values[-1], label, color=INK2, fontsize=10, va="center")
    ax.set_xticks(sorted(cv_table["window"].unique()))
    ax.set_xlabel("Rolling window (cycles)", color=INK2, fontsize=11)
    ax.set_title(title, color=INK, fontsize=12.5, loc="left")
    ax.set_xlim(2, 48)
leg = axes[0].legend(frameon=False, fontsize=10, loc="upper left")
fig.suptitle("5-fold grouped CV on training engines (mean, bars = 1 std across folds)",
             color=INK, fontsize=13.5, x=0.015, ha="left")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(FIGURES_DIR, "turbofan_cv_window.png"), dpi=170, facecolor=SURFACE)
plt.close(fig)

# ================= CHART 2: model comparison on test =================
# Every bar is the mean over 5 random seeds, with a whisker for 1 std (step 20
# for the tree models and the tuned LSTM, step 12 for the untuned LSTM).
# Seed-averaged numbers are fairer than one run with random_state=42.
seed_table = pd.read_csv(os.path.join(RESULTS_DIR, "seed_robustness.csv"))
lstm_untuned = pd.read_csv(os.path.join(RESULTS_DIR, "lstm_seed_results.csv"))
lstm_untuned["model"] = "LSTM untuned"
seed_table = pd.concat([seed_table, lstm_untuned], ignore_index=True)

#              name in seed_robustness.csv     label on the chart
BAR_ORDER = [("RF window 20",                "RF\nw=20"),
             ("RF window 5",                 "RF\nw=5"),
             ("XGBoost window 5",            "XGB\nw=5"),
             ("XGBoost tuned (lambda 10)",   "XGB tuned\nλ=10"),
             ("XGBoost asymmetric",          "XGB\nasym."),
             ("LSTM untuned",                "LSTM\nuntuned"),
             ("LSTM tuned",                  "LSTM\ntuned")]
names = []
values = {"rmse": [], "mae": [], "phm08": []}
spreads = {"rmse": [], "mae": [], "phm08": []}
for model_name, label in BAR_ORDER:
    rows = seed_table[seed_table["model"] == model_name]
    names.append(label)
    for metric in values:
        values[metric].append(rows[metric].mean())
        spreads[metric].append(rows[metric].std())

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), facecolor=SURFACE)
for ax, metric, title in [(axes[0], "rmse", "RMSE (cycles)"), (axes[1], "mae", "MAE (cycles)"),
                          (axes[2], "phm08", "PHM08 score")]:
    style(ax)
    best = int(np.argmin(values[metric]))
    colours = []
    for i in range(len(names)):
        if i == best:
            colours.append(ORANGE)
        else:
            colours.append(LIGHT_BAR)
    positions = np.arange(len(names))
    ax.bar(positions, values[metric], color=colours, width=0.62, zorder=3)
    for i in range(len(names)):
        if spreads[metric][i] > 0:
            ax.errorbar(positions[i], values[metric][i], yerr=spreads[metric][i],
                        color=INK2, capsize=4, linewidth=1, zorder=4)
        text = ("%.2f" % values[metric][i]) if values[metric][i] < 100 else ("%.0f" % values[metric][i])
        ax.text(positions[i], values[metric][i] + spreads[metric][i] + max(values[metric]) * 0.02,
                text, ha="center", color=INK, fontsize=9.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(names, fontsize=8.5, color=INK2)
    tallest = 0
    for i in range(len(names)):
        tallest = max(tallest, values[metric][i] + spreads[metric][i])
    ax.set_ylim(0, tallest * 1.12)
    ax.set_title(title, color=INK, fontsize=12.5, loc="left")
fig.suptitle("100 held-out test engines, mean over 5 random seeds (whisker = 1 std; orange = lowest mean)",
             color=INK, fontsize=13, x=0.015, ha="left")
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(FIGURES_DIR, "turbofan_model_compare.png"), dpi=170, facecolor=SURFACE)
plt.close(fig)

# ================= CHART 3: predicted vs true RUL over every cycle =================
# Retrain the step-9 Random Forest (window 20) and predict EVERY test cycle,
# not just the last one, to show how the prediction evolves as the engine runs.
train_data, test_data = load_normalized()
train_fe = add_rolling_features(train_data, 20)
test_fe = add_rolling_features(test_data, 20)
feature_columns = get_feature_columns(train_fe)
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(train_fe[feature_columns], train_fe["RUL"])
test_fe["predicted"] = rf.predict(test_fe[feature_columns])

# pick 4 long-running test engines that are close to failure at the cut-off,
# so the "knee" where RUL starts dropping is visible
lengths = test_fe.groupby("unit_number")["cycle"].max()
final_rul = test_fe.groupby("unit_number")["RUL"].last()
candidates = lengths[final_rul <= 50].sort_values(ascending=False)
chosen = list(candidates.index[:4])
print("Engines shown:", chosen)

fig, axes = plt.subplots(2, 2, figsize=(11, 6.6), facecolor=SURFACE, sharey=True)
for k in range(4):
    ax = axes[k // 2][k % 2]
    style(ax)
    engine_rows = test_fe[test_fe["unit_number"] == chosen[k]]
    ax.plot(engine_rows["cycle"], engine_rows["RUL"], color=BLUE, linewidth=2.2, label="True RUL", zorder=3)
    ax.plot(engine_rows["cycle"], engine_rows["predicted"], color=ORANGE, linewidth=1.6,
            label="Predicted RUL", zorder=4)
    ax.set_title("Test engine %d" % chosen[k], color=INK, fontsize=12, loc="left")
    ax.set_ylim(0, 140)
    if k >= 2:
        ax.set_xlabel("Cycle", color=INK2, fontsize=11)
    if k % 2 == 0:
        ax.set_ylabel("RUL (cycles)", color=INK2, fontsize=11)
leg = axes[0][0].legend(frameon=False, fontsize=10, loc="lower left")
fig.suptitle("Random Forest (window 20) prediction at every cycle, until the test cut-off",
             color=INK, fontsize=13.5, x=0.015, ha="left")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(FIGURES_DIR, "turbofan_engine_tracks.png"), dpi=170, facecolor=SURFACE)
plt.close(fig)
print("charts written")
