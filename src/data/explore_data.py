import pandas as pd
from pathlib import Path

base_dir = Path(__file__).resolve().parents[2]
(base_dir / "reports" / "figures").mkdir(parents=True, exist_ok=True)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

col_names = (
    ["unit_id", "cycle", "setting_1", "setting_2", "setting_3"]
    + [f"s_{i}" for i in range(1, 22)]
)

df = pd.read_csv(base_dir / "data" / "raw" / "train_FD001.txt", sep=r"\s+", header=None, names=col_names)

print("=== 1. GENERAL DATA QUALITY ===")
print(f"Total Rows: {df.shape[0]}, Total Columns: {df.shape[1]}")
print(f"Total Missing (NaN) Values: {df.isna().sum().sum()}")
print(f"Total Engines: {df['unit_id'].nunique()}")

lifespans = df.groupby("unit_id")["cycle"].max()
print("\n=== 2. ENGINE LIFESPAN STATS (IN CYCLES) ===")
print(f"Shortest-lived engine: {lifespans.min()} cycles (Engine {lifespans.idxmin()})")
print(f"Longest-lived engine:  {lifespans.max()} cycles (Engine {lifespans.idxmax()})")
print(f"Average lifespan:      {lifespans.mean():.1f} cycles")

sensor_cols = [f"s_{i}" for i in range(1, 22)]
setting_cols = ["setting_1", "setting_2", "setting_3"]

stds = df[setting_cols + sensor_cols].std()
flat_cols = stds[stds < 0.01].index.tolist()
active_sensors = [s for s in sensor_cols if s not in flat_cols]

print("\n=== 3. SENSOR VARIANCE SUMMARY ===")
print(df[setting_cols + sensor_cols].agg(["min", "max", "std"]).T.to_string())
print("\nFlat columns (std < 0.001):", flat_cols)
print(f"Active Sensors ({len(active_sensors)}):", active_sensors)

plt.figure(figsize=(8, 4))
sns.histplot(lifespans, bins=20, kde=True)
plt.title("Distribution of Engine Lifespans in train_FD001")
plt.xlabel("Total Cycles Until Failure")
plt.ylabel("Number of Engines")
plt.tight_layout()
plt.savefig(base_dir / "reports" / "figures" / "1_engine_lifespans.png", dpi=150)
plt.close()

sample_engines = [1, 2, 3, 4, 5]
sample_df = df[df["unit_id"].isin(sample_engines)]

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
for idx, setting in enumerate(setting_cols):
    ax = axes[idx]
    for eng_id in sample_engines:
        eng_data = sample_df[sample_df["unit_id"] == eng_id]
        ax.plot(eng_data["cycle"], eng_data[setting], label=f"Engine {eng_id}", alpha=0.8)
    status = "[CONSTANT]" if setting in flat_cols else "[NOISE / NO TREND]"
    ax.set_title(f"{setting} {status}", color="darkred", fontweight="bold")
    ax.set_xlabel("Flight Cycle")
    ax.set_ylabel(setting)
    ax.set_facecolor("#fff5f5")
    if idx == 0:
        ax.legend(loc="upper right")

plt.suptitle("Operational Settings Over Cycles (Sample Engines 1-5)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(base_dir / "reports" / "figures" / "2_operational_settings_all.png", dpi=150)
plt.close()

fig, axes = plt.subplots(7, 3, figsize=(18, 22), sharex=True)
axes = axes.flatten()

for idx, sensor in enumerate(sensor_cols):
    ax = axes[idx]
    for eng_id in sample_engines:
        eng_data = sample_df[sample_df["unit_id"] == eng_id]
        p = ax.plot(eng_data["cycle"], eng_data[sensor], alpha=0.15)
        color = p[0].get_color()
        smoothed_data = eng_data[sensor].rolling(window=10, min_periods=1).mean()
        ax.plot(eng_data["cycle"], smoothed_data, color=color, label=f"Engine {eng_id}", linewidth=2, alpha=0.9)
    
    if sensor in flat_cols:
        ax.set_title(f"{sensor} [CONSTANT - DROP]", color="crimson", fontweight="bold")
        ax.set_facecolor("#ffeef0")
    else:
        ax.set_title(f"{sensor} [ACTIVE - KEEP]", color="darkgreen", fontweight="bold")
    
    ax.set_ylabel(sensor)
    if idx == 0:
        ax.legend(loc="upper left", fontsize=8)

for ax in axes[-3:]:
    ax.set_xlabel("Flight Cycle")

plt.suptitle("All 21 Sensors Over Cycles (Red = Constant/Dead, Green = Active)", fontsize=16, fontweight="bold", y=1.002)
plt.tight_layout()
plt.savefig(base_dir / "reports" / "figures" / "3_all_21_sensors_trends.png", dpi=150)
plt.close()

plt.figure(figsize=(12, 10))
corr_active = df[active_sensors].corr()
sns.heatmap(corr_active, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title(f"Clean Correlation Heatmap: {len(active_sensors)} Active Sensors Only")
plt.tight_layout()
plt.savefig(base_dir / "reports" / "figures" / f"4_correlation_{len(active_sensors)}_active_sensors.png", dpi=150)
plt.close()

print("\nSaved 4 plots:")
print("1. 1_engine_lifespans.png")
print("2. 2_operational_settings_all.png")
print("3. 3_all_21_sensors_trends.png")
print(f"4. 4_correlation_{len(active_sensors)}_active_sensors.png")