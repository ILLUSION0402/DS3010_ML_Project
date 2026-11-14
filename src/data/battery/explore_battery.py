import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_project_root():
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / 'data').exists() and (parent / 'src').exists():
            return parent
    return current.parents[3]

PROJECT_ROOT = get_project_root()
BATTERY_PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed', 'battery')
BATTERY_FIGURES_DIR = os.path.join(PROJECT_ROOT, 'reports', 'battery', 'figures')
os.makedirs(BATTERY_FIGURES_DIR, exist_ok=True)

def main():
    sns.set_theme(style='whitegrid', font_scale=1.1)

    train_df = pd.read_csv(os.path.join(BATTERY_PROCESSED_DIR, 'train_battery.csv'))
    test_df = pd.read_csv(os.path.join(BATTERY_PROCESSED_DIR, 'test_battery.csv'))

    # 1. Capacity Degradation Over Cycles
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    colors = {5: '#1f77b4', 6: '#2ca02c', 7: '#ff7f0e'}
    for unit in [5, 6, 7]:
        sub = train_df[train_df['unit_number'] == unit]
        ax1.plot(sub['cycle'], sub['capacity'], label=f'Battery {unit} (Train)', color=colors[unit], linewidth=2)

    sub18 = test_df[test_df['unit_number'] == 18]
    ax1.plot(sub18['cycle'], sub18['capacity'], label='Battery 18 (Held-out Test)', color='#9467bd', linewidth=2.5, linestyle='--')
    ax1.axhline(y=1.4, color='#d62728', linestyle=':', label='EOL Threshold (1.4 Ahr)', linewidth=2.5)
    ax1.set_title('NASA Battery Dataset: Capacity Fade Across Cycles', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Discharge Cycle Index', fontsize=12)
    ax1.set_ylabel('Measured Capacity (Ahr)', fontsize=12)
    ax1.legend(frameon=True, loc='upper right')
    fig1.tight_layout()
    fig1.savefig(os.path.join(BATTERY_FIGURES_DIR, 'capacity_fade.png'), dpi=300)
    plt.close(fig1)

    # 2. RUL Trajectory Comparison
    preds_file = os.path.join(BATTERY_PROCESSED_DIR, 'battery_18_all_model_predictions.csv')
    if not os.path.exists(preds_file):
        preds_file = os.path.join(BATTERY_PROCESSED_DIR, 'battery_18_predictions.csv')
    preds_df = pd.read_csv(preds_file)

    fig2, ax2 = plt.subplots(figsize=(11, 6))
    ax2.plot(preds_df['cycle'], preds_df['true_RUL'], label='Ground Truth RUL', color='black', linewidth=3, zorder=5)

    if 'Gradient_Boosting_Pred' in preds_df.columns:
        ax2.plot(preds_df['cycle'], preds_df['Gradient_Boosting_Pred'], label='Baseline Gradient Boosting (MSE)', color='#ff7f0e', linestyle='--', linewidth=2)
    if 'Asymmetric_XGBoost_Pred' in preds_df.columns:
        ax2.plot(preds_df['cycle'], preds_df['Asymmetric_XGBoost_Pred'], label='Asymmetric XGBoost (Custom PHM Objective)', color='#2ca02c', linewidth=2.5)
    if 'PyTorch_LSTM_Pred' in preds_df.columns:
        ax2.plot(preds_df['cycle'], preds_df['PyTorch_LSTM_Pred'], label='PyTorch LSTM (Sequence W=10)', color='#1f77b4', linestyle=':', linewidth=2)

    ax2.axvline(x=97, color='red', linestyle='--', alpha=0.7, label='Battery 18 EOL Cycle (Cycle 97)')
    ax2.set_title('Battery 18: Remaining Useful Life (RUL) Trajectory Comparison', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Discharge Cycle Index', fontsize=12)
    ax2.set_ylabel('Remaining Useful Life (Cycles)', fontsize=12)
    ax2.legend(frameon=True, loc='upper right')
    fig2.tight_layout()
    fig2.savefig(os.path.join(BATTERY_FIGURES_DIR, 'rul_predictions_comparison.png'), dpi=300)
    fig2.savefig(os.path.join(BATTERY_FIGURES_DIR, 'rul_predictions.png'), dpi=300)
    plt.close(fig2)

    # 3. Late-Stage Optimism Zoom
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    tail_df = preds_df[preds_df['cycle'] >= 70]
    ax3.plot(tail_df['cycle'], tail_df['true_RUL'], label='Ground Truth RUL', color='black', linewidth=3)
    if 'Gradient_Boosting_Pred' in tail_df.columns:
        ax3.plot(tail_df['cycle'], tail_df['Gradient_Boosting_Pred'], label='Baseline (MSE) - Exhibits Late Optimism', color='#ff7f0e', linestyle='--', linewidth=2)
    if 'Asymmetric_XGBoost_Pred' in tail_df.columns:
        ax3.plot(tail_df['cycle'], tail_df['Asymmetric_XGBoost_Pred'], label='Asymmetric XGBoost - Corrected Overprediction', color='#2ca02c', linewidth=2.5)

    ax3.axvline(x=97, color='red', linestyle=':', label='EOL Point (Cycle 97)', linewidth=2)
    if 'Gradient_Boosting_Pred' in tail_df.columns:
        ax3.fill_between(tail_df['cycle'], tail_df['true_RUL'], tail_df['Gradient_Boosting_Pred'], where=(tail_df['Gradient_Boosting_Pred'] > tail_df['true_RUL']), color='red', alpha=0.15, label='Dangerous Late Penalty Region')
    ax3.set_title('Late-Stage Optimism Correction (Cycles 70 - 132)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Discharge Cycle Index', fontsize=12)
    ax3.set_ylabel('Remaining Useful Life (Cycles)', fontsize=12)
    ax3.legend(frameon=True, loc='upper right')
    fig3.tight_layout()
    fig3.savefig(os.path.join(BATTERY_FIGURES_DIR, 'late_stage_overprediction_fix.png'), dpi=300)
    plt.close(fig3)

    print(f"All battery figures successfully saved to: {BATTERY_FIGURES_DIR}")

if __name__ == '__main__':
    main()
