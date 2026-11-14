import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

PROCESSED_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed'))
FIGURES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'reports', 'figures'))
os.makedirs(FIGURES_DIR, exist_ok=True)

def main():
    sns.set_theme(style='whitegrid', font_scale=1.1)

    # 1. Capacity Degradation Over Cycles
    train_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'train_battery.csv'))
    test_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'test_battery.csv'))

    plt.figure(figsize=(10, 6))
    colors = {5: '#1f77b4', 6: '#2ca02c', 7: '#ff7f0e'}
    for unit in [5, 6, 7]:
        sub = train_df[train_df['unit_number'] == unit]
        plt.plot(sub['cycle'], sub['capacity'], label=f'Battery {unit} (Train)', color=colors[unit], linewidth=2)

    sub18 = test_df[test_df['unit_number'] == 18]
    plt.plot(sub18['cycle'], sub18['capacity'], label='Battery 18 (Held-out Test)', color='#9467bd', linewidth=2.5, linestyle='--')

    plt.axhline(y=1.4, color='#d62728', linestyle=':', label='EOL Threshold (1.4 Ahr)', linewidth=2.5)
    plt.title('NASA Battery Dataset: Capacity Fade Across Cycles', fontsize=14, fontweight='bold')
    plt.xlabel('Discharge Cycle Index', fontsize=12)
    plt.ylabel('Measured Capacity (Ahr)', fontsize=12)
    plt.legend(frameon=True, loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'capacity_fade.png'), dpi=300)
    plt.close()

    # 2. True vs Predicted RUL: Full Comparison
    preds_file = os.path.join(PROCESSED_DATA_DIR, 'battery_18_all_model_predictions.csv')
    if not os.path.exists(preds_file):
        preds_file = os.path.join(PROCESSED_DATA_DIR, 'battery_18_predictions.csv')

    preds_df = pd.read_csv(preds_file)

    plt.figure(figsize=(11, 6))
    plt.plot(preds_df['cycle'], preds_df['true_RUL'], label='Ground Truth RUL', color='black', linewidth=3, zorder=5)

    if 'Gradient_Boosting_Pred' in preds_df.columns:
        plt.plot(preds_df['cycle'], preds_df['Gradient_Boosting_Pred'], label='Baseline Gradient Boosting (MSE)', color='#ff7f0e', linestyle='--', linewidth=2)
    if 'Asymmetric_XGBoost_Pred' in preds_df.columns:
        plt.plot(preds_df['cycle'], preds_df['Asymmetric_XGBoost_Pred'], label='Asymmetric XGBoost (Custom PHM Objective)', color='#2ca02c', linewidth=2.5)
    if 'PyTorch_LSTM_Pred' in preds_df.columns:
        plt.plot(preds_df['cycle'], preds_df['PyTorch_LSTM_Pred'], label='PyTorch LSTM (Sequence W=10)', color='#1f77b4', linestyle=':', linewidth=2)

    plt.axvline(x=97, color='red', linestyle='--', alpha=0.7, label='Battery 18 EOL Cycle (Cycle 97)')
    plt.title('Battery 18: Remaining Useful Life (RUL) Trajectory Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Discharge Cycle Index', fontsize=12)
    plt.ylabel('Remaining Useful Life (Cycles)', fontsize=12)
    plt.legend(frameon=True, loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'rul_predictions_comparison.png'), dpi=300)
    plt.close()

    # Also keep rul_predictions.png backward-compatible for existing reports
    plt.figure(figsize=(10, 6))
    plt.plot(preds_df['cycle'], preds_df['true_RUL'], label='Ground Truth RUL', color='black', linewidth=3)
    if 'Asymmetric_XGBoost_Pred' in preds_df.columns:
        plt.plot(preds_df['cycle'], preds_df['Asymmetric_XGBoost_Pred'], label='Asymmetric XGBoost (Custom PHM Objective)', color='#2ca02c', linewidth=2.5)
    if 'Gradient_Boosting_Pred' in preds_df.columns:
        plt.plot(preds_df['cycle'], preds_df['Gradient_Boosting_Pred'], label='Baseline Gradient Boosting (MSE)', color='#ff7f0e', linestyle='--', linewidth=2)
    plt.title('Battery 18: True vs Asymmetric XGBoost Predicted RUL', fontsize=14, fontweight='bold')
    plt.xlabel('Cycle', fontsize=12)
    plt.ylabel('Remaining Useful Life (Cycles)', fontsize=12)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'rul_predictions.png'), dpi=300)
    plt.close()

    # 3. Late-Stage Overprediction Zoom (Cycles 70 to 132)
    plt.figure(figsize=(10, 6))
    tail_df = preds_df[preds_df['cycle'] >= 70]
    plt.plot(tail_df['cycle'], tail_df['true_RUL'], label='Ground Truth RUL', color='black', linewidth=3)
    if 'Gradient_Boosting_Pred' in tail_df.columns:
        plt.plot(tail_df['cycle'], tail_df['Gradient_Boosting_Pred'], label='Baseline (MSE) - Exhibits Late Optimism', color='#ff7f0e', linestyle='--', linewidth=2)
    if 'Asymmetric_XGBoost_Pred' in tail_df.columns:
        plt.plot(tail_df['cycle'], tail_df['Asymmetric_XGBoost_Pred'], label='Asymmetric XGBoost - Corrected Overprediction', color='#2ca02c', linewidth=2.5)

    plt.axvline(x=97, color='red', linestyle=':', label='EOL Point (Cycle 97)', linewidth=2)
    plt.fill_between(tail_df['cycle'], tail_df['true_RUL'], tail_df['Gradient_Boosting_Pred'], where=(tail_df['Gradient_Boosting_Pred'] > tail_df['true_RUL']), color='red', alpha=0.15, label='Dangerous Late Penalty Region')
    plt.title('Late-Stage Optimism Correction (Cycles 70 - 132)', fontsize=14, fontweight='bold')
    plt.xlabel('Discharge Cycle Index', fontsize=12)
    plt.ylabel('Remaining Useful Life (Cycles)', fontsize=12)
    plt.legend(frameon=True, loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'late_stage_overprediction_fix.png'), dpi=300)
    plt.close()

    print(f"All figures successfully generated in: {FIGURES_DIR}")

if __name__ == '__main__':
    main()
