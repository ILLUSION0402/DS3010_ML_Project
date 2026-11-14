import os
from pathlib import Path
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def get_project_root():
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / 'data').exists() and (parent / 'src').exists():
            return parent
    return current.parents[3]

PROJECT_ROOT = get_project_root()
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
BATTERY_PROCESSED_DIR = os.path.join(PROCESSED_DATA_DIR, 'battery')
TRAIN_PATH = os.path.join(PROCESSED_DATA_DIR, 'train_battery.csv')
TEST_PATH = os.path.join(PROCESSED_DATA_DIR, 'test_battery.csv')

FEATURE_COLS = ['v_mean', 'v_min', 'v_max', 'c_mean', 'c_min', 'c_max', 't_mean', 't_min', 't_max', 'capacity']
RUL_CAP = 125.0
WINDOW_SIZE = 10

def phm_score(y_true, y_pred):
    d = np.array(y_pred) - np.array(y_true)
    return float(np.sum(np.where(d >= 0, np.exp(d / 13.0) - 1.0, np.exp(-d / 10.0) - 1.0)))

def asymmetric_phm_objective(y_true, y_pred):
    d = np.clip(y_pred - y_true, -35.0, 35.0)
    # Heavily penalize late predictions (y_pred > y_true)
    grad = np.where(d > 0, (1.0 / 10.0) * np.exp(d / 10.0), -(1.0 / 13.0) * np.exp(-d / 13.0))
    hess = np.where(d > 0, (1.0 / 100.0) * np.exp(d / 10.0), (1.0 / 169.0) * np.exp(-d / 13.0))
    hess = np.clip(hess, 1e-4, 100.0)
    return grad, hess

class BatteryLSTM(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=48, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 24),
            nn.ReLU(),
            nn.Linear(24, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last_step = out[:, -1, :]
        return self.fc(last_step).squeeze(-1)

def create_sliding_sequences(df, feature_cols, window_size=WINDOW_SIZE):
    X_seq, y_seq, cycles = [], [], []
    for _, group in df.groupby('unit_number'):
        feats = group[feature_cols].values
        targets = group['target'].values
        cycs = group['cycle'].values
        n = len(group)
        for i in range(n):
            if i < window_size - 1:
                pad_len = (window_size - 1) - i
                pad = np.repeat(feats[0:1], pad_len, axis=0)
                seq = np.vstack([pad, feats[:i + 1]])
            else:
                seq = feats[i - window_size + 1 : i + 1]
            X_seq.append(seq)
            y_seq.append(targets[i])
            cycles.append(cycs[i])
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32), np.array(cycles)

def main():
    os.makedirs(BATTERY_PROCESSED_DIR, exist_ok=True)
    train_data = pd.read_csv(TRAIN_PATH)
    test_data = pd.read_csv(TEST_PATH)

    train_data['target'] = train_data['target'].clip(upper=RUL_CAP)
    test_data['target'] = test_data['target'].clip(upper=RUL_CAP)

    X_train_raw = train_data[FEATURE_COLS]
    y_train = train_data['target']
    X_test_raw = test_data[FEATURE_COLS]
    y_test = test_data['target']

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    results = []
    predictions = {
        'cycle': test_data['cycle'].values,
        'true_RUL': y_test.values
    }

    # 1. Baseline Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train_raw, y_train)
    rf_preds = rf.predict(X_test_raw)
    predictions['Random_Forest_Pred'] = np.round(rf_preds, 2)
    results.append({
        'Model': 'Random Forest (Baseline)',
        'Paradigm': 'Ensemble Trees (Bagging)',
        'Loss Function': 'Standard MSE',
        'RMSE': round(np.sqrt(mean_squared_error(y_test, rf_preds)), 2),
        'MAE': round(mean_absolute_error(y_test, rf_preds), 2),
        'PHM Score': round(phm_score(y_test, rf_preds), 2)
    })

    # 2. Baseline Gradient Boosting
    gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb.fit(X_train_raw, y_train)
    gb_preds = gb.predict(X_test_raw)
    predictions['Gradient_Boosting_Pred'] = np.round(gb_preds, 2)
    results.append({
        'Model': 'Gradient Boosting (Baseline)',
        'Paradigm': 'Ensemble Trees (Boosting)',
        'Loss Function': 'Standard MSE',
        'RMSE': round(np.sqrt(mean_squared_error(y_test, gb_preds)), 2),
        'MAE': round(mean_absolute_error(y_test, gb_preds), 2),
        'PHM Score': round(phm_score(y_test, gb_preds), 2)
    })

    # 3. Asymmetric XGBoost (Custom PHM Objective)
    xgb_asym = xgb.XGBRegressor(
        objective=asymmetric_phm_objective,
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42
    )
    xgb_asym.fit(X_train_raw, y_train)
    xgb_asym_preds = xgb_asym.predict(X_test_raw)
    predictions['Asymmetric_XGBoost_Pred'] = np.round(xgb_asym_preds, 2)
    results.append({
        'Model': 'Asymmetric XGBoost',
        'Paradigm': 'Ensemble Trees (Gradient Boosting)',
        'Loss Function': 'Custom Asymmetric PHM',
        'RMSE': round(np.sqrt(mean_squared_error(y_test, xgb_asym_preds)), 2),
        'MAE': round(mean_absolute_error(y_test, xgb_asym_preds), 2),
        'PHM Score': round(phm_score(y_test, xgb_asym_preds), 2)
    })

    # 4. PyTorch LSTM
    train_df_scaled = train_data.copy()
    test_df_scaled = test_data.copy()
    train_df_scaled[FEATURE_COLS] = X_train_scaled
    test_df_scaled[FEATURE_COLS] = X_test_scaled

    X_train_seq, y_train_seq, _ = create_sliding_sequences(train_df_scaled, FEATURE_COLS, WINDOW_SIZE)
    X_test_seq, y_test_seq, _ = create_sliding_sequences(test_df_scaled, FEATURE_COLS, WINDOW_SIZE)

    torch.manual_seed(42)
    np.random.seed(42)
    lstm_model = BatteryLSTM(input_dim=len(FEATURE_COLS), hidden_dim=48, num_layers=2)
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.003, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss()

    dataset = TensorDataset(torch.tensor(X_train_seq), torch.tensor(y_train_seq))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    lstm_model.train()
    for _ in range(65):
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(lstm_model(bx), by)
            loss.backward()
            optimizer.step()

    lstm_model.eval()
    with torch.no_grad():
        lstm_preds = lstm_model(torch.tensor(X_test_seq)).numpy()

    predictions['PyTorch_LSTM_Pred'] = np.round(lstm_preds, 2)
    results.append({
        'Model': 'PyTorch LSTM',
        'Paradigm': 'Recurrent Neural Network',
        'Loss Function': 'Smooth L1 (Sequence Window W=10)',
        'RMSE': round(np.sqrt(mean_squared_error(y_test, lstm_preds)), 2),
        'MAE': round(mean_absolute_error(y_test, lstm_preds), 2),
        'PHM Score': round(phm_score(y_test, lstm_preds), 2)
    })

    results_df = pd.DataFrame(results)
    preds_df = pd.DataFrame(predictions)

    print("\n" + "="*85)
    print(" " * 20 + "FINAL BATTERY MODEL EVALUATION (CELL 18)")
    print("="*85)
    print(results_df.to_markdown(index=False))
    print("="*85 + "\n")

    for d in [PROCESSED_DATA_DIR, BATTERY_PROCESSED_DIR]:
        results_df.to_csv(os.path.join(d, 'final_model_comparison.csv'), index=False)
        results_df.to_csv(os.path.join(d, 'battery_final_model_comparison.csv'), index=False)
        preds_df.to_csv(os.path.join(d, 'battery_18_all_model_predictions.csv'), index=False)

    print(f"Comparison metrics and predictions saved to:\n  {PROCESSED_DATA_DIR}\n  {BATTERY_PROCESSED_DIR}")

if __name__ == '__main__':
    main()
