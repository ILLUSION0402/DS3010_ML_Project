import os
from pathlib import Path
import pandas as pd
import numpy as np
import math
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

def get_project_root():
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / 'data').exists() and (parent / 'src').exists():
            return parent
    return current.parents[3]

PROJECT_ROOT = get_project_root()
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
BATTERY_PROCESSED_DIR = os.path.join(PROCESSED_DATA_DIR, 'battery')
TRAIN_PATH = os.path.join(PROCESSED_DATA_DIR, "train_battery.csv")
TEST_PATH = os.path.join(PROCESSED_DATA_DIR, "test_battery.csv")

def phm_score(y_true, y_pred):
    score = 0
    for true, pred in zip(y_true, y_pred):
        d = pred - true
        score += math.exp(d / 13) - 1 if d >= 0 else math.exp(-d / 10) - 1
    return score

def main():
    train_data = pd.read_csv(TRAIN_PATH)
    test_data = pd.read_csv(TEST_PATH)
    
    base_features = ["v_mean", "v_min", "v_max", "c_mean", "c_min", "c_max", "t_mean", "t_min", "t_max", "capacity"]
    feature_cols = [c for c in base_features if train_data[c].nunique() > 5]
    
    y_train = train_data["target"].clip(upper=125)
    y_test = test_data["target"].clip(upper=125)
    
    scalers = {
        "Raw (No Scaling)": None,
        "Standard (Z-Score)": StandardScaler(),
        "MinMax (0 to 1)": MinMaxScaler()
    }
    
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    results = []
    for scaler_name, scaler in scalers.items():
        if scaler is None:
            x_train = train_data[feature_cols]
            x_test = test_data[feature_cols]
        else:
            x_train = pd.DataFrame(scaler.fit_transform(train_data[feature_cols]), columns=feature_cols)
            x_test = pd.DataFrame(scaler.transform(test_data[feature_cols]), columns=feature_cols)
            
        for model_name, model in models.items():
            model.fit(x_train, y_train)
            preds = model.predict(x_test)
            
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            mae = mean_absolute_error(y_test, preds)
            
            results.append({
                "Preprocessing": scaler_name,
                "Model": model_name,
                "RMSE": round(rmse, 2),
                "MAE": round(mae, 2),
                "PHM Score": round(phm_score(y_test, preds), 2)
            })
            
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*70)
    print(" " * 18 + "BATTERY BASELINE COMPARISON RESULTS")
    print("="*70)
    print(results_df.to_markdown(index=False))
    print("="*70 + "\n")
    
    for d in [PROCESSED_DATA_DIR, BATTERY_PROCESSED_DIR]:
        results_df.to_csv(os.path.join(d, "model_comparison_results.csv"), index=False)
    
    best_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    best_model.fit(train_data[feature_cols], y_train)
    final_preds = best_model.predict(test_data[feature_cols])
    
    pred_df = pd.DataFrame({
        'cycle': test_data['cycle'],
        'true_RUL': y_test,
        'predicted_RUL': np.round(final_preds, 1)
    })
    
    for d in [PROCESSED_DATA_DIR, BATTERY_PROCESSED_DIR]:
        pred_df.to_csv(os.path.join(d, "battery_18_predictions.csv"), index=False)
        
    print("Battery baseline comparisons completed and saved.")

if __name__ == "__main__":
    main()
