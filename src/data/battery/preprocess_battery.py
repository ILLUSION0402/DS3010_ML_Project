import os
from pathlib import Path
import scipy.io
import pandas as pd
import numpy as np

def get_project_root():
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / 'data').exists() and (parent / 'src').exists():
            return parent
    return current.parents[3]

PROJECT_ROOT = get_project_root()
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw', 'Battery', '5. Battery Data Set', 'Extracted')
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
BATTERY_PROCESSED_DIR = os.path.join(PROCESSED_DATA_DIR, 'battery')

def process_battery(mat_file, unit_number):
    mat = scipy.io.loadmat(mat_file)
    b_data = mat[f'B000{unit_number}' if unit_number < 10 else f'B00{unit_number}']
    cycles = b_data['cycle'][0, 0][0]
    
    records = []
    discharge_cycle = 1
    
    for c in cycles:
        if c['type'][0] == 'discharge':
            data = c['data'][0, 0]
            voltage, current, temp = data['Voltage_measured'][0], data['Current_measured'][0], data['Temperature_measured'][0]
            capacity = data['Capacity'][0, 0]
            
            records.append({
                'unit_number': unit_number,
                'cycle': discharge_cycle,
                'v_mean': np.mean(voltage), 'v_min': np.min(voltage), 'v_max': np.max(voltage),
                'c_mean': np.mean(current), 'c_min': np.min(current), 'c_max': np.max(current),
                't_mean': np.mean(temp), 't_min': np.min(temp), 't_max': np.max(temp),
                'capacity': capacity
            })
            discharge_cycle += 1
            
    return pd.DataFrame(records)

def main():
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(BATTERY_PROCESSED_DIR, exist_ok=True)
    
    df = pd.concat([process_battery(os.path.join(RAW_DATA_DIR, f'B00{b:02d}.mat'), b) for b in [5, 6, 7, 18]], ignore_index=True)
    
    for unit, group in df.groupby('unit_number'):
        eol_idx = group.index[group['capacity'] < 1.4].min()
        eol_cycle = group.loc[eol_idx, 'cycle'] if not pd.isna(eol_idx) else group['cycle'].max()
        df.loc[group.index, 'target'] = (eol_cycle - group['cycle']).clip(lower=0)
        
    df = df[df['target'] >= 0]
    
    train_df = df[df['unit_number'].isin([5, 6, 7])]
    test_df = df[df['unit_number'] == 18]
    
    # Save to data/processed
    train_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'train_battery.csv'), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'test_battery.csv'), index=False)
    
    # Also save to data/processed/battery
    train_df.to_csv(os.path.join(BATTERY_PROCESSED_DIR, 'train_battery.csv'), index=False)
    test_df.to_csv(os.path.join(BATTERY_PROCESSED_DIR, 'test_battery.csv'), index=False)
    print("Battery dataset successfully preprocessed and saved.")

if __name__ == '__main__':
    main()
