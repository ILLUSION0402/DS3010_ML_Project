import os
import scipy.io
import pandas as pd
import numpy as np

RAW_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'Battery', '5. Battery Data Set', 'Extracted'))
PROCESSED_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed'))

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
                'unit_number': unit_number, 'cycle': discharge_cycle,
                'v_mean': np.mean(voltage), 'v_min': np.min(voltage), 'v_max': np.max(voltage),
                'c_mean': np.mean(current), 'c_min': np.min(current), 'c_max': np.max(current),
                't_mean': np.mean(temp), 't_min': np.min(temp), 't_max': np.max(temp),
                'capacity': capacity
            })
            discharge_cycle += 1
            
    return pd.DataFrame(records)

def main():
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    df = pd.concat([process_battery(os.path.join(RAW_DATA_DIR, f'B00{b:02d}.mat'), b) for b in [5, 6, 7, 18]], ignore_index=True)
    
    for unit, group in df.groupby('unit_number'):
        eol_idx = group.index[group['capacity'] < 1.4].min()
        eol_cycle = group.loc[eol_idx, 'cycle'] if not pd.isna(eol_idx) else group['cycle'].max()
        df.loc[group.index, 'target'] = (eol_cycle - group['cycle']).clip(lower=0)
        
    df = df[df['target'] >= 0]
    
    df[df['unit_number'].isin([5, 6, 7])].to_csv(os.path.join(PROCESSED_DATA_DIR, 'train_battery.csv'), index=False)
    df[df['unit_number'] == 18].to_csv(os.path.join(PROCESSED_DATA_DIR, 'test_battery.csv'), index=False)

if __name__ == '__main__':
    main()
