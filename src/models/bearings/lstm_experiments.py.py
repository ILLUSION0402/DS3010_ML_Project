"""IMS bearings: Extra Trees vs LSTM (MSE vs asymmetric loss).

Uses bearings_out/features_raw.pkl created by bearings_pipeline(2).ipynb Cell 4.
Validation is leave-one-TEST-SET-out, matching the original notebook.
Features never use a future observation; healthy reference is hours 40--70.
Requires numpy, pandas, scipy, scikit-learn, matplotlib, torch, joblib.

python bearings_future_work.py --cache bearings_out/features_raw.pkl --epochs 20
"""
from __future__ import annotations
import argparse
from pathlib import Path
import copy
import json
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FAULTS = ['BPFO', 'BPFI', 'BSF', 'FTF']
CORE = (['rms','kurt','p2p','crest','std','skew','impulse','margin','centroid','spec_std']
        + [f'band{i}' for i in range(8)]
        + [f'raw_{k}' for k in FAULTS] + [f'env_{k}' for k in FAULTS])
COLS = CORE + [x + '_slope' for x in CORE]
INDICATORS = ['rms','kurt','env_BPFO','env_BPFI','env_BSF','band5','band6','band7']
BASE_START, BASE_END = 40.0, 70.0
WARMUP = 40.0
SEED = 42


def prepare(raw):
    """Match v2 notebook's target and 52 features. No manual onset overrides."""
    parts = []
    for run, g in raw.groupby('run', sort=False):
        g = g.sort_values('timestamp').copy().reset_index(drop=True)
        dt = pd.to_datetime(g.timestamp).diff().dt.total_seconds().fillna(0).clip(0, 600)
        g['t_hours'] = dt.cumsum() / 3600.0
        bmask = (g.t_hours >= BASE_START) & (g.t_hours < BASE_END)
        if bmask.sum() < 10:
            raise ValueError(f'{run}: fewer than ten baseline snapshots between 40--70 h')
        f = pd.DataFrame(index=g.index)
        for c in CORE:
            base = g.loc[bmask, c].astype(float)
            v = g[c].astype(float)
            # Choose transform using baseline observations ONLY (not future values).
            if (base > 0).all():
                z = np.log(v.clip(lower=1e-12) / (base.median() + 1e-12))
            else:
                z = (v - base.median()) / (base.std() + 1e-12)
            z = z.rolling(5, min_periods=1).median().ewm(span=10, adjust=False).mean()
            f[c] = z
            f[c+'_slope'] = z.diff(12).fillna(0)
        # Original v2 onset logic (threshold computed using only healthy baseline).
        candidates = []
        t_last_baseline = g.loc[bmask, 't_hours'].max()
        for c in INDICATORS:
            b = f.loc[bmask, c]
            thr = b.mean() + 4*b.std() + 0.1
            exceed = ((f[c] > thr).astype(int).rolling(6).sum() >= 6) & (g.t_hours > t_last_baseline)
            if exceed.any():
                pos = int(np.flatnonzero(exceed.to_numpy())[0])
                candidates.append(float(g.t_hours.iloc[max(0, pos-5)]))
        candidates.sort()
        onset = candidates[1] if len(candidates) > 1 else (candidates[0] if candidates else 0.0)
        end = float(g.t_hours.iloc[-1])
        # Pseudo ground truth: uses known failure time. DO NOT use as online input feature.
        y = np.clip((end - g.t_hours.to_numpy()) / max(end - onset, 1e-6), 0, 1)
        out = pd.concat([g[['run','test','bearing','timestamp','t_hours']], f], axis=1)
        out['target'] = y
        out['onset_hours'] = onset
        out[COLS] = out[COLS].replace([np.inf,-np.inf], np.nan).fillna(0)
        parts.append(out)
    return pd.concat(parts, ignore_index=True)


class WindowDataset(Dataset):
    """One sample per end timestamp; pad beginning with first observed vector."""
    def __init__(self, runs, length=36, stride=1):
        self.items = []
        self.length = length
        for _, g in runs:
            x = g['_scaled'].to_list()  # stored as vectors, without future information
            X = np.asarray(x, dtype=np.float32)
            y = g['target'].to_numpy(dtype=np.float32)
            self.items.append((X, y))
        self.indices = [(i, j) for i, (x,y) in enumerate(self.items)
                        for j in range(0, len(y), stride)]

    def __len__(self): return len(self.indices)

    def __getitem__(self, index):
        i,j = self.indices[index]
        X,y = self.items[i]
        a = max(0, j-self.length+1)
        w = X[a:j+1]
        if len(w) < self.length:
            w = np.concatenate([np.repeat(X[:1], self.length-len(w), axis=0), w])
        return torch.from_numpy(w), torch.tensor(y[j], dtype=torch.float32)


class LSTMRegressor(nn.Module):
    def __init__(self, features, hidden=64, layers=2, dropout=0.15):
        super().__init__()
        self.lstm = nn.LSTM(features, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden,32), nn.ReLU(), nn.Linear(32,1), nn.Sigmoid())
    def forward(self, x):
        x,_ = self.lstm(x)
        return self.head(x[:,-1]).squeeze(-1)


def loss_fn(pred, truth, mode='mse', late_boost=2.0):
    e = pred - truth
    if mode == 'mse': return e.square().mean()
    # When predicted RUL > real RUL: potentially dangerous late maintenance.
    # PHM08-inspired directional asymmetry, adapted to a 0--1 health index.
    # Weight late-life samples more strongly to address late-stage RMSE.
    weight = torch.where(e > 0, 3.0, 1.0)
    weight = weight * torch.where(truth < 0.5, late_boost, 1.0)
    return (weight * e.square()).mean()


def postprocess(pred, hours, warm=40, win=25, alpha=0.1, monotone=True):
    pred = np.asarray(pred, dtype=float)
    hours = np.asarray(hours, dtype=float)
    out = np.ones(len(pred)); mask = hours >= warm
    if mask.any():
        s = pd.Series(pred[mask]).rolling(win, min_periods=1).median().ewm(alpha=alpha,adjust=False).mean().to_numpy()
        out[mask] = np.minimum.accumulate(s) if monotone else s
    return np.clip(out, 0, 1)


def metrics(y,p,t):
    y,p,t = map(np.asarray,(y,p,t))
    late = y < .5
    healthy = (y >= .999999) & (t >= 40)
    return {'RMSE': float(np.sqrt(np.mean((p-y)**2))),
            'MAE': float(np.mean(np.abs(p-y))),
            'NRMSE': float(np.sqrt(np.mean((p-y)**2))), # target has fixed range 0--1
            'RMSE_late': float(np.sqrt(np.mean((p[late]-y[late])**2))) if late.any() else np.nan,
            'late_overestimate_mean': float(np.maximum(p[late]-y[late],0).mean()) if late.any() else np.nan,
            'false_alarm_rate': float((p[healthy] < .9).mean()) if healthy.any() else np.nan}


def train_lstm(train_groups, scaler, args, mode, device):
    groups=[]
    for run,g in train_groups:
        z=g.copy()
        z['_scaled'] = list(scaler.transform(z[COLS].to_numpy()))
        groups.append((run,z))
    ds=WindowDataset(groups, length=args.seq_len, stride=args.train_stride)
    loader=DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model=LSTMRegressor(len(COLS), hidden=args.hidden).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    for epoch in range(args.epochs):
        model.train(); total=0.; count=0
        for x,y in loader:
            x,y=x.to(device),y.to(device)
            opt.zero_grad()
            pred=model(x)
            loss=loss_fn(pred,y,mode,args.late_boost)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
            total += loss.item()*len(y); count+=len(y)
        if (epoch+1) % max(1,args.epochs//4)==0:
            print(f'  {mode}: epoch {epoch+1}/{args.epochs}; training loss {total/count:.5f}',flush=True)
    return model


@torch.no_grad()
def infer_lstm(model,g,scaler,seq_len,device,batch_size):
    z=g.copy()
    z['_scaled']=list(scaler.transform(g[COLS].to_numpy()))
    ds=WindowDataset([(str(g.run.iloc[0]),z)],length=seq_len,stride=1)
    loader=DataLoader(ds,batch_size=batch_size,shuffle=False)
    model.eval()
    return np.concatenate([model(x.to(device)).cpu().numpy() for x,_ in loader])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache',default='bearings_out/features_raw.pkl')
    p.add_argument('--out',default='bearings_out/future_work')
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--seq-len',type=int,default=36,help='number of snapshots per sample (~6 hours at 10 min intervals)')
    p.add_argument('--train-stride',type=int,default=3)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--hidden',type=int,default=64)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--late-boost',type=float,default=2.)
    p.add_argument('--device',default='auto',choices=['auto','cpu','cuda'])
    args=p.parse_args()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if not Path(args.cache).exists():
        raise FileNotFoundError(f'{args.cache} not found. Run cells 0--4 of bearings_pipeline(2).ipynb first.')
    raw=pd.read_pickle(args.cache)
    needed=set(CORE)|{'run','test','bearing','timestamp'}
    if missing:=needed-set(raw.columns): raise ValueError(f'Feature cache missing columns: {sorted(missing)}')
    f=prepare(raw).sort_values(['run','timestamp']).reset_index(drop=True)
    tests=sorted(f.test.unique())
    if len(tests)<3: raise ValueError('Need at least three independent test sets for leave-one-test-set-out evaluation')
    print('Runs:',f.groupby('run').size().to_dict(),'\nHeld-out test folders:',tests,flush=True)
    device=torch.device('cuda' if args.device=='auto' and torch.cuda.is_available() else ('cpu' if args.device=='auto' else args.device))
    print('Device:',device,flush=True)
    rows=[]; curves={}
    for held in tests:
        train=f[f.test!=held]
        test=f[f.test==held]
        print(f'\nHELD OUT: {held}; training on {sorted(train.test.unique())}',flush=True)
        scaler=StandardScaler().fit(train[COLS].to_numpy()) # only training trajectories
        train_groups=list(train.groupby('run'))
        et=ExtraTreesRegressor(n_estimators=500,min_samples_leaf=50,max_features=.3,n_jobs=-1,random_state=0)
        et.fit(train[COLS],train.target)
        models={mode:train_lstm(train_groups,scaler,args,mode,device) for mode in ['mse','asymmetric']}
        for run,g in test.groupby('run'):
            y=g.target.to_numpy();t=g.t_hours.to_numpy()
            predictions={'ExtraTrees':et.predict(g[COLS])}
            for mode, model in models.items():
                predictions['LSTM_'+mode]=infer_lstm(model,g,scaler,args.seq_len,device,args.batch_size)
            curves[run]={'t':t,'y':y,'pred':{}}
            for name,pred in predictions.items():
                pp=postprocess(pred,t)
                curves[run]['pred'][name]=pp
                rows.append({'held_test':held,'run':run,'model':name,**metrics(y,pp,t)})
            print('  evaluated',run,flush=True)
    df=pd.DataFrame(rows)
    df.to_csv(out/'per_run_metrics.csv',index=False)
    summary=df.groupby('model',sort=False)[['RMSE','MAE','NRMSE','RMSE_late','late_overestimate_mean','false_alarm_rate']].mean()
    summary.to_csv(out/'summary_metrics.csv')
    print('\nMEAN OF 4 HELD-OUT BEARING ERRORS\n',summary.round(4).to_string())
    for run, d in curves.items():
        fig,ax=plt.subplots(figsize=(9,4))
        ax.plot(d['t'],d['y'],label='target (auto onset)',linewidth=2)
        for name,pred in d['pred'].items(): ax.plot(d['t'],pred,label=name,alpha=.8)
        ax.set(xlabel='operating hours',ylabel='health index (0--1)',title=run,ylim=(-.05,1.05))
        ax.legend();fig.tight_layout();fig.savefig(out/f'{run}_comparison.png',dpi=150);plt.close(fig)
    config=vars(args).copy();config['notes']='Only 4 failed bearings from 3 experiments; labels are pseudo-targets based on end-of-run failure and auto onset. No validation or external generalization claimed.'
    (out/'experiment_config.json').write_text(json.dumps(config,indent=2))
    print('\nSaved metrics and plots in',out.resolve())
    print('Do NOT compare 0.418 from the slide directly unless target/onset scoring matches exactly.')
    print('Normalized PHM08-style score is not a standardized PHM08 score; asymmetry is evaluated through signed late overestimate.')

if __name__=='__main__':main()
