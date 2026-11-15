"""NASA IMS bearing diagnostics: alert lead time, censored-bearing warnings, ablations.

Place beside bearings_future_work.py, bearings_improvement.py and bearing_predictor2.py.
- Uses frozen out-of-fold failure predictions from the tuning experiment.
- Fits out-of-fold baseline/tuned models to ALL 12 bearings for warning analysis.
- Uses the same training experiment partitions and preselected tuning choices.
- Treats non-failed bearings as 'no documented failure', NOT proof of health.
- Uses retrospective proxy health target ONLY for the four documented failures.

Run from project folder:
  python bearings_diagnostics.py --data-root bearings --n-jobs 4
"""
from __future__ import annotations
import argparse
from datetime import datetime
from pathlib import Path
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.metrics import mean_squared_error

import bearings_future_work as fw
import bearings_improvement as bi
from bearing_predictor2 import _file_features_multi

CHANNELS = {
    '1st_test': {'b1':[0,1], 'b2':[2,3], 'b3':[4,5], 'b4':[6,7]},
    '2nd_test': {'b1':[0], 'b2':[1], 'b3':[2], 'b4':[3]},
    '3rd_test': {'b1':[0], 'b2':[1], 'b3':[2], 'b4':[3]},
}
FAILED = {'1st_test_b3','1st_test_b4','2nd_test_b1','3rd_test_b3'}
FAULTS = ['BPFO','BPFI','BSF','FTF']
FEATURE_GROUPS = {
    'time_domain':['rms','kurt','p2p','crest','std','skew','impulse','margin'],
    'spectrum':['centroid','spec_std']+[f'band{i}' for i in range(8)],
    'fault_frequency':[f'raw_{k}' for k in FAULTS]+[f'env_{k}' for k in FAULTS],
}


def _snapshot_files(folder):
    return sorted((p for p in Path(folder).rglob('*')
                   if p.is_file() and p.name.count('.') == 5), key=lambda p: p.name)


def load_all_bearings(raw_fail, data_root, cache_dir, n_jobs=4):
    """Only extract the other 8 channels; reuse already cached failed bearing features.

    Expects the NASA IMS files locally at data_root/<test>/YYYY.MM.DD.HH.MM.SS.
    Each test's extracted features are cached separately to support reruns.
    """
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    chunks = [raw_fail.copy()]
    for test, mapping in CHANNELS.items():
        pending = {b: ch for b,ch in mapping.items() if f'{test}_{b}' not in FAILED}
        target = cache_dir / f'{test}_additional_bearings.pkl'
        if target.exists():
            print('Reusing:', target, flush=True)
            extra = pd.read_pickle(target)
        else:
            files = _snapshot_files(Path(data_root)/test)
            if not files:
                raise FileNotFoundError(f'No timestamp vibration files in {Path(data_root)/test}. '
                                        'Point DATA_ROOT to your extracted NASA IMS folder.')
            print(f'Extracting {test}: {len(files)} snapshots, bearings {list(pending)}', flush=True)
            result = Parallel(n_jobs=n_jobs, verbose=0)(
                delayed(_file_features_multi)(str(p), pending) for p in files)
            errors = [msg for obj,msg in result if obj is None]
            if errors:
                raise RuntimeError(f'{test}: {len(errors)} unusable snapshots; first: {errors[0]}')
            rows = []
            for path,(values,_) in zip(files,result):
                ts = datetime.strptime(path.name,'%Y.%m.%d.%H.%M.%S')
                for b, features in values.items():
                    rows.append({'test':test,'bearing':b,'run':f'{test}_{b}',
                                 'timestamp':ts, **features})
            extra = pd.DataFrame(rows)
            extra.to_pickle(target)
            print('Saved:', target, flush=True)
        chunks.append(extra)
    all_raw = pd.concat(chunks, ignore_index=True)
    all_raw['timestamp'] = pd.to_datetime(all_raw['timestamp'])
    all_raw = all_raw.sort_values(['run','timestamp']).reset_index(drop=True)
    if set(all_raw.run.unique()) != {f'{test}_{b}' for test, bs in CHANNELS.items() for b in bs}:
        raise ValueError('Expected 12 unique bearing trajectories; inspect cache and raw data.')
    if all_raw.duplicated(['run','timestamp']).any():
        raise ValueError('Duplicate time stamps; the feature cache does not match this dataset.')
    return all_raw


def sustained_alarms(t, p, threshold=.9, persistence=6, warm=70.0):
    """Return beginning time of each non-overlapping sustained warning episode.

    Alarm requires persistence consecutive readings below threshold, after the
    healthy-reference interval. Start time is the CONFIRMATION time (causal).
    """
    t = np.asarray(t, dtype=float); p = np.asarray(p, dtype=float)
    bad = (p < threshold) & (t >= warm)
    starts=[]; streak=0; in_alarm=False
    for i, flagged in enumerate(bad):
        if flagged:
            streak += 1
            if streak >= persistence and not in_alarm:
                starts.append(float(t[i])); in_alarm=True
        else:
            streak=0; in_alarm=False
    return starts


def assess_alarms(frame, thresholds=(.9,.8,.7), persistence=6):
    """No-failure cases are right-censored; no measured RUL is assigned."""
    rows=[]
    for (run,model), g in frame.groupby(['run','model'], sort=True):
        g=g.sort_values('t_hours')
        failed=run in FAILED
        end=float(g.t_hours.iloc[-1])
        for threshold in thresholds:
            alert=sustained_alarms(g.t_hours,g.health_prediction,threshold,persistence)
            # For failures, end is known run-to-failure endpoint (retrospective only).
            # For others, end is merely the last observation (right-censored).
            rows.append({'run':run,'test':g.test.iloc[0],'model':model,
                         'documented_failure':failed,'threshold':threshold,
                         'alerts':len(alert), 'first_alert_hours':alert[0] if alert else np.nan,
                         'alerted':bool(alert),
                         'lead_hours_if_failed':end-alert[0] if failed and alert else np.nan,
                         'hours_observed':end})
    return pd.DataFrame(rows)


def make_out_of_fold_predictions(all_raw, choices, trees=500):
    """Fit on failed-bearing tracks from other experiments; score every bearing.

    Configurations are read from the preceding nested-validation experiment,
    NOT chosen using these held-out predictions or healthy-bearing outcomes.
    """
    prepared = fw.prepare(all_raw)
    prepared, history_cols = bi.causal_history(prepared)
    base_cols=list(fw.COLS)
    out=[]
    for held in sorted(CHANNELS):
        train=prepared[(prepared.test != held)&(prepared.run.isin(FAILED))]
        test=prepared[prepared.test == held]
        if not len(train) or not len(test):
            raise ValueError('Missing held-out/test experiment data')
        selected=choices[choices.outer_test == held]
        if len(selected)!=1: raise ValueError(f'Missing/ambiguous configuration for {held}')
        entry=selected.iloc[0]
        cfg={k: entry[k] for k in bi.configurations()[0]}
        basecfg=bi.configurations()[0]
        for model_name,config,cols in [
            ('ExtraTrees_baseline',basecfg,base_cols),
            ('ExtraTrees_tuned',cfg,base_cols if cfg['feature_set']=='base' else history_cols)
        ]:
            model=bi.train_model(train,cols,config,trees)
            for run,g in test.groupby('run'):
                p=bi.predict_trajectory(model,g,cols,config)
                out.append(pd.DataFrame({'run':run,'test':held,'bearing':g.bearing.to_numpy(),
                    'timestamp':g.timestamp.to_numpy(),'t_hours':g.t_hours.to_numpy(),
                    'model':model_name,'health_prediction':p,
                    'documented_failure':run in FAILED}))
        print(f'Finished held-out experiment {held}',flush=True)
    return pd.concat(out,ignore_index=True),prepared


def ablation_study(failed_features, choices, trees=250):
    """Train frozen-configuration models with feature families removed.

    This is an explanatory experiment, not inner tuned ablation; results must
    not be used for additional model selection on these held-out folds.
    """
    f=failed_features[failed_features.run.isin(FAILED)].copy()
    base_cols=list(fw.COLS)
    groups={key: [x for c in value for x in [c,c+'_slope']]
            for key,value in FEATURE_GROUPS.items()}
    variants={'all_52':base_cols}
    for group,excluded in groups.items():
        variants['without_'+group]=[c for c in base_cols if c not in excluded]
        variants['only_'+group]=excluded
    rows=[]
    for held in sorted(CHANNELS):
        train=f[f.test!=held]; test=f[f.test==held]
        entry=choices[choices.outer_test==held].iloc[0]
        cfg={k:entry[k] for k in bi.configurations()[0]}
        # Use selected leaf/weights/smoothing but original 52 features for clean ablation.
        for variant,columns in variants.items():
            model=bi.train_model(train,columns,cfg,trees)
            for run,g in test.groupby('run'):
                pred=bi.predict_trajectory(model,g,columns,cfg)
                rows.append({'outer_test':held,'run':run,'variant':variant,
                             **fw.metrics(g.target.to_numpy(),pred,g.t_hours.to_numpy())})
        print(f'Finished ablations for {held}',flush=True)
    return pd.DataFrame(rows)


def plot_hard_bearing(prepared, predictions, output_dir, run='3rd_test_b3'):
    g=prepared[prepared.run==run].sort_values('t_hours')
    p=predictions[(predictions.run==run)&(predictions.model=='ExtraTrees_tuned')].sort_values('t_hours')
    if len(g)!=len(p):raise ValueError('Feature and prediction timestamps misaligned')
    fig,axes=plt.subplots(3,1,figsize=(11,10),sharex=True)
    axes[0].plot(g.t_hours, g.target, '--', label='Retrospective reference')
    axes[0].plot(p.t_hours,p.health_prediction,label='Out-of-fold tuned prediction')
    axes[0].legend();axes[0].set_ylabel('Health index')
    for c in ('rms','kurt','p2p'):
        axes[1].plot(g.t_hours,g[c],label=c)
    axes[1].legend();axes[1].set_ylabel('Baseline-normalized features')
    for c in ('env_BPFO','env_BPFI','env_BSF','env_FTF'):
        axes[2].plot(g.t_hours,g[c],label=c)
    axes[2].legend();axes[2].set_ylabel('Envelope frequency features')
    axes[2].set_xlabel('Operating hours');fig.suptitle(run+' — hard case diagnostics')
    fig.tight_layout();fig.savefig(Path(output_dir)/'hard_bearing_diagnostics.png',dpi=140)
    plt.close(fig)


def plot_alerts(predictions, alarms, output_dir):
    for test in sorted(CHANNELS):
        fig,axes=plt.subplots(4,1,figsize=(10,10),sharex=True)
        for ax,b in zip(axes,CHANNELS[test]):
            run=f'{test}_{b}'
            for model in ['ExtraTrees_baseline','ExtraTrees_tuned']:
                p=predictions[(predictions.run==run)&(predictions.model==model)]
                ax.plot(p.t_hours,p.health_prediction,label=model.replace('ExtraTrees_',''))
            ax.axhline(.9,color='gray',ls=':',lw=1)
            aa=alarms[(alarms.run==run)&(alarms.model=='ExtraTrees_tuned')&
                      (np.isclose(alarms.threshold,.9))]
            if len(aa) and bool(aa.iloc[0].alerted):
                ax.axvline(float(aa.iloc[0].first_alert_hours),color='red',ls='--',lw=1)
            ax.set_ylabel(b+' health');ax.set_ylim(-.05,1.05)
            ax.set_title('documented failure' if run in FAILED else 'no documented failure', fontsize=9)
        axes[-1].set_xlabel('Operating hours')
        axes[0].legend(loc='lower left',fontsize=8)
        fig.suptitle(test+': held-out bearing predictions / confirmed warnings')
        fig.tight_layout();fig.savefig(Path(output_dir)/f'{test}_all_bearings.png',dpi=140)
        plt.close(fig)


def run_diagnostics(raw_fail, data_root='bearings', improvement_dir='bearings_out/improvement',
                    out_dir='bearings_out/diagnostics', n_jobs=4, trees=500, ablation_trees=250):
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    choices_path=Path(improvement_dir)/'selected_configurations.csv'
    if not choices_path.exists():
        raise FileNotFoundError(f'Missing {choices_path}; run bearings_improvement.ipynb first.')
    choices=pd.read_csv(choices_path)
    all_raw=load_all_bearings(raw_fail,data_root,out/'raw_feature_cache',n_jobs)
    predictions, prepared=make_out_of_fold_predictions(all_raw,choices,trees)
    predictions.to_csv(out/'all_bearing_predictions.csv',index=False)
    alarms=assess_alarms(predictions)
    alarms.to_csv(out/'alert_evaluation.csv',index=False)
    grouped=alarms.groupby(['model','threshold','documented_failure'],as_index=False).agg(
        n_bearings=('run','size'),n_bearings_alerted=('alerted','sum'),
        mean_lead_hours=('lead_hours_if_failed','mean'))
    grouped['fraction_bearings_alerted']=grouped.n_bearings_alerted/grouped.n_bearings
    grouped.to_csv(out/'alert_summary.csv',index=False)
    ablation=ablation_study(prepared,choices,trees=ablation_trees)
    ablation.to_csv(out/'feature_ablation_per_run.csv',index=False)
    agg=ablation.groupby('variant')[['RMSE','RMSE_late','late_overestimate_mean']].mean()
    agg.to_csv(out/'feature_ablation_summary.csv')
    plot_alerts(predictions,alarms,out)
    plot_hard_bearing(prepared,predictions,out)
    (out/'READ_ME.txt').write_text(
        'Warnings are generated from models trained on other IMS experiments.\n'
        'Without documented failure means right-censored, NOT known healthy.\n'
        'Lead hours = time from first sustained prediction-based warning to known final failure timestamp.\n'
        'Thresholds (.9, .8, .7) and persistence (6) were fixed in advance; do not cherry-pick on test data.\n'
        'The failure reference health index is a retrospective proxy, not a measured RUL target.\n'
        'Ablations are explanatory; do not tune/final-select on these outer test outcomes.\n'
        'The 12 channels share 3 rigs/tests; they are NOT 12 independent machines.\n')
    print('\nALERT SUMMARY\n',grouped.to_string(index=False))
    print('\nABLATION SUMMARY\n',agg.round(4).to_string())
    print('\nFiles saved to',out.resolve())
    return {'predictions':predictions,'alerts':alarms,'alert_summary':grouped,
            'ablation':ablation,'ablation_summary':agg,'features':prepared}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root',default='bearings')
    ap.add_argument('--cache',default='bearings_out/features_raw.pkl')
    ap.add_argument('--improvement-dir',default='bearings_out/improvement')
    ap.add_argument('--out',default='bearings_out/diagnostics')
    ap.add_argument('--n-jobs',type=int,default=4)
    ap.add_argument('--trees',type=int,default=500)
    ap.add_argument('--ablation-trees',type=int,default=250)
    args=ap.parse_args()
    raw=pd.read_pickle(args.cache)
    run_diagnostics(raw,args.data_root,args.improvement_dir,args.out,
                    args.n_jobs,args.trees,args.ablation_trees)

if __name__=='__main__':main()
