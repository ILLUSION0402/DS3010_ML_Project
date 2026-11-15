"""IMS Bearings: leakage-conscious Extra Trees improvement experiment.

Run in same folder as bearings_future_work.py:
  python bearings_improvement.py --cache bearings_out/features_raw.pkl

Compares original Extra Trees versus nested-validation-selected:
  * causal rolling vibration-history features
  * sample weighting of the late degradation stage
  * leaf size / feature subsampling
  * prediction smoothing

Outer holdout = one IMS test experiment; selection uses OTHER experiments only.
Uses the same retrospective proxy health labels as bearings_future_work.py.
Not a physically validated true remaining-life predictor.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
import joblib

from bearings_future_work import prepare, COLS, postprocess, metrics

HISTORY_FEATURES = ['rms', 'kurt', 'p2p', 'crest', 'env_BPFO', 'env_BPFI', 'env_BSF', 'band6']
ROLL_WINDOWS = (6, 24)  # ~1 and ~4 hours, at ten-minute intervals


def causal_history(frame):
    """Build lagged stats separately per run; no future snapshots used."""
    chunks = []
    for _, g in frame.groupby('run', sort=False):
        g = g.sort_values('timestamp').copy()
        for c in HISTORY_FEATURES:
            for w in ROLL_WINDOWS:
                past = g[c].rolling(w, min_periods=1)
                g[f'{c}_mean_{w}'] = past.mean()
                g[f'{c}_std_{w}'] = past.std().fillna(0.0)
        chunks.append(g)
    result = pd.concat(chunks, ignore_index=True)
    result = result.sort_values(['run', 'timestamp']).reset_index(drop=True)
    new_cols = [c for c in result.columns if any(c.endswith(f'_{stat}_{w}')
                for stat in ('mean', 'std') for w in ROLL_WINDOWS)]
    return result, list(COLS) + [c for c in new_cols if c not in COLS]


def build_model(cfg, n_estimators=180, seed=42):
    return ExtraTreesRegressor(n_estimators=n_estimators,
                               min_samples_leaf=cfg['leaf'],
                               max_features=cfg['max_features'],
                               n_jobs=-1, random_state=seed)


def sample_weights(y, late_weight):
    # y is a retrospective proxy health index; 0 = failure, 1 = healthy.
    # Increase weight late in life instead of changing target label.
    return np.where(np.asarray(y) < 0.5, late_weight, 1.0)


def predict_trajectory(model, g, columns, cfg):
    p = model.predict(g[columns])
    return postprocess(p, g.t_hours.to_numpy(), warm=40,
                       win=cfg['win'], alpha=cfg['alpha'], monotone=True)


def evaluate_runs(model, frame, columns, cfg):
    rows = []
    for run, g in frame.groupby('run', sort=True):
        y = g.target.to_numpy()
        p = predict_trajectory(model, g, columns, cfg)
        rows.append({'run': run, **metrics(y, p, g.t_hours.to_numpy())})
    return pd.DataFrame(rows)


def objective(table, lambda_late=1.5, lambda_alarm=0.25):
    """Training-only model selection. Penalize late error and early alarms.

    Equal weight per failed bearing, rather than per snapshot, avoids one long
    trajectory dominating validation.
    """
    late = table.RMSE_late.dropna()
    loss = table.RMSE.mean() + lambda_late * (late.mean() if len(late) else 0)
    alarms = table.false_alarm_rate.dropna()
    return float(loss + lambda_alarm * (alarms.mean() if len(alarms) else 0))


def configurations():
    # Small predetermined grid to avoid overfitting the tiny 3-experiment dataset.
    return [
        dict(feature_set='base', leaf=50, max_features=.3, late_weight=1, win=25, alpha=.1),
        dict(feature_set='history', leaf=50, max_features=.3, late_weight=1, win=25, alpha=.1),
        dict(feature_set='base', leaf=50, max_features=.3, late_weight=3, win=25, alpha=.1),
        dict(feature_set='history', leaf=50, max_features=.3, late_weight=3, win=25, alpha=.1),
        dict(feature_set='history', leaf=20, max_features=.5, late_weight=2, win=15, alpha=.2),
        dict(feature_set='history', leaf=20, max_features=.5, late_weight=4, win=15, alpha=.2),
        dict(feature_set='history', leaf=75, max_features=.3, late_weight=3, win=10, alpha=.2),
        dict(feature_set='base', leaf=20, max_features=.5, late_weight=2, win=15, alpha=.2),
    ]


def train_model(train, cols, cfg, trees):
    model = build_model(cfg, trees)
    model.fit(train[cols], train.target, sample_weight=sample_weights(train.target, cfg['late_weight']))
    return model


def choose_config(train, columns_base, columns_history, trees):
    """Inner leave-one-EXPERIMENT-out validation using outer TRAIN only."""
    experiments = sorted(train.test.unique())
    if len(experiments) < 2:
        raise ValueError('Need at least two independent experiments for inner validation')
    results = []
    for idx, cfg in enumerate(configurations()):
        cols = columns_base if cfg['feature_set'] == 'base' else columns_history
        tables = []
        for val_exp in experiments:
            fitting = train[train.test != val_exp]
            validation = train[train.test == val_exp]
            model = train_model(fitting, cols, cfg, trees)
            tables.append(evaluate_runs(model, validation, cols, cfg))
        checked = pd.concat(tables, ignore_index=True)
        results.append({'candidate': idx, 'score': objective(checked),
                        'inner_RMSE': checked.RMSE.mean(),
                        'inner_RMSE_late': checked.RMSE_late.mean(),
                        'inner_false_alarm': checked.false_alarm_rate.mean(),
                        **cfg})
    scores = pd.DataFrame(results).sort_values('score').reset_index(drop=True)
    return scores, configurations()[int(scores.iloc[0].candidate)]


def run_experiment(raw, output_dir, inner_trees=150, final_trees=500):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    f = prepare(raw)
    f, all_cols = causal_history(f)
    original_cols = list(COLS)
    experiments = sorted(f.test.unique())
    if len(experiments) < 3:
        raise ValueError('Need at least 3 experiments for independent inner/outer validation')
    per_run, selections, candidate_tables, predictions = [], [], [], []
    base_cfg = configurations()[0]
    for held in experiments:
        train, test = f[f.test != held], f[f.test == held]
        print(f'Outer test {held}; inner tuning only on {sorted(train.test.unique())}', flush=True)
        candidates, best = choose_config(train, original_cols, all_cols, inner_trees)
        candidates.insert(0, 'outer_test', held)
        candidate_tables.append(candidates)
        selections.append({'outer_test': held, **best})
        chosen_cols = original_cols if best['feature_set'] == 'base' else all_cols
        baseline = train_model(train, original_cols, base_cfg, final_trees)
        tuned = train_model(train, chosen_cols, best, final_trees)
        for run, g in test.groupby('run'):
            y = g.target.to_numpy()
            t = g.t_hours.to_numpy()
            pred = {}
            for name, model, cols, cfg in (
                    ('ExtraTrees_baseline', baseline, original_cols, base_cfg),
                    ('ExtraTrees_tuned', tuned, chosen_cols, best)):
                pp = predict_trajectory(model, g, cols, cfg)
                pred[name] = pp
                per_run.append({'outer_test': held, 'run': run, 'model': name,
                                **metrics(y, pp, t)})
            predictions.append(pd.DataFrame({'outer_test': held, 'run': run,
                'timestamp': g.timestamp.to_numpy(), 't_hours': t, 'target': y,
                **pred}))

    per_run = pd.DataFrame(per_run)
    per_run.to_csv(output_dir/'per_run_metrics.csv', index=False)
    summary = per_run.groupby('model')[['RMSE', 'MAE', 'NRMSE', 'RMSE_late',
            'late_overestimate_mean','false_alarm_rate']].mean()
    summary.to_csv(output_dir/'summary_metrics.csv')
    pd.DataFrame(selections).to_csv(output_dir/'selected_configurations.csv', index=False)
    pd.concat(candidate_tables, ignore_index=True).to_csv(output_dir/'inner_validation.csv', index=False)
    pred_df = pd.concat(predictions, ignore_index=True)
    pred_df.to_csv(output_dir/'outer_test_predictions.csv', index=False)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for run, g in pred_df.groupby('run'):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(g.t_hours, g.target, 'k--', label='Reference proxy health', lw=1.7)
        ax.plot(g.t_hours, g.ExtraTrees_baseline, label='Original Extra Trees')
        ax.plot(g.t_hours, g.ExtraTrees_tuned, label='Tuned Extra Trees')
        ax.set(xlabel='Operating hours', ylabel='Health index (0–1)',
               title=f'{run}: fully held-out predictions', ylim=(-.05, 1.05))
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir/f'{run}_tuned_vs_baseline.png', dpi=140)
        plt.close(fig)

    # OPTIONAL deployment-oriented artifacts; trained on ALL data for later inference,
    # never use these to calculate the outer test metrics above.
    # Since configs differ by outer fold, choose the most frequent selected configuration,
    # breaking ties in a deterministic, documented way. This is exploratory only.
    # Deterministic most-common configuration across the outer folds.
    cfg_ids = [next(i for i, c in enumerate(configurations())
                    if all(c[k] == selected[k] for k in c))
               for selected in selections]
    counts = np.bincount(cfg_ids, minlength=len(configurations()))
    chosen = configurations()[int(np.argmax(counts))]
    global_cols = original_cols if chosen['feature_set'] == 'base' else all_cols
    final = train_model(f, global_cols, chosen, final_trees)
    joblib.dump({'model': final, 'columns': global_cols, 'configuration': chosen,
                 'note': 'Exploratory model; retrospective pseudo-labels; not validated for industrial use'},
                output_dir/'exploratory_final_model.joblib')
    (output_dir/'README_results.txt').write_text(
        'Held-out scores come ONLY from the outer test predictions; the final saved model uses all data.\n'
        'Target is a proxy based on known failure endpoint and detected onset, NOT measured actual RUL.\n'
        'Selected configs depend on only 2 independent inner experiments; large uncertainty.\n'
        'Avoid choosing a model based on inspecting outer holdout outcomes and rerunning repeatedly.\n')
    print('\nMEAN ACROSS HELD-OUT BEARINGS\n' + summary.round(4).to_string())
    print('\nSaved to:', output_dir.resolve())
    return summary, per_run, pd.DataFrame(selections), pred_df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', default='bearings_out/features_raw.pkl')
    parser.add_argument('--out', default='bearings_out/improvement')
    parser.add_argument('--inner-trees', type=int, default=150)
    parser.add_argument('--final-trees', type=int, default=500)
    args = parser.parse_args()
    if not Path(args.cache).exists():
        raise FileNotFoundError(f'Cannot find {args.cache}. Check notebook working directory.')
    raw = pd.read_pickle(args.cache)
    run_experiment(raw, args.out, args.inner_trees, args.final_trees)

if __name__ == '__main__':
    main()
