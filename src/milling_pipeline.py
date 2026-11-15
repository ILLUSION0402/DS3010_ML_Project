#!/usr/bin/env python3
"""Milling wear prediction - Stage 2

NASA PCoE milling data (mill.mat), with:
  * raw-waveform feature extraction
  * whole-case train/test split (cases 1-8 train, 9-16 test)
  * measured-VB-only supervision/evaluation (no interpolated labels)
  * Random Forest and symmetric XGBoost baselines
  * asymmetric-Huber XGBoost objective
  * sequence-to-one LSTM using the current and preceding cuts within each case
  * symmetric and asymmetric LSTM ablations
  * RMSE, MAE, train-range-normalised metrics, underprediction metrics

Milling target is flank wear VB in mm. No RUL target or RUL cap is imposed by
this script because the dataset README does not specify a numeric failure
threshold. A user-specified VB threshold would be required to derive RUL.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SENSOR_FIELDS = ["smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]
META_FIELDS = ["case", "run", "VB", "DOC", "feed", "material"]
TRAIN_CASES = list(range(1, 9))
TEST_CASES = list(range(9, 17))
SEED = 42


def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # CPU is used by default for reproducible small-data training.
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


def load_mill_mat(path: Path) -> pd.DataFrame:
    mat = loadmat(str(path), squeeze_me=True, struct_as_record=False)
    if "mill" not in mat:
        raise KeyError(f"No 'mill' struct found in {path}")
    rows = []
    for entry in np.atleast_1d(mat["mill"]):
        row = {field: getattr(entry, field) for field in META_FIELDS}
        for sensor in SENSOR_FIELDS:
            row[sensor] = np.asarray(getattr(entry, sensor), dtype=float).reshape(-1)
        rows.append(row)
    return pd.DataFrame(rows)


def waveform_features(values: np.ndarray) -> Dict[str, float]:
    x = np.asarray(values, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {k: np.nan for k in ("mean", "std", "rms", "max_abs", "kurtosis", "skew", "energy")}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        k = float(kurtosis(x, bias=False)) if x.size > 3 else np.nan
        s = float(skew(x, bias=False)) if x.size > 2 else np.nan
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "rms": float(np.sqrt(np.mean(np.square(x)))),
        "max_abs": float(np.max(np.abs(x))),
        "kurtosis": k,
        "skew": s,
        "energy": float(np.sum(np.square(x))),
    }


def build_feature_table(raw: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in raw.iterrows():
        item = {
            "case": int(row["case"]),
            "run": int(row["run"]),
            "VB": pd.to_numeric(row["VB"], errors="coerce"),
            "DOC": float(row["DOC"]),
            "feed": float(row["feed"]),
            "material": row["material"],
        }
        for sensor in SENSOR_FIELDS:
            for stat, value in waveform_features(row[sensor]).items():
                item[f"{sensor}_{stat}"] = value
        records.append(item)
    frame = pd.DataFrame(records).sort_values(["case", "run"]).reset_index(drop=True)
    frame = pd.get_dummies(frame, columns=["material"], prefix="material", dtype=float)
    return frame


def make_features(frame: pd.DataFrame, train_cases: List[int]) -> Tuple[pd.DataFrame, List[str], Dict[str, float], np.ndarray, np.ndarray]:
    metadata = {"case", "run", "VB"}
    candidate_cols = [c for c in frame.columns if c not in metadata]
    train_run_mask = frame["case"].isin(train_cases).to_numpy()

    # Fit the near-constant filter and imputation values on training cases only.
    retained = []
    for col in candidate_cols:
        if frame.loc[train_run_mask, col].nunique(dropna=True) > 1:
            retained.append(col)
    dropped = sorted(set(candidate_cols) - set(retained))
    if not retained:
        raise ValueError("No non-constant predictor features remain")

    medians = frame.loc[train_run_mask, retained].median(axis=0).fillna(0.0)
    x_df = frame[retained].copy().fillna(medians).fillna(0.0)
    # Train-only feature standardisation.
    x_scaler = __import__("sklearn.preprocessing", fromlist=["StandardScaler"]).StandardScaler()
    x_scaler.fit(x_df.loc[train_run_mask])
    x_scaled = x_scaler.transform(x_df).astype(np.float32)
    return x_df, retained, medians.to_dict(), x_scaled, train_run_mask


def build_sequences(
    frame: pd.DataFrame,
    x_scaled: np.ndarray,
    lookback: int,
    target_cases: List[int],
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    sequences, targets, meta_rows = [], [], []
    # Only rows with an actually measured VB become supervised targets.
    for case, group in frame.groupby("case", sort=True):
        if int(case) not in target_cases:
            continue
        row_ids = group.index.to_list()
        group = group.sort_values("run")
        row_ids = group.index.to_list()
        for pos, row_id in enumerate(row_ids):
            y = frame.loc[row_id, "VB"]
            if pd.isna(y):
                continue
            start = max(0, pos - lookback + 1)
            window_ids = row_ids[start:pos + 1]
            seq = np.zeros((lookback, x_scaled.shape[1]), dtype=np.float32)
            values = x_scaled[window_ids]
            seq[-len(values):, :] = values
            sequences.append(seq)
            targets.append(float(y))
            meta_rows.append({"case": int(case), "run": int(frame.loc[row_id, "run"]), "VB": float(y)})
    if not sequences:
        raise ValueError(f"No measured VB targets found for cases {target_cases}")
    return np.stack(sequences), np.asarray(targets, dtype=np.float32), pd.DataFrame(meta_rows)


class WearLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        return self.head(hidden[-1]).squeeze(-1)


def weighted_huber_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    asymmetric: bool,
    late_threshold: float,
    underprediction_weight: float = 2.0,
    late_stage_weight: float = 1.5,
    delta: float = 0.5,
) -> torch.Tensor:
    error = prediction - target
    # Smooth pseudo-Huber penalty (differentiable everywhere).
    huber = (delta ** 2) * (torch.sqrt(1.0 + (error / delta).square()) - 1.0)
    if not asymmetric:
        return huber.mean()
    under_weight = torch.where(error < 0, underprediction_weight, 1.0)
    late_weight = torch.where(target >= late_threshold, late_stage_weight, 1.0)
    return (huber * under_weight * late_weight).mean()


def _fit_lstm_epochs(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray | None,
    y_val: np.ndarray | None,
    asymmetric: bool,
    epochs: int,
    seed: int,
    batch_size: int = 16,
    patience: int = 28,
) -> Tuple[WearLSTM, int, float]:
    set_seeds(seed)
    device = torch.device("cpu")
    model = WearLSTM(input_size=x_train.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0025, weight_decay=1e-3)
    generator = torch.Generator().manual_seed(seed)

    xt = torch.tensor(x_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32)
    loader = DataLoader(TensorDataset(xt, yt), batch_size=min(batch_size, len(yt)), shuffle=True, generator=generator)
    vxt = torch.tensor(x_val, dtype=torch.float32) if x_val is not None else None
    vyt = torch.tensor(y_val, dtype=torch.float32) if y_val is not None else None
    late_threshold = float(np.quantile(y_train, 0.75))

    best_state = None
    best_val = float("inf")
    best_epoch = epochs
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch_x)
            loss = weighted_huber_loss(pred, batch_y, asymmetric, late_threshold)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

        if vxt is not None and vyt is not None:
            model.eval()
            with torch.no_grad():
                val_pred = model(vxt)
                val_loss = float(weighted_huber_loss(val_pred, vyt, asymmetric, late_threshold).item())
            if val_loss < best_val - 1e-5:
                best_val = val_loss
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_val = float("nan")
    return model, int(best_epoch), float(best_val)


def train_lstm_variant(
    x_train: np.ndarray,
    y_train_mm: np.ndarray,
    meta_train: pd.DataFrame,
    x_test: np.ndarray,
    asymmetric: bool,
    max_epochs: int,
    seed: int,
) -> Tuple[WearLSTM, np.ndarray, Dict[str, float]]:
    # Standardise the regression target using training labels only.
    # Group-level validation: hold out training cases 4 and 8; do not mix runs
    # from the same case across the fit/validation boundary.
    val_cases = [c for c in (4, 8) if c in set(meta_train["case"].astype(int))]
    val_mask = meta_train["case"].isin(val_cases).to_numpy()
    fit_mask = ~val_mask
    if fit_mask.sum() < 10 or val_mask.sum() < 4:
        # Small-data fallback: use a fixed training duration.
        val_cases = []
        fit_mask = np.ones(len(y_train_mm), dtype=bool)
        val_mask = np.zeros(len(y_train_mm), dtype=bool)

    # Fit the inner-validation target scaler only on the inner fit cases.
    inner_mean = float(np.mean(y_train_mm[fit_mask]))
    inner_std = float(np.std(y_train_mm[fit_mask]))
    if inner_std < 1e-8:
        inner_std = 1.0
    y_inner = ((y_train_mm - inner_mean) / inner_std).astype(np.float32)

    _, best_epoch, best_val = _fit_lstm_epochs(
        x_train[fit_mask], y_inner[fit_mask],
        x_train[val_mask] if val_mask.any() else None,
        y_inner[val_mask] if val_mask.any() else None,
        asymmetric=asymmetric,
        epochs=max_epochs,
        seed=seed,
    )

    # Refit target scaling on all training cases, then refit from scratch using
    # every training case for the selected epoch count.
    y_mean = float(np.mean(y_train_mm))
    y_std = float(np.std(y_train_mm))
    if y_std < 1e-8:
        y_std = 1.0
    y_train = ((y_train_mm - y_mean) / y_std).astype(np.float32)
    final_model, _, _ = _fit_lstm_epochs(
        x_train, y_train, None, None,
        asymmetric=asymmetric,
        epochs=max(1, best_epoch),
        seed=seed + 1,
    )
    final_model.eval()
    with torch.no_grad():
        pred_scaled = final_model(torch.tensor(x_test, dtype=torch.float32)).cpu().numpy()
    pred_mm = pred_scaled * y_std + y_mean
    pred_mm = np.maximum(pred_mm, 0.0)  # physical lower bound only; no upper VB cap.
    info = {
        "epochs_selected": int(best_epoch),
        "validation_loss": float(best_val) if np.isfinite(best_val) else None,
        "target_mean_mm": y_mean,
        "target_std_mm": y_std,
        "validation_cases": [int(c) for c in val_cases],
    }
    return final_model, pred_mm, info


def make_xgb_custom_objective(
    late_threshold: float,
    asymmetric: bool,
    underprediction_weight: float = 2.0,
    late_stage_weight: float = 1.5,
    huber_delta: float = 0.5,
):
    def objective(predt: np.ndarray, dtrain: xgb.DMatrix):
        y = dtrain.get_label()
        error = predt - y
        scale = 1.0 + np.square(error / huber_delta)
        # Exact gradient/Hessian of pseudo-Huber, scaled for numerical stability.
        grad = error / np.sqrt(scale)
        hess = 1.0 / np.power(scale, 1.5)
        if asymmetric:
            weight = np.where(error < 0.0, underprediction_weight, 1.0)
            weight *= np.where(y >= late_threshold, late_stage_weight, 1.0)
            grad = grad * weight
            hess = hess * weight
        return grad, hess
    return objective


def train_xgb(
    x_train: np.ndarray,
    y_train_mm: np.ndarray,
    x_test: np.ndarray,
    asymmetric: bool,
    num_boost_round: int,
) -> Tuple[xgb.Booster, np.ndarray, float, float]:
    y_mean = float(np.mean(y_train_mm))
    y_std = float(np.std(y_train_mm))
    if y_std < 1e-8:
        y_std = 1.0
    y_train = ((y_train_mm - y_mean) / y_std).astype(np.float32)
    late_threshold = float(np.quantile(y_train, 0.75))

    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test)
    params = {
        "max_depth": 2,
        "eta": 0.035,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 3.0,
        "lambda": 8.0,
        "alpha": 0.1,
        "base_score": 0.0,
        "tree_method": "hist",
        "seed": SEED,
        "nthread": 4,
        "disable_default_eval_metric": 1,
    }
    if not asymmetric:
        params["objective"] = "reg:squarederror"
        booster = xgb.train(params, dtrain, num_boost_round=num_boost_round, verbose_eval=False)
    else:
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            obj=make_xgb_custom_objective(late_threshold, asymmetric=True),
            verbose_eval=False,
        )
    pred_scaled = booster.predict(dtest)
    pred_mm = np.maximum(pred_scaled * y_std + y_mean, 0.0)
    return booster, pred_mm, y_mean, y_std


def metric_row(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_range: float,
    late_cutoff: float,
) -> Dict[str, float | str]:
    err = y_pred - y_true
    abs_err = np.abs(err)
    under = err < 0.0  # VB underprediction is the unsafe/optimistic direction.
    late = y_true >= late_cutoff
    stage_w = np.where(late, 1.5, 1.0)
    direction_w = np.where(under, 2.0, 1.0)
    denom = max(float(train_range), 1e-8)
    return {
        "model": model_name,
        "n_test_measured": int(len(y_true)),
        "RMSE_mm": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE_mm": float(mean_absolute_error(y_true, y_pred)),
        "NRMSE_train_range": float(np.sqrt(mean_squared_error(y_true, y_pred)) / denom),
        "NMAE_train_range": float(mean_absolute_error(y_true, y_pred) / denom),
        "WearAsymMAE_mm": float(np.mean(abs_err * direction_w * stage_w)),
        "Underprediction_rate_pct": float(100.0 * np.mean(under)),
        "Underprediction_MAE_mm": float(np.mean(np.abs(err[under]))) if np.any(under) else 0.0,
        "High_wear_RMSE_mm": float(np.sqrt(mean_squared_error(y_true[late], y_pred[late]))) if np.any(late) else float("nan"),
        "High_wear_n": int(np.sum(late)),
    }


def save_eda(frame: pd.DataFrame, predictions: pd.DataFrame, out_dir: Path) -> None:
    # Measured labels only: do not draw interpolated pseudo-labels.
    fig, ax = plt.subplots(figsize=(12, 7))
    for case, group in frame.groupby("case"):
        observed = group[group["VB"].notna()].sort_values("run")
        ax.plot(observed["run"], observed["VB"], marker="o", linewidth=1.1, alpha=0.75, label=f"Case {int(case)}")
    ax.set_xlabel("Run / cut index")
    ax.set_ylabel("Measured flank wear VB (mm)")
    ax.set_title("NASA Milling: measured VB trajectories (unmeasured runs omitted)")
    ax.grid(alpha=0.25)
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "measured_vb_by_case.png", dpi=180)
    plt.close(fig)

    # Test predictions overlaid by case for each learned model.
    model_cols = [c for c in predictions.columns if c.startswith("pred_")]
    cases = sorted(predictions["case"].unique())
    fig, axes = plt.subplots(4, 2, figsize=(14, 15), sharey=True)
    for ax, case in zip(axes.flat, cases):
        group = predictions[predictions["case"] == case].sort_values("run")
        ax.plot(group["run"], group["VB"], "o-", color="black", linewidth=1.5, label="Measured VB")
        for col in model_cols:
            ax.plot(group["run"], group[col], "--", marker=".", alpha=0.85, label=col.replace("pred_", ""))
        ax.set_title(f"Test case {case}")
        ax.set_xlabel("Run")
        ax.set_ylabel("VB (mm)")
        ax.grid(alpha=0.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=3, fontsize=9)
    fig.suptitle("Milling test set: predictions at measured VB runs", y=0.99, fontsize=14)
    fig.tight_layout(rect=[0, 0.055, 1, 0.965])
    fig.savefig(out_dir / "test_predictions_by_case.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Residual versus observed wear (shows whether late-stage wear is underpredicted).
    fig, ax = plt.subplots(figsize=(9, 6))
    for col in model_cols:
        residual = predictions[col].to_numpy() - predictions["VB"].to_numpy()
        ax.scatter(predictions["VB"], residual, alpha=0.65, s=24, label=col.replace("pred_", ""))
    ax.axhline(0, linestyle="--", linewidth=1, color="black")
    ax.set_xlabel("Measured VB (mm)")
    ax.set_ylabel("Prediction error: predicted - measured (mm)")
    ax.set_title("Prediction error by wear level (negative = wear underprediction)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "prediction_error_by_wear.png", dpi=180)
    plt.close(fig)

    # NRMSE chart is useful in the four-component comparison.
    if (out_dir / "model_comparison_measured_only.csv").exists():
        table = pd.read_csv(out_dir / "model_comparison_measured_only.csv")
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(table["model"], table["NRMSE_train_range"])
        ax.set_ylabel("NRMSE (normalised by training VB range)")
        ax.set_title("Milling model comparison on measured VB labels")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / "model_nrmse_comparison.png", dpi=180)
        plt.close(fig)


def run(args: argparse.Namespace) -> None:
    set_seeds(args.seed)
    output_dir = Path(args.output).resolve()
    model_dir = output_dir / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    raw = load_mill_mat(Path(args.data))
    frame = build_feature_table(raw)
    observed_count = int(frame["VB"].notna().sum())
    missing_count = int(frame["VB"].isna().sum())

    x_raw, feature_cols, medians, x_scaled, train_run_mask = make_features(frame, TRAIN_CASES)
    train_label_mask = frame["case"].isin(TRAIN_CASES).to_numpy() & frame["VB"].notna().to_numpy()
    test_label_mask = frame["case"].isin(TEST_CASES).to_numpy() & frame["VB"].notna().to_numpy()
    x_train = x_scaled[train_label_mask]
    x_test = x_scaled[test_label_mask]
    y_train = frame.loc[train_label_mask, "VB"].to_numpy(dtype=np.float32)
    y_test = frame.loc[test_label_mask, "VB"].to_numpy(dtype=np.float32)
    test_meta = frame.loc[test_label_mask, ["case", "run", "VB"]].reset_index(drop=True)
    if len(y_train) == 0 or len(y_test) == 0:
        raise RuntimeError("Train/test measured target set is empty")

    train_range = float(np.max(y_train) - np.min(y_train))
    if train_range <= 1e-8:
        train_range = 1.0
    late_cutoff_mm = float(np.quantile(y_train, 0.75))

    predictions = test_meta.copy()
    metrics = []
    models_for_importance = {}

    print(f"Loaded {len(frame)} runs; measured VB={observed_count}, missing VB={missing_count}")
    print(f"Train cases={TRAIN_CASES}: runs={int(train_run_mask.sum())}, measured labels={int(train_label_mask.sum())}")
    print(f"Test cases={TEST_CASES}: runs={int((~train_run_mask).sum())}, measured labels={int(test_label_mask.sum())}")
    print(f"Retained {len(feature_cols)} features; dropped constants in train={len(set(frame.columns)-set(feature_cols)-{'case','run','VB'})}")
    print(f"Late-stage training emphasis threshold: Q75(VB_train)={late_cutoff_mm:.4f} mm")

    # Random Forest baseline, trained and evaluated only on originally measured VB.
    rf = RandomForestRegressor(
        n_estimators=500,
        max_features=0.8,
        min_samples_leaf=2,
        random_state=args.seed,
        n_jobs=-1,
    )
    rf.fit(x_train, y_train)
    rf_pred = np.maximum(rf.predict(x_test), 0.0)
    predictions["pred_RandomForest"] = rf_pred
    metrics.append(metric_row("RandomForest", y_test, rf_pred, train_range, late_cutoff_mm))
    models_for_importance["RandomForest"] = pd.Series(rf.feature_importances_, index=feature_cols)
    joblib.dump(rf, model_dir / "random_forest_measured_vb.joblib")

    # Symmetric XGBoost baseline.
    xgb_sym, xgb_sym_pred, xgb_y_mean, xgb_y_std = train_xgb(
        x_train, y_train, x_test, asymmetric=False, num_boost_round=args.xgb_rounds
    )
    predictions["pred_XGBoost_symmetric"] = xgb_sym_pred
    metrics.append(metric_row("XGBoost_symmetric", y_test, xgb_sym_pred, train_range, late_cutoff_mm))
    xgb_sym.save_model(str(model_dir / "xgboost_symmetric.json"))
    xgb_gain = xgb_sym.get_score(importance_type="gain")
    models_for_importance["XGBoost_symmetric"] = pd.Series({feature_cols[int(k[1:])]: v for k, v in xgb_gain.items() if k.startswith("f")})

    # Asymmetric-Huber XGBoost: lower VB predictions are more costly, and high-wear
    # labels (>= training Q75) receive an additional late-stage weight.
    xgb_asym, xgb_asym_pred, asym_y_mean, asym_y_std = train_xgb(
        x_train, y_train, x_test, asymmetric=True, num_boost_round=args.xgb_rounds
    )
    predictions["pred_XGBoost_asymmetric"] = xgb_asym_pred
    metrics.append(metric_row("XGBoost_asymmetric", y_test, xgb_asym_pred, train_range, late_cutoff_mm))
    xgb_asym.save_model(str(model_dir / "xgboost_asymmetric.json"))
    xgb_asym_gain = xgb_asym.get_score(importance_type="gain")
    models_for_importance["XGBoost_asymmetric"] = pd.Series({feature_cols[int(k[1:])]: v for k, v in xgb_asym_gain.items() if k.startswith("f")})

    # LSTM sequence data: each supervised VB label receives the last `lookback`
    # cuts from its own case, including the current cut; no cross-case windows.
    seq_train, seq_y_train, seq_train_meta = build_sequences(frame, x_scaled, args.lookback, TRAIN_CASES)
    seq_test, seq_y_test, seq_test_meta = build_sequences(frame, x_scaled, args.lookback, TEST_CASES)
    # Align sequence windows to the exact test-label order used by the tree models.
    seq_lookup = {(int(row.case), int(row.run)): idx for idx, row in enumerate(seq_test_meta.itertuples(index=False))}
    order = [seq_lookup[(int(case), int(run))] for case, run in zip(test_meta["case"], test_meta["run"])]
    seq_test = seq_test[order]
    seq_y_test = seq_y_test[order]
    seq_test_meta = test_meta.copy()

    # Sequence LSTM - symmetric Huber ablation.
    lstm_sym, lstm_sym_pred, lstm_sym_info = train_lstm_variant(
        seq_train, seq_y_train, seq_train_meta,
        seq_test, asymmetric=False, max_epochs=args.lstm_epochs, seed=args.seed + 10,
    )
    predictions["pred_LSTM_symmetric"] = lstm_sym_pred
    metrics.append(metric_row("LSTM_symmetric", y_test, lstm_sym_pred, train_range, late_cutoff_mm))
    torch.save({
        "state_dict": lstm_sym.state_dict(), "feature_columns": feature_cols,
        "lookback": args.lookback, "hidden_size": 32, "training_info": lstm_sym_info,
        "target": "VB_mm", "target_scaler_mean_mm": lstm_sym_info["target_mean_mm"],
        "target_scaler_std_mm": lstm_sym_info["target_std_mm"],
    }, model_dir / "lstm_symmetric.pt")

    # Sequence LSTM - asymmetric Huber objective.
    lstm_asym, lstm_asym_pred, lstm_asym_info = train_lstm_variant(
        seq_train, seq_y_train, seq_train_meta,
        seq_test, asymmetric=True, max_epochs=args.lstm_epochs, seed=args.seed + 20,
    )
    predictions["pred_LSTM_asymmetric"] = lstm_asym_pred
    metrics.append(metric_row("LSTM_asymmetric", y_test, lstm_asym_pred, train_range, late_cutoff_mm))
    torch.save({
        "state_dict": lstm_asym.state_dict(), "feature_columns": feature_cols,
        "lookback": args.lookback, "hidden_size": 32, "training_info": lstm_asym_info,
        "target": "VB_mm", "target_scaler_mean_mm": lstm_asym_info["target_mean_mm"],
        "target_scaler_std_mm": lstm_asym_info["target_std_mm"],
    }, model_dir / "lstm_asymmetric.pt")

    # Prediction and metric tables.
    metrics_df = pd.DataFrame(metrics).sort_values("NRMSE_train_range").reset_index(drop=True)
    metrics_df.to_csv(output_dir / "model_comparison_measured_only.csv", index=False)
    predictions.to_csv(output_dir / "test_predictions_measured_only.csv", index=False)

    # Per-case score breakdown.
    per_case_rows = []
    pred_cols = [c for c in predictions.columns if c.startswith("pred_")]
    for case, group in predictions.groupby("case"):
        for col in pred_cols:
            row = metric_row(col.replace("pred_", ""), group["VB"].to_numpy(), group[col].to_numpy(), train_range, late_cutoff_mm)
            row["case"] = int(case)
            per_case_rows.append(row)
    pd.DataFrame(per_case_rows).to_csv(output_dir / "per_case_metrics_measured_only.csv", index=False)

    # Feature importances (tree models only).
    importance_df = pd.DataFrame(index=feature_cols)
    for model_name, series in models_for_importance.items():
        importance_df[model_name] = series.reindex(feature_cols).fillna(0.0)
    importance_df.to_csv(output_dir / "tree_feature_importance.csv", index_label="feature")

    # Save preprocessing details and model card/config.
    x_scaler = __import__("sklearn.preprocessing", fromlist=["StandardScaler"]).StandardScaler()
    x_scaler.fit(x_raw.loc[train_run_mask])
    joblib.dump({
        "feature_columns": feature_cols,
        "train_medians": medians,
        "feature_scaler": x_scaler,
        "train_cases": TRAIN_CASES,
        "test_cases": TEST_CASES,
        "lookback": args.lookback,
    }, model_dir / "preprocessing.joblib")

    config = {
        "dataset": "NASA PCoE Milling / mill.mat",
        "rows": int(len(frame)),
        "observed_vb_labels": observed_count,
        "missing_vb_labels": missing_count,
        "train_cases": TRAIN_CASES,
        "test_cases": TEST_CASES,
        "train_measured_labels": int(len(y_train)),
        "test_measured_labels": int(len(y_test)),
        "features": feature_cols,
        "lookback_cuts": int(args.lookback),
        "lstm": {"hidden_size": 32, "max_epochs": int(args.lstm_epochs), "symmetric": lstm_sym_info, "asymmetric": lstm_asym_info},
        "xgboost": {"rounds": int(args.xgb_rounds), "max_depth": 2, "learning_rate": 0.035,
                    "target_mean_mm": xgb_y_mean, "target_std_mm": xgb_y_std},
        "asymmetric_loss": {
            "form": "weighted Huber on standardised VB",
            "underprediction_weight": 2.0,
            "late_stage_weight": 1.5,
            "late_stage_cutoff_mm": late_cutoff_mm,
            "late_stage_cutoff_definition": "75th percentile of measured training VB labels; training emphasis only, not a physical failure threshold",
            "direction": "VB underprediction (predicted wear < measured wear) is penalised more, analogous to overestimating RUL",
        },
        "normalization": {"name": "NRMSE_train_range", "denominator_mm": train_range, "formula": "RMSE / (max(VB_train) - min(VB_train))"},
        "rul": {"constructed": False, "cap": None, "reason": "The source README provides direct VB measurements but no numeric wear-limit threshold; no RUL cap can be justified without a declared threshold."},
        "phm08_score": {"reported": False, "reason": "PHM08 is a RUL scoring function; this model predicts direct flank wear VB in mm. The asymmetric training loss is PHM08-inspired in risk direction, not the official PHM08 RUL score."},
        "prediction_floor_mm": 0.0,
        "prediction_upper_cap_mm": None,
        "seed": int(args.seed),
    }
    (output_dir / "experiment_config.json").write_text(json.dumps(config, indent=2))
    (model_dir / "xgboost_target_scalers.json").write_text(json.dumps({
        "symmetric": {"mean_mm": xgb_y_mean, "std_mm": xgb_y_std},
        "asymmetric": {"mean_mm": asym_y_mean, "std_mm": asym_y_std},
    }, indent=2))

    # Save a compact audit of observed/missing labels per case.
    audit = frame.groupby("case").agg(
        runs=("run", "count"),
        measured_vb=("VB", "count"),
        missing_vb=("VB", lambda s: int(s.isna().sum())),
        vb_min_mm=("VB", "min"),
        vb_max_mm=("VB", "max"),
    ).reset_index()
    audit.to_csv(output_dir / "case_label_audit.csv", index=False)

    # Prior stage result preserved only as a historical reference because that
    # run trained/evaluated on linearly interpolated VB for every run.
    legacy = pd.DataFrame([
        {"model": "RandomForest_legacy_interpolated", "RMSE_mm": 0.2689071843425587, "MAE_mm": 0.18276773049645403,
         "label_protocol": "all runs; 21 missing VB values linearly interpolated within case"},
        {"model": "XGBoost_legacy_interpolated", "RMSE_mm": 0.26941275259386144, "MAE_mm": 0.1856635961253593,
         "label_protocol": "all runs; 21 missing VB values linearly interpolated within case"},
    ])
    legacy.to_csv(output_dir / "legacy_interpolated_reference.csv", index=False)

    save_eda(frame, predictions, output_dir)

    summary = [
        "MILLING STAGE 2 - RUN SUMMARY",
        "==============================",
        f"Runs loaded: {len(frame)}",
        f"Original measured VB labels: {observed_count}",
        f"VB not measured: {missing_count} (kept missing; not interpolated for Stage 2 model fitting/evaluation)",
        f"Train cases: {TRAIN_CASES} | measured labels: {len(y_train)}",
        f"Test cases: {TEST_CASES} | measured labels: {len(y_test)}",
        f"Sequence lookback: {args.lookback} cuts (same case; current cut included)",
        f"Late-stage cutoff for loss weighting: {late_cutoff_mm:.4f} mm (Q75 of training labels; not a failure threshold)",
        "",
        "MODEL COMPARISON (MEASURED LABELS ONLY)",
        metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "",
        "RUL / CAP DECISION",
        "No RUL is derived and no target cap is applied. The dataset reports direct flank wear VB in mm, and the supplied README does not specify a numeric VB failure threshold. The observed data reaches 1.53 mm, so clipping to a conventional assumed value would discard recorded data.",
        "",
        "PHM08 DECISION",
        "Official PHM08 RUL score is not reported because this is direct-VB regression. The asymmetric training objective penalises VB underprediction (the wear-space analogue of RUL overprediction) and adds extra weight to the upper quartile of training wear labels.",
        "",
        "LEGACY REFERENCE (NOT APPLE-TO-APPLE WITH STAGE 2)",
        "The previous RF/XGBoost scores were produced after interpolating all 21 unmeasured VB labels. They are retained in legacy_interpolated_reference.csv only for traceability; compare Stage 2 models using measured-only rows in model_comparison_measured_only.csv.",
    ]
    (output_dir / "RESULTS_SUMMARY_STAGE2.txt").write_text("\n".join(summary) + "\n")

    print("\n=== Measured-label model comparison ===")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nLSTM symmetric training info:", lstm_sym_info)
    print("LSTM asymmetric training info:", lstm_asym_info)
    print("\nOutputs:", output_dir)
    for path in sorted(output_dir.iterdir()):
        if path.is_file():
            print(" -", path.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-2 NASA Milling sequence and asymmetric-loss pipeline")
    parser.add_argument("--data", default="mill.mat", help="Path to extracted mill.mat")
    parser.add_argument("--output", default="milling_stage2_outputs", help="Output folder")
    parser.add_argument("--lookback", type=int, default=5, help="Number of preceding/current cuts per LSTM sequence")
    parser.add_argument("--lstm-epochs", type=int, default=160, help="Maximum LSTM epochs; training-case validation selects epoch count")
    parser.add_argument("--xgb-rounds", type=int, default=160, help="XGBoost boosting rounds")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.lookback < 1:
        parser.error("--lookback must be >= 1")
    if args.lstm_epochs < 1 or args.xgb_rounds < 1:
        parser.error("training iterations must be >= 1")
    return args


if __name__ == "__main__":
    run(parse_args())
