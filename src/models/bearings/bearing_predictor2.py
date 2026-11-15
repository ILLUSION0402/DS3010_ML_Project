"""
bearing_predictor.py  (v2)
==========================
Inference wrapper for the IMS bearing RUL model (rul_model_final_v2.joblib).

Single bearing:
    pred = BearingRULPredictor("models/bearings/rul_model_final_v2.joblib")
    res  = pred.predict(folder="data/raw/bearings/2nd_test", channels=[0], n_jobs=-1)
    print(res.summary()); res.plot()

Whole machine (files are read ONCE; all bearings scored together):
    m = pred.predict_many({"b1": [0], "b2": [1], "b3": [2], "b4": [3]},
                          folder="data/raw/bearings/2nd_test", n_jobs=-1)
    print(m.summary()); m.plot()

Needs: a time-ordered series of raw vibration snapshots (20,480 rows, 20 kHz, whitespace-separated,
filename = timestamp YYYY.MM.DD.HH.MM.SS) from the START of monitoring; hours 40-70 are the
healthy reference. `channels` = 0-indexed column(s) of the accelerometer(s) on ONE bearing.

Limits: trained on 4 run-to-failure runs from one rig (2000 RPM, 6000 lb, Rexnord ZA-2115).
Not validated on other bearings/speeds/sensors. Research prototype only.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import signal, stats

# --------------------------------------------------------------------------------------
# Constants (feature-related ones must match the training notebook)
# --------------------------------------------------------------------------------------
FS = 20_000
N_SAMPLES = 20_480
FAULT_HZ = {"BPFO": 236.4, "BPFI": 296.9, "BSF": 139.9, "FTF": 14.8}
FAULT_NAMES = {"BPFO": "outer race", "BPFI": "inner race",
               "BSF": "rolling element", "FTF": "cage"}
CORE = (["rms", "kurt", "p2p", "crest", "std", "skew", "impulse", "margin", "centroid", "spec_std"]
        + [f"band{i}" for i in range(8)]
        + [f"raw_{k}" for k in FAULT_HZ] + [f"env_{k}" for k in FAULT_HZ])

ONSET_INDICATORS = ["rms", "kurt", "env_BPFO", "env_BPFI", "env_BSF", "band5", "band6", "band7"]
K_SIGMA, MARGIN, SUSTAIN = 4, 0.1, 6
MAX_GAP_S = 600

# Interpretation settings (heuristics from held-out behaviour; NOT validated limits)
HEALTHY_MIN = 0.9
CRITICAL_MAX = 0.5
DEG_STAGE_H = (12.0, 95.0)
NEIGHBOUR_GAP = 0.2        # another bearing this much lower -> add a "may be influenced" note

_SOS_ENV = signal.butter(4, [2000, 9000], btype="bandpass", fs=FS, output="sos")


class InputError(ValueError):
    """Raised when the supplied data cannot be used."""


# --------------------------------------------------------------------------------------
# Feature extraction (identical to the notebook)
# --------------------------------------------------------------------------------------
def _time_feats(x):
    rms = np.sqrt(np.mean(x ** 2)); peak = np.max(np.abs(x)); mabs = np.mean(np.abs(x))
    return {"rms": rms, "kurt": stats.kurtosis(x, fisher=False), "p2p": np.ptp(x),
            "crest": peak / rms, "std": x.std(), "skew": stats.skew(x),
            "impulse": peak / mabs, "margin": peak / (np.mean(np.sqrt(np.abs(x))) ** 2)}


def _band_peak(f, mag, f0, width=3.0, harmonics=(1, 2, 3)):
    out = 0.0
    for h in harmonics:
        m = (f > h * f0 - width) & (f < h * f0 + width)
        if m.any():
            out += mag[m].max()
    return out


def _freq_feats(x):
    x = x - x.mean(); n = len(x); win = np.hanning(n)
    mag = np.abs(np.fft.rfft(x * win)) / n; f = np.fft.rfftfreq(n, 1 / FS)
    d = {}
    edges = np.linspace(0, 10_000, 9)
    for i in range(8):
        m = (f >= edges[i]) & (f < edges[i + 1]); d[f"band{i}"] = np.sum(mag[m] ** 2)
    p = mag ** 2
    d["centroid"] = np.sum(f * p) / np.sum(p)
    d["spec_std"] = np.sqrt(np.sum(((f - d["centroid"]) ** 2) * p) / np.sum(p))
    for k, hz in FAULT_HZ.items():
        d[f"raw_{k}"] = _band_peak(f, mag, hz)
    env = np.abs(signal.hilbert(signal.sosfiltfilt(_SOS_ENV, x))); env -= env.mean()
    emag = np.abs(np.fft.rfft(env * win)) / n
    for k, hz in FAULT_HZ.items():
        d[f"env_{k}"] = _band_peak(f, emag, hz)
    return d


def _file_features_multi(path, bearings):
    """Read one file once; return ({bearing: feature dict} | None, error | None)."""
    name = os.path.basename(path)
    try:
        data = pd.read_csv(path, sep=r"\s+", header=None).values
    except Exception as e:                                   # noqa: BLE001
        return None, f"{name}: could not be read ({e})"
    if data.shape[0] != N_SAMPLES:
        return None, f"{name}: {data.shape[0]} rows, expected {N_SAMPLES} (1 s at 20 kHz)"
    need = max(max(ch) for ch in bearings.values())
    if need >= data.shape[1]:
        return None, f"{name}: has {data.shape[1]} columns, channel {need} requested"
    out = {}
    for b, channels in bearings.items():
        per = []
        for c in channels:
            x = data[:, c].astype(float)
            per.append({**_time_feats(x), **_freq_feats(x)})
        out[b] = {k: float(np.mean([d[k] for d in per])) for k in per[0]}
    return out, None


# --------------------------------------------------------------------------------------
# Normalisation / onset / post-processing (identical logic to the notebook)
# --------------------------------------------------------------------------------------
def _normalise(raw, base_start, base_len):
    bmask = (raw["t_hours"] >= base_start) & (raw["t_hours"] < base_start + base_len)
    eps = 1e-12
    feats = pd.DataFrame(index=raw.index)
    for c in CORE:
        base = raw.loc[bmask, c]
        if (raw[c] > 0).all():
            z = np.log((raw[c] + eps) / (base.median() + eps))
        else:
            z = (raw[c] - base.median()) / (base.std() + eps)
        z = z.rolling(5, min_periods=1).median().ewm(span=10, adjust=False).mean()
        feats[c] = z
        feats[c + "_slope"] = z.diff(12).fillna(0)
    return feats, bmask


def _detect_onset(t_hours, feats, bmask):
    t_end_base = t_hours[bmask].max()
    cands = []
    for c in ONSET_INDICATORS:
        s = feats[c]; b = s[bmask]
        thr = b.mean() + K_SIGMA * b.std() + MARGIN
        ok = ((s > thr).astype(int).rolling(SUSTAIN).sum() >= SUSTAIN) & (t_hours > t_end_base)
        if ok.any():
            pos = int(np.argmax(ok.values))
            cands.append(float(t_hours.iloc[max(pos - SUSTAIN + 1, 0)]))
    cands = sorted(cands)
    if len(cands) >= 2:
        return cands[1]
    return cands[0] if cands else None


def _postprocess(pred, t, warm, mono, win, alpha):
    pred = np.asarray(pred); t = np.asarray(t)
    out = np.ones(len(pred)); w = t >= warm
    s = pd.Series(pred[w])
    if win > 1:
        s = s.rolling(win, min_periods=1).median()
    if alpha < 1.0:
        s = s.ewm(alpha=alpha, adjust=False).mean()
    s = s.values
    out[w] = np.minimum.accumulate(s) if mono else s
    return np.clip(out, 0, 1)


# --------------------------------------------------------------------------------------
# Result objects
# --------------------------------------------------------------------------------------
@dataclass
class BearingResult:
    status: str                              # insufficient_data | healthy | elevated | critical
    n_files: int
    hours_monitored: float
    onset_hours: float | None = None
    rul_fraction: float | None = None        # 1 = healthy, 0 = failure (model's final value)
    rul_hours_range: tuple | None = None     # very rough
    fault_evidence: dict = field(default_factory=dict)   # descriptive only, NOT a diagnosis
    warnings: list = field(default_factory=list)
    timeline: pd.DataFrame | None = None
    features: pd.DataFrame | None = None

    def to_dict(self):
        return {"status": self.status, "n_files": self.n_files,
                "hours_monitored": self.hours_monitored, "onset_hours": self.onset_hours,
                "rul_fraction": self.rul_fraction, "rul_hours_range": self.rul_hours_range,
                "fault_evidence": self.fault_evidence, "warnings": self.warnings}

    def summary(self):
        L = [f"Status: {self.status.upper()}",
             f"Monitored: {self.hours_monitored:.1f} operating hours ({self.n_files} snapshots)"]
        if self.status == "insufficient_data":
            L += ["", *[f"! {w}" for w in self.warnings]]
            return "\n".join(L)
        L.append("Degradation onset detected at: " +
                 (f"{self.onset_hours:.1f} h" if self.onset_hours is not None else "none detected"))
        L.append(f"Model health index: {self.rul_fraction:.2f}  (1 = healthy, 0 = failure)")
        if self.rul_hours_range:
            lo, hi = self.rul_hours_range
            L.append(f"Very rough remaining life: {lo:.0f}-{hi:.0f} h  (uncertain; see warnings)")
        if self.fault_evidence and self.status != "healthy":
            parts = [f"{k} ({FAULT_NAMES[k]}) {v['change_pct']:+.0f}%" for k, v in self.fault_evidence.items()]
            L.append("Envelope energy at fault frequencies vs own baseline: " + ", ".join(parts))
            L.append("  (descriptive only; this is NOT a fault diagnosis)")
        if self.warnings:
            L += ["", *[f"! {w}" for w in self.warnings]]
        return "\n".join(L)

    def plot(self):
        import matplotlib.pyplot as plt
        if self.timeline is None or "rul_fraction" not in self.timeline:
            return None
        t = self.timeline["t_hours"]
        fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        ax[0].plot(t, self.timeline["rul_fraction"], label="model health index")
        ax[0].axhline(HEALTHY_MIN, ls=":", color="gray"); ax[0].axhline(CRITICAL_MAX, ls=":", color="gray")
        if self.onset_hours is not None:
            ax[0].axvline(self.onset_hours, color="r", ls="--", label="detected onset")
        ax[0].set_ylim(-0.05, 1.05); ax[0].set_ylabel("health index"); ax[0].legend()
        for c in ["rms", "kurt", "env_BPFO", "env_BPFI", "env_BSF"]:
            ax[1].plot(t, self.timeline[c], label=c, alpha=0.8)
        if self.onset_hours is not None:
            ax[1].axvline(self.onset_hours, color="r", ls="--")
        ax[1].set_xlabel("operating hours"); ax[1].set_ylabel("change vs own baseline (log ratio)")
        ax[1].legend(fontsize=8)
        fig.tight_layout()
        return fig


@dataclass
class MachineResult:
    bearings: dict                                   # name -> BearingResult
    notes: list = field(default_factory=list)

    def table(self):
        rows = []
        for name, r in self.bearings.items():
            rows.append({"bearing": name, "status": r.status,
                         "health_index": None if r.rul_fraction is None else round(r.rul_fraction, 3),
                         "onset_h": None if r.onset_hours is None else round(r.onset_hours, 1),
                         "hours": round(r.hours_monitored, 1)})
        return pd.DataFrame(rows).set_index("bearing")

    def summary(self):
        L = [self.table().to_string(), ""]
        L += [f"* {n}" for n in self.notes]
        return "\n".join(L)

    def plot(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4))
        for name, r in self.bearings.items():
            if r.timeline is not None and "rul_fraction" in r.timeline:
                ax.plot(r.timeline["t_hours"], r.timeline["rul_fraction"], label=name)
        ax.axhline(HEALTHY_MIN, ls=":", color="gray"); ax.axhline(CRITICAL_MAX, ls=":", color="gray")
        ax.set_ylim(-0.05, 1.05); ax.set_xlabel("operating hours"); ax.set_ylabel("health index")
        ax.legend(); fig.tight_layout()
        return fig


# --------------------------------------------------------------------------------------
# Predictor
# --------------------------------------------------------------------------------------
class BearingRULPredictor:
    def __init__(self, model_path):
        bundle = joblib.load(model_path)
        missing = {"model", "cols", "post", "baseline_hours"} - set(bundle)
        if missing:
            raise InputError(f"model file is missing keys: {sorted(missing)}")
        self.model = bundle["model"]
        self.cols = bundle["cols"]
        self.post = bundle["post"]
        self.base_start, self.base_len = bundle["baseline_hours"]

    @staticmethod
    def _list_files(folder, files):
        if files is not None:
            return sorted(files, key=os.path.basename)
        if folder is None:
            raise InputError("provide folder= or files=")
        found = [f for f in glob.glob(os.path.join(folder, "**", "*"), recursive=True)
                 if os.path.isfile(f) and os.path.basename(f).count(".") == 5]
        if not found:
            raise InputError(f"no snapshot files (names like 2004.02.12.10.32.39) found under {folder}")
        return sorted(found, key=os.path.basename)

    @staticmethod
    def _timestamps(files, fallback_minutes):
        try:
            ts = [datetime.strptime(os.path.basename(f), "%Y.%m.%d.%H.%M.%S") for f in files]
            return pd.to_datetime(ts), False
        except ValueError:
            start = pd.Timestamp("2000-01-01")
            return pd.to_datetime([start + pd.Timedelta(minutes=fallback_minutes * i)
                                   for i in range(len(files))]), True

    # ---- single bearing (convenience) ----
    def predict(self, channels, folder=None, files=None, n_jobs=1, fallback_minutes=10):
        m = self.predict_many({"bearing": list(channels)}, folder=folder, files=files,
                              n_jobs=n_jobs, fallback_minutes=fallback_minutes)
        return m.bearings["bearing"]

    # ---- several bearings of one machine, files read once ----
    def predict_many(self, bearings, folder=None, files=None, n_jobs=1, fallback_minutes=10):
        bearings = {name: list(ch) for name, ch in bearings.items()}
        if not bearings:
            raise InputError("bearings is empty")
        warnings = []
        paths = self._list_files(folder, files)
        res = Parallel(n_jobs=n_jobs)(delayed(_file_features_multi)(p, bearings) for p in paths)

        errs = [e for _, e in res if e is not None]
        if errs:
            if len(errs) > 0.02 * len(paths):
                raise InputError(f"{len(errs)} of {len(paths)} files unusable, e.g.: " + "; ".join(errs[:3]))
            warnings.append(f"{len(errs)} unusable file(s) skipped, e.g. {errs[0]}")
        good_idx = [i for i, (f, _) in enumerate(res) if f is not None]
        if len(good_idx) < 50:
            raise InputError(f"only {len(good_idx)} usable snapshots; need a long series from the start of monitoring")
        ts, fallback = self._timestamps([paths[i] for i in good_idx], fallback_minutes)
        if fallback:
            warnings.append(f"filenames are not timestamps; assumed one snapshot every {fallback_minutes} min")

        out = {}
        for name in bearings:
            raw = pd.DataFrame([res[i][0][name] for i in good_idx])
            raw["timestamp"] = ts
            out[name] = self._assess(raw, list(warnings))

        # machine-level context: a much worse neighbour may be transmitting vibration
        notes = []
        scored = {n: r for n, r in out.items() if r.rul_fraction is not None}
        if len(scored) > 1:
            worst = min(scored, key=lambda n: scored[n].rul_fraction)
            wr = scored[worst]
            if wr.status in ("elevated", "critical"):
                for n, r in scored.items():
                    if n != worst and r.status != "healthy" and r.rul_fraction - wr.rul_fraction >= NEIGHBOUR_GAP:
                        msg = (f"Bearing '{worst}' on the same machine is in a much worse state "
                               f"(index {wr.rul_fraction:.2f} vs {r.rul_fraction:.2f}). This bearing's rise in "
                               f"vibration may be partly transmitted from it; check '{worst}' first.")
                        r.warnings.insert(0, msg)
                        notes.append(f"'{n}': flagged {r.status}, but may be influenced by '{worst}'.")
            notes.insert(0, f"Most affected bearing: '{worst}' (health index {wr.rul_fraction:.2f}, {wr.status}).")
        return MachineResult(out, notes)

    # ---- assessment of one bearing ----
    def _assess(self, raw, warnings):
        raw = raw.sort_values("timestamp").reset_index(drop=True)
        dt = raw["timestamp"].diff().dt.total_seconds().fillna(0).clip(lower=0, upper=MAX_GAP_S)
        raw["t_hours"] = dt.cumsum() / 3600
        n, hours = len(raw), float(raw["t_hours"].iloc[-1])

        bmask = (raw["t_hours"] >= self.base_start) & (raw["t_hours"] < self.base_start + self.base_len)
        if bmask.sum() < 10:
            warnings.append(
                f"Need data covering at least {self.base_start + self.base_len:.0f} operating hours "
                f"(healthy reference = hours {self.base_start:.0f}-{self.base_start + self.base_len:.0f}); "
                f"only {hours:.1f} h supplied, or the reference window has too few snapshots.")
            return BearingResult("insufficient_data", n, hours, warnings=warnings)

        feats, bmask = _normalise(raw, self.base_start, self.base_len)
        onset = _detect_onset(raw["t_hours"], feats, bmask)
        health = _postprocess(self.model.predict(feats[self.cols]), raw["t_hours"].values, **self.post)
        frac = float(health[-1])

        if frac <= CRITICAL_MAX:
            status = "critical"
        elif frac < HEALTHY_MIN or onset is not None:
            status = "elevated"
        else:
            status = "healthy"
            
        rul_range = None if status == "healthy" else (frac * DEG_STAGE_H[0], frac * DEG_STAGE_H[1])

        evidence = {}
        for k in FAULT_HZ:
            z_last = float(feats[f"env_{k}"].iloc[-12:].mean())
            evidence[k] = {"change_pct": float((np.exp(z_last) - 1) * 100)}

        warnings += [
            "Model trained on 4 run-to-failure tests from one rig (2000 RPM, 6000 lb); "
            "not validated on other bearings, speeds or sensors.",
            "Near failure the model tends to OVERESTIMATE remaining life; treat 'critical' as urgent.",
        ]
        if status == "healthy":
            warnings.append("'Healthy' means no degradation detected; one test bearing (outer-race failure) "
                            "showed almost no warning before failure.")
        else:
            warnings.append(f"Remaining-hours range assumes a degradation stage of {DEG_STAGE_H[0]:.0f}-"
                            f"{DEG_STAGE_H[1]:.0f} h, the spread seen in 4 runs; it is very rough.")

        timeline = pd.concat([raw[["t_hours", "timestamp"]],
                              pd.Series(health, name="rul_fraction"),
                              feats[["rms", "kurt"] + [f"env_{k}" for k in FAULT_HZ]]], axis=1)
        return BearingResult(status, n, hours, onset, frac, rul_range, evidence, warnings, timeline, feats)