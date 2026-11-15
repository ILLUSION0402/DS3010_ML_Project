import numpy as np
import torch

from src.milling_pipeline import waveform_features, weighted_huber_loss


def test_waveform_features_basic_statistics():
    features = waveform_features(np.array([1.0, 2.0, 3.0, 4.0]))
    assert features["mean"] == 2.5
    assert features["std"] > 0
    assert features["rms"] > 0
    assert features["max_abs"] == 4.0


def test_empty_waveform_returns_nan_features():
    features = waveform_features(np.array([], dtype=float))
    assert np.isnan(features["mean"])
    assert np.isnan(features["energy"])


def test_asymmetric_loss_penalises_underprediction_more():
    target = torch.tensor([1.0, 1.0])
    under = torch.tensor([0.5, 0.5])
    over = torch.tensor([1.5, 1.5])
    under_loss = weighted_huber_loss(
        under, target, asymmetric=True, late_threshold=2.0
    )
    over_loss = weighted_huber_loss(
        over, target, asymmetric=True, late_threshold=2.0
    )
    assert under_loss.item() > over_loss.item()
