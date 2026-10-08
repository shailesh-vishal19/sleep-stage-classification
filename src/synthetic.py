"""Synthetic sleep-like EEG for testing the pipeline without downloading data.

NOT real data. Stage-dependent band amplitudes plus a per-subject gain on each
band, so that subjects differ (as real people do) and random epoch splits leak.
"""
from __future__ import annotations

import numpy as np

from .features import BANDS

# rows: W, N1, N2, N3, REM ; cols: delta, theta, alpha, sigma, beta
STAGE_AMP = np.array([
    [0.3, 0.4, 2.0, 0.3, 1.0],   # W
    [0.5, 1.4, 0.8, 0.3, 0.5],   # N1
    [1.0, 1.0, 0.4, 1.6, 0.4],   # N2
    [3.0, 0.5, 0.2, 0.3, 0.2],   # N3
    [0.4, 1.2, 0.5, 0.2, 0.4],   # REM
])
TRANS = np.array([
    [0.90, 0.08, 0.02, 0.00, 0.00],
    [0.10, 0.60, 0.28, 0.00, 0.02],
    [0.02, 0.03, 0.85, 0.08, 0.02],
    [0.00, 0.00, 0.10, 0.90, 0.00],
    [0.04, 0.03, 0.03, 0.00, 0.90],
])


def make_subject(rng, n_epochs=240, sfreq=100.0, n_channels=2, subject_spread=0.5):
    stages = np.empty(n_epochs, dtype=int)
    stages[0] = 0
    for t in range(1, n_epochs):
        stages[t] = rng.choice(5, p=TRANS[stages[t - 1]])

    n_samp = int(30 * sfreq)
    t = np.arange(n_samp) / sfreq
    band_gain = np.exp(rng.normal(0, subject_spread, size=len(BANDS)))  # per subject
    X = rng.normal(0, 1.0, size=(n_epochs, n_channels, n_samp))
    for k, (lo, hi) in enumerate(BANDS.values()):
        f = rng.uniform(lo, hi, size=(n_epochs, n_channels, 1))
        ph = rng.uniform(0, 2 * np.pi, size=(n_epochs, n_channels, 1))
        amp = STAGE_AMP[stages, k][:, None, None] * band_gain[k]
        amp = amp * (1 + 0.3 * rng.normal(size=(n_epochs, n_channels, 1)))
        X += 3.0 * amp * np.sin(2 * np.pi * f * t + ph)
    return X, stages


def make_dataset(n_subjects=8, n_epochs=240, seed=0, sfreq=100.0):
    rng = np.random.default_rng(seed)
    Xs, ys, gs = [], [], []
    for s in range(n_subjects):
        X, y = make_subject(rng, n_epochs=n_epochs, sfreq=sfreq)
        Xs.append(X); ys.append(y); gs.append(np.full(len(y), s))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs), sfreq
