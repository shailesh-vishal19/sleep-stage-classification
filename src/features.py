"""Spectral and time-domain features per 30-s epoch."""
from __future__ import annotations

import numpy as np
from scipy.signal import welch

_trapz = getattr(np, "trapezoid", None) or np.trapz  # NumPy 1.x / 2.x

BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "sigma": (12.0, 15.0),   # spindle band
    "beta": (15.0, 30.0),
}


def band_powers(X, sfreq):
    """Absolute power per band. X: (n, ch, samples) -> (n, ch, n_bands)."""
    nperseg = int(4 * sfreq)
    freqs, psd = welch(X, fs=sfreq, nperseg=nperseg, axis=-1)
    out = []
    for lo, hi in BANDS.values():
        m = (freqs >= lo) & (freqs < hi)
        out.append(_trapz(psd[..., m], freqs[m], axis=-1))
    return np.stack(out, axis=-1), freqs, psd


def hjorth(X):
    """Hjorth activity, mobility, complexity. X: (n, ch, samples) -> (n, ch, 3)."""
    d1 = np.diff(X, axis=-1)
    d2 = np.diff(d1, axis=-1)
    v0, v1, v2 = X.var(-1), d1.var(-1), d2.var(-1)
    eps = 1e-12
    mobility = np.sqrt(v1 / (v0 + eps))
    complexity = np.sqrt(v2 / (v1 + eps)) / (mobility + eps)
    return np.stack([v0, mobility, complexity], axis=-1)


def spectral_entropy(psd):
    p = psd / (psd.sum(-1, keepdims=True) + 1e-12)
    return -(p * np.log2(p + 1e-12)).sum(-1)


def extract_features(X, sfreq):
    """Feature matrix (n_epochs, n_features) for epochs X (n, ch, samples)."""
    bp, _, psd = band_powers(X, sfreq)
    rel = bp / (bp.sum(-1, keepdims=True) + 1e-12)
    logbp = np.log10(bp + 1e-12)
    i = {b: k for k, b in enumerate(BANDS)}
    ratios = np.stack([
        np.log10((bp[..., i["delta"]] + 1e-12) / (bp[..., i["beta"]] + 1e-12)),
        np.log10((bp[..., i["theta"]] + 1e-12) / (bp[..., i["alpha"]] + 1e-12)),
    ], axis=-1)
    hj = hjorth(X)
    hj[..., 0] = np.log10(hj[..., 0] + 1e-12)   # log activity
    ent = spectral_entropy(psd)[..., None]
    F = np.concatenate([logbp, rel, ratios, hj, ent], axis=-1)
    return F.reshape(len(X), -1)


def add_context(F, groups, offsets=(-2, -1, 1, 2)):
    """Append features of neighbouring epochs (within each subject only).

    Sleep stages are temporally structured, so neighbours help. Epochs must be
    in time order within each subject. Edges repeat the first/last epoch.
    """
    out = [F]
    for off in offsets:
        shifted = np.empty_like(F)
        for g in np.unique(groups):
            idx = np.where(groups == g)[0]
            src = np.clip(np.arange(len(idx)) + off, 0, len(idx) - 1)
            shifted[idx] = F[idx][src]
        out.append(shifted)
    return np.concatenate(out, axis=1)
