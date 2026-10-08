"""Band-power features + leave-one-subject-out (LOSO) evaluation for sleep staging.

Put this file next to data.py (src/) and run from the project root:

    python src/loso_eval_v2.py --subjects 0 1 2 3 4 5 6 7 8 9
    python src/loso_eval_v2.py --subjects 0 1 2 3 4 5 6 7 8 9 --context 2 --balanced
    python src/loso_eval_v2.py --subjects 0 1 2 3 4 5 6 7 8 9 --shuffle   # sanity check

Outputs go to results/loso_<options>/ (e.g. results/loso_ctx2_bal/):
    classification_report.txt, confusion_matrix.png, per_subject.csv,
    spectrum_by_stage.png, feature_importance.png
"""
from __future__ import annotations

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import LeaveOneGroupOut

STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "sigma": (12.0, 15.0),
    "beta": (15.0, 30.0),
}
CHANNEL_NAMES = ("Fpz-Cz", "Pz-Oz")


def _welch(X, sfreq):
    """PSD with 4-s windows (0.25 Hz resolution). X: (..., n_samples)."""
    freqs, psd = welch(X, fs=sfreq, nperseg=int(4 * sfreq), axis=-1)
    return freqs, psd.astype(np.float64)


def band_power_features(X, sfreq, channels=CHANNEL_NAMES):
    """Per channel and band: log10 absolute power and relative power.

    X: (n_epochs, n_channels, n_samples). Returns F (n_epochs, n_features), names.
    """
    assert X.shape[1] == len(channels), "channels must match X.shape[1]"
    freqs, psd = _welch(X, sfreq)
    df = freqs[1] - freqs[0]
    total = psd[..., (freqs >= 0.5) & (freqs < 30.0)].sum(-1) * df  # (n_epochs, n_ch)

    feats, names = [], []
    for ci, ch in enumerate(channels):
        for band, (lo, hi) in BANDS.items():
            p = psd[:, ci, (freqs >= lo) & (freqs < hi)].sum(-1) * df
            feats.append(np.log10(p + 1e-20))
            names.append(f"{ch}_{band}_logabs")
            feats.append(p / (total[:, ci] + 1e-20))
            names.append(f"{ch}_{band}_rel")
    return np.stack(feats, axis=1), names


def add_context(F, groups, k=2):
    """Stack features of k epochs before and after, within the same subject only.

    Edges are padded by repeating the first/last epoch. Assumes epochs are in
    time order within each subject (load_dataset returns them that way).
    """
    if k == 0:
        return F
    out = np.empty((F.shape[0], F.shape[1] * (2 * k + 1)), dtype=F.dtype)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        padded = np.pad(F[idx], ((k, k), (0, 0)), mode="edge")
        out[idx] = np.concatenate([padded[i:i + len(idx)] for i in range(2 * k + 1)], axis=1)
    return out


def run_loso(F, y, groups, model=None):
    """Leave-one-subject-out. Returns pooled predictions and per-subject metrics."""
    if model is None:
        model = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
    y_pred = np.empty_like(y)
    rows = []
    for train_idx, test_idx in LeaveOneGroupOut().split(F, y, groups):
        m = clone(model).fit(F[train_idx], y[train_idx])
        y_pred[test_idx] = m.predict(F[test_idx])
        yt, yp = y[test_idx], y_pred[test_idx]
        rows.append(
            {
                "subject": int(groups[test_idx][0]),
                "n_epochs": len(test_idx),
                "acc": accuracy_score(yt, yp),
                "macro_f1": f1_score(yt, yp, average="macro"),
                "kappa": cohen_kappa_score(yt, yp),
            }
        )
    return {"y_true": y, "y_pred": y_pred, "per_subject": rows}


def save_report(res, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    y_true, y_pred, rows = res["y_true"], res["y_pred"], res["per_subject"]

    report = classification_report(
        y_true, y_pred, labels=range(len(STAGE_NAMES)),
        target_names=STAGE_NAMES, digits=3, zero_division=0,
    )
    print(report)
    with open(os.path.join(out_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred, labels=range(len(STAGE_NAMES)), normalize="true")
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay(cm, display_labels=STAGE_NAMES).plot(
        ax=ax, cmap="Blues", values_format=".2f", colorbar=False
    )
    ax.set_title("Confusion matrix (LOSO, row-normalised)")
    fig.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    with open(os.path.join(out_dir, "per_subject.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    for key in ("acc", "macro_f1", "kappa"):
        v = np.array([r[key] for r in rows])
        print(f"{key:9s} mean={v.mean():.3f}  std={v.std(ddof=1) if len(v) > 1 else 0:.3f}  "
              f"min={v.min():.3f}  max={v.max():.3f}  (n_subjects={len(v)})")


def plot_spectrum_by_stage(X, y, sfreq, path, ch=0, ch_name="Fpz-Cz"):
    freqs, psd = _welch(X[:, ch], sfreq)
    m = (freqs >= 0.5) & (freqs <= 30.0)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k, name in enumerate(STAGE_NAMES):
        if (y == k).any():
            ax.semilogy(freqs[m], psd[y == k][:, m].mean(0), label=name)
    for lo, hi in BANDS.values():
        ax.axvline(lo, color="grey", lw=0.5, ls=":")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Mean PSD (V$^2$/Hz)")
    ax.set_title(f"Average power spectrum by sleep stage ({ch_name})")
    ax.legend()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(F, y, names, path, top=15):
    """Descriptive only: RF fitted on all data. Not a performance estimate."""
    rf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1).fit(F, y)
    order = np.argsort(rf.feature_importances_)[::-1][:top][::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh([names[i] for i in order], rf.feature_importances_[order])
    ax.set_xlabel("Random forest importance")
    ax.set_title("Top band-power features")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    from data import load_dataset  # imported here so the functions above work without MNE

    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--context", type=int, default=0,
                    help="neighbouring epochs on each side (0 = single epoch)")
    ap.add_argument("--balanced", action="store_true",
                    help="class_weight='balanced' in the random forest")
    ap.add_argument("--shuffle", action="store_true",
                    help="sanity check: shuffle labels, accuracy should fall to chance")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    k = args.context
    tag = f"ctx{k}" + ("_bal" if args.balanced else "") + ("_shuffled" if args.shuffle else "")
    out = args.out or os.path.join("results", f"loso_{tag}")

    X, y, groups, sfreq = load_dataset(args.subjects)
    X = X.astype(np.float32)
    print(f"{len(y)} epochs, {len(set(groups))} subjects, class counts {np.bincount(y)}")
    print(f"options: context={k}, balanced={args.balanced}, shuffle={args.shuffle}")

    F, names = band_power_features(X, sfreq)
    F = add_context(F, groups, k)
    if k > 0:
        names = [f"{n}_t{o:+d}" for o in range(-k, k + 1) for n in names]

    y_eval = np.random.default_rng(0).permutation(y) if args.shuffle else y
    model = RandomForestClassifier(
        n_estimators=300, random_state=0, n_jobs=-1,
        class_weight="balanced" if args.balanced else None,
    )
    res = run_loso(F, y_eval, groups, model)
    os.makedirs(out, exist_ok=True)
    save_report(res, out)
    if not args.shuffle:
        plot_spectrum_by_stage(X, y, sfreq, os.path.join(out, "spectrum_by_stage.png"))
        plot_feature_importance(F, y, names, os.path.join(out, "feature_importance.png"))
    print(f"saved to {out}/")


if __name__ == "__main__":
    main()
