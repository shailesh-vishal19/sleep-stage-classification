"""Evaluation: subject-wise vs random CV, metrics, plots."""
from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                             confusion_matrix, f1_score)
from sklearn.model_selection import GroupKFold, StratifiedKFold

from .data import STAGE_NAMES


def metrics(y_true, y_pred):
    labels = list(range(len(STAGE_NAMES)))
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "per_class_f1": {n: float(v) for n, v in zip(STAGE_NAMES, per_class)},
    }


def cross_validate(model, F, y, groups, scheme="subject", n_splits=5, seed=0):
    """Out-of-fold predictions.

    scheme="subject": GroupKFold, no subject appears in both train and test.
    scheme="random":  stratified shuffled epoch split (leaks subject identity;
                      shown only to quantify how much it inflates scores).
    """
    n_splits = min(n_splits, len(np.unique(groups))) if scheme == "subject" else n_splits
    if scheme == "subject":
        splits = GroupKFold(n_splits=n_splits).split(F, y, groups)
    elif scheme == "random":
        splits = StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(F, y)
    else:
        raise ValueError(scheme)

    y_pred = np.full_like(y, -1)
    for tr, te in splits:
        m = clone(model).fit(F[tr], y[tr])
        y_pred[te] = m.predict(F[te])
    return y_pred


def plot_confusion(y_true, y_pred, path, title="Subject-wise CV"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(range(len(STAGE_NAMES)))
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(labels, STAGE_NAMES); ax.set_yticks(labels, STAGE_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in labels:
        for j in labels:
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_leakage(subject_m, random_m, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = ["accuracy", "macro_f1", "kappa"]
    x = np.arange(len(keys)); w = 0.38
    fig, ax = plt.subplots(figsize=(5, 3.8))
    ax.bar(x - w / 2, [subject_m[k] for k in keys], w, label="Subject-wise split")
    ax.bar(x + w / 2, [random_m[k] for k in keys], w, label="Random epoch split (leaky)")
    ax.set_xticks(x, ["Accuracy", "Macro-F1", "Kappa"]); ax.set_ylim(0, 1)
    ax.legend(fontsize=8); ax.set_title("Effect of split strategy")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
