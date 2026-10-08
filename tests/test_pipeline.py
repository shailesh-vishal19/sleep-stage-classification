import numpy as np
import mne

from src.data import raw_to_epochs
from src.features import add_context, extract_features
from src.synthetic import make_dataset


def _fake_raw(stage_seq, sfreq=100.0, n_ch=2):
    """RawArray with one 30-s annotation per stage, like a Sleep-EDF hypnogram."""
    n = len(stage_seq)
    data = np.random.default_rng(0).normal(size=(n_ch, int(n * 30 * sfreq)))
    raw = mne.io.RawArray(data, mne.create_info(["a", "b"], sfreq, "eeg"), verbose=False)
    names = {0: "Sleep stage W", 1: "Sleep stage 1", 2: "Sleep stage 2",
             3: "Sleep stage 3", 4: "Sleep stage R"}
    raw.set_annotations(mne.Annotations(
        onset=np.arange(n) * 30.0, duration=[30.0] * n,
        description=[names[s] for s in stage_seq]), emit_warning=False)
    return raw


def test_epoching_labels_and_wake_cropping():
    # 100 wake, 20 sleep (N2 then REM), 100 wake -> keep 30 min = 60 wake epochs each side
    seq = [0] * 100 + [2] * 10 + [4] * 10 + [0] * 100
    X, y, sfreq = raw_to_epochs(_fake_raw(seq), wake_margin_min=30.0)
    assert X.shape[1:] == (2, 3000) and sfreq == 100.0
    assert len(y) == 60 + 20 + 60
    assert (y == 2).sum() == 10 and (y == 4).sum() == 10
    # order preserved: first kept wake epoch precedes the sleep block
    assert list(y[60:70]) == [2] * 10


def test_stage_3_and_4_merge_into_n3():
    raw = _fake_raw([0, 2, 3, 3, 2])
    raw.annotations.description[3] = "Sleep stage 4"
    _, y, _ = raw_to_epochs(raw, wake_margin_min=30.0)
    assert list(y) == [0, 2, 3, 3, 2]


def test_features_shape_and_finite():
    X, y, groups, sfreq = make_dataset(n_subjects=2, n_epochs=30)
    F = extract_features(X, sfreq)
    assert F.shape[0] == len(y) and np.isfinite(F).all()


def test_context_does_not_cross_subjects():
    F = np.arange(6, dtype=float).reshape(6, 1)
    groups = np.array([0, 0, 0, 1, 1, 1])
    out = add_context(F, groups, offsets=(-1, 1))
    # epoch 3 is the first of subject 1: its "previous" must be itself, not epoch 2
    assert out[3, 1] == 3.0
    # epoch 2 is the last of subject 0: its "next" must be itself, not epoch 3
    assert out[2, 2] == 2.0
