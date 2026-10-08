"""Loading Sleep-EDF recordings and turning them into labelled 30-s epochs."""
from __future__ import annotations

import numpy as np
import mne

STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]

# Stages 3 and 4 are merged into N3 (AASM convention). Ids start at 1 because
# MNE does not like event id 0; they are shifted to 0..4 afterwards.
ANNOT_TO_EVENT_ID = {
    "Sleep stage W": 1,
    "Sleep stage 1": 2,
    "Sleep stage 2": 3,
    "Sleep stage 3": 4,
    "Sleep stage 4": 4,
    "Sleep stage R": 5,
}
DEFAULT_CHANNELS = ("EEG Fpz-Cz", "EEG Pz-Oz")
EPOCH_SEC = 30.0


def raw_to_epochs(raw, wake_margin_min: float = 30.0):
    """Cut a raw recording (with hypnogram annotations) into 30-s epochs.

    Wake far from the sleep period is dropped: only `wake_margin_min` minutes
    of wake before the first and after the last sleep epoch are kept, otherwise
    wake dominates the class balance.

    Returns X (n_epochs, n_channels, n_samples), y (0..4, see STAGE_NAMES), sfreq.
    """
    events, _ = mne.events_from_annotations(
        raw, event_id=ANNOT_TO_EVENT_ID, chunk_duration=EPOCH_SEC, verbose=False
    )
    if len(events) == 0:
        raise ValueError("No sleep-stage annotations found in recording.")

    sfreq = raw.info["sfreq"]
    wake_id = ANNOT_TO_EVENT_ID["Sleep stage W"]
    sleep_rows = events[events[:, 2] != wake_id]
    if len(sleep_rows):
        margin = int(wake_margin_min * 60 * sfreq)
        lo = sleep_rows[0, 0] - margin
        hi = sleep_rows[-1, 0] + int(EPOCH_SEC * sfreq) + margin
        events = events[(events[:, 0] >= lo) & (events[:, 0] < hi)]

    tmax = EPOCH_SEC - 1.0 / sfreq
    epochs = mne.Epochs(
        raw, events, event_id=None, tmin=0.0, tmax=tmax, baseline=None,
        preload=True, on_missing="ignore", verbose=False,
    )
    X = epochs.get_data()
    y = epochs.events[:, 2].astype(int) - 1
    return X, y, sfreq


def load_subject(psg_path, hyp_path, channels=DEFAULT_CHANNELS,
                 l_freq: float = 0.3, h_freq: float = 35.0,
                 wake_margin_min: float = 30.0):
    """Read one night (PSG + hypnogram EDF pair), filter, and epoch it."""
    raw = mne.io.read_raw_edf(psg_path, include=list(channels), preload=True, verbose=False)
    raw.set_annotations(mne.read_annotations(hyp_path), emit_warning=False)
    raw.pick(list(channels))
    raw.filter(l_freq, h_freq, fir_design="firwin", verbose=False)
    return raw_to_epochs(raw, wake_margin_min)


def load_dataset(subject_ids, data_path=None, recording=(1,)):
    """Download (via MNE) and load several Sleep-EDF cassette subjects.

    Returns X, y, groups (subject id per epoch; epochs in time order), sfreq.
    """
    from mne.datasets.sleep_physionet.age import fetch_data

    files = fetch_data(subjects=list(subject_ids), recording=list(recording),
                       path=data_path, verbose=False)
    Xs, ys, gs, sfreq = [], [], [], None
    for (psg, hyp) in files:
        # file names look like SC4001E0-PSG.edf -> subject number 00
        sid = int(psg.split("SC4")[-1][:2])
        X, y, sfreq = load_subject(psg, hyp)
        Xs.append(X); ys.append(y); gs.append(np.full(len(y), sid))
        print(f"subject {sid:02d}: {len(y)} epochs")
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs), sfreq
