import pytest
import numpy as np 
import mne 
import os
import sys
from unittest.mock import Mock, patch
from typing import List, Tuple
from unittest.mock import Mock, patch

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
# Insert the project root into the system path so Python can find 'src'
sys.path.insert(0, project_root)
from src.preprocessing import apply_filters, create_epochs


@pytest.fixture 
def mock_raw():
    info = mne.create_info(
        ch_names=["Cz", "Pz"], 
        sfreq=256.0,
        ch_types='eeg'
    )
    data = np.vstack([
        np.random.randn(2, 256 * 200),

    ])
    raw = mne.io.RawArray(
        data, info, verbose=False
        )
    return raw

@pytest.fixture
def mock_raw_with_events():
    # Create info for EEG channels + events
    info = mne.create_info(
        ch_names=["Cz", "Pz", "StimulusBegin", "StimulusType"],
        sfreq=256.0,
        ch_types=['eeg', 'eeg', 'stim', 'stim']
    )
    n_samples = 256 * 10
    eeg_data = np.random.randn(2, n_samples)
    # Simulate 3 events at sample 100, 200, 300
    stimulus_begin = np.zeros(n_samples)
    stimulus_type = np.zeros(n_samples)
    # Onset events: will create two events with type 1, one with 0
    event_indices = [100, 200, 300]
    stimulus_begin[event_indices] = 1.0
    stimulus_type[event_indices] = [1, 0, 1]

    data = np.vstack([eeg_data, stimulus_begin[np.newaxis, :], stimulus_type[np.newaxis, :]])

    raw = mne.io.RawArray(data, info, verbose=False)
    return raw

# --- UNIT TESTS ---
def test_apply_filters_return_raw(mock_raw):
    raw = apply_filters(mock_raw)
    assert isinstance(raw, mne.io.RawArray)
    assert raw.info['ch_names'] == ["Cz", "Pz"]
    assert raw.info['sfreq'] == 256.0
    assert raw.get_data().shape == (2, 256 * 200)


def test_create_epochs_return_epochs(mock_raw_with_events):
    events = np.array([[100, 0, 1], [200, 0, 0], [300, 0, 1]    ])
    epochs = create_epochs(mock_raw_with_events, events)
    assert isinstance(epochs, mne.Epochs)
    assert epochs.info['ch_names'] == ["Cz", "Pz", "StimulusBegin", "StimulusType"]
    assert epochs.info['sfreq'] == 256.0
    