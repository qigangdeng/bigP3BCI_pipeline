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
from src.data_loader import BCIDataLoader # Assuming your class is in src/data_loader.py

# --- REAL edf file to check if the dataloader can give correct 
# sfreq and time windows 
raw_data_dir = r"/Users/qigangdeng/Downloads/BIGP3BCI/physionet.org/files/bigp3bci/1.0.0/bigP3BCI-data/StudyA/A_01/SE001/Test/RC/A_01_SE001_RC_Test07.edf"

# --- FIXTURES FOR MOCK RAW OBJECTS ---

@pytest.fixture
def mock_raw_512hz():
    """Returns a mock Raw object with a high sampling rate (512 Hz)."""
    sfreq_high = 512.0
    info = mne.create_info(
        ch_names=['EEG 001', 'EEG 002'], 
        sfreq=sfreq_high, 
        ch_types='eeg'
    )
    data = np.random.randn(2, 512 * 5) # 5 seconds of data
    return mne.io.RawArray(data, info, verbose=False)

@pytest.fixture
def mock_raw_256hz():
    """Returns a mock Raw object with the target sampling rate (256 Hz)."""
    sfreq_target = 256.0
    info = mne.create_info(
        ch_names=['EEG 001', 'EEG 002'], 
        sfreq=sfreq_target, 
        ch_types='eeg'
    )
    data = np.random.randn(2, 256 * 5) # 5 seconds of data
    return mne.io.RawArray(data, info, verbose=False)

@pytest.fixture 
def mock_raw_events():
    """
    Returns a mock Raw object with EEG channels,
    as well as StimulusBegin and StimulusType state channels
    suitable for event extraction tests.
    """
    sfreq = 256.0
    n_samples = int(sfreq * 2)  # 2 seconds of data

    # Define all channels (2 EEG, 2 state)
    ch_names = ['EEG 001', 'EEG 002', 'StimulusBegin', 'StimulusType']
    ch_types = ['eeg', 'eeg', 'stim', 'stim']
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    eeg_data = np.random.randn(2, n_samples)

    # StimulusBegin transitions from 0 to 1 to mark event onset, else 0
    stimulus_begin = np.zeros(n_samples)
    stimulus_type = np.zeros(n_samples)

    # Inject an "event" at sample 100
    onset_idx = 100
    stimulus_begin[onset_idx] = 1.0  # A rising edge—indicating new event

    # Mark StimulusType as active (1) at that event, rest 0
    stimulus_type[onset_idx] = 1.0

    # Stack all data
    all_data = np.vstack([eeg_data, stimulus_begin[np.newaxis, :], stimulus_type[np.newaxis, :]])

    raw = mne.io.RawArray(all_data, info, verbose=False)
    return raw

@pytest.fixture
def mock_raw_with_events():
    """Fixture simulating BCI2000 state channels at 10Hz for easy index checking."""
    sfreq = 10.0
    info = mne.create_info(
        ch_names=['EEG', 'StimulusBegin', 'StimulusType'],
        sfreq=sfreq,
        ch_types=['eeg', 'stim', 'stim']
    )
    
    # Create data for 10 samples (1 second)
    # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 
    
    # Target flash at index 2, Non-Target flash at index 6
    stimulus_begin = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0])
    stimulus_type  = np.array([0, 0, 1, 1, 0, 0, 0, 0, 0, 0]) 
    
    data = np.vstack([
        np.random.randn(1, 10), # EEG channel (ignored by _extract_events)
        stimulus_begin, 
        stimulus_type
    ])
    
    raw = mne.io.RawArray(data, info, verbose=False)
    # We set resample_rate to 10.0 Hz to ensure no resampling occurs during testing
    loader = BCIDataLoader(resample_rate=10.0) 
    return raw, loader

@pytest.fixture
def mock_none_stimulusbegin():
    sfreq = 10.0
    info = mne.create_info(
        ch_names=['EEG', 'StimulusBegin', 'StimulusType'],
        sfreq=sfreq,
        ch_types=['eeg', 'stim', 'stim']
    )
    
    # Create data for 10 samples (1 second)
    # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 
    
    # Target flash at index 2, Non-Target flash at index 6
    stimulus_begin = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    stimulus_type  = np.array([0, 0, 1, 1, 0, 0, 0, 0, 0, 0]) 
    
    data = np.vstack([
        np.random.randn(1, 10), # EEG channel (ignored by _extract_events)
        stimulus_begin, 
        stimulus_type
    ])
    
    raw = mne.io.RawArray(data, info, verbose=False)
    # We set resample_rate to 10.0 Hz to ensure no resampling occurs during testing
    loader = BCIDataLoader(resample_rate=10.0) 
    return raw

# --- UNIT TESTS ---

def test_resample_standardise_resample_needed(mock_raw_512hz):
    """
    Tests the condition where raw.sfreq != target_sfreq (512Hz -> 256Hz).
    Verifies that the new sampling frequency is correct.
    """
    # 1. Arrange
    target_sfreq = 256.0
    loader = BCIDataLoader(resample_rate=target_sfreq)
    
    # Check initial state
    assert mock_raw_512hz.info['sfreq'] == 512.0
    assert mock_raw_512hz.n_times == 512 * 5

    # 2. Act
    resampled_raw = loader._resample_standardise(mock_raw_512hz)

    # 3. Assert
    # Check that the sampling frequency has been correctly updated
    assert resampled_raw.info['sfreq'] == pytest.approx(target_sfreq)
    
    # Check that the data length has changed (downsampling reduces n_times)
    # Original length was 512 * 5 = 2560 samples. New length should be 256 * 5 = 1280 samples.
    assert resampled_raw.n_times < mock_raw_512hz.n_times


def test_resample_standardise_no_resample(mock_raw_256hz):
    """
    Tests the condition where raw.sfreq is already close to target_sfreq (256Hz -> 256Hz).
    Verifies that the sampling frequency and data length remain unchanged.
    """
    # 1. Arrange
    target_sfreq = 256.0
    loader = BCIDataLoader(resample_rate=target_sfreq)
    
    # Store original data length for comparison
    original_n_times = mock_raw_256hz.n_times

    # 2. Act
    unchanged_raw = loader._resample_standardise(mock_raw_256hz)

    # 3. Assert
    # Check that the sampling frequency is still the target frequency
    assert unchanged_raw.info['sfreq'] == pytest.approx(target_sfreq)
    
    # Check that the data length is unchanged (no resampling occurred)
    assert unchanged_raw.n_times == original_n_times

    # Check that the returned object is the same underlying data structure (if possible)
    # Using np.array_equal is safer than checking object identity here due to MNE internals
    assert np.array_equal(unchanged_raw.get_data(), mock_raw_256hz.get_data())


def test_extract_events(mock_raw_events):
    loader = BCIDataLoader()
    events = loader._extract_events(mock_raw_events)
    assert events.shape == (1, 3)
    assert events[0, 0] == 100
    assert events[0, 1] == 0
    assert events[0, 2] == 1

def test_extract_events_no_events(mock_raw_256hz):
    loader = BCIDataLoader()
    with pytest.raises(ValueError, match="Missing BCI2000 state channels: 'StimulusBegin', 'StimulusType'."):
        loader._extract_events(mock_raw_256hz)

def test_extract_events_standard_and_alignment(mock_raw_with_events):
    """
    Checks if the onsets (column 0) are correct and if labels (column 2) 
    align correctly with StimulusType at those onsets.
    """
    raw, loader = mock_raw_with_events
    
    # 1. Act
    events = loader._extract_events(raw)
    
    # 2. Assert
    
    # Expected onsets (0-based sample indices from the table above)
    expected_onsets = np.array([2, 6])
    
    # Expected MNE labels (1=Target, 0=Non-Target)
    expected_labels = np.array([1, 0])
    
    # --- CHECK 1: The Critical Onset Detection (Column 0) ---
    # Assert that the first column of the events array matches the expected onsets
    assert np.array_equal(events[:, 0], expected_onsets), "Onset indices are incorrect"

    # --- CHECK 2: The Critical Label Alignment (Column 2) ---
    # Assert that the third column (the event ID/label) matches the expected labels
    assert np.array_equal(events[:, 2], expected_labels), "Labels are incorrectly aligned or mapped"
    
    # Check overall shape: (2 events, 3 columns)
    assert events.shape == (2, 3)

def test_extract_events_none_stimulusbegin(mock_none_stimulusbegin):
    loader = BCIDataLoader()
    events = loader._extract_events(mock_none_stimulusbegin)
    assert events.size == 0

def test_epoch_length_and_sfreq_on_real_file():
    loader = BCIDataLoader(resample_rate=256.0)

    # Pick a single known EDF path to keep this test light
    edf_path = raw_data_dir

    raw, events = loader.load_raw_and_events(edf_path)
    assert raw.info["sfreq"] == pytest.approx(256.0)

    tmin, tmax = -0.2, 0.8
    epochs = loader.create_epochs(raw, events, tmin=tmin, tmax=tmax, baseline=(tmin, 0.0))

    sfreq = epochs.info["sfreq"]
    n_times = epochs.get_data().shape[-1]
    times = epochs.times  # length n_times

    # 1) sfreq should be ~256 Hz
    assert sfreq == pytest.approx(256.0, rel=1e-4)

    # 2) Time axis should start/end where we expect (within one sample)
    assert times[0] == pytest.approx(tmin, abs=1.0 / sfreq)
    assert times[-1] == pytest.approx(tmax, abs=1.0 / sfreq)

    # 3) Duration should be close to (tmax - tmin)
    duration = times[-1] - times[0]
    assert duration == pytest.approx(tmax - tmin, rel=1e-3, abs=1e-3)

    # (Optional) sanity check on n_times: allow ±1 sample due to floating-point / EDF sfreq
    theoretical = (tmax - tmin) * sfreq
    assert abs(n_times - theoretical) <= 1, (
        f"Epoch length {n_times} too far from theoretical {theoretical}"
    )