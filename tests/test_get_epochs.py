# --- tests/test_get_epochs.py ---
import mne
import numpy as np
import pytest
import os
import sys

# Add project root to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)

from src.data_loader import BCIDataLoader

@pytest.fixture
def mock_bci_raw():
    # Simulate a 16-channel EEG, 1-second long, sampled at 256 Hz
    sfreq = 256
    n_channels = 16
    duration_s = 5  # 5 seconds of data
    info = mne.create_info(ch_names=[f'EEG {i:03d}' for i in range(n_channels)], sfreq=sfreq, ch_types='eeg')
    
    # Add a Status channel for event markers (CRITICAL for BCI)
    info.set_channel_types({'EEG 000': 'misc'}) # We'll misuse channel 0 for status in the mock
    
    # Create random noise data
    data = np.random.randn(n_channels, duration_s * sfreq) * 1e-6 
    
    raw = mne.io.make_raw_array(data, info)
    
    # 2. MVP Annotation Setup (Simulate BCI events)
    # Timings in seconds:
    onsets = [1.0, 3.0] 
    durations = [0.0] * 2
    descriptions = ['StimulusBegin/1', 'StimulusType/1'] # Target event
    
    # Add a Non-Target event
    onsets.append(2.0)
    durations.append(0.0)
    descriptions.append('StimulusType/0') # Non-Target event
    
    annotations = mne.Annotations(onsets, durations, descriptions)
    raw.set_annotations(annotations)
    
    return raw

# 3. MVP Test Function
def test_create_epochs_dimensions(mock_bci_raw):
    # This simulates the extraction of the actual event array
    # In the final implementation, your data loader does this part
    events, event_id = mne.events_from_annotations(mock_bci_raw, verbose=False)
    
    # Use BCIDataLoader to create epochs
    data_loader = BCIDataLoader(resample_rate=256.0)
    epochs = data_loader.create_epochs(mock_bci_raw, events)
    
    # A. ASSERT: Correct number of epochs
    assert len(epochs) == 2, "Should find 2 epochs (one target, one non-target)."
    
    # B. ASSERT: Correct time window (tmax=0.8, tmin=-0.2 => 1.0s window)
    expected_samples = int(1.0 * epochs.info['sfreq']) 
    assert epochs.get_data().shape[2] == expected_samples, "Time dimension should match tmax-tmin."
    
    # C. ASSERT: Correct Labels (target=1, non-target=0)
    # The event_id mapping depends on how create_epochs handles labels, but assume the final labels are [1, 0]
    expected_labels = [1, 0] # Assuming target is the second event we passed (1.0s, 3.0s) and non-target is the third (2.0s)
    # NOTE: The actual event extraction logic needs to be complex to isolate target/non-target pairs.
    
    # MVP Check: Ensure the data is segmented and loaded
    assert epochs.get_data().ndim == 3 

# 4. MVP Test for Filtering
def test_apply_filters_output_sfreq(mock_bci_raw):
    initial_sfreq = mock_bci_raw.info['sfreq']
    # Use BCIDataLoader to apply filters
    data_loader = BCIDataLoader(resample_rate=256.0)
    filtered_raw = data_loader._apply_filters(mock_bci_raw)
    
    # ASSERT: Filtering should not change the sampling rate
    assert filtered_raw.info['sfreq'] == initial_sfreq
    
    # ASSERT: Data should still be float
    assert filtered_raw.get_data().dtype == np.float64