import pytest
import numpy as np 
import mne 
import sys
import os

from unittest.mock import Mock, patch
from typing import List, Tuple

# Get the absolute path to the directory two levels up (the project root)
# The test file is in 'project_root/tests/', so we go up twice.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))

# Insert the project root into the system path so Python can find 'src'
sys.path.insert(0, project_root)
from src import run_session
from src import data_loader 


BASE_CHS = ["F3", "Fz"]


# --- MOCK FIXTURE ---

# Helper function to create a realistic mock raw/events output
def create_mock_raw_events(n_samples: int, local_event_index: int) -> Tuple[mne.io.Raw, np.ndarray]:
    """Creates a synthetic raw object and a single event marker."""
    
    # 1. Create a minimal info structure (e.g., 2 EEG channels at 256 Hz)
    info = mne.create_info(ch_names=BASE_CHS, sfreq=256.0, ch_types='eeg')
    
    # 2. Create data (2 channels x N_samples)
    data = np.random.randn(2, n_samples)
    raw = mne.io.RawArray(data, info, verbose=False)
    
    # 3. Create the MNE event array ([sample_index, previous_value, event_ID])
    events = np.array([[local_event_index, 0, 1]], dtype=int)
    
    return raw, events

@pytest.fixture
def mock_loader_output() -> List[Tuple[mne.io.Raw, np.ndarray]]:
    """Fixture returning the data sequence for the mocked loader."""
    
    # File 1: Length 1000, Event at 500
    raw1, events1 = create_mock_raw_events(n_samples=1000, local_event_index=500)
    
    # File 2: Length 1500, Event at 300
    raw2, events2 = create_mock_raw_events(n_samples=1500, local_event_index=300)

    return [
        (raw1, events1),
        (raw2, events2)
    ]

# --- UNIT TEST ---

# We use @patch to replace the real BCIDataLoader with a Mock object
@patch('src.run_session.BCIDataLoader')
def test_event_time_alignment(MockBCIDataLoader, mock_loader_output):
    """
    Tests if the load_data_from_session function correctly shifts event indices 
    after concatenation.
    """
    
    # Configure the Mock Loader to return our fixture data sequentially
    # The Mock needs to return an object (the loader instance) whose method 
    # returns the data.
    mock_instance = MockBCIDataLoader.return_value
    
    # The mock must simulate the behavior of calling the method for each file
    # The side_effect makes it return an item from the list on each call
    mock_instance.load_raw_and_events.side_effect = mock_loader_output

    # --- EXECUTION ---
    # We pass None for file_list, but we don't need real files since the loader is mocked
    combined_raw, combined_events = run_session.load_data_from_session(
        session_dir="/fake/dir", 
        phase="Train", 
        file_list=["fake_file1.edf", "fake_file2.edf"] # Inject two file paths
    )

    # --- ASSERTIONS ---
    
    # 1. Check Total Length (Data Integrity)
    # Total length should be 1000 + 1500 = 2500 samples
    assert combined_raw.n_times == 2500, "Combined raw length is incorrect"

    # 2. Check Event Alignment (The CRITICAL Test)
    # The first event (from File 1) should be at its local index (500)
    assert combined_events[0, 0] == 500, "First event was incorrectly offset"

    # The second event (from File 2) should be shifted by the length of File 1 (1000 samples)
    # Local index was 300. Global index should be 1000 + 300 = 1300
    expected_second_index = 1000 + 300
    assert combined_events[1, 0] == expected_second_index, "Second event was not correctly shifted"

    # 3. Check Final Shape
    assert combined_events.shape == (2, 3), "Combined events should have 2 rows (2 files/events)"