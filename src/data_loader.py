import mne 
import numpy as np 
from typing import Tuple, Dict, List, Optional 

class BCIDataLoader: 
    """
    A data loader for BCI2000 data in EDF format. 
    It handles file loading, channel standardization, and event extraction from
    BCI2000 states.
    return raw and events array as (onsets, dummy, labels)
    """
    def __init__(
        self, 
        file_path: str, 
        resample_rate: float = 256.0
        ):
        self.resample_rate = resample_rate

    def _resample_standardise(self, raw:mne.io.Raw) -> mne.io.Raw:
        if abs(raw.info["sfreq"]-self.resample_rate) > 1e-6:
            return raw.resample(self.resample_rate)
        return raw

    def _extract_events(self, raw: mne.io.Raw) -> np.ndarray:
        """
        Extracts MNE-formatted events from BCI2000 state channels 
        (StimulusBegin/StimulusType).
        """
        # 1. Access Data Efficiently (The core improvement)
        # We only copy/pick the necessary channels once.
        if not ({"StimulusBegin", "StimulusType"}.issubset(set(raw.ch_names))):
            # Handle error: channels not found 
            # (e.g., if you only picked EEG channels previously)
            raise ValueError(
                "Missing BCI2000 state channels: 'StimulusBegin', 'StimulusType'."
            )

        # Get the data for the two state channels (Efficiently using .get_data())
        stimulus_begin = raw.copy().pick(["StimulusBegin"]).get_data().ravel()
        stimulus_type = raw.copy().pick(["StimulusType"]).get_data().ravel()
        
        # 2. Extract Onsets (Timing the Flash)
        # Find the transition from 'off' (<0.5) to 'on' (>=0.5)
        # We add +1 to get the sample index *after* the change starts
        onsets = np.where(
            (stimulus_begin[:-1] < 0.5) & 
            (stimulus_begin[1:] >= 0.5)
            )[0] + 1
        
        if onsets.size == 0:
            return np.array([]) # Return empty array if no stimuli are found

        # 3. Extract Labels (Target or Non-Target)
        # Get the StimulusType value (0 or 1) at each onset index
        labels = (stimulus_type[onsets] > 0).astype(int)
        
        # 4. Construct the MNE Events Array
        # Format: [sample_index, previous_value (0), event_ID (0 or 1)]
        events = np.column_stack([
            onsets, 
            np.zeros_like(onsets), 
            labels]).astype(int)
        
        return events

    # I/O operations
    def load_raw_and_events(self, file_path: str) -> Tuple[mne.io.Raw, List[int]]:
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        raw = self._resample_standardise(raw)
        events = self._extract_events(raw)
        return raw, events



