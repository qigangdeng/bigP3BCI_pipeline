import mne 
import numpy as np
from typing import Tuple

def apply_filters(
    raw: mne.io.Raw, 
    notch_freq: float = 50.0, 
    bp_low: float = 0.1, 
    bp_high: float = 15.0
    ) -> mne.io.Raw:
    # 1. Notch Filter: Remove power line noise
    # Use a narrow IIR filter for speed and minimal phase distortion
    raw.notch_filter(freqs=notch_freq, method='iir', picks='eeg') 
    
    # 2. Bandpass Filter: Isolate P300 frequencies
    # Method='fir' or 'iir' is based on preference, 'fir' is often safer for BCI
    raw.filter(l_freq=bp_low, h_freq=bp_high, method='fir', phase='zero-double', picks='eeg')
    
    # 4. Map montage to standard 1020
    raw.set_montage("standard_1020", match_case=False, on_missing="ignore")
    return raw # Return the filtered Raw object

def create_epochs(
    raw: mne.io.Raw, 
    events: np.ndarray, 
    tmin: float = -0.2, 
    tmax: float = 0.8, 
    baseline: Tuple[float, float] = (-0.2, 0.0)
    ) -> mne.Epochs:
    """
    Creates MNE Epochs from the raw data and events, applying baseline correction.
    """
    # Define your standard P300 Event IDs (Target=1, Non-Target=0)
    event_id = {"non": 0, "target": 1}
    
    # MNE epochs creation segments the data
    epochs = mne.Epochs(
        raw=raw, 
        events=events, 
        event_id=event_id, 
        tmin=tmin, 
        tmax=tmax, 
        baseline=baseline, # CRITICAL: Baseline correction happens here
        preload=True,      # Preload to RAM for faster access
        reject_by_annotation=False # Assuming we handled artifacts elsewhere
    )
    epochs.set_eeg_reference(ref_channels="average", projection=True)
    return epochs # Return the clean, segmented Epochs object
