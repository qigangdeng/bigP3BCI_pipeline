import mne 
import numpy as np 
from pathlib import Path
from typing import Tuple, Dict, List, Optional 
from .utils import edf_list 

class BCIDataLoader: 
    """
    A data loader for BCI2000 data in EDF format. 
    It handles file loading, channel standardization, and event extraction from
    BCI2000 states.
    return raw and events array as (onsets, dummy, labels)
    """
    def __init__(
        self, 
        resample_rate: float = 256.0
        ):
        self.resample_rate = resample_rate

    def _resample_standardise(self, raw:mne.io.Raw) -> mne.io.Raw:
        if abs(raw.info["sfreq"]-self.resample_rate) > 1e-6:
            return raw.copy().resample(self.resample_rate)
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

    def _apply_filters(
        self,
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
        self,
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
        epochs.set_eeg_reference(ref_channels="average")
        return epochs # Return the clean, segmented Epochs object

    def load_and_epoch_phase(self, session_dir: Path, phase: str, **epoch_params) -> Tuple[np.ndarray, np.ndarray, mne.Info]:
            """Loads all EDFs for a phase, processes them, concatenates epochs, and returns data."""
            
            # 1. Get file list: Use the utility
            edf_files = edf_list(str(session_dir), phase)
            
            all_epochs_data = []
            all_labels = []
            info_ref = None
            X_phase, y_phase = [], []

            for edf_path in edf_files:
                try: 
                    # 2. Load, Preprocess, and Epoch (File-by-File)
                    # This logic now handles the internal complexity of one file
                    raw, events = self.load_raw_and_events(edf_path)
                    
                    # **IMPORTANT:** Handle all filtering here before epoching (Apply Filters)
                    raw = self._apply_filters(raw) # Assuming a private helper method for filters

                    raw.set_montage("standard_1020", match_case=False, on_missing="ignore")
                    
                    epochs = self.create_epochs(raw, events, **epoch_params) # Uses your src.epochs function
                    
                    # 3. Collect Data
                    all_epochs_data.append(epochs.get_data())
                    all_labels.append(epochs.events[:, 2])
                    
                    if info_ref is None:
                        info_ref = epochs.info.copy()
                except (RuntimeError, ValueError, Exception) as e: 
                    print(f"  Warning: Skipping corrupted file {edf_path}: {e}")
                    continue 
            # 4. Concatenate using NumPy 
            try:
                X_phase = np.concatenate(all_epochs_data, axis=0)
                y_phase = np.concatenate(all_labels, axis=0)
            except Exception as e:
                print(f"  Error during concatenation: {e}")
            
            return X_phase, y_phase, info_ref
