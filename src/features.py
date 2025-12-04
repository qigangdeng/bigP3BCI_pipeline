import mne 
import numpy as np
from pathlib import Path

BASE_CHS = ["F3", "Fz", "F4", "C3", "Cz", "C4", "P3", "Pz", "P4", "PO7", "PO8", "Oz", "Fp1", "Fp2", "O1", "O2"]

class FeatureExtractor:
    def __init__(self, npz_file: str):
        data = np.load(npz_file)
        self.X = data["X"]
        self.y = data["y"]
        # Create MNE Info object
        self.ch_names = list(data["ch_names"])
        self.sfreq = float(data["sfreq"])

        self.ref_info = mne.create_info(
            ch_names=self.ch_names,
            sfreq=self.sfreq,
            ch_types='eeg'
        )

    def extract_amplitude_features(
        self, 
        tmin: float = -0.2,
        tmax: float = 0.8
        ) -> np.ndarray:
        ch_to_idx = {ch: i for i, ch in enumerate(self.ch_names)}
        selected_indices = [ch_to_idx[ch] for ch in BASE_CHS if ch in ch_to_idx]
    
        X_selected = self.X[:, selected_indices, :]


        window_indices = self.get_time_window_indices(X_selected, tmin, tmax)

        window1_mean = X_selected[:, :, window_indices[0][0]:window_indices[0][1]].mean(axis=2)
        window2_mean = X_selected[:, :, window_indices[1][0]:window_indices[1][1]].mean(axis=2)
        window3_mean = X_selected[:, :, window_indices[2][0]:window_indices[2][1]].mean(axis=2)
        window4_mean = X_selected[:, :, window_indices[3][0]:window_indices[3][1]].mean(axis=2)
        # Concatenate results along the feature axis (axis=1)
        X_features = np.hstack([window1_mean, window2_mean, window3_mean, window4_mean])

        assert X_features.shape[1]==64, f"Expect 64 features, got {X_features.shape[1]}"
        return X_features, self.y 

    def get_time_window_indices(
        self, 
        X: np.array,
        tmin: float = 0.2,
        tmax: float = 0.8,
        ):
        start_idx = int(tmin * self.sfreq)
        window_size = int(0.15 * self.sfreq)
        return [(start_idx + i*window_size, start_idx + (i+1)*window_size) for i in range(4)]