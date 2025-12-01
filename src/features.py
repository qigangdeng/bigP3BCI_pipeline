import mne 
import numpy as np

BASE_CHS = ["F3", "Fz", "F4", "C3", "Cz", "C4", "P3", "Pz", "P4", "PO7", "PO8", "Oz", "Fp1", "Fp2", "O1", "O2"]


def extract_amplitude_features(
    epochs: mne.Epochs,
    tmin: float = -0.2,
    tmax: float = 0.8
    ) -> np.ndarray:

    # Select requested channels 
    epochs_selected = epochs.copy().pick_channels(BASE_CHS)
    # Get data array X and labels y
    X, y = epochs_selected.get_data(), epochs_selected.events[:,2]

    window_indices = get_time_window_indices(epochs_selected, tmin, tmax)

    window1_mean = X[:, :, window_indices[0][0]:window_indices[0][1]].mean(axis=2)
    window2_mean = X[:, :, window_indices[1][0]:window_indices[1][1]].mean(axis=2)
    window3_mean = X[:, :, window_indices[2][0]:window_indices[2][1]].mean(axis=2)
    window4_mean = X[:, :, window_indices[3][0]:window_indices[3][1]].mean(axis=2)
    # Concatenate results along the feature axis (axis=1)
    X_features = np.hstack([window1_mean, window2_mean, window3_mean, window4_mean])

    assert X_features.shape[1]==64, f"Expect 64 features, got {X_features.shape[1]}"
    return X_features, y 

def get_time_window_indices(
    epochs: mne.Epochs,
    tmin: float = 0.2,
    tmax: float = 0.8,
    ):
    sfreq = epochs.info['sfreq']
    start_idx = epochs.time_as_index(tmin)[0]
    window_size = int(0.15 * sfreq)
    return [(start_idx + i*window_size, start_idx + (i+1)*window_size) for i in range(4)]