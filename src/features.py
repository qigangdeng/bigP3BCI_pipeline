import mne 
import numpy as np

def extract_amplitude_features(epochs: mne.Epochs) -> np.ndarray:
    X, y = epochs.get_data(), epochs.events[:,2]