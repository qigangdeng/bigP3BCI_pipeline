import mne 
import numpy as np

from data_loader import BCIDataLoader
from utils import edf_list


def load_data_from_session(
    session_dir: str, 
    phase: str, 
    resample_freq: float = 256.0
    ):
    edf_files = edf_list(session_dir, phase)






