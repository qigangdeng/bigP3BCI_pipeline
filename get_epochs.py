# This script will get the epochs from the raw data

import mne 
import numpy as np 
import os 
from src.preprocessing import apply_filters, create_epochs
from src.run_session import load_data_from_session

session_dir = r"/Users/qigangdeng/Downloads/physionet.org/files/bigp3bci/1.0.0/bigP3BCI-data/StudyA"


raw, events = load_data_from_session()