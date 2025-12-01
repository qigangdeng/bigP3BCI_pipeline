import mne 
import numpy as np
from .data_loader import BCIDataLoader
from .utils import edf_list
from typing import List, Optional 


def load_data_from_session(
    session_dir: str, 
    phase: str, 
    resample_rate: float = 256.0,
    # Optional file list injection
    file_list: Optional[List[str]] = None
    ):
    data_loader = BCIDataLoader(resample_rate=resample_rate)
    all_raws = []
    all_events = []
    time_offset = 0 # to counter each file time offset from 0 when 

    # Allow for unit testing by injecting a file list
    if file_list is None:
        edf_files = edf_list(session_dir, phase)
    else:
        edf_files = file_list
        
    for file in edf_files: 
        result = data_loader.load_raw_and_events(file)
        # Check if the result is valid
        if result is None:
            print(f"Warning: Failed to load or extract events from {f}. Skipping file.")
            continue
        raw, events = result
        events[:,0] += time_offset
        time_offset += raw.n_times

        all_raws.append(raw)
        all_events.append(events)
    ## Check if any data was loaded
    if not all_raws:
            raise RuntimeError(f"No valid data loaded for session phase '{phase}' in {session_dir}")
    
    combined_raw = mne.concatenate_raws(all_raws)
    combined_events = np.concatenate(all_events)

    return combined_raw, combined_events





