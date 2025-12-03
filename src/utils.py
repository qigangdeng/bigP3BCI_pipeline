import os 
import glob 
from tkinter import NONE
from typing import List

def edf_list(session_dir: str, phase: str) -> List[str]:
    """Find all EDF files in the specific session and phase directory"""
    return sorted(
        glob.glob(os.path.join(session_dir, phase, "**", "*.edf"), 
        recursive=True))


def id_lists(base_data_dir, participant_str=NONE, sub_folder_str=NONE):
    if sub_folder_str:
        return sorted((base_data_dir / participant / sub_folder_str) for participant in base_data_dir.glob(participant_str))
    else:
        return sorted((base_data_dir / participant) for participant in base_data_dir.glob(participant_str))