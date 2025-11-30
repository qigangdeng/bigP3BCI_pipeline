import os 
import glob 
from typing import List

def edf_list(session_dir: str, phase: str) -> List[str]:
    """Find all EDF files in the specific session and phase directory"""
    return sorted(
        glob.glob(os.path.join(session_dir, phase, "**", "*.edf"), 
        recursive=True))
