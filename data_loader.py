import os
import glob
import json
import argparse
from pathlib import Path
import numpy as np
import mne


def edf_list(session_dir, phase):
    return sorted(glob.glob(os.path.join(session_dir, phase, "**", "*.edf"), recursive=True))