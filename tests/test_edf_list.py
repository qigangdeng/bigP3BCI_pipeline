import pytest
import numpy as np 
import mne 
import os
import sys
from unittest.mock import Mock, patch
from typing import List, Tuple
from unittest.mock import Mock, patch

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
# Insert the project root into the system path so Python can find 'src'
sys.path.insert(0, project_root)
from src.utils import edf_list # Assuming your class is in src/data_loader.py


@py.fixture 
def mock_edf_list():
    