import pytest
import numpy as np
import os
import sys
from unittest.mock import Mock, patch
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)

from src.features import FeatureExtractor


@pytest.fixture
def mock_npz_file(tmp_path):
    """Create a temporary .npz file with mock data"""
    npz_file = tmp_path / "test_epochs.npz"
    
    # Create mock data
    n_epochs = 10
    n_channels = 16
    n_times = 256  # 1 second at 256 Hz
    
    X = np.random.randn(n_epochs, n_channels, n_times)
    y = np.random.randint(0, 2, n_epochs)
    sfreq = 256.0
    ch_names = ["F3", "Fz", "F4", "C3", "Cz", "C4", "P3", "Pz", "P4", "PO7", "PO8", "Oz", "Fp1", "Fp2", "O1", "O2"]
    
    np.savez_compressed(
        npz_file,
        X=X,
        y=y,
        sfreq=sfreq,
        ch_names=ch_names
    )
    
    return str(npz_file)


def test_get_time_window_indices_basic(mock_npz_file):
    """Test that get_time_window_indices returns correct indices"""
    extractor = FeatureExtractor(mock_npz_file)
    
    # Create dummy X array (not actually used in the method)
    X_dummy = np.random.randn(10, 16, 256)
    
    # Test with tmin=0.0 (start of epoch after baseline)
    tmin = 0.0
    tmax = 0.8
    window_indices = extractor.get_time_window_indices(X_dummy, tmin, tmax)
    
    # Verify we get 4 windows
    assert len(window_indices) == 4, "Should return 4 windows"
    
    # Verify each window is a tuple of (start, end)
    for window in window_indices:
        assert isinstance(window, tuple), "Each window should be a tuple"
        assert len(window) == 2, "Each window should have (start, end)"
        assert window[0] < window[1], "Start index should be less than end index"
    
    # Verify windows are consecutive (no gaps, no overlaps)
    for i in range(len(window_indices) - 1):
        assert window_indices[i][1] == window_indices[i+1][0], \
            f"Window {i} end should equal window {i+1} start"


def test_get_time_window_indices_calculation(mock_npz_file):
    """Test that indices are calculated correctly"""
    extractor = FeatureExtractor(mock_npz_file)
    sfreq = extractor.sfreq  # Should be 256.0
    
    X_dummy = np.random.randn(10, 16, 256)
    
    # Test with tmin=0.0
    tmin = 0.0
    window_indices = extractor.get_time_window_indices(X_dummy, tmin)
    
    # Expected calculations:
    # start_idx = int(0.0 * 256) = 0
    # window_size = int(0.15 * 256) = 38
    # Windows: (0, 38), (38, 76), (76, 114), (114, 152)
    
    expected_start_idx = int(tmin * sfreq)
    expected_window_size = int(0.15 * sfreq)
    
    assert window_indices[0][0] == expected_start_idx, \
        f"First window should start at {expected_start_idx}"
    
    assert window_indices[0][1] - window_indices[0][0] == expected_window_size, \
        f"Window size should be {expected_window_size} samples"
    
    # Verify all windows have the same size
    for i, (start, end) in enumerate(window_indices):
        assert end - start == expected_window_size, \
            f"Window {i} should have size {expected_window_size}"


def test_get_time_window_indices_with_different_tmin(mock_npz_file):
    """Test with different tmin values"""
    extractor = FeatureExtractor(mock_npz_file)
    sfreq = extractor.sfreq
    
    X_dummy = np.random.randn(10, 16, 256)
    
    # Test with tmin=0.2 (200ms after epoch start)
    tmin = 0.2
    window_indices = extractor.get_time_window_indices(X_dummy, tmin)
    
    expected_start_idx = int(tmin * sfreq)  # int(0.2 * 256) = 51
    expected_window_size = int(0.15 * sfreq)  # 38
    
    assert window_indices[0][0] == expected_start_idx, \
        f"With tmin={tmin}, start_idx should be {expected_start_idx}"
    
    # Verify the windows are correctly offset
    assert window_indices[0][0] == int(0.2 * 256), "Start index should be 51"
    assert window_indices[0][1] == int(0.2 * 256) + int(0.15 * 256), "End index should be 89"


def test_get_time_window_indices_window_coverage(mock_npz_file):
    """Test that windows cover the expected time range"""
    extractor = FeatureExtractor(mock_npz_file)
    
    X_dummy = np.random.randn(10, 16, 256)
    
    tmin = 0.0
    window_indices = extractor.get_time_window_indices(X_dummy, tmin)
    
    # Total window coverage: Calculate the same way as the implementation
    # window_size = int(0.15 * sfreq), then 4 * window_size
    window_size = int(0.15 * extractor.sfreq)
    expected_total_samples = 4 * window_size  # 4 * 38 = 152
    
    total_samples = window_indices[-1][1] - window_indices[0][0]
    
    assert total_samples == expected_total_samples, \
        f"Total window coverage should be {expected_total_samples} samples (4 windows * {window_size} samples each)"