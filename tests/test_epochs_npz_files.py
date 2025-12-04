import numpy as np
from pathlib import Path

preprocessed_dir = Path("preprocessed_data")

expected_subjects = [f"A_{i:02d}" for i in range(1, 20)]
expected_phase = ["Train", "Test"]

def test_all_expected_epochs_exist():
    files = list(preprocessed_dir.glob("*_epochs.npz"))
    found = {(f.name.split("_")[0], f.name.split("_")[1]) for f in files}
    # found will contain (subject, phase) like ("A_02", "Train")

    missing = []
    for subj in expected_subjects:
        for phase in expected_phase:
            if (subj, phase) not in found:
                missing.append((subj, phase))

    assert not missing, f"Missing epochs files for: {missing}"

def iter_npz_files():
    return preprocessed_dir.glob("*_epochs.npz")

def test_npz_files_load_and_have_required_keys():
    files = list(iter_npz_files())
    assert files, "No *_epochs.npz files found in preprocessed_data/"

    for f in files:
        data = np.load(f)
        for key in ["X", "y", "sfreq", "ch_names"]:
            assert key in data, f"{f} missing key {key}"
        X = data["X"]
        y = data["y"]
        assert X.ndim == 3, f"{f}: X should be (n_epochs, n_channels, n_times)"
        assert y.shape[0] == X.shape[0], f"{f}: y len != n_epochs"