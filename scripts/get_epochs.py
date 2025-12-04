# This script will get the epochs from the raw data
import numpy as np
import mne
import yaml
from pathlib import Path

from src import utils  # type: ignore  # noqa: E402
from src.data_loader import BCIDataLoader  # type: ignore  # noqa: E402

# Resolve the config path relative to this file, not the working directory
_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
with open(_CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)


# Define input and output dirs from config
input_dir = Path(config["data"]["raw_root"])

# Use configured preprocessed_root; fall back to default if missing
preprocessed_root = config["data"].get("preprocessed_root", "./preprocessed_data")
output_dir = Path(preprocessed_root)
output_dir.mkdir(parents=True, exist_ok=True)

# Initiase data loader with 256 frequency sampling rate
data_loader = BCIDataLoader(resample_rate=256.0)

for subject_session_dir in utils.id_lists(input_dir, "A_*", "SE001"):
    subject_id = subject_session_dir.parent.name
    for phase in ["Train", "Test"]:
        output_file = output_dir / f"{subject_id}_{phase}_epochs.npz"
        if output_file.exists():
            print(f"Skipping {subject_id} - {phase}: {output_file.name} already exists.")
            continue
        print(f"\nProcessing {subject_id} - {phase}")
        try: 
            X_phase, y_phase, info_ref = data_loader.load_and_epoch_phase(
                subject_session_dir, 
                phase,
            )
        
        except RuntimeError as e:
            print(f"Skiping {subject_id} {phase}: {e}")
            continue 
        try: 
            np.savez_compressed(
                output_file, 
                X=X_phase, 
                y=y_phase,
                sfreq=info_ref['sfreq'], # Save key metadata
                ch_names=info_ref['ch_names']
            )
        except Exception as e: 
            print(f"Error saving {subject_id} {phase}: {e}")
            continue
            
        
        print(f"Saved {phase} epochs: {X_phase.shape} to {output_file.name}")
