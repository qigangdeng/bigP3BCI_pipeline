# This script will get the epochs from the raw data
import numpy as np 
import mne 
from pathlib import Path
from src import utils
from src.data_loader import BCIDataLoader

# Define input and output dirs 
# Insert the dir of your dataset folder 
input_dir = Path(
    r"/Users/qigangdeng/Downloads/physionet.org/files/bigp3bci/1.0.0/bigP3BCI-data/StudyA"
)
# Create the output_dir if not existed
output_dir = Path("./preprocessed_data")
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
