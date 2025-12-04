from src.features import FeatureExtractor
import numpy as np
from pathlib import Path

# Define input and output directories
input_dir = Path("./preprocessed_data")  # Where epoch .npz files are
output_dir = Path("./features")
output_dir.mkdir(parents=True, exist_ok=True)

# Process each epoch file
for epoch_file in input_dir.glob("*_epochs.npz"):
    subject_phase = epoch_file.stem.replace("_epochs", "")  # e.g., "A_01_Train"
    
    print(f"\nProcessing {subject_phase}")
    
    try:
        # Load epochs and extract features
        extractor = FeatureExtractor(epoch_file)
        X_features, y = extractor.extract_amplitude_features()
        
        # Save features to .npz
        output_file = output_dir / f"{subject_phase}_features.npz"
        np.savez_compressed(
            output_file,
            X=X_features,
            y=y
        )
        
        print(f"Saved features: {X_features.shape} to {output_file.name}")
        
    except Exception as e:
        print(f"Error processing {subject_phase}: {e}")
        continue