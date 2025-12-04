# bigP3BCI_pipeline
This is a public pipeline that based on dataset PhysioNet bigP3BCI EEG, do things from data loading to data visual.

# Research Question
A grid comparison from classical logistic regression to Transformer on performance metrics in binary classification of target vs non-target epochs in a visual P300 BCI task.

## Description 
Name: BigP3BCI – An Open, Diverse, and Machine‑Learning‑Ready P300‑based BCI Dataset
Host: PhysioNet
URL: [https://physionet.org/content/bigp3bci/1.0.0/](https://physionet.org/content/bigp3bci/1.0.0/#files-panel)
What we use: Subjects A_01 to A_19 from Session SE001 (both Train/ Test; CB and RD tasks). The code works for other sessions in the same layout.

## Environment Setup
### 1. Clone the repository
```
git clone https://github.com/qigangdeng/bigP3BCI_pipeline
```
### 2. Create and activate a Python virutal environment
```
python -m venv .venv
source .venv/bin/activate  # (Linux/Mac)
# or: .venv\Scripts\activate  # (Windows PowerShell)
```
### 3. Install the required dependencies:
```
pip install -r requirements.txt
```
`requirements.txt` includes:
```
numpy
pandas
scikit-learn
mne
joblib
matplotlib
tqdm
phe            # Paillier (python-paillier / phe)
```
## Project Layout

## Reproduce the pipeline
