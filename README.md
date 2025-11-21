# bigP3BCI_pipeline
This is a public pipeline that based on dataset PhysioNet bigP3BCI EEG, do things from data loading to data visual.

## Description 
Name: BigP3BCI – An Open, Diverse, and Machine‑Learning‑Ready P300‑based BCI Dataset
Host: PhysioNet
URL: https://physionet.org/content/bigp3bci/1.0.0/
What we use: Subjects A_01 to A_19 from Session SE001 (both Train/ Test; CB and RD tasks). The code works for other sessions in the same layout.

## Dataset Download 
The code to download them: 

`wget -r -N -c -np --reject="index.html*" https://physionet.org/files/bigp3bci/1.0.0/bigP3BCI-data/StudyA/`

To use wget, you can download it from brew: 

`brew install wget`

## Environment Setup
1. Clone the repository
``
2. Create and activate a Python virutal environment
``
3. Install the required dependencies:

## Project Layout

## Reproduce the pipeline
