# bigP3BCI_pipeline
This is a public pipeline that based on dataset PhysioNet bigP3BCI EEG, do things from data loading to data visual.


The Pipeline data can be downloaded from https://physionet.org/content/bigp3bci/1.0.0/#files-panel while in this pipeline, we are using StudyA with 19 participants: 

The code to download them: 

wget -r -N -c -np --reject="index.html*" https://physionet.org/files/bigp3bci/1.0.0/bigP3BCI-data/StudyA/

To use wget, you can download it from brew: 

brew install wget

