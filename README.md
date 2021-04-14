# EEG-offline-processing

This processing package includes EEG preprocessing, EMG preprocessing, epoching, evoking. 
This is designed for processing data collected using [EEG online experiment GUI](https://github.com/WILLSNIU186/uw_eboinics_experimental_interface)

## Content

### EEG preprocessing

1. Read data from 'record/subject_name/Run1/raw_eeg.csv'
2. Band pass filter
3. Use ICA to remove eye blink corrupted IC.
4. Project back to sensor space.
5. save as processed_raw.fif 

### EMG preprocessing

1. Read data from 'record/subject_name/Run1/raw_eeg.csv'
2. Band pass filter
3. Remove baseline
4. TKEO
5. Smoothing
6. Choose threshold
7. Convert to binary 
8. Extract EMG onset
9. Save as dictionary

### Epoching

1. Load processed raw data from 'processed_data/consecutive_data/processed_raw.fif '
2. Visualize bad epochs and remove
3. Save as epochs.fif

### Evoking

1. Load epochs.fif
2. Calculate grand average trial for each class
3. Visualize interesting plots.

_Note : processed_data and records folder does not include real EEG data due to github data size limit. You should collect your own
data from EEG online GUI._
## Contact

Jiansheng Niu: jiansheng.niu1@uwaterloo.ca

_Developed in Ebionics lab, University of Waterloo, Canada._
