
import numpy as np

import pandas as pd

import mne

class DataLoader():
    total_exp_number = 1

    def __init__(self, exp_counter):
        self.exp_counter = exp_counter
        self.trial_removel_list = []

    def init_task_dependent_variables(self):
        # set up task-dependent variables
        if self.exp_counter ==1 :
            self.fs = 500
            self.channel_dict = {'P7': 1,
                                 'P4': 2,
                                 'Cz': 3,
                                 'Pz': 4,
                                 'P3': 5,
                                 'P8': 6,
                                 'EMG_WE_right': 7,
                                 'EMG_IE_right': 8,
                                 'T8': 9,
                                 'F8': 10,
                                 'C4': 11,
                                 'F4': 12,
                                 'Fp2': 13,
                                 'Fz': 14,
                                 'C3': 15,
                                 'F3': 16,
                                 'Fp1': 17,
                                 'T7': 18,
                                 'F7': 19,
                                 'EMG_IE_left': 21,
                                 'FC6': 22,
                                 'FC2': 23,
                                 'C2': 24,
                                 'CP6': 25,
                                 'CP2': 26,
                                 'CP1': 27,
                                 'CP5': 28,
                                 'FC1': 29,
                                 'FC5': 30,
                                 'C1': 31,
                                 'EMG_WE_left': 32}
                                 
            self.channel_type = ['eeg','eeg','eeg','eeg','eeg','eeg','emg','emg','eeg',
                                 'eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg',
                                 'eeg','emg','eeg','eeg','eeg','eeg','eeg','eeg',
                                 'eeg','eeg','eeg','eeg','emg']
                                 
            self.channel_names = list(self.channel_dict.keys())
            self.channel_ind = list(self.channel_dict.values())
            self.channel_ind = [x + 1 for x in self.channel_ind]
            self.n_channel = len(self.channel_ind)
            

    def load_data(self):
        if self.exp_counter == 1:
            self.exp_name = "sub1"
            self.base_folder = r"..\records\sub1\Run1"
            self.mapping = {10: "IE_l", 20: "IE_r", 30: "WE_l", 40: "WE_r", 100: "EMG_WE_l",
                            200: "EMG_WE_r", 300: "EMG_IE_l", 400: "EMG_IE_r",
                            1: "Idle", 2: 'focus', 3: 'prepare', 4: 'two', 5: 'one', 6: 'task'}


    def create_raw_object(self):
        # read raw data from csv file
        raw_eeg_path = self.base_folder + "//raw_eeg.csv"
        df = pd.read_csv(raw_eeg_path, header=None)
        self.raw_data = df.values
        # self.raw_emg = self.raw_data[:, 15]
        self.raw_eeg = self.raw_data[:, self.channel_ind]



        info = mne.create_info(sfreq = self.fs, ch_names = self.channel_names, ch_types=self.channel_type)
        info.set_montage('standard_1020')

        self.raw_array = mne.io.RawArray(np.transpose(self.raw_eeg/10**6), info)# convert uV to V


    def create_event(self):
        event_path = self.base_folder + "//event.csv"
        event_df = pd.read_csv(event_path, header=None)
        self.events = event_df.values
        self.origin_time = self.raw_data[0, 0]
        self.onsets = self.events[:, 1] - self.origin_time
        self.durations = np.zeros_like(self.onsets)
        self.event_array = np.column_stack(((self.onsets * self.fs).astype(int), np.zeros_like(self.onsets, dtype = int), self.events[:, 0].astype(int)))
        self.descriptions = [self.mapping[event_id] for event_id in self.event_array[:, 2]]
        self.annot_from_events = mne.Annotations(onset=self.onsets, duration=self.durations, description=self.descriptions)
        self.raw_array.set_annotations(self.annot_from_events)
        
    def remove_nan_from_array(self, array):
        return array[~np.isnan(array)]
    
    def add_EMG_event(self):
        file = r"{}\{}".format(self.base_folder, 'EMG_onsets.csv')
        dt = pd.read_csv(file, index_col=0, header=None)
        task_name_list = ['WE_l', 'WE_r', 'IE_l', 'IE_r']
        EMG_onsets_dict = dict.fromkeys(task_name_list)
        EMG_onsets_dict['WE_l'] = self.remove_nan_from_array(dt.values[0, :])
        EMG_onsets_dict['WE_r'] = self.remove_nan_from_array(dt.values[1, :])
        EMG_onsets_dict['IE_l'] = self.remove_nan_from_array(dt.values[2, :])
        EMG_onsets_dict['IE_r'] = self.remove_nan_from_array(dt.values[3, :])
        onsets_for_annotation = np.concatenate((self.raw_array.annotations.onset,
                                                EMG_onsets_dict['WE_l'],
                                                EMG_onsets_dict['WE_r'],
                                                EMG_onsets_dict['IE_l'],
                                                EMG_onsets_dict['IE_r']))
        durations_for_annotation = np.zeros_like(onsets_for_annotation)
        descriptions_for_annotation = np.concatenate((self.raw_array.annotations.description,
                                                      np.array(['EMG_WE_l'] * len(EMG_onsets_dict['WE_l'])),
                                                      np.array(['EMG_WE_r'] * len(EMG_onsets_dict['WE_r'])),
                                                      np.array(['EMG_IE_l'] * len(EMG_onsets_dict['IE_l'])),
                                                      np.array(['EMG_IE_r'] * len(EMG_onsets_dict['IE_r']))))
        annot_from_events = mne.Annotations(onset=onsets_for_annotation,
                                            duration=durations_for_annotation,
                                            description=descriptions_for_annotation)
        self.raw_array.set_annotations(annot_from_events)
    
    

