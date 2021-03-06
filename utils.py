import pandas as pd
import numpy as np
import pdb
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, lfilter
from scipy import io

import mne
from mne.time_frequency import tfr_morlet, psd_multitaper, psd_welch

class Utils:

    @staticmethod
    def butter_bandpass_scope(highcut, lowcut, fs):
        low = lowcut / (0.5 * fs)
        high = highcut / (0.5 * fs)
        b, a = butter(2, [low, high], btype='band')  # second order for MRCP, 4th order for SMR
        return b, a

    @staticmethod
    def lap(data_pre_lap):
        lap_filter = [-1 / 4] * 4
        lap_filter.insert(0, 1)  # center channel is channel 0
        lap_filter = np.asarray(lap_filter)
        data_lap_filtered = np.matmul(data_pre_lap, np.transpose(lap_filter))
        return data_lap_filtered

    @staticmethod
    def lap_2_ch(data_pre_lap):
        lap_filter = [-1 / 2] * 2
        lap_filter.insert(0, 1)  # center channel is channel 0
        lap_filter = np.asarray(lap_filter)
        data_lap_filtered = np.matmul(data_pre_lap, np.transpose(lap_filter))
        return data_lap_filtered
