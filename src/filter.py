import numpy as np
from dataset import EEGDataset
from mne.filter import filter_data
from scipy.signal import convolve2d

    
def bandpass_filter(data: EEGDataset, l_freq, h_freq):
    return filter_data(data.trials.astype(np.float64),
                       sfreq=data.sampling_frequency, 
                       l_freq=l_freq, 
                       h_freq=h_freq)


def CAR_filter(data: EEGDataset):
    return data.trials - np.mean(data.trials, axis=2, keepdims=True)
    
def small_laplacian_filter(data: EEGDataset):
    d = -1
    laplacian_kernel = np.array([[0, d, 0],
                                 [d, 4, d],
                                 [0, d, 0]])
    
    return np.array([
        convolve2d(trial, laplacian_kernel, mode='same', boundary='symm') for trial in data.trials
        ])

def large_laplacian_filter(data: EEGDataset):
    d = -0.5
    laplacian_kernel = np.array([[0, 0, d, 0, 0],
                                 [0, 0, 0, 0, 0],
                                 [d, 0, 2, 0, d],
                                 [0, 0, 0, 0, 0],
                                 [0, 0, d, 0, 0]])
    
    return np.array([
        convolve2d(trial, laplacian_kernel, mode='same', boundary='symm') for trial in data.trials
        ])
    