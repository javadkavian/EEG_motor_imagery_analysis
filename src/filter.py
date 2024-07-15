import numpy as np
from dataset import EEGDataset
from mne.filter import filter_data
from scipy.signal import convolve2d
import copy
    
    
def bandpass_filter(data: EEGDataset, l_freq, h_freq):
    filtered_data = copy.deepcopy(data)
    filtered_data.trials = filter_data(data.trials,
                            sfreq=data.sampling_frequency, 
                            l_freq=l_freq, 
                            h_freq=h_freq)
    
    return filtered_data

def CAR_filter(data: EEGDataset):
    filtered_data = copy.deepcopy(data)
    filtered_data.trials -= np.mean(data.trials, axis=2, keepdims=True)
    return filtered_data
    
def small_laplacian_filter(data: EEGDataset):
    filtered_data = copy.deepcopy(data)
    d = -1
    laplacian_kernel = np.array([[0, d, 0],
                                 [d, 4, d],
                                 [0, d, 0]])
    
    filtered_data.trials = np.array([
        convolve2d(trial, laplacian_kernel, mode='same', boundary='symm') for trial in data.trials
        ])
    
    return filtered_data

def large_laplacian_filter(data: EEGDataset):
    filtered_data = copy.deepcopy(data)
    d = -0.5
    laplacian_kernel = np.array([[0, 0, d, 0, 0],
                                 [0, 0, 0, 0, 0],
                                 [d, 0, 2, 0, d],
                                 [0, 0, 0, 0, 0],
                                 [0, 0, d, 0, 0]])
    
    filtered_data.trials = np.array([
        convolve2d(trial, laplacian_kernel, mode='same', boundary='symm') for trial in data.trials
        ])
    
    return filtered_data
    