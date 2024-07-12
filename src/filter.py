import numpy as np
from scipy.signal import butter, filtfilt, convolve2d
from dataset import EEGDataset

class Filter:
    def __init__(self):
        pass
    
    @staticmethod
    def bandpass_filter():
        pass

    @staticmethod
    def CAR_filter(data: EEGDataset):
        return data.trials - np.mean(data.trials, axis=2, keepdims=True)
    
    @staticmethod
    def laplacian_filter(data: EEGDataset):
        
        # laplacian_kernel = np.array([[0, -1, 0],
        #                             [-1, 4, -1],
        #                             [0, -1, 0]])
        
        d = -0.5
        laplacian_kernel = np.array([[0, 0, d, 0, 0],
                                     [0, 0, 0, 0, 0],
                                     [d, 0, 2, 0, d],
                                     [0, 0, 0, 0, 0],
                                     [0, 0, d, 0, 0]])

        return np.array([
            convolve2d(trial, laplacian_kernel, mode='same', boundary='symm') for trial in data.trials
            ])
        
        