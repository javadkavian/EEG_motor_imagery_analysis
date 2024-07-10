import numpy as np
from scipy.signal import butter, filtfilt, convolve2d
from dataset import Dataset

class Filter:
    def __init__(self):
        pass

    @staticmethod
    def __butter_band_pass(data: Dataset,
                           low_cut,
                           high_cut,
                           order):
        
        nyq = 0.5 * data.sampling_frequency
        low = low_cut / nyq
        high = high_cut / nyq
        return butter(order, [low, high], btype='band')
    
    @staticmethod
    def bandpass_filter(data: Dataset,
                        low_cut=8,
                        high_cut=30,
                        order = 3):
        b, a = Filter.__butter_band_pass(data=data,
                                     low_cut=low_cut,
                                     high_cut=high_cut,
                                     order=order)
        return filtfilt(b = b,
                        a = a,
                        x=data.trials,
                        axis=0)

    @staticmethod
    def CAR_filter(data: Dataset):
        return data.trials - np.mean(data.trials, axis=2, keepdims=True)
    
    @staticmethod
    def laplacian_filter(data: Dataset):
        
        laplacian_kernel = np.array([[0, 1, 0],
                                    [1, -4, 1],
                                    [0, 1, 0]])

        return np.array([
            convolve2d(trial, laplacian_kernel, mode='same', boundary='symm') for trial in data.trials
            ])
        
        