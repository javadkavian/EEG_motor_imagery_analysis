import numpy as np
from scipy.io import loadmat 

class EEGDataset:
    def __init__(self,
                 file_path,
                 window_length=2):
        
        self.data = loadmat(file_name=file_path)
        self.window_length = window_length
        
        self.sampling_frequency = self.data['nfo']['fs'][0][0][0][0]
        self.EEGsignals = self.data['cnt'].T
        self.n_channels, self.n_samples = self.EEGsignals.shape
        
        self.cue_positions = self.data['mrk']['pos'][0][0][0]
        self.targets = self.data['mrk']['y'][0][0][0]
        self.n_trails = len(self.cue_positions)
        
        self.class_labels = [x[0] for x in self.data['nfo']['classes'][0][0][0]]
        self.n_classes = len(self.class_labels)
        
        self.window = np.arange(0, int(self.window_length*self.sampling_frequency))
        self.n_samples_per_trail = len(self.window)
        
        self.trials = []
        for pos in self.cue_positions:
            self.trials.append(
                self.EEGsignals[:, self.window + pos]
            )
        self.trials = np.array(self.trials)
        
    