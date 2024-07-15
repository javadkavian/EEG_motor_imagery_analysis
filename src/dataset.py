import numpy as np
from scipy.io import loadmat 


class EEGDataset:
    def __init__(self,
                 file_path,
                 start_window=0.5,
                 end_window=2.5):
        
        self.data = loadmat(file_name=file_path)
        self.start_window = start_window
        self.end_window = end_window
        
        self.sampling_frequency = self.data['nfo']['fs'][0][0][0][0]
        self.EEGsignals = self.data['cnt'].T
        self.n_channels, self.n_samples = self.EEGsignals.shape
        
        self.cue_positions = self.data['mrk']['pos'][0][0][0]
        self.targets = np.array([1 if target == 1 else 0 for target in self.data['mrk']['y'][0][0][0]])
        self.n_trails = len(self.cue_positions)
        
        self.class_labels = [x[0] for x in self.data['nfo']['classes'][0][0][0]]
        self.n_classes = len(self.class_labels)
        
        self.window = np.arange(int(start_window * self.sampling_frequency), int(end_window * self.sampling_frequency))
        
        self.n_samples_per_trail = len(self.window)
        
        self.trials = []
        for pos in self.cue_positions:
            self.trials.append(
                self.EEGsignals[:, self.window + pos]
            )
            
        self.trials = np.array(self.trials).astype(np.float64)
        self.targets.view(np.float64)

    def get_window_length(self):
        return self.end_window - self.start_window