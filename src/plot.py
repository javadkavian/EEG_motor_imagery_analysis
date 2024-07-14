import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from dataset import EEGDataset
import numpy as np

def plot_eeg(data: EEGDataset, index: int, title):
    t = np.linspace(0, data.get_window_length(), data.trials.shape[2])
    for i in range(0, 59, 15):
        plt.plot(t, data.trials[index, i], label="Ch" + str(i).zfill(2))
    plt.grid()
    plt.xlabel('Time (seconds)')
    plt.ylabel('Voltage')
    plt.title(title)
    plt.legend()

def plot_scatter_trials(trials, y, labels, title):
    X = trials.reshape(trials.shape[0], -1)
    tsne = TSNE(n_components=2, random_state=42)
    X = tsne.fit_transform(X)
    classes = np.unique(y)
    for i, c in zip(range(len(classes)), classes):
        plt.scatter(X[y==c, 0], X[y==c, 1], label=labels[i])
    plt.xlabel('comp-1')
    plt.ylabel('comp-2')
    plt.grid()
    plt.legend()
    plt.title(title)