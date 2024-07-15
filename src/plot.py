import numpy as np
from dataset import EEGDataset
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


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
    
def plot_kth_nn_distance(k, kth_nearest_distance, EPS=None):
    n_samples = len(kth_nearest_distance)
    plt.plot(sorted(kth_nearest_distance))
    if EPS != None:
        plt.plot(EPS * np.ones(n_samples), '--')
        
    if k == 3:
        postfix = 'rd'
    else:
        postfix = 'th'
        
    plt.xlabel(f'Points Sorted According to \n Distance of {k}{postfix} Nearest Neighbor')
    plt.ylabel(f'{k}{postfix} Nearest Neighbor Distance')
    plt.xlim((0,len(kth_nearest_distance)))
    plt.grid()
    
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
    
def plot_clusters(X, y, labels, title):
    pca = PCA(n_components=2)
    X = pca.fit_transform(X)
    classes = np.unique(y)
    for i, c in zip(range(len(classes)), classes):
        plt.scatter(X[y==c, 0], X[y==c, 1], label=labels[i])
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(title)
    plt.grid()
    plt.legend()
    return pca
    
def plot_kmeans_elbow(X, optimal_k):
    inertias = []
    for i in range(1, 15):
        kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    plt.plot(np.arange(1, 15), inertias)
    plt.grid()
    plt.xlabel('K')
    plt.axvline(x=optimal_k, linestyle='--', c='red')
    plt.ylabel('Inertia')
    plt.title('Optimal K') 

def plot_kmeans(X, y, centers, title):
    pca = plot_clusters(X, y, [f"class {i}" for i in range(len(centers))], title)
    centers = pca.transform(centers)
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=150, alpha=0.5)