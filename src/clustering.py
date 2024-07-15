import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from plot import plot_kmeans, plot_clusters
from metrics import clustring_report


def apply_kmeans_clustering(X, y, k):
    model = KMeans(n_clusters=k, init='k-means++', n_init=10)
    labels = model.fit_predict(X)

    clustring_report(X, y, labels)
    plot_kmeans(X, labels, model.cluster_centers_, 'K-means Clustering')
    plt.savefig(f'../assets/{k}means.png')


def apply_kernel_kmeans_clustering(X, y, k):
    model = SpectralClustering(n_clusters=k, assign_labels='kmeans', affinity='nearest_neighbors')
    labels = model.fit_predict(X)
    
    clustring_report(X, y, labels)
    plot_clusters(X, labels, [f"class {i}" for i in range(k)], "Kernel K-means Clustering")
    plt.savefig(f'../assets/kernel_{k}means.png')


def find_kth_nn(data_matrix, k):
    kth_nearest_distance = []
    for vector in data_matrix:
        distances = [np.linalg.norm(vector - neighbor) for neighbor in data_matrix]
        kth_nearest_distance.append(sorted(distances)[k - 1])
    return kth_nearest_distance

