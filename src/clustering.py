import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.cluster import homogeneity_score, silhouette_score
from sklearn.cluster import SpectralClustering
import copy


def plot_kmeans_elbow(X):
    inertias = []
    for i in range(1, 15):
        kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    plt.plot(np.arange(1, 15), inertias)
    plt.grid()
    plt.xlabel('K')
    plt.axvline(x=2, linestyle='--', c='red')
    plt.ylabel('Inertia')
    plt.title('Elbow')
    plt.show()    


def apply_kmeans_clustering(X, y, k):
    model = KMeans(n_clusters=k, init='k-means++', n_init=10)
    model.fit(X)
    labels = model.labels_
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], c='red', s=200, alpha=0.5)
    plt.title('K-Means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid()
    plt.show()
    print("silhouette_score : ", silhouette_score(X, labels))
    print("homogeneity_score : ", homogeneity_score(y, labels))


def apply_kernel_kmeans_clustering(X, y, k):
    clustering = SpectralClustering(n_clusters=k, assign_labels='kmeans', affinity='nearest_neighbors').fit(X)
    plt.scatter(X[:, 0], X[:, 1], c=clustering.labels_, cmap='viridis')  
    plt.title("Kernel K-means Clustering")  
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid()
    plt.show()  
    print("silhouette_score : ", silhouette_score(X, clustering.labels_))
    print("omogeneity_score : ", homogeneity_score(y, clustering.labels_))    
    

def plot_K_nearest_neighbor_distance(X, K):
    n_samples = X.shape[0]
    distances = np.empty((n_samples, n_samples), dtype=float)
    for i in range(n_samples):
        for j in range(n_samples):
            dist = np.sqrt((X[i][0] - X[j][0])**2 + (X[i][1] - X[j][1])**2)
            distances[i][j] = dist
            distances[j][i] = dist
    distances = np.sort(distances)
    k_th_nearest_neighbor = distances[:, K].flatten()
    plt.plot(np.sort(k_th_nearest_neighbor))
    plt.axhline(y=0.09, linestyle='--', c='red')
    plt.title('K_nearest_neighbor distance')
    plt.xlabel("sorted point " +  str(K) + "-th nearest neighbor")
    plt.ylabel(str(K) + '-th nearest neighbor distance')
    plt.grid()


def apply_DB_scan_clustering(eps, min_samples, X, y):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)


    unique_labels = set(labels)

    plt.figure(figsize=(8, 6))
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for label, color in zip(unique_labels, colors):
        if label == -1:
            color = [0, 0, 0, 1]

        class_member_mask = (labels == label)
        xy = X[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(color),
                 markeredgecolor='k', markersize=14)

    plt.title('DBSCAN Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid()
    plt.show()
    print("silhouette_score: ", silhouette_score(X, labels))
    print("homogeneity_score: ", homogeneity_score(y, labels))

