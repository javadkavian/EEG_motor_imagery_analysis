# Machine Learning Course Project

## Comparative Analysis of Machine Learning Algorithms for EEG Signal Classification and Clustering


### Abstract
One of the most powerful cognitive processes that in-
volves mental simulation of movement without physical
execution is motor imagery. In this study, our focus is
on extracting and preprocessing EEG signals and feed-
ing them to machine-learning models for motor imagery
classification and clustering.

---

### Introduction
In this project, we will first get to know EEG signal data and explore different ways to prepare the data, clean it, and remove any unwanted noise, which is common with real-world signals. Then, using techniques for extracting features from these signals that can be useful.

Additionally, we try to classify and cluster the data using the features extracted earlier with different machine-learning algorithms. And finally, the results will be compared and analyzed.

---

### More details

#### Dataset
[**Data sets**](https://www.bbci.de/competition/iv/desc_1.html) provided by the [**Berlin BCI**](https://www.bbci.de/) group.

---

#### Preprocessing

In the preprocessing stage, we employed a variety of techniques, including:
* Bandpass filtering 
* Common Average Referencing (CAR)
* Laplacian Filtering 
* Principal Component Analysis (PCA) 
* Normalization

---

#### Feature Extraction

For feature extraction, we utilized following algorithms:
* Independent Component Analysis (ICA)
* Common Spatial Patterns (CSP)

---
#### Classification
In classification section, we explored multiple algorithms, such as:
* Logistic Regression 
* Support Vector Machines (SVM) 
* K-Nearest Neighbors (KNN) 
* Multi-Layer Perceptron (MLP) 
* AdaBoost 
* XGBoost

Furthermore, to thoroughly evaluate the performance of these classifiers, we calculated an array of metrics, including:
* Accuracy 
* Confusion Matrix 
* Receiver Operating Characteristic (ROC) Curve

---
#### Clustering
Lastly, in the clustering phase, we applied following models:
* DBSCAN 
* K-means
* Kernel-based K-means 

and analyzed the results using these scores:
* Silhouette Score 
* Homogeneity Score

---
For more information, please refer to the [project report](https://github.com/javadkavian/EEG_motor_imagery_analysis/blob/main/Report.pdf).

[**javadkavian**](https://github.com/javadkavian) & [**MehdiJmlkh**](https://github.com/MehdiJmlkh)   