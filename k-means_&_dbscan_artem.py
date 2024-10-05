#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

# Step 1: Retrieve and load the Olivetti faces dataset
faces_data = fetch_olivetti_faces(shuffle=True, random_state=42)

# X contains the flattened image data, y contains the labels
X = faces_data.data
y = faces_data.target

print(f"Shape of X: {X.shape}")  # (400, 4096) as each image is 64x64 pixels
print(f"Shape of y: {y.shape}")  # (400,) since there are 400 images

# Step 2: Split the Dataset using Stratified Sampling
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print(f"Training set size: {X_train.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Verify stratified sampling by checking the distribution of images per person
unique_train, counts_train = np.unique(y_train, return_counts=True)
unique_val, counts_val = np.unique(y_val, return_counts=True)
unique_test, counts_test = np.unique(y_test, return_counts=True)

print("Train set counts per person:", dict(zip(unique_train, counts_train)))
print("Validation set counts per person:", dict(zip(unique_val, counts_val)))
print("Test set counts per person:", dict(zip(unique_test, counts_test)))

# Step 3: Train a Classifier Using k-Fold Cross Validation
svm_classifier = SVC(kernel='linear', random_state=42)
cv_scores = cross_val_score(svm_classifier, X_train, y_train, cv=5)
print(f"Cross-validation accuracy scores: {cv_scores}")
print(f"Mean cross-validation accuracy: {cv_scores.mean()}")

svm_classifier.fit(X_train, y_train)
y_val_pred = svm_classifier.predict(X_val)

print(classification_report(y_val, y_val_pred))

# Step 4: Dimensionality Reduction Using K-Means
range_n_clusters = range(2, 110)
silhouette_avg_scores = []
best_silhouette_score = -1
best_k = None

# Test different numbers of clusters and calculate the silhouette score for each
for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_train)
    silhouette_avg = silhouette_score(X_train, cluster_labels)
    silhouette_avg_scores.append(silhouette_avg)
    
    if silhouette_avg > best_silhouette_score:
        best_silhouette_score = silhouette_avg
        best_k = n_clusters
    
print(f"\nThe best number of clusters is {best_k} with a silhouette score of {best_silhouette_score}")

# Plot the silhouette scores for each K
plt.figure(figsize=(8, 6))
plt.plot(range_n_clusters, silhouette_avg_scores, 'bo-')
plt.xlabel("Number of clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score for Different Numbers of Clusters")
plt.grid(True)
plt.show()

kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(X_train) for k in range(1, 110)]
inertias = [model.inertia_ for model in kmeans_per_k]

# Plot the elbow method to visually inspect the optimal number of clusters
plt.figure(figsize=(8, 6))
plt.plot(range(1, 110), inertias, 'bo-')
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal K in K-Means")
plt.grid(True)
plt.show()

# Fit K-Means with the optimal number of clusters (best_k)
kmeans = KMeans(n_clusters=best_k, random_state=42)
kmeans.fit(X_train)

# Step 5: Visualize K-Means Clustering Using PCA
# Reduce the dimensionality of the dataset to 2 components for visualization
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_train_pca_kmeans = kmeans.predict(X_train)

# Plot the clusters in the PCA-reduced 2D space
# Displays images grouped by their clusters
plt.figure(figsize=(8, 6))
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=X_train_pca_kmeans, cmap='viridis', s=10)
plt.title("K-Means Clustering Visualization on PCA-Reduced Olivetti Faces Data")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar()
plt.show()

# Visualize images in K-Means clusters
def plot_cluster_images(X, cluster_labels, n_clusters, images_per_cluster=10):
    for cluster in range(n_clusters):
        indices = np.where(cluster_labels == cluster)[0]
        selected_indices = indices[:images_per_cluster]
        
        plt.figure(figsize=(12, 6))
        for i, index in enumerate(selected_indices):
            plt.subplot(2, images_per_cluster//2, i + 1)
            plt.imshow(X[index].reshape(64, 64), cmap='gray')
            plt.axis('off')
            plt.title(f"Cluster {cluster}")
        
        plt.suptitle(f"Cluster {cluster} Images", fontsize=16)
        plt.show()

# Visualize images from K-Means clusters
plot_cluster_images(X_train, kmeans.labels_, best_k)

# Train SVM Classifier on Reduced Dimensionality Data
# K-Means is used to reduce the dimensionality before training the classifier
X_train_reduced = kmeans.transform(X_train)
X_val_reduced = kmeans.transform(X_val)
X_test_reduced = kmeans.transform(X_test)

# Train the SVM on the reduced data
svm_classifier_reduced = SVC(kernel='linear', random_state=42)
svm_classifier_reduced.fit(X_train_reduced, y_train)


# Evaluate the classifier on the validation set
val_accuracy = svm_classifier_reduced.score(X_val_reduced, y_val)
print(f"Validation accuracy with reduced dimensionality: {val_accuracy}")

y_val_pred = svm_classifier_reduced.predict(X_val_reduced)

print("Classification Report on Validation Set:")
print(classification_report(y_val, y_val_pred))

# Step 6: Apply DBSCAN Clustering
dbscan = DBSCAN(eps=7.5, min_samples=2, metric='euclidean')
dbscan.fit(X_train)

# Extract labels from DBSCAN and count the number of clusters and noise points
dbscan_labels = dbscan.labels_
n_noise_points = np.sum(dbscan_labels == -1)
n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)

print(f"Number of clusters found by DBSCAN: {n_clusters_dbscan}")
print(f"Number of noise points: {n_noise_points}")

# Visualize DBSCAN clustering in 2D using PCA
X_train_pca_dbscan = pca.transform(X_train)

plt.figure(figsize=(8, 6))
plt.scatter(X_train_pca_dbscan[:, 0], X_train_pca_dbscan[:, 1], c=dbscan_labels, cmap='Paired', s=10)
plt.title("DBSCAN Clustering Visualization on PCA-Reduced Olivetti Faces Data")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar()
plt.show()

# Visualize images in DBSCAN clusters
plot_cluster_images(X_train, dbscan.labels_, n_clusters_dbscan)
