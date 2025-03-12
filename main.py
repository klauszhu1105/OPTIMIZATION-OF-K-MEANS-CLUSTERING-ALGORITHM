import time
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import MinMaxScaler


# Class for K-Means and K-Means++ implementation in PyTorch
class KMeansTorch:
    def __init__(self, n_clusters, max_iters=100, tol=1e-4, init='random', random_state=None):
        """
        Initialize the KMeansTorch class.

        Parameters:
        - n_clusters: Number of clusters
        - max_iters: Maximum number of iterations
        - tol: Tolerance for convergence
        - init: Initialization method ('random' or 'k-means++')
        - random_state: Random state for reproducibility
        """
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.init = init
        self.random_state = random_state
        self.centroids = None

    def initialize_centroids(self, X):
        """
        Initialize centroids for K-Means based on the chosen initialization method.

        Parameters:
        - X: Input data

        Returns:
        - Initial centroids as a PyTorch tensor
        """
        if self.init == 'random':
            # Randomly select centroids from the input data points
            indices = torch.randperm(X.size(0))[:self.n_clusters]
            return X[indices]
        elif self.init == 'k-means++':
            # K-Means++ initialization for better centroid placement
            centroids = []
            initial_index = torch.randint(0, X.size(0), (1,)).item()
            centroids.append(X[initial_index])
            for _ in range(1, self.n_clusters):
                distances = torch.min(torch.stack([torch.cdist(X, c.unsqueeze(0)).squeeze() for c in centroids]),
                                      dim=0).values
                prob_distribution = distances ** 2 / distances.sum()
                next_index = torch.multinomial(prob_distribution, 1).item()
                centroids.append(X[next_index])
            return torch.stack(centroids)
        else:
            raise ValueError("Invalid initialization method. Choose 'random' or 'k-means++'.")

    def fit(self, X):
        """
        Fit the K-Means model to the data.

        Parameters:
        - X: Input data
        """
        if self.random_state:
            # Set the random seed for reproducibility
            torch.manual_seed(self.random_state)

        # Initialize centroids based on the chosen method
        self.centroids = self.initialize_centroids(X)

        for i in range(self.max_iters):
            # Calculate the distance between each data point and the centroids
            distances = torch.cdist(X, self.centroids)
            # Assign each data point to the nearest centroid
            labels = torch.argmin(distances, dim=1)
            # Update centroids as the mean of the points assigned to each cluster
            new_centroids = torch.stack([X[labels == j].mean(dim=0) for j in range(self.n_clusters)])

            # Check for convergence (if centroids do not change significantly)
            if torch.all(torch.abs(new_centroids - self.centroids) < self.tol):
                break

            # Update centroids for the next iteration
            self.centroids = new_centroids

        # Store the final labels and calculate the inertia (sum of squared distances)
        self.labels_ = labels.numpy()
        self.inertia_ = sum((torch.cdist(X[labels == i], self.centroids[i].unsqueeze(0)) ** 2).sum() for i in
                            range(self.n_clusters)).item()


# Class for Mini-Batch K-Means implementation in PyTorch
class MiniBatchKMeansTorch:
    def __init__(self, n_clusters, batch_size, max_iters=100, tol=1e-4, random_state=None):
        """
        Initialize the MiniBatchKMeansTorch class.

        Parameters:
        - n_clusters: Number of clusters
        - batch_size: Size of the mini-batch for each iteration
        - max_iters: Maximum number of iterations
        - tol: Tolerance for convergence
        - random_state: Random state for reproducibility
        """
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        self.centroids = None

    def fit(self, X):
        """
        Fit the Mini-Batch K-Means model to the data.

        Parameters:
        - X: Input data
        """
        if self.random_state:
            # Set the random seed for reproducibility
            torch.manual_seed(self.random_state)

        # Randomly initialize centroids from the input data
        indices = torch.randperm(X.size(0))[:self.n_clusters]
        self.centroids = X[indices]

        for i in range(self.max_iters):
            # Select a random mini-batch of data points
            batch_indices = torch.randperm(X.size(0))[:self.batch_size]
            X_batch = X[batch_indices]

            # Calculate the distance between batch data points and centroids
            distances = torch.cdist(X_batch, self.centroids)
            # Assign each data point in the batch to the nearest centroid
            labels = torch.argmin(distances, dim=1)

            # Update centroids using the batch data
            for j in range(self.n_clusters):
                points = X_batch[labels == j]
                if points.size(0) > 0:
                    self.centroids[j] = (self.centroids[j] + points.mean(dim=0)) / 2

        # Store the final labels and calculate the inertia (sum of squared distances)
        self.labels_ = torch.argmin(torch.cdist(X, self.centroids), dim=1).numpy()
        self.inertia_ = sum((torch.cdist(X[self.labels_ == i], self.centroids[i].unsqueeze(0)) ** 2).sum() for i in
                            range(self.n_clusters)).item()


# Function for Spectral Clustering implementation
def spectral_clustering_torch(X, n_clusters):
    """
    Perform Spectral Clustering using PyTorch.

    Parameters:
    - X: Input data
    - n_clusters: Number of clusters

    Returns:
    - labels: Cluster labels
    - inertia: Sum of squared distances within clusters
    """
    # Compute the similarity matrix using the RBF kernel
    similarity_matrix = rbf_kernel(X.numpy(), gamma=1)
    similarity_tensor = torch.tensor(similarity_matrix, dtype=torch.float32)

    # Construct the Laplacian matrix
    D = torch.diag(similarity_tensor.sum(dim=1))
    L = D - similarity_tensor

    # Perform eigenvalue decomposition to get eigenvectors
    eigenvalues, eigenvectors = torch.linalg.eigh(L)
    # Select the top-k eigenvectors as the new feature space
    top_k_eigenvectors = eigenvectors[:, :n_clusters]

    # Run K-Means on the new feature space
    kmeans = KMeansTorch(n_clusters=n_clusters, max_iters=100, random_state=42)
    kmeans.fit(top_k_eigenvectors)

    return kmeans.labels_, kmeans.inertia_


# Function to evaluate clustering performance
def evaluate_clustering(algorithm_name, labels, inertia, start_time, end_time, data):
    """
    Evaluate the performance of a clustering algorithm and return metrics.

    Parameters:
    - algorithm_name: Name of the algorithm
    - labels: Cluster labels
    - inertia: Sum of squared distances within clusters
    - start_time: Start time of the algorithm
    - end_time: End time of the algorithm
    - data: Original input data for calculating metrics

    Returns:
    - Dictionary of performance metrics
    """
    return {
        'Algorithm': algorithm_name,
        'SSE': inertia,
        'Silhouette Score': silhouette_score(data, labels),
        'Davies-Bouldin Index': davies_bouldin_score(data, labels),
        'Calinski-Harabasz Index': calinski_harabasz_score(data, labels),
        'Run Time (s)': end_time - start_time
    }


# Function to recompute SSE in the original space for Spectral Clustering
def recompute_sse_original_space(X_original, labels):
    """
    Recompute SSE using the original data and the mean of the points in each cluster.

    Parameters:
    - X_original: Original input data (numpy array)
    - labels: Cluster labels from Spectral Clustering

    Returns:
    - SSE value in the original space
    """
    sse = 0
    unique_labels = set(labels)

    for i in unique_labels:
        # Extract all points belonging to the current cluster
        cluster_points = X_original[labels == i]
        if cluster_points.shape[0] > 0:  # Ensure the cluster has points
            # Calculate the mean of the cluster
            cluster_mean = cluster_points.mean(axis=0)
            # Compute the squared distance from each point to the cluster mean
            distances = ((cluster_points - cluster_mean) ** 2).sum(axis=1)
            sse += distances.sum()

    return sse


# Load the Iris dataset
iris = load_iris()
X = torch.tensor(iris.data, dtype=torch.float32)
results = []

# Basic dataset exploration
print("Dataset Shape:", iris.data.shape)
print("Feature Names:", iris.feature_names)
print("Class Distribution:", dict(zip(*np.unique(iris.target, return_counts=True))))

# Summary statistics
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(iris_df.describe())

# Pairwise scatterplot to visualize relationships
sns.pairplot(pd.concat([iris_df, pd.DataFrame(iris.target, columns=['Target'])], axis=1), hue='Target')
plt.show()

# Run Mini-Batch K-Means and record performance
mini_batch_kmeans = MiniBatchKMeansTorch(n_clusters=3, batch_size=30, max_iters=100, random_state=42)
start_time = time.time()
mini_batch_kmeans.fit(X)
end_time = time.time()
results.append(
    evaluate_clustering('Mini-Batch K-Means', mini_batch_kmeans.labels_, mini_batch_kmeans.inertia_, start_time,
                        end_time, iris.data))

# Run Spectral Clustering and record performance
start_time = time.time()
labels_spectral, inertia_spectral = spectral_clustering_torch(X, n_clusters=3)
end_time = time.time()
sse_original_space = recompute_sse_original_space(iris.data, labels_spectral)
results.append(
    evaluate_clustering('Spectral Clustering (Recomputed SSE)', labels_spectral, sse_original_space, start_time,
                        end_time, iris.data))

# Run K-Means and K-Means++ and record performance
kmeans_random = KMeansTorch(n_clusters=3, init='random', max_iters=100, random_state=42)
start_time = time.time()
kmeans_random.fit(X)
end_time = time.time()
results.append(
    evaluate_clustering(f'K-Means ({'random'})', kmeans_random.labels_, kmeans_random.inertia_, start_time, end_time, iris.data))
    
kmeans_pp = KMeansTorch(n_clusters=3, init='k-means++', max_iters=100, random_state=42)
start_time = time.time()
kmeans_pp.fit(X)
end_time = time.time()
results.append(
    evaluate_clustering(f'K-Means ({'k-means++'})', kmeans_pp.labels_, kmeans_pp.inertia_, start_time, end_time, iris.data))
    

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Normalize metrics for comparison (SSE should be minimized; others should be maximized)
normalized_df = results_df.copy()
scaler = MinMaxScaler()
normalized_df[['SSE', 'Silhouette Score', 'Davies-Bouldin Index', 'Calinski-Harabasz Index',
               'Run Time (s)']] = scaler.fit_transform(
    normalized_df[['SSE', 'Silhouette Score', 'Davies-Bouldin Index', 'Calinski-Harabasz Index', 'Run Time (s)']]
)

# Calculate a composite score for overall ranking (invert SSE, Davies-Bouldin, and Run Time for scoring)
normalized_df['Composite Score'] = (
        (1 - normalized_df['SSE']) +  # Invert SSE
        normalized_df['Silhouette Score'] +
        (1 - normalized_df['Davies-Bouldin Index']) +  # Invert DBI
        normalized_df['Calinski-Harabasz Index'] +
        (1 - normalized_df['Run Time (s)'])  # Invert Run Time
)

# Sort by composite score and display original data
results_df['Composite Score'] = normalized_df['Composite Score']
results_df = results_df.sort_values(by='Composite Score', ascending=False)

# Print the results to the console
print("\nClustering Algorithm Performance Evaluation (Sorted by Composite Score):")
print(results_df.to_string(index=False))

# Results
"""
Clustering Algorithm Performance Evaluation (Sorted by Composite Score):
                           Algorithm        SSE  Silhouette Score  Davies-Bouldin Index  Calinski-Harabasz Index  Run Time (s)  Composite Score
                 K-Means (k-means++)  78.855606          0.551192              0.666039               561.593732      0.007050         4.582201
                  Mini-Batch K-Means  80.163277          0.546540              0.672969               558.404613      0.049489         3.568433
Spectral Clustering (Recomputed SSE) 131.247042          0.528335              0.546889               308.076137      0.021679         2.468692
                    K-Means (random) 145.452789          0.499983              0.953419               270.809469      0.001004         1.000000
"""


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(iris.data)

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
algorithms = ['Mini-Batch K-Means', 'Spectral Clustering', 'K-Means (random)', 'K-Means (k-means++)']
labels_list = [mini_batch_kmeans.labels_, labels_spectral, kmeans_random.labels_, kmeans_pp.labels_]

for i, ax in enumerate(axs.ravel()):
    ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels_list[i], cmap='viridis', s=30)
    ax.set_title(algorithms[i])
plt.tight_layout()
plt.show()

from scipy.stats import mode
import numpy as np

# Map each cluster to the most frequent true label
labels_mapped = np.zeros_like(kmeans_pp.labels_)
for i in range(3):  # Number of clusters
    mask = kmeans_pp.labels_ == i
    labels_mapped[mask] = mode(iris.target[mask])[0]

# Calculate accuracy
accuracy = np.mean(labels_mapped == iris.target)
print("Accuracy of K-Means++ Clustering:", accuracy)

# Map each cluster to the most frequent true label
labels_mapped = np.zeros_like(kmeans_random.labels_)
for i in range(3):  # Number of clusters
    mask = kmeans_random.labels_ == i
    labels_mapped[mask] = mode(iris.target[mask])[0]

# Calculate accuracy
accuracy = np.mean(labels_mapped == iris.target)
print("Accuracy of K-Means Clustering:", accuracy)

# Map each cluster to the most frequent true label
labels_mapped = np.zeros_like(mini_batch_kmeans.labels_)
for i in range(3):  # Number of clusters
    mask = mini_batch_kmeans.labels_ == i
    labels_mapped[mask] = mode(iris.target[mask])[0]

# Calculate accuracy
accuracy = np.mean(labels_mapped == iris.target)
print("Accuracy of Mini-Batch K-Means Clustering:", accuracy)

# Map each cluster to the most frequent true label
labels_mapped = np.zeros_like(labels_spectral)
for i in range(3):  # Number of clusters
    mask = labels_spectral == i
    labels_mapped[mask] = mode(iris.target[mask])[0]

# Calculate accuracy
accuracy = np.mean(labels_mapped == iris.target)
print("Accuracy of Spectral Clustering:", accuracy)

"""
Accuracy of K-Means++ Clustering: 0.8933333333333333
Accuracy of K-Means Clustering: 0.6666666666666666
Accuracy of Mini-Batch K-Means Clustering: 0.9
Accuracy of Spectral Clustering: 0.6933333333333334
"""
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Assuming KMeansTorch is already defined in the environment

# Load Iris Dataset
iris = load_iris()
X = torch.tensor(iris.data, dtype=torch.float32)

# 1. Elbow Method
sse_values = []
k_values = range(2, 11)
for k in k_values:
    kmeans = KMeansTorch(n_clusters=k, max_iters=100, random_state=42)
    kmeans.fit(X)
    sse_values.append(kmeans.inertia_)

# Plot SSE for Elbow Method
plt.figure(figsize=(8, 6))
plt.plot(k_values, sse_values, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Sum of Squared Errors (SSE)")
plt.grid()
plt.show()

# 2. Silhouette Analysis
silhouette_scores = []
for k in k_values:
    kmeans = KMeansTorch(n_clusters=k, max_iters=100, random_state=42)
    kmeans.fit(X)
    silhouette_scores.append(silhouette_score(iris.data, kmeans.labels_))

# Plot Silhouette Scores
plt.figure(figsize=(8, 6))
plt.plot(k_values, silhouette_scores, marker='o')
plt.title("Silhouette Analysis")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.grid()
plt.show()

# 3. Visualize Clusters for Selected k
selected_k_values = [2, 3, 4]  # Modify as needed based on the results
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(iris.data)

fig, axs = plt.subplots(1, len(selected_k_values), figsize=(15, 5))

for i, k in enumerate(selected_k_values):
    kmeans = KMeansTorch(n_clusters=k, max_iters=100, random_state=42)
    kmeans.fit(X)
    labels = kmeans.labels_

    axs[i].scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap='viridis', s=30)
    axs[i].set_title(f"k = {k}")

plt.tight_layout()
plt.show()

from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
import numpy as np

gamma_values = [0.01, 0.1, 0.5, 1, 5, 10]
best_gamma = None
best_score = -1

for gamma in gamma_values:
    clustering = SpectralClustering(n_clusters=3, affinity='rbf', gamma=gamma, random_state=42)
    labels = clustering.fit_predict(X)
    score = silhouette_score(X, labels)
    print(f"Gamma: {gamma}, Silhouette Score: {score:.4f}")
    
    if score > best_score:
        best_score = score
        best_gamma = gamma

print(f"\nBest Gamma: {best_gamma}, Best Silhouette Score: {best_score:.4f}")