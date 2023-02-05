from collections import deque
import numpy as np


class QueueClustering:
    def __init__(self, n_clusters, distance_metric='euclidean'):
        self.n_clusters = n_clusters
        self.distance_metric = distance_metric

    def fit_predict(self, X):
        n_samples, _ = X.shape

        # Check if number of desired clusters is greater than number of samples
        if self.n_clusters > n_samples:
            raise ValueError("Number of desired clusters cannot be greater than number of samples")

        # Check if there are NaN values in the input
        if np.isnan(X).any():
            raise ValueError("Input contains NaN values")

        # Check if there are identical samples in the input
        unique_rows, unique_indices = np.unique(X, return_index=True, axis=0)
        if unique_rows.shape[0] != X.shape[0]:
            raise ValueError("Input contains identical samples")

        distances = self._calculate_distance_matrix(X)
        nearest_neighbors = self._calculate_nearest_neighbors(distances)
        cluster_assignments = np.arange(n_samples)
        current_cluster_count = n_samples

        while current_cluster_count > self.n_clusters:
            min_distance = np.inf
            min_i, min_j = None, None
            for i in range(n_samples):
                j = nearest_neighbors[i][0]
                if cluster_assignments[i] != cluster_assignments[j] and distances[i, j] < min_distance:
                    min_distance = distances[i, j]
                    min_i, min_j = i, j

            cluster_assignments[cluster_assignments == cluster_assignments[min_j]] = cluster_assignments[min_i]
            current_cluster_count -= 1
            nearest_neighbors[min_i].extend(nearest_neighbors[min_j])
            nearest_neighbors.pop(min_j)
            self._update_nearest_neighbors(nearest_neighbors, min_i, distances)

        return cluster_assignments

    def _calculate_distance_matrix(self, X):
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                distances[i, j] = np.linalg.norm(X[i] - X[j])
                distances[j, i] = distances[i, j]
        return distances

    def _update_nearest_neighbors(self, nearest_neighbors, min_i, distances):
        n_samples = distances.shape[0]
        for j in range(n_samples):
            if min_i == j:
                continue
            if j in nearest_neighbors[min_i]:
                nearest_neighbors[min_i].remove(j)
            if distances[min_i, j] < distances[j, nearest_neighbors[j][0]]:
                nearest_neighbors[j][0] = min_i
            else:
                nearest_neighbors[j].append(min_i)

    def _calculate_nearest_neighbors(self, distances):
        n_samples = distances.shape[0]
        nearest_neighbors = [deque([j for j in range(n_samples) if j != i]) for i in range(n_samples)]
        for i in range(n_samples):
            nearest_neighbors[i] = deque(sorted(nearest_neighbors[i], key=lambda j: distances[i, j]))
        return nearest_neighbors
