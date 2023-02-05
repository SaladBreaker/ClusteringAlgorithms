import numpy as np


class AgglomerativeClustering1:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

    def fit_predict(self, X):
        n_samples, _ = X.shape
        distances = self._calculate_distance_matrix(X)
        cluster_assignments = np.arange(n_samples)
        current_cluster_count = n_samples

        while current_cluster_count > self.n_clusters:
            min_distance = np.inf
            min_i, min_j = None, None
            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    if cluster_assignments[i] != cluster_assignments[j] and distances[i, j] < min_distance:
                        min_distance = distances[i, j]
                        min_i, min_j = i, j
            cluster_assignments[cluster_assignments == cluster_assignments[min_j]] = cluster_assignments[min_i]
            current_cluster_count -= 1

        return cluster_assignments

    def _calculate_distance_matrix(self, X):
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                distances[i, j] = np.linalg.norm(X[i] - X[j])
                distances[j, i] = distances[i, j]
        return distances
