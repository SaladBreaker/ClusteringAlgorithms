import heapq
import numpy as np


class QueueClustering:
    def _init_(self, n_clusters):
        self.n_clusters = n_clusters

    def fit_predict(self, X):
        n_samples, _ = X.shape
        distances = self._calculate_distance_matrix(X)
        cluster_assignments = np.arange(n_samples)
        current_cluster_count = n_samples

        heap = []
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                heap.append((distances[i, j], i, j))
        heapq.heapify(heap)

        while current_cluster_count > self.n_clusters:
            min_distance, min_i, min_j = heapq.heappop(heap)
            if cluster_assignments[min_i] != cluster_assignments[min_j]:
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