import pandas as pd
import numpy as np

from models.queue_model import AgglomerativeClustering3

DUMMY_ENTRY = {'CustomerID': [1, 2, 3], 'Gender': [1, 0, 1], 'Age': [30, 25, 35],
               'Annual_Income': [50000, 60000, 55000], 'Spending_Score': [70, 80, 75], 'cluster': [1, 2, 3]}


class TestQueueImplementation:
    alg = AgglomerativeClustering3(3)

    def test_constructor(self):
        assert self.alg.n_clusters == 3

    def test_calculate_distance_matrix_return_shape(self):
        X = pd.DataFrame(DUMMY_ENTRY).values
        distances = self.alg._calculate_distance_matrix(X)
        assert distances.shape == (3, 3), f"Expected shape (3, 3), got {distances.shape}"

    def test_calculate_distance_matrix_distance(self):
        X = pd.DataFrame(DUMMY_ENTRY).values
        distances = self.alg._calculate_distance_matrix(X)
        expected_distances = np.array([[0, 14.1421356, 10.0], [14.1421356, 0, 10.0], [10.0, 10.0, 0]])
        assert distances.all() == expected_distances.all()

    def test_calculate_distance_matrix_on_empty_input(self):
        X = pd.DataFrame(
            {'CustomerID': [], 'Gender': [], 'Age': [], 'Annual_Income': [], 'Spending_Score': [],
             'cluster': []}).values
        distances = self.alg._calculate_distance_matrix(X)
        assert distances.shape == (0, 0), f"Expected shape (0, 0), got {distances.shape}"

    def test_model_fit_works_without_error(self):
        X = pd.DataFrame(DUMMY_ENTRY).values
        self.alg.fit_predict(X)

    def test_calculate_nearest_neighbors_returns_3(self):
        distances = np.array([[0, 10, 5], [10, 0, 8], [5, 8, 0]])
        nearest_neighbors = self.alg._calculate_nearest_neighbors(distances)
        assert len(nearest_neighbors) == 3, f"Expected 3 nearest neighbors, got {len(nearest_neighbors)}"

    def test_calculate_nearest_neighbors_on_1x1(self):
        distances = np.array([[0]])
        nearest_neighbors = self.alg._calculate_nearest_neighbors(distances)
        assert len(nearest_neighbors) == 1, f"Expected 1 nearest neighbor, got {len(nearest_neighbors)}"
        assert len(nearest_neighbors[0]) == 0, f"Expected 0 nearest neighbors, got {len(nearest_neighbors[0])}"
