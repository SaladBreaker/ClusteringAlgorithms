import pandas as pd

DUMMY_ENTRY = {'CustomerID': [1, 2, 3], 'Gender': ['Male', 'Female', 'Male'], 'Age': [30, 25, 35],
               'Annual_Income': [50000, 60000, 55000], 'Spending_Score': [70, 80, 75], 'cluster': [1, 2, 3]}


class TestQueueImplementation:
    alg = None

    def test_constructor(self):
        n_clusters = 100
        self.alg(n_clusters)
        assert self.alg.n_clusters == n_clusters
        assert self.alg.distance_metric == 'euclidean'

    def test_calculate_distance_matrix_return_shape(self):
        pass

    def test_calculate_distance_matrix_distance(self):
        pass

    def test_calculate_distance_matrix_on_empty_input(self):
        pass

    def test_model_fit_works_without_error(self):
        pass

    def test_calculate_nearest_neighbors_returns_3(self):
        pass

    def test_calculate_nearest_neighbors_order(self):
        pass

    def test_calculate_nearest_neighbors_on_1x1(self):
        pass
