import pandas as pd

DUMMY_ENTRY = {'CustomerID': [1, 2, 3], 'Gender': ['Male', 'Female', 'Male'], 'Age': [30, 25, 35],
               'Annual_Income': [50000, 60000, 55000], 'Spending_Score': [70, 80, 75], 'cluster': [1, 2, 3]}

class TestNaiveImplementation:
    alg = None

    def test_constructor(self):
        n_clusters = 100
        alg(n_clusters)
        assert alg.n_clusters == n_clusters

    def test_calculate_distance_matrix_return_shape(self):
        pass


    def test_calculate_distance_matrix_distance(self):
        pass

    def test_calculate_distance_matrix_on_empty_input(self):
        pass

    def test_model_fit_works_without_error(self):
        pass
