import pandas as pd
import numpy as np

from models.naive_model import AgglomerativeClustering1

DUMMY_ENTRY = {'CustomerID': [1, 2, 3], 'Gender': ['Male', 'Female', 'Male'], 'Age': [30, 25, 35],
               'Annual_Income': [50000, 60000, 55000], 'Spending_Score': [70, 80, 75], 'cluster': [1, 2, 3]}

class TestNaiveImplementation:
    alg = AgglomerativeClustering1(100)

    def test_constructor(self):
        assert self.alg.n_clusters == n_clusters

    def test_calculate_distance_matrix_return_shape(self):
        X = pd.DataFrame(DUMMY_ENTRY)
        distances = self.alg._calculate_distance_matrix(X)
        assert distances.shape == (3, 3), f"Expected shape (3, 3), got {distances.shape}"

    def test_calculate_distance_matrix_distance(self):
        X = pd.DataFrame(DUMMY_ENTRY)
        distances = self.alg._calculate_distance_matrix(X)
        expected_distances = np.array([[0, 14.1421356, 10.0], [14.1421356, 0, 10.0], [10.0, 10.0, 0]])
        np.testing.assert_array_almost_equal(distances, expected_distances, decimal=6,
                                             err_msg="Incorrect calculation of distances")

    def test_calculate_distance_matrix_on_empty_input(self):
        X = pd.DataFrame(
            {'CustomerID': [], 'Gender': [], 'Age': [], 'Annual_Income': [], 'Spending_Score': [], 'cluster': []})
        distances = self.alg._calculate_distance_matrix(X)
        assert distances.shape == (0, 0), f"Expected shape (0, 0), got {distances.shape}"

    def test_model_fit_works_without_error(self):
        X = pd.DataFrame(DUMMY_ENTRY)
        self.alg.fit_predict(X)
