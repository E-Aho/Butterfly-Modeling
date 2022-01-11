import numpy as np
import pytest

from main import get_nearest_neighbor_dists, get_entropy, get_mi, preprocess


class TestGetMinChebyshevDistances:

    def test_for_simple_case_returns_correctly(self):

        X = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 1]
        ])

        output = get_nearest_neighbor_dists(X, 1)
        assert output == [1, 1, 1]

    def test_for_5_floats_returns_correctly(self):

        X = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.1, 0.0, 1.5, 0.0, 0.0],
            [1.0, -2.0, 0.0, 0.0, -3.4],
            [0.0, 0.0, 1.6, 0.1, 0.0]
        ])

        output = get_nearest_neighbor_dists(X, 1)
        assert output == [1.5, 1.1, 3.4, 1.1]

    def test_works_for_larger_k(self):

        X = np.array([
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [1.1, 0.0, 1.5, 0.0, 0.0],
            [1.0, -2.0, 0.0, 0.0, -3.4],
            [0.0, 0.0, 1.6, 0.1, 0.0]
        ])

        output_2 = get_nearest_neighbor_dists(X, 2)
        output_3 = get_nearest_neighbor_dists(X, 3)

        assert output_2 == [1.3, 1.2, 3.4, 1.3]
        assert output_3 == [3.9, 3.4, 3.9, 3.4]


class TestGetEntropy:

    def test_base_case_returns_correctly(self):

        X = np.array([
            [1.0, 1.0, 2.0, 0.0],
            [1.0, 0.5, 0.0, 0.0],
            [1.2, 0.0, 3.0, 1.0]
        ])

        output = get_entropy(X)

        assert 5.94 < output < 5.96  # flexible to avoid any floating point issues

    def test_base_ordering_makes_sense(self):

        X = np.array([
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 1.1, 2.0, 1.0]
        ])

        Y = np.array([
            [2.0, 1.0, 0.0, 0.0],
            [0.2, 4.0, 0.0, 3.0],
            [1.0, 1.1, 5.0, 1.0]
        ])

        Z = np.array([
            [10.0, 1.0, -20.0, 2.0],
            [0.2, 4.0, 0.9, 3.0],
            [1.1, 11.1, 5.0, 1.0]
        ])

        e_X = get_entropy(X)
        e_Y = get_entropy(Y)
        e_Z = get_entropy(Z)

        assert e_X < e_Y < e_Z

class TestGetMI:

    def test_for_base_case_returns_correctly(self):
        X = np.array([
            np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            np.array([-0.1, -0.2, 0, 0.1, 0.2]),
            np.array([0.2, 0.3, 0.1, 0.3, 0.1]),
            np.array([-0.1, 0, 0, 0, -0.4])
        ])

        Y = np.array([
            0.1, 0.2, 0.3, 0.4
        ]).reshape(-1, 1)

        feature = 0
        selected = [2, 3]

        mi = get_mi(feature, selected, X, Y)
        assert mi > 0

    def test_ordering_makes_sense(self):
        low_x = np.array([
            np.array([0, 1, 2, 3, 4]),
            np.array([1, 1, 0, 0, 0]),
            np.array([2, 3, 1, 3, 1]),
            np.array([-1, 0, 0, 0, 0])
        ])

        high_x = np.array([
            np.array([1, 2, 3]),
            np.array([2, 4, 5]),
            np.array([1.5, 4.5, 1.3]),
            np.array([9, 8, -3.4])
        ])

        Y = np.array([
            1, 3, 2, 6
        ]).reshape(-1, 1)

        low_x = preprocess(low_x)
        high_x = preprocess(high_x)
        Y = preprocess(Y)

        feature = 1
        selected = [0]

        low_mi = get_mi(feature, selected, low_x, Y)
        high_mi = get_mi(feature, selected, high_x, Y)

        assert low_mi > 0
        assert high_mi > 0
        assert low_mi < high_mi
