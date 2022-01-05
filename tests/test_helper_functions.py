import numpy as np
import pytest

from main import get_min_chebyshev_distances, get_entropy


class TestGetMinChebyshevDistances:

    def test_for_simple_case_returns_correctly(self):

        X = [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 1]
        ]

        output = get_min_chebyshev_distances(X)
        assert output == [1, 1, 1]

    def test_for_5_floats_returns_correctly(self):

        X = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.1, 0.0, 1.5, 0.0, 0.0],
            [1.0, -2.0, 0.0, 0.0, -3.4],
            [0.0, 0.0, 1.6, 0.1, 0.0]
        ])

        output = get_min_chebyshev_distances(X)
        assert output == [1.5, 1.1, 3.4, 1.1]


class TestGetEntropy:

    def test_base_case_returns_correctly(self):

        X = np.array([
            [1.0, 1.0, 2.0, 0.0],
            [1.0, 0.5, 0.0, 0.0],
            [1.2, 0.0, 3.0, 1.0]
        ])

        output = get_entropy(X)

        assert 10.081 < output < 10.083  # flexible to avoid any floating point issues

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
