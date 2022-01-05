import pytest

from main import get_min_chebyshev_distances


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

        X = [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.1, 0.0, 1.5, 0.0, 0.0],
            [1.0, -2.0, 0.0, 0.0, -3.4],
            [0.0, 0.0, 1.6, 0.1, 0.0]
        ]
        output = get_min_chebyshev_distances(X)
        assert output == [1.5, 1.1, 3.4, 1.1]

