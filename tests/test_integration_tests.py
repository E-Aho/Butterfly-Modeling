import numpy as np
import pandas as pd
import pytest

from main import compute_best_features


class TestComputeBestFeatures:
    def test_for_simple_input_returns_correctly(self):
        label = pd.DataFrame(
            np.array([1, 2, 3, 4]), columns=["label"]
        )


        #Low info first feature, high info second feature
        features = pd.DataFrame(
        np.array([
            [1, 1, 1],
            [1, 0.9, 1.2],
            [1, 1, 2],
            [1, 1.1, 4],
        ]), columns=["a", "b", "c"],
        )

        output_feat, output_info = compute_best_features(label=label, features=features, produce_plots=False)

        # Checking features returned as expected
        assert output_feat == ["c", "b", "a"]

        # Checking information is reasonable
        assert sorted(output_info) == output_info
        assert min(output_info) > 0
