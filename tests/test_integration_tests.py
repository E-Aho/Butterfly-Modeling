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
            [0, 1, 1],
            [0, 1, 1.2],
            [0, 2, 2],
            [0, 2, 4],
        ]), columns=["a", "b", "c"],
        )

        output_feat, output_info = compute_best_features(label=label, feature_array=features, produce_plots=False)

        # Checking features returned as expected
        assert output_feat == ["c", "b", "a"]

        # Checking information is reasonable
        assert min(output_info) > 0
