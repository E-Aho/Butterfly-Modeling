import math
from typing import List

import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy.special._ufuncs import gamma, psi


def get_inputs(filename: str = "data/ButterflyFeatures.csv") -> pd.DataFrame:

    output_df = pd.read_csv(filename)
    return output_df


def compute_jmi(
        label: pd.DataFrame,
        features: pd.DataFrame,
        feature_count: int = math.inf,
        produce_plots: bool = False
) -> [List, List]:
    """Performs feature selection based on JMI (Joint Mutual Information),
     and returns a list of the ordered features and the amount of information they add.

     :param label: the target variable (continuous) to compute information against
     :param features: the features to use in feature selection
     :param feature_count: the limit of features to compute. Defaults to computing all features
     :param produce_plots: whether to print and save plots. Defaults to False
     :returns:  a list of the ordered features, and a list of the amount of information they each add"""

    information = 0
    selected_features = []
    information_gains = []
    features_map = {col: i for col, i in sorted(enumerate(list(features.columns)))}
    remaining_features = list(features.columns)

    is_discrete = get_discrete_dict(features.values)

    first_info, first_feature = get_first_mi(Y=label, X=features.values, is_discrete=is_discrete)

    while len(selected_features) <= min(feature_count, len(features)):
        pass

    return selected_features, information_gains


def get_first_mi(X, Y, is_discrete) -> [float, int]:
    max_mi = 0
    max_i = 0
    for i in range(len(Y)):
        pass
    return


def get_mi(X, Y, is_discrete) -> [float, int]:
    if is_discrete:
        return
    else:
        all_vars = np.hstack(X)

def get_entropy(X: np.array):
    """Based on implementation in Daniel Homola's MIFS repo"""
    distances = get_min_chebyshev_distances(X)
    n, dimension = X.shape
    unit_volume = np.pi ** (0.5*dimension) / gamma(dimension / 2 + 1)
    entropy = (dimension * np.mean(np.log(distances) + np.log(unit_volume) + np.log(n-1) - np.log(1)))
    return entropy



def get_min_chebyshev_distances(X) -> List[float]:
        """Returns a list of the minimum chebyshev distance"""
        out = []
        for i in range(len(X)):
            out.append(
                min([
                    distance.chebyshev(X[i], X[j]) if j != i else math.inf
                    for j in range(len(X))
                ])
            )
        return out

def get_discrete_dict(features):
    """Returns a dictionary that tracks if a feature is discrete or continuous"""
    #     NB: For this project, the only discrete feature is binary, so we only check for binary variables.
    d = {}
    for col in features.columns:
        if set(features[col]) == {0, 1} or {True, False}:
            d[col] = True
        else:
            d[col] = False
    return d


def main_entrypoint():
    """Main entrypoint for the code used to analyse factors relating to species richness of butterflies in nations"""
    input_df = get_inputs()
    input_df.sort_index()

    label_col = "logSpeciesDensity"
    non_feature_cols = [label_col, "Country", "SpeciesDensity"]

    label = input_df[label_col]
    features = input_df[[col for col in input_df.columns if col not in non_feature_cols]]

    compute_jmi(label, features)




if __name__ == "__main__":
    main_entrypoint()
