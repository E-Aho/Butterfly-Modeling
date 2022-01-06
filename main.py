import math
from typing import List

import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy.special._ufuncs import gamma, psi
from sklearn.preprocessing import StandardScaler


def get_inputs(filename: str = "data/ButterflyFeatures.csv") -> pd.DataFrame:

    # TODO: Normalise each field here before inputting

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

    first_info, first_feature = get_first_mi(Y=label, X=features.values)

    while len(selected_features) <= min(feature_count, len(features)):
        pass

    return selected_features, information_gains


def get_first_mi(X, Y) -> [float]:
    max_mi = 0
    max_i = 0
    for i in range(len(Y)):
        pass
    return


def get_mi(feature, selected, X, Y) -> float:

    n, p = X.shape
    Y = Y.reshape((n, 1))

    joint_data = X[:, (feature + selected)]
    all_data = (joint_data, Y)
    stacked_data = np.hstack(all_data)

    return sum([get_entropy(z) for z in all_data]) - get_entropy(stacked_data)


def get_entropy(vars: np.array, k: int = 2) -> float:
    """Based on Kraskov et al, 2004, Estimating mutual information, and work by Daniel Homola in MIFS repo"""
    distances = get_nearest_neighbor_dists(vars, k)
    if len(vars.shape) == 1:
        vars = vars.reshape((-1, 1))
    n, dimension = vars.shape
    unit_volume = np.pi ** (0.5*dimension) / gamma(dimension / 2 + 1)
    entropy = (
            dimension * np.mean(np.log(distances + np.finfo(vars.dtype).eps))
            + np.log(unit_volume) + psi(n) - psi(k)
    )

    # np.finfo(...).eps required to solve floating point issues which led to taking log of 0
    return entropy


def get_nearest_neighbor_dists(vars, k: int = 2) -> List[float]:
        """Returns a list of the kth minimum chebyshev distance"""
        out = []
        if k > len(vars) - 1:
            raise Exception(f"The given k is too large for the given input.\n K must be <= len(input) - 1, but here:\n"
                            f"K = {k}, len(vars) = {len(vars)}")
        for i in range(len(vars)):
            out.append(
                sorted([
                    distance.chebyshev(vars[i], vars[j]) if j != i else math.inf
                    for j in range(len(vars))
                ])[k-1]
            )
        return out


def preprocess(arr) -> np.ndarray:
    # TODO: Refactor this, make it work for either X or Y, not X and Y as input and return input as scaled.
    scaler = StandardScaler() #TODO: Refactor this, maybe write our own scaler, minimize need for other pkgs

    if len(arr.shape) == 1:
        arr = arr.reshape(-1,1)

    return scaler.fit_transform(arr)

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
