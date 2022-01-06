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
        produce_plots: bool = False,
        k: int = 3
) -> [List, List]:
    """Performs feature selection based on JMI (Joint Mutual Information),
     and returns a list of the ordered features and the amount of information they add.

     :param label: the target variable (continuous) to compute information against
     :param features: the features to use in feature selection
     :param feature_count: the limit of features to compute. Defaults to computing all features
     :param produce_plots: whether to print and save plots. Defaults to False
     :param k: the kth neighbor to take the distance from when computing entropy. Defaults to 3
     :returns:  a list of the ordered features, and a list of the amount of information they each add"""

    X = features.to_numpy()
    Y = label.to_numpy()

    X = preprocess(X)
    Y = preprocess(Y)

    n_obs, n_features = X.shape

    selected_features = []
    remaining_features = list(range(n_features))
    information_list = []

    features_map = {col: i for col, i in sorted(enumerate(list(features.columns)))}

    mi_matrix = np.zeros((n_obs, n_features))
    mi_matrix[:] = np.nan

    first_info, first_feature = select_first_feature(Y=Y, X=X, k=k)
    selected_features.append(first_feature)
    information_list.append(first_info)
    remaining_features.remove(first_feature)

    while len(selected_features) <= min(feature_count, len(features)):
        s = len(selected_features) - 1

        for feature in remaining_features:
            mi_matrix[s, feature] = get_mi(feature=feature, selected=selected_features, X=X, Y=Y, k=k)

        current_mi_matrix = mi_matrix[:len(selected_features), remaining_features]

        selected = remaining_features[np.nanargmax(np.nansum(current_mi_matrix, axis=0))]

        information_list.append(np.nanmax(np.nanmin(current_mi_matrix, axis=0)))
        selected_features.append(selected)
        remaining_features.remove(selected)
    return selected_features, information_list

def select_first_feature(X, Y, k: int = 2) -> [float, int]:
    mi_array = get_first_mi_vector(X=X, Y=Y, k=k)
    min_val = np.nanmin(mi_array)
    min_i = np.nanargmin(mi_array)
    return min_val, min_i


def get_first_mi_vector(X, Y, k=2) -> np.ndarray:
    n, f = X.shape
    mi_array = []
    for i in range(f): # could be parallelized for speed up
        vars = (X[:, i].reshape((n, 1)), Y)
        stacked_vars = np.hstack(vars)
        mi = sum([get_entropy(X, k) for X in vars]) - get_entropy(stacked_vars, k)
        mi_array.append(mi)
    return np.array(mi_array)

def get_mi_vector(X, Y, selected: list, k=2) -> np.ndarray:
    n, f = X.shape
    mi_array = []
    for i in range(f):
        mi = get_mi(feature=i, selected=selected, X=X, Y=Y, k=3)
        mi_array.append(mi)
    return np.array(mi_array)

def get_mi(feature: int, selected: list[int], X: np.ndarray, Y: np.ndarray, k: int = 2) -> float:

    n, p = X.shape
    Y = Y.reshape((n, 1))

    joint_data = X[:, ([feature] + selected)]
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
            dimension * np.mean(np.log(distances + np.finfo(vars.dtype).eps))  # np.finfo(...).eps required to solve floating point issues which led to taking log of 0
            + np.log(unit_volume) + psi(n) - psi(k)
    )

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
        arr = arr.reshape(-1, 1)

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
