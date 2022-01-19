import argparse
import math
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial import distance
from scipy.special._ufuncs import gamma, psi


def get_inputs(filename: str) -> pd.DataFrame:
    output_df = pd.read_csv(filename)
    return output_df


def compute_best_features(
        label: pd.DataFrame,
        feature_array: pd.DataFrame,
        feature_count: int = math.inf,
        produce_plots: bool = False,
        k: int = 3
) -> [List, List]:
    """Performs feature selection based on JMI (Joint Mutual Information),
     and returns a list of the ordered features and the amount of information they add.

     :param label: the target variable (continuous) to compute information against
     :param feature_array: the features to use in feature selection
     :param feature_count: the limit of features to compute. Defaults to computing all features
     :param produce_plots: whether to print and save plots. Defaults to False
     :param k: the kth neighbor to take the distance from when computing entropy. Defaults to 3
     :returns:  a list of the ordered features, and a list of the amount of information they each add"""

    features_map = {col: i for col, i in sorted(enumerate(list(feature_array.columns)))}

    feature_array = feature_array.to_numpy()
    target_array = label.to_numpy()

    feature_array = preprocess(feature_array)
    target_array = preprocess(target_array)

    n_obs, n_features = feature_array.shape

    selected_features = []
    remaining_features = list(range(n_features))
    information_list = []

    mi_matrix = np.zeros((n_obs, n_features))
    mi_matrix[:] = np.nan

    first_info, first_feature = select_first_feature(target_matrix=target_array, feature_matrix=feature_array, k=k)
    selected_features.append(first_feature)
    information_list.append(first_info)
    remaining_features.remove(first_feature)

    while len(selected_features) <= min(feature_count, len(feature_array)) and len(remaining_features) >= 1:
        s = len(selected_features) - 1

        mi_vector = get_mi_vector(feature_matrix=feature_array, target_matrix=target_array, selected=selected_features,
                                  remaining=remaining_features, k=k)
        mi_matrix[s, remaining_features] = mi_vector
        current_mi_matrix = mi_matrix[:s + 1, remaining_features]

        selected = remaining_features[np.nanargmax(np.nanmin(current_mi_matrix, axis=0))]
        information_list.append(np.nanmax(np.nanmin(current_mi_matrix, axis=0)))

        selected_features.append(selected)
        remaining_features.remove(selected)

    output_features = [features_map[feature] for feature in selected_features]

    if produce_plots:
        plot_figs(output_features, information_list)

    return output_features, information_list





def select_first_feature(feature_matrix, target_matrix, k: int = 2) -> [float, int]:
    """Uses an independent features approach to find the feature in X with the highest MI with Y

    Args:
        feature_matrix (np.ndarray): The features
        target_matrix (np.ndarray): The target variable
        k (int): Number of neighbours used in the calculation of entropy. Defaults to 2.

    Returns:
        min_val, min_i ([float, int]): the value of the minimum MI found and the index of the feature
    """
    mi_array = get_first_mi_vector(feature_matrix=feature_matrix, target_matrix=target_matrix, k=k)
    min_val = np.nanmax(mi_array)
    min_i = np.nanargmax(mi_array)
    return min_val, min_i


def get_first_mi_vector(feature_matrix, target_matrix, k=2) -> np.ndarray:
    """Finds the MI of each feature in X with the target Y

    Args:
        feature_matrix (np.ndarray): The features
        target_matrix (np.ndarray): The target variable
        k (int): Number of neighbours used in the calculation of entropy. Defaults to 2.

    Returns:
        np.ndarray: An array containing the MIs'
    """
    n, f = feature_matrix.shape
    mi_array = []
    for i in range(f):  # could be parallelized for speed up
        variables = (feature_matrix[:, i].reshape((n, 1)), target_matrix)
        stacked_vars = np.hstack(variables)
        mi = sum([get_entropy(X, k) for X in variables]) - get_entropy(stacked_vars, k)
        mi_array.append(mi)
    return np.array(mi_array)


def get_mi_vector(
        feature_matrix: np.ndarray,
        target_matrix: np.ndarray,
        selected: list,
        remaining: list, k=3
) -> np.ndarray:
    """Finds the MI of each feature in X and all the features in selected given Y

    Args:
        feature_matrix (np.ndarray): The features
        target_matrix (np.ndarray): The target variable
        selected (list[int]): The indexes of the features selected
        remaining(list[int]): the indexes of the remaining features
        k (int): Number of neighbours used in the calculation of entropy. Defaults to 2.

    Returns:
        np.ndarray: An array containing the MIs' for each unselected feature against the already selected features
    """
    mi_array = []
    for i in remaining:
        mi = get_mi(feature=i, selected=selected, feature_matrix=feature_matrix, target_matrix=target_matrix, k=k)
        mi_array.append(mi)
    return np.array(mi_array)


def get_mi(
        feature: int,
        selected: list[int],
        feature_matrix: np.ndarray,
        target_matrix: np.ndarray,
        k: int = 2
) -> float:
    """Find the specific MI of a single feature and all the features in selected given Y

    Args:
        feature (int): The index of the feature in X
        selected (list[int]): The indexes of the features selected in X
        feature_matrix (np.ndarray): The features
        target_matrix (np.ndarray): The target variable
        k (int, optional): Number of neighbours used in the calculation of entropy. Defaults to 2.

    Returns:
        float: The MI for the input feature against the already selected features
    """
    n, p = feature_matrix.shape
    target_matrix = target_matrix.reshape((n, 1))

    joint_data = feature_matrix[:, ([feature] + selected)]
    all_data = (joint_data, target_matrix)
    stacked_data = np.hstack(all_data)

    info = sum([get_entropy(z) for z in all_data]) - get_entropy(stacked_data)
    if info < 0:
        return np.nan  # information gain cannot be negative
    return info


def get_entropy(
        variables: np.array,
        k: int = 2
) -> float:
    """Based on Kraskov et al, 2004, Estimating mutual information, and work by Daniel Homola in MIFS repo
    Calculates the joint entropy of all the features in vars

    Args:
        variables (np.array): The features to calculate the entropy of
        k (int, optional): Number of neighbours used in the calculation of entropy. Defaults to 2.

    Returns:
        float: The joint entropy H(features in vars)
    """
    distances = get_nearest_neighbor_dists(variables, k)
    if len(variables.shape) == 1:
        variables = variables.reshape((-1, 1))
    n, dimension = variables.shape
    unit_volume = np.pi ** (0.5 * dimension) / gamma(dimension / 2 + 1)
    entropy = (
            dimension * np.mean(
                np.log(distances + np.finfo(variables.dtype).eps)  # np.finfo(...).eps required to solve floating point issues which led to taking log of 0
            ) + np.log(unit_volume) + psi(n) - psi(k)
    )

    return entropy


def get_nearest_neighbor_dists(
        variables,
        k: int = 2
) -> List[float]:
    """Returns a list of the kth minimum chebyshev distance

    Args:
        variables (np.ndarray): The dataset to calculate the pairwise distances on
        k (int, optional): The number of closest neighbours to return . Defaults to 2.

    Raises:
        Exception: If the k is greater than the number of data points minus 1

    Returns:
        List[List[int]]: the sorted distances of the k closest neighbours to each point 
    """
    out = []
    if k > len(variables) - 1:
        raise Exception(f"The given k is too large for the given input.\n K must be <= len(input) - 1, but here:\n"
                        f"K = {k}, len(vars) = {len(variables)}")
    for i in range(len(variables)):
        out.append(
            sorted([
                distance.chebyshev(variables[i], variables[j]) if j != i else math.inf
                for j in range(len(variables))
            ])[k - 1]
        )
    return out


def preprocess(arr) -> np.ndarray:
    """Uses np linalg to
    Assumes that columns are features, and rows are observations

    Args:
        arr (np.ndarray): The array to be scaled

    Returns:
        np.ndarray: A centred and standardised version of the data
    """

    if len(arr.shape) == 1:
        arr = arr.reshape(-1, 1)

    norm_arr = np.linalg.norm(arr, axis=0, ord=2)  # l2 norm for each feature
    # note that because we are using Chebyshev distances, we don't need to adjust mean, just variance

    return np.nan_to_num(arr / norm_arr, nan=np.finfo(norm_arr.dtype).eps)


def plot_figs(output_features: List[str], information_list: List[int]) -> None:
    with plt.style.context('science'):
        x = range(1, len(output_features) + 1)
        y = information_list
        fig, ax = plt.subplots(dpi=1600)
        plt.bar(x, y, width=0.8)
        plt.xlim(0, 30)
        ax.legend(title="Incremental information gain")
        ax.set(xlabel="$n^{th}$ selected feature", ylabel="Information gain/ Shannons")
        plt.xticks(np.arange(1, 30, step=10), fontsize=10, )
        plt.savefig("figures/bar_chart.pdf")


def main_entrypoint():
    """Main entrypoint for the code used to analyse factors relating to species richness of butterflies in nations"""

    parser = argparse.ArgumentParser(description="Main entrypoint to the Feature Selection algorithm")

    parser.add_argument("-p", "--print-plots",
                        help="Enables the plot printing function. Will print to the /figures folder",
                        action="store_true")

    parser.add_argument("--file",
                        help="Optional path to csv data. Defaults to using data/ButterflyFeatures.csv",
                        default="data/ButterflyFeatures.csv")

    args = parser.parse_args()
    input_df = get_inputs(filename=args.file)
    input_df.sort_index()

    label_col = "log(species)/log(area)"
    non_feature_cols = [label_col, "Country", "SpeciesDensity"]

    label = input_df[label_col]
    features = input_df[[col for col in input_df.columns if col not in non_feature_cols]]

    feature_list, info = compute_best_features(label, features, produce_plots=args.print_plots)
    print(f"list of features: {feature_list}")
    print(f"info: {info}")


if __name__ == "__main__":
    main_entrypoint()
