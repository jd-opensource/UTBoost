# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from typing import Union, Tuple, Sequence, List

from sklearn.datasets import make_classification


def logloss(preds: np.ndarray, labels: np.ndarray):
    return np.mean(
        -np.log(np.where(labels, preds, 1 - preds))
    )

def mse(preds: np.ndarray, labels: np.ndarray):
    return np.mean(np.square(preds - labels))


def segment_metric(preds: np.ndarray, labels: np.ndarray, tid: np.ndarray, metric: str = 'logloss'):
    tid = tid.flatten().astype(np.int32)
    metric = globals()[metric]
    preds_new = np.zeros(len(labels))
    for t in sorted(set(tid)):
        idx = (tid == t)
        preds_new[idx] = preds[idx, t]
    return metric(preds_new, labels)


def make_uplift_problem(
        n_samples_per_group: int = 1000,
        n_treatments: int = 2,
        delta_uplift: Union[Sequence[float], float] = 0.1,
        n_features: int = 10,
        n_informative: int = 5,
        n_uplift_mix_informative: int = 5,
        n_redundant: int = 0,
        n_repeated: int = 0,
        flip_y: float = 0.,
        pos_weight: float = 0.5,
        random_seed: int = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Generate a synthetic dataset for classification uplift problem.

    Parameters
    ----------
    n_samples_per_group : int, optional (default=1000)
        The number of samples per group.
    n_treatments : int, optional (default=2)
        The number of treated groups.
    delta_uplift : float or array-like, optional (default=0.1)
        The delta treatment effect. If an array-like, it should contain the treatment effects
        for each treatment group.
    n_features : int, optional (default=10)
        The total number of features.
    n_informative : int, optional (default=5)
        The number of informative features.
    n_uplift_mix_informative : int, optional (default=5)
        The number of mixed informative features.
    n_redundant : int, optional (default=0)
        The number of redundant features.
    n_repeated : int, optional (default=0)
        The number of duplicated features, drawn randomly from the informative
        and the redundant features.
    flip_y : float, optional (default=0.0)
        The fraction of samples whose class is assigned randomly.
    pos_weight : float, optional (default=0.5)
        The proportion of samples assigned to the positive class.
    random_seed : int, optional (default=None)
        Random seed for reproducibility.

    Returns
    -------
    output : DataFrame
        Data frame with the treatment, label, features, and effect.
    features : list of strings
        Feature names in the output DataFrame.

    References
    ----------
    [1] I. Guyon, "Design of experiments for the NIPS 2003 variable selection benchmark", 2003.
    """
    if random_seed is not None:
        np.random.seed(seed=random_seed)
    # dataset dataframe
    dataset = pd.DataFrame()
    n_samples = n_samples_per_group * (n_treatments + 1)
    # generate treatments
    treatments = np.repeat(np.arange(n_treatments + 1), n_samples_per_group)
    treatments = np.random.permutation(treatments)
    dataset["treatment"] = treatments
    # generate base dataset
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_informative,
                               n_redundant=n_redundant, n_repeated=n_repeated, n_clusters_per_class=1,
                               weights=[1 - pos_weight, pos_weight], flip_y=flip_y)
    features = []
    x_informative_name = []
    for i in range(n_informative):
        x_name = "x" + str(len(features) + 1) + "_informative"
        features.append(x_name)
        x_informative_name.append(x_name)
        dataset[x_name] = X[:, i]
    for i in range(n_redundant):
        x_name = "x" + str(len(features) + 1) + "_redundant"
        features.append(x_name)
        dataset[x_name] = X[:, n_informative + i]
    for i in range(n_repeated):
        x_name = "x" + str(len(features) + 1) + "_repeated"
        features.append(x_name)
        dataset[x_name] = X[:, n_informative + n_redundant + i]

    # Generate irrelevant features
    for i in range(n_features - n_informative - n_redundant - n_repeated):
        x_name = "x" + str(len(features) + 1) + "_irrelevant"
        features.append(x_name)
        dataset[x_name] = np.random.normal(0, 1, n_samples)

    # delta values
    if isinstance(delta_uplift, float):
        delta_uplift = [delta_uplift] * n_treatments
    else:
        assert len(delta_uplift) == n_treatments

    yt = y.copy()

    for tid, delta in enumerate(delta_uplift, start=1):
        suffix, direct = ("_increase", 1) if delta >= 0 else ("_decrease", -1)
        idx = dataset.index[dataset["treatment"] == tid].tolist()
        adjust_abs_pos_weight = abs(delta) / (1 - pos_weight)
        X_delta, y_delta = make_classification(n_samples=n_samples, n_features=n_informative,
                                               n_informative=n_informative, n_redundant=0,
                                               n_clusters_per_class=1,
                                               weights=[1 - adjust_abs_pos_weight, adjust_abs_pos_weight])
        x_uplift_delta_name = []
        for i in range(n_informative):  # uplift informative x
            x_name = "x" + str(len(features) + 1) + "_t" + str(tid) + "_uplift" + suffix
            features.append(x_name)
            x_uplift_delta_name.append(x_name)
            dataset[x_name] = X_delta[:, i]

        if n_uplift_mix_informative > 0:  # mix informative x
            for i in range(n_uplift_mix_informative):
                x_name = "x" + str(len(features) + 1) + "_t" + str(tid) + "_uplift_mix" + suffix
                features.append(x_name)
                dataset[x_name] = (
                        np.random.uniform(-1, 1) * dataset[np.random.choice(x_informative_name)]
                        + np.random.uniform(-1, 1) * dataset[np.random.choice(x_uplift_delta_name)]
                )

        yt[idx] += y_delta[idx] * direct

    yt = np.clip(yt, 0, 1)

    dataset["label"] = yt
    dataset["effect"] = yt - y
    return dataset, features
