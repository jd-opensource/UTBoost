# -*- coding:utf-8 -*-
import itertools
import numpy as np
import pandas as pd

from numpy import ndarray
from pandas import DataFrame
from typing import Tuple, Union

from .validation import check_binary, check_consistent_length, is_numpy_2d_array


def make_uplift_curve(
        actual_outcome: ndarray,
        uplift_score: ndarray,
        treatment_flag: ndarray,
        drop_duplicates: bool = True) -> DataFrame:
    """ Compute Uplift curve.
    This function calculates the points on an uplift curve. The uplift
    curve represents the incremental impact of a treatment, sorted by the predicted
    likelihood of an individual responding to the treatment.

    Parameters
    ----------
    actual_outcome : array-like matrix
        Binary target values.
    uplift_score : array-like matrix
        Predicted uplift scores.
    treatment_flag : array-like matrix
        Binary treatment assignment indicators.
    drop_duplicates : bool, optional (default=True)
        Whether drop duplicates to keep only the last cumulative value for each unique predicted uplift.

    Returns
    -------
    DataFrame: A DataFrame with the uplift curve points, including the number of samples,
               cumulative outcomes for treatment and control, and uplift values.
    """

    check_consistent_length(actual_outcome, uplift_score, treatment_flag)
    check_binary(treatment_flag)

    # Create a DataFrame for easier manipulation
    # Sort the data by predicted uplifts in descending order
    sorted_df = DataFrame({
        "threshold": np.r_[np.inf, uplift_score.flatten()],
        "_actual_outcome": np.r_[0, actual_outcome.flatten()],
        "_treatment_flag": np.r_[0, treatment_flag.flatten()]
    }, dtype=np.float32).sort_values(by="threshold", ascending=False, ignore_index=True)

    sorted_df["count"] = sorted_df.index.values.astype(np.float32)

    # Create control and treatment groups
    sorted_df["_control_outcome"] = sorted_df["_actual_outcome"] * (1 - sorted_df["_treatment_flag"])
    sorted_df["_treatment_outcome"] = sorted_df["_actual_outcome"] * sorted_df["_treatment_flag"]

    # Calculate cumulative sums for both groups
    sorted_df[["cumulative_treatment", "cumulative_treatment_outcome", "cumulative_control_outcome"]] = \
        sorted_df[["_treatment_flag", "_treatment_outcome", "_control_outcome"]].cumsum().values
    sorted_df["cumulative_control"] = sorted_df["count"] - sorted_df["cumulative_treatment"]

    # Drop duplicates to keep only the last cumulative value for each unique predicted uplift
    if drop_duplicates:
        sorted_df = sorted_df.drop_duplicates(subset=["threshold"], keep="last", ignore_index=True)

    # Calculate the uplift curve values
    sorted_df["treatment_outcome_mean"] = (sorted_df["cumulative_treatment_outcome"] /
                                           sorted_df["cumulative_treatment"]).replace([np.inf, -np.inf, np.nan], 0)

    sorted_df["control_outcome_mean"] = (sorted_df["cumulative_control_outcome"] /
                                         sorted_df["cumulative_control"]).replace([np.inf, -np.inf, np.nan], 0)

    sorted_df["uplift_curve"] = sorted_df["treatment_outcome_mean"] - sorted_df["control_outcome_mean"]

    return sorted_df.drop([i for i in sorted_df.columns if i.startswith("_")], axis=1)


def _extract_best_auuc_curve(uplift_stat: DataFrame) -> Tuple[ndarray, ndarray]:
    """ Extract the theoretical best AUUC curve from uplift statistics.

    This function calculates the coordinates of the best possible AUUC curve based on
    cumulative statistics of treatment and control groups. The best AUUC curve represents
    the ideal scenario where the maximum possible uplift is achieved.

    Parameters
    ----------
    uplift_stat : DataFrame
        A DataFrame containing cumulative statistics necessary for AUUC curve calculations.
        Expected columns are:
        - "count": the total number of samples
        - "cumulative_control": the cumulative count of control group samples
        - "cumulative_control_outcome": the cumulative count of positive outcomes in the control group
        - "cumulative_treatment_outcome": the cumulative count of positive outcomes in the treatment group
        - "cumulative_treatment": the cumulative count of treatment group samples
        - "uplift_curve": the calculated uplift values at each decile of the scored data

    Returns
    -------
    Tuple[ndarray, ndarray]
        Two numpy arrays representing the x and y coordinates of the best AUUC curve.
    """

    # Extract cumulative numbers and uplift curve values from the DataFrame
    num_samples = uplift_stat["count"].iloc[-1]
    num_control = uplift_stat["cumulative_control"].iloc[-1]
    num_control_pos = uplift_stat["cumulative_control_outcome"].iloc[-1]
    num_treatment_pos = uplift_stat["cumulative_treatment_outcome"].iloc[-1]
    num_treatment = uplift_stat["cumulative_treatment"].iloc[-1]
    num_control_neg = num_control - num_control_pos
    num_treatment_neg = num_treatment - num_treatment_pos

    # Determine the best coordinates based on the cumulative statistics
    if num_control_pos > num_treatment_neg:
        x = num_samples - num_treatment_neg
        uplift = 1.0 - num_control_pos / num_control
    else:
        x = num_samples - num_control_pos
        uplift = num_treatment_pos / num_treatment

    best_xs = [0, num_treatment_pos + num_control_neg, x, num_samples]
    best_ys = [0, num_treatment_pos + num_control_neg, uplift * x, uplift_stat["uplift_curve"].iloc[-1] * num_samples]

    return np.array(best_xs), np.array(best_ys)


def _multi_treatment_to_binary(actual_outcome: ndarray,
                               uplift_score: ndarray,
                               treatment_flag: ndarray) -> Tuple[ndarray, ndarray, ndarray]:
    """
    Convert the multi-treatment problem into a standard binary problem.

    The transformation retains all control group samples,
    and when the true treatment group is equal to the one with the largest predicted uplift,
    the value is taken as the corresponding score.

    Note that other treatment group samples that do not meet the conditions are discarded,
    which means that the sample set will change as the score changes.

    Raises
    ------
    ValueError
        If the treatment flags are not binary or incremental starting from 0, if the uplift
        score is not a two-dimensional array, or if the dimensions of the uplift score do
        not match the number of treatment groups minus one, a ValueError is raised with an
        appropriate message.

    Example
    -------
    >>> actual_outcome = np.array([1, 0, 0, 1])
    >>> uplift_score = np.array([[0.2, 0.8], [0.5, 0.5], [0.9, 0.1], [0.4, 0.6]])
    >>> treatment_flag = np.array([1, 0, 1, 2])
    >>> _multi_treatment_to_binary(actual_outcome, uplift_score, treatment_flag)
    (array([0, 0, 1]), array([0.5, 0.9, 0.6]), array([0, 1, 1]))
    """
    check_consistent_length(actual_outcome, uplift_score, treatment_flag)

    try:
        check_binary(treatment_flag)
        return actual_outcome, uplift_score, treatment_flag
    except ValueError:
        if not is_numpy_2d_array(uplift_score):
            raise ValueError("Input score should be presented as a two-dimensional array.")

        treatment_flag = treatment_flag.astype(np.int32).flatten()
        if uplift_score.shape[1] != np.max(treatment_flag):
            raise ValueError("The number of columns in the uplift score must be equal to the number "
                             "of distinct treatment groups minus one.")

        control_flag = treatment_flag == 0
        # Determine the boolean index for filtering based on the treatment flag and uplift score
        bool_idx = (np.argmax(uplift_score, axis=1) == (treatment_flag - 1)) | control_flag
        # Select the maximum uplift score across the second dimension
        uplift_score = np.max(uplift_score, axis=1)

        return actual_outcome[bool_idx], uplift_score[bool_idx], np.clip(treatment_flag[bool_idx], 0, 1)


def auuc_score(
        actual_outcome: ndarray,
        uplift_score: ndarray,
        treatment_flag: ndarray,
        normalize: bool = True) -> float:
    """ Calculate the Area Under the Uplift Curve (AUUC) for an uplift model.

    The AUUC is a metric that quantifies the cumulative gain from an uplift model, which
    can be normalized by the best possible uplift to provide a relative measure of model
    performance. A higher AUUC indicates a more effective model at targeting individuals
    that are more likely to respond positively to the treatment.

    Parameters
    ----------
    actual_outcome : array-like matrix
        Target values.
    uplift_score : array-like matrix
        Predicted uplift scores from the model, indicating the likelihood of the treatment having
        a positive effect on the individual.
    treatment_flag : array-like matrix
        Binary indicators showing which individuals received the treatment.
    normalize : bool, optional (default=True)
        If True, normalizes the AUUC score by the best possible uplift.

    Returns
    -------
    float: The AUUC score, representing the cumulative gain from the uplift model,
           optionally normalized by the best possible uplift.

    Raises
    -------
    ValueError: If the actual outcomes are not binary when `normal=True`.
    """

    if normalize:
        try:
            check_binary(actual_outcome)
        except ValueError:
            raise ValueError("When using `normal=True`, make sure the outcome is binary.")

    # If there are multiple treatments, transform the inputs.
    actual_outcome, uplift_score, treatment_flag = \
        _multi_treatment_to_binary(actual_outcome, uplift_score, treatment_flag)

    # Generate the uplift curve using the make_uplift_curve function
    uplift_df = make_uplift_curve(actual_outcome, uplift_score, treatment_flag)

    # Extract cumulative numbers and uplift curve values from the DataFrame
    actual_x = uplift_df["count"].values
    actual_y = uplift_df["uplift_curve"].values * actual_x

    # Calculate the areas under the curves using the trapezoidal rule
    baseline_area = np.trapz(x=[0.0, actual_x[-1]], y=[0.0, actual_y[-1]])
    actual_area = np.trapz(x=actual_x, y=actual_y) - baseline_area

    if normalize:
        best_x, best_y = _extract_best_auuc_curve(uplift_df)
        best_area = np.trapz(x=best_x, y=best_y) - baseline_area
        return actual_area / best_area
    return actual_area


def _extract_best_qini_curve(uplift_stat: DataFrame, negative_effect: bool = True) -> Tuple[ndarray, ndarray]:
    """
    Extract the theoretical best Qini curve from uplift statistics.

    The best Qini curve represents the ideal scenario where all positive outcomes come from the treated
    group and none from the control group until all positive outcomes in the treatment group are exhausted.

    Parameters
    ----------
    uplift_stat : DataFrame
        A DataFrame containing cumulative statistics necessary for Qini curve
        calculations. Expected columns are:
        - "count": the number of samples
        - "cumulative_control": the cumulative count of control group samples
        - "cumulative_control_outcome": the cumulative count of positive outcomes in the control group
        - "cumulative_treatment_outcome": the cumulative count of positive outcomes in the treatment group
        - "cumulative_treatment": the cumulative count of treatment group samples
    negative_effect : bool, optional (default=True)
        If True, the curve accounts for the possibility of a negative effect of the treatment,
        which affects the shape of the best possible Qini curve.

    Returns
    -------
    Tuple[ndarray, ndarray]
        Two numpy arrays representing the x and y coordinates of the best Qini curve.
    """

    # Extract cumulative numbers and uplift curve values from the DataFrame
    num_samples = uplift_stat["count"].iloc[-1]
    num_control = uplift_stat["cumulative_control"].iloc[-1]
    num_control_pos = uplift_stat["cumulative_control_outcome"].iloc[-1]
    num_treatment_pos = uplift_stat["cumulative_treatment_outcome"].iloc[-1]
    num_treatment = uplift_stat["cumulative_treatment"].iloc[-1]

    # Calculate the endpoint of the Qini curve
    end_y = num_treatment_pos - (num_control_pos / num_control) * num_treatment
    best_x, best_y = [0.0, num_samples], [0.0, end_y]

    # Adjust the best Qini curve for negative effects if applicable
    if negative_effect:
        # Insert the point where the treatment group is exhausted
        best_x.insert(1, num_treatment_pos)
        best_y.insert(1, num_treatment_pos)
        # Insert the point where the control group"s positive outcomes are subtracted
        best_x.insert(2, num_samples - num_control_pos)
        best_y.insert(2, num_treatment_pos)
    else:
        # If no negative effect is considered, insert an intermediate point for a sharper curve
        best_x.insert(1, end_y)
        best_y.insert(1, end_y)

    # Return the coordinates of the best Qini curve as numpy arrays
    return np.array(best_x), np.array(best_y)


def qini_score(
        actual_outcome: ndarray,
        uplift_score: ndarray,
        treatment_flag: ndarray,
        normalize: bool = True,
        negative_effect=True) -> float:
    """ Calculate the Qini score for an uplift model.

    The score can be normalized by the area under the best possible Qini curve, which represents
    the maximum potential incremental benefit.

    Parameters
    ----------
    actual_outcome : array-like matrix
        Target values.
    uplift_score : array-like matrix
        Predicted uplift scores from the model, indicating the likelihood of the treatment having
        a positive effect on the individual.
    treatment_flag : array-like matrix
        Binary indicators showing which individuals received the treatment.
    normalize : bool, optional (default=True)
        If True, the calculated Qini score will be normalized by the best possible Qini score,
        providing a relative measure of model performance.
    negative_effect : bool, optional (default=True)
        If True, the model accounts for the potential negative effects of the treatment,
        which can be reflected in the Qini calculation.

    Returns
    -------
    float: The Qini score, representing the efficiency of the uplift model compared
           to a perfect model. A higher score indicates a better model.

    Raises
    -------
    ValueError: If the actual outcomes are not binary when `normal=True`.
    """

    if normalize:
        try:
            check_binary(actual_outcome)
        except ValueError:
            raise ValueError("When using `normal=True`, make sure the outcome is binary.")

    # If there are multiple treatments, transform the inputs.
    actual_outcome, uplift_score, treatment_flag = \
        _multi_treatment_to_binary(actual_outcome, uplift_score, treatment_flag)

    # Generate the uplift curve using the make_uplift_curve function
    uplift_df = make_uplift_curve(actual_outcome, uplift_score, treatment_flag)

    # Extract cumulative numbers and uplift curve values from the DataFrame
    actual_x = uplift_df["count"].values
    actual_y = uplift_df["uplift_curve"].values * uplift_df["cumulative_treatment"].values

    # Calculate the areas under the curves using the trapezoidal rule
    baseline_area = np.trapz(x=[0.0, actual_x[-1]], y=[0.0, actual_y[-1]])
    actual_area = np.trapz(x=actual_x, y=actual_y) - baseline_area

    # Calculate and return the Qini score
    if normalize:
        best_x, best_y = _extract_best_qini_curve(uplift_df, negative_effect)
        best_area = np.trapz(x=best_x, y=best_y) - baseline_area
        return actual_area / best_area
    return actual_area


def uplift_curve_by_percentile(
        actual_outcome: ndarray,
        uplift_score: ndarray,
        treatment_flag: ndarray,
        n_bins: int = 10) -> DataFrame:
    """
    Calculate the uplift curve by percentile for a given set of predictions and actual outcomes.

    This function computes the uplift curve, which is used to evaluate the performance of an uplift
    modeling approach. The curve is computed by dividing the population into bins based on percentiles
    of the uplift score, and then calculating the average treatment effect within each bin.

    Parameters
    ----------
    actual_outcome : array-like matrix
        Target values.
    uplift_score : array-like matrix
        Predicted uplift scores from the model, indicating the likelihood of the treatment having
        a positive effect on the individual.
    treatment_flag : array-like matrix
        Binary indicators showing which individuals received the treatment.
    n_bins : Union[int, float], optional (default=10)
        The number of bins to divide the population into based on percentiles of the uplift score.

    Returns
    -------
    - DataFrame: A pandas DataFrame containing the uplift curve data, including the following columns:
        - "bin": The percentile bin.
        - "score_upper": The upper threshold of the uplift score for the bin.
        - "n_control": The number of instances in the control group for the bin.
        - "n_treatment": The number of instances in the treatment group for the bin.
        - "control_outcome_mean": The mean actual outcome for the control group in the bin.
        - "treatment_outcome_mean": The mean actual outcome for the treatment group in the bin.
        - "bin_uplift": The calculated uplift (difference in mean outcomes) for the bin.
        - "cumulative_uplift": The cumulative uplift value up to the current bin.
    """

    # If there are multiple treatments, transform the inputs.
    actual_outcome, uplift_score, treatment_flag = \
        _multi_treatment_to_binary(actual_outcome, uplift_score, treatment_flag)

    # Generate the uplift curve
    df = DataFrame({
        "threshold": uplift_score.flatten(),
        "actual_outcome": actual_outcome.flatten(),
        "treatment_flag": treatment_flag.flatten()
    }, dtype=np.float32)

    qs = [1.0 / n_bins * i for i in range(1, n_bins)]
    _, thresholds = pd.qcut(df["threshold"], q=qs, retbins=True)
    df["bin"] = pd.cut(
        df["threshold"],
        bins=[float("-inf")] + list(thresholds) + [float("inf")],
        labels=[str(n_bins - i) for i in range(n_bins)]
    )

    # Create control and treatment groups
    df["control_outcome"] = df["actual_outcome"] * (1 - df["treatment_flag"])
    df["treatment_outcome"] = df["actual_outcome"] * df["treatment_flag"]

    # Aggregate data by bin
    pivoted = df.groupby(by="bin").agg(
        {
            "threshold": ["max"],
            "control_outcome": ["sum"],
            "treatment_outcome": ["sum"],
            "treatment_flag": ["sum"],
            "actual_outcome": ["count"]
        }
    ).reset_index().sort_values(by="bin", ascending=False, ignore_index=True)

    pivoted.iloc[0, 1] = np.inf  # Adjust the upper score for the first bin
    pivoted.columns = ["bin", "score_upper", "_control_outcome_sum", "_treatment_outcome_sum", "n_treatment", "total"]

    # Calculate cumulative sums for both groups
    pivoted["n_control"] = pivoted["total"] - pivoted["n_treatment"]
    pivoted[[
        "cumulative_treatment", "cumulative_treatment_outcome", "cumulative_control", "cumulative_control_outcome"
    ]] = pivoted[["n_treatment", "_treatment_outcome_sum", "n_control", "_control_outcome_sum"]].cumsum().values

    pivoted["control_outcome_mean"] = (pivoted["_control_outcome_sum"] /
                                       pivoted["n_control"]).replace([np.inf, -np.inf, np.nan], 0)
    pivoted["treatment_outcome_mean"] = (pivoted["_treatment_outcome_sum"] /
                                         pivoted["n_treatment"]).replace([np.inf, -np.inf, np.nan], 0)
    pivoted["cumulative_control_outcome_mean"] = (pivoted["cumulative_control_outcome"] /
                                                  pivoted["cumulative_control"]).replace([np.inf, -np.inf, np.nan], 0)
    pivoted["cumulative_treatment_outcome_mean"] = (pivoted["cumulative_treatment_outcome"] /
                                                    pivoted["cumulative_treatment"]).replace([np.inf, -np.inf, np.nan],
                                                                                             0)
    # Calculate uplift values for each bin and cumulatively
    pivoted["uplift"] = pivoted["treatment_outcome_mean"] - pivoted["control_outcome_mean"]
    pivoted["cumulative_uplift"] = pivoted["cumulative_treatment_outcome_mean"] - \
                                   pivoted["cumulative_control_outcome_mean"]
    # Return relevant columns for the uplift curve
    return pivoted.loc[:, ["bin", "score_upper", "n_control", "n_treatment", "control_outcome_mean",
                           "treatment_outcome_mean", "uplift", "cumulative_uplift"]]


def cumulative_uplift_top_k(
        actual_outcome: ndarray,
        uplift_score: ndarray,
        treatment_flag: ndarray,
        k: Union[int, float] = 0.1) -> float:
    """
    Calculate the cumulative uplift at the top k percent or top k instances of the ranked population.

    Parameters
    ----------
    actual_outcome : array-like matrix
        Target values.
    uplift_score : array-like matrix
        Predicted uplift scores from the model, indicating the likelihood of the treatment having
        a positive effect on the individual.
    treatment_flag : array-like matrix
        Binary indicators showing which individuals received the treatment.
    k : Union[int, float], optional (default=10)
        The cutoff for calculating cumulative uplift.
        If k is a float less than 1, it represents the top percentage of the population.
        If k is an integer or a float greater than or equal to 1, it represents the top k instances.

    Returns
    -------
    float : The cumulative uplift value at the top k percent or top k instances of the ranked population.
    """

    # If there are multiple treatments, transform the inputs.
    actual_outcome, uplift_score, treatment_flag = \
        _multi_treatment_to_binary(actual_outcome, uplift_score, treatment_flag)

    n = 0
    if k > 1.0:
        n = int(k)
    elif k > 0:
        n = int(actual_outcome.shape[0] * k)

    if n < 1:
        raise ValueError("Target index must be a positive value.")

    uplift_df = make_uplift_curve(actual_outcome, uplift_score, treatment_flag, drop_duplicates=False)

    return uplift_df["uplift_curve"].values[n]


def treatment_balance_score(X: ndarray, ti: ndarray, n_bins: int = 10) -> Tuple[Union[float, ndarray], DataFrame]:
    """
    Calculates the treatment balance score for the given features and treatment indicators.

    Parameters
    ----------
    X : ndarray
        The feature dataset which can be one or two-dimensional.
    ti : ndarray
        The treatment indicator array, representing the treatment group for each observation.
    n_bins : int, optional
        The number of bins to divide the feature data into, defaults to 10.

    Returns
    -------
    ndarray
        An array of balance scores for each feature.
    DataFrame
        A DataFrame containing feature indices, bin numbers, and proportion for each treatment group.

    Raises
    -------
    ValueError
        If the number of different treatment groups is less than 2.
    """

    check_consistent_length(X, ti)

    uniques, counts = np.unique(ti, return_counts=True)
    if len(uniques) < 2:
        raise ValueError("Number of different treatment groups for input variables less than 2.")

    x = X.reshape((-1, 1)) if len(X.shape) == 1 else np.copy(X)
    ti = ti.flatten()

    block = round(x.shape[0] / n_bins)

    # Calculate the expected treatment distribution ratio
    expect = counts / np.sum(counts)

    buffer = []
    out_df = DataFrame(
        itertools.product([i for i in range(x.shape[1])], [i for i in range(n_bins)]),
        columns=["fidx", "bin"]
    )

    # Iterate over each feature
    for col in range(x.shape[1]):
        sorted_idx = np.argsort(x[:, col])
        sorted_treatment = ti[sorted_idx]
        begin = 0
        for i in range(1, n_bins + 1):
            end = max(i * block, ti.shape[0]) if i == n_bins else i * block
            _, actual_counts = np.unique(np.r_[uniques, sorted_treatment[begin: end]], return_counts=True)
            buffer.append(np.clip((actual_counts - 1.0) / (end - begin), 1e-5, 1))
            begin += block

    group_cols = ["group:" + str(i) for i in uniques]
    out_df[group_cols] = np.array(buffer)

    # Calculate the balance score (PSI) for each bin
    out_df["psi"] = np.sum((expect - out_df[group_cols].values) * np.log(expect / out_df[group_cols].values), axis=1)

    # Aggregate the PSI for each feature
    fpsi = out_df.groupby(by="fidx").agg({"psi": sum}).reset_index().sort_values(
        by="fidx", ascending=True, ignore_index=True).values[:, 1]

    return fpsi[0] if len(fpsi) == 1 else fpsi, out_df[["fidx", "bin"] + group_cols]

