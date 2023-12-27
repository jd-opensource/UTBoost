# -*- coding:utf-8 -*-
import numpy as np

from typing import Any


def check_binary(x: np.ndarray):
    uniques = np.unique(x).astype(np.int32)
    if not np.allclose(uniques, np.array([0, 1])):
        raise ValueError(f"Input array is not binary. "
                         f"Array should contain only int or float binary values 0 (or 0.) and 1 (or 1.). "
                         f"Got values {uniques}.")


def check_consistent_length(*arrays):
    """Check that all arrays have consistent first dimensions.

    Parameters
    ----------
    *arrays : list or tuple of arrays.
    """
    lengths = [X.shape[0] for X in arrays if X is not None]
    uniques = np.unique(lengths)
    if len(uniques) > 1:
        raise ValueError("Found input variables with inconsistent numbers of samples: %r" % [int(l) for l in lengths])


def check_incremental_group(indicator: np.ndarray) -> np.ndarray:
    """
    Check if the treatment group identifiers in the indicator array are incremental.

    Parameters
    ----------
    indicator : np.ndarray
        An array containing integer identifiers for treatment groups. The identifiers
        should start at 0 and increment by 1 for each group.

    Returns
    -------
    np.ndarray
        The unique, sorted array of group identifiers if they are correctly incremental.

    Raises
    ------
    ValueError
        If the group identifiers are not consecutive integers starting from 0, a
        ValueError is raised with an appropriate message.
    """
    uniques = np.unique(indicator.astype(np.int32))
    max_indicator = np.max(uniques)
    if not np.allclose(uniques, np.arange(max_indicator + 1)):
        raise ValueError("Integer treatment group identifiers should start at 0 and be incremented consecutively.")
    else:
        return uniques


def is_numpy_2d_array(data: Any) -> bool:
    return isinstance(data, np.ndarray) and len(data.shape) == 2 and data.shape[1] > 1


def is_numpy_1d_array(data: Any) -> bool:
    return isinstance(data, np.ndarray) and len(data.shape) == 1
