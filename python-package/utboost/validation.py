# -*- coding:utf-8 -*-
import os
import re
import numpy as np

from typing import Any, Tuple


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


def check_numpy_nd_array(data: Any, nd: int):
    """ Check if obj is ndarray with expected dimension. """
    if not isinstance(data, np.ndarray):
        raise TypeError("The input data is not an numpy.ndarray object, "
                         "it is of type {}.".format(type(data).__name__))

    if len(data.shape) != nd:
        raise ValueError("The dimension of the input data ({:d}) "
                         "is not equal to the expected dimension ({:d}).".format(len(data.shape), nd))


def check_libsvm_file(file_path: str) -> Tuple[int, int]:
    """ Check if the file matches the libsvm format. """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")

    with open(file_path, 'r') as file:
        first_line = file.readline().strip()

    patterns = (
        # <index1>:<value1> <index2>:<value2> ... <indexN>:<valueN>
        (re.compile(r'^((\d+:\S+\s*)+)$'), (-1, -1)),
        # <label> <index1>:<value1> <index2>:<value2> ... <indexN>:<valueN>
        (re.compile(r'^\d+\s+((\d+:\S+\s*)+)$'), (0, -1)),
        # <label> <treatment> <index1>:<value1> <index2>:<value2> ... <indexN>:<valueN>
        (re.compile(r'^\d+\s+\d+\s+((\d+:\S+\s*)+)$'), (0, 1)),
    )

    for pattern, ret in patterns:
        if pattern.match(first_line):
            return ret

    raise IOError(f"The file '{file_path}' does not match the libsvm format.")
