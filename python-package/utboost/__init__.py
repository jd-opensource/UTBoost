# -*- coding:utf-8 -*-
import os

from .core import UTBClassifier, UTBRegressor
from .basic import Dataset, read_libsvm

with open(os.path.join(os.path.dirname(__file__), "VERSION")) as f:
    __version__ = f.read().strip()

__all__ = [
    "Dataset",
    "UTBClassifier",
    "UTBRegressor",
    "read_libsvm"
]
