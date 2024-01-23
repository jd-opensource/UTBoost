# -*- coding:utf-8 -*-
import ctypes
import warnings
import json
import re
import scipy.sparse as sp
import numpy as np

from numpy import ndarray
from copy import deepcopy
from typing import Any, Dict, List, Tuple, Union, Optional

from .libpath import find_lib_path
from .validation import check_consistent_length, check_numpy_nd_array, check_libsvm_file, check_incremental_group

_EvalRetType = Tuple[str, str, float]  # data, metric, score
_ModelHandle = ctypes.c_void_p
_DataHandle = ctypes.c_void_p
_LIB: ctypes.CDLL = ctypes.cdll.LoadLibrary(find_lib_path()[0])
_LIB.UTB_GetLastError.restype = ctypes.c_char_p


class Logger:

    @staticmethod
    def info(msg: str):
        print(msg)

    @staticmethod
    def warn(msg):
        warnings.warn(msg, stacklevel=3)


class TrainingStopException(Exception):
    pass


def _c_str(string: str) -> ctypes.c_char_p:
    """Convert a Python string to C string."""
    return ctypes.c_char_p(string.encode("utf-8"))


def _safe_call(ret: int) -> None:
    """Check the return value from C API call.

    Parameters
    ----------
    ret : int
        The return value from C API calls.
    """
    if ret != 0:
        raise RuntimeError(_LIB.UTB_GetLastError().decode("utf-8"))


class FileParser:

    def __init__(self,
                 filename: str,
                 max_fidx: int, label_idx: int, tr_idx: int = -1,
                 default_value: float = 0.0,
                 n_threads: int = -1):
        """
        Initialize file parser

        Parameters
        ----------
        filename : str
            file path
        max_fidx : int
            maximum feature index
        label_idx : int
            label index
        tr_idx : int, optional (default=-1)
            Treatment indicator index.
        default_value : float, optional (default=0.0)
            The default feature value when the key-value pair is missing.
        n_threads : int, optional (default=-1)
            Number of parallel threads to use for training. (<=0 means using all threads)
        """
        self.filename = filename
        self.label_idx = label_idx
        self.tr_idx = tr_idx
        self.max_fidx = max_fidx
        self.n_threads = n_threads
        self.default = default_value
        self.handle = None
        self.num_rows = None

    def parse(self):
        out_len = ctypes.c_int32(0)
        self.handle = ctypes.c_void_p()
        _safe_call(_LIB.UTB_ParseLibsvm(
            _c_str(self.filename),
            ctypes.c_int32(self.label_idx),
            ctypes.c_int32(self.tr_idx),
            ctypes.c_int32(self.n_threads),
            ctypes.byref(out_len),
            ctypes.byref(self.handle)
        ))

        self.num_rows = out_len.value

    def get_data(self):
        """
        Get the parsed data.

        Returns
        -------
        feature : ndarray
            Array of features.
        label : ndarray
            Array of labels.
        treatment : ndarray, optional
            Array of treatment indicators (if tr_idx > 0).
        """
        if self.handle and (self.num_rows > 0):
            feature = np.full(
                shape=self.num_rows * self.max_fidx,
                fill_value=self.default,
                dtype=np.float32
            )
            label = np.zeros(self.num_rows, dtype=np.float32)
            treatment = np.zeros(self.num_rows, dtype=np.int32)
            feature_ptr = feature.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            label_ptr = label.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            treatment_ptr = treatment.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
            _safe_call(_LIB.UTB_MoveLibsvm(
                self.handle,
                ctypes.c_int(self.max_fidx),
                feature_ptr,
                label_ptr,
                treatment_ptr,
            ))
            feature = feature.reshape((self.num_rows, self.max_fidx))
            if self.tr_idx > 0:  # with treatment
                return feature, label, treatment
            else:
                return feature, label
        return None

    def __del__(self):
        """
        Destructor to free resources.
        """
        if self.handle:
            _safe_call(_LIB.UTB_FreeParser(self.handle))


def read_libsvm(
        filename: str,
        max_fidx: int,
        with_treatment: bool = False,
        default_value: float = np.nan,
        n_threads: int = -1):
    """
    Read data from a libsvm file.

    The LIBSVM data format represents each data instance as a single line, with the following structure:
    <label> <index1>:<value1> <index2>:<value2> ... <indexN>:<valueN>
    if `with_treatment=True`, the structure following:
    <label> <treatment_idx> <index1>:<value1> <index2>:<value2> ... <indexN>:<valueN>

    Note:
        `treatment_idx` should start from 0 and increase continuously.

    Parameters
    ----------
    filename : str
        File path.
    max_fidx : int
        Maximum feature index.
    with_treatment : bool, optional (default=False)
        Whether the file contains treatment indicators.
    default_value : float, optional (default=np.nan)
        The default feature value when the key-value pair is missing.
    n_threads : int, optional (default=-1)
        Number of parallel threads to use for training. (<=0 means using all threads)

    Returns
    -------
    feature : ndarray
        Array of features.
    label : ndarray
        Array of labels.
    treatment : ndarray, optional
        Array of treatment indicators (if with_treatment=True).
    """
    parser = FileParser(
        filename=filename,
        max_fidx=max_fidx,
        label_idx=0,
        tr_idx=1 if with_treatment else -1,
        default_value=default_value,
        n_threads=n_threads
    )
    parser.parse()
    return parser.get_data()


# template to convert model
_model_py = """from math import exp

# Model meta
class UTBoostModel:
    tree_count = {n_trees:d}
    n_treatment = {n_treatments:d}
    max_feature_idx = {max_fidx:d}
    sigmoid = {sigmoid:d}
    average = {avg:d}
    trees = [{trees}]


class Result:

    def __init__(self, model):
        self.num_treatments = model.n_treatment
        self.sigmoid = model.sigmoid > 0
        self.average = model.average > 0
        self._res = [0.0] * self.num_treatments
        self._cnt = 0.0

    def add(self, other):
        for i in range(self.num_treatments):
            self._res[i] += other[i]
        self._cnt += 1.0

    def get(self):
        ret = [self._res[0]] * self.num_treatments
        for i in range(1, self.num_treatments):
            ret[i] += self._res[i]
        if self.average:
            for i in range(self.num_treatments):
                ret[i] /= self._cnt
        if self.sigmoid:
            for i in range(self.num_treatments):
                ret[i] = 1.0 / (1.0 + exp(-ret[i]))
        return ret


def get_leaf_value(node, features):
    if "left_child" in node.keys():
        feature = features[node["split_feature"]]
        if feature is None:
            if node["missing_as_zero"]:
                feature = 0.0
            elif node["default_left"]:
                return get_leaf_value(node["left_child"], features)
            else:
                return get_leaf_value(node["right_child"], features)
        if feature <= node["threshold"]:
            return get_leaf_value(node["left_child"], features)
        else:
            return get_leaf_value(node["right_child"], features)
    else:
        return node["leaf_value"]


def apply_model(float_features):
    \"\"\"
    Applies the model built by UTBoost.
    
    Parameters
    ----------
    float_features : lift of float
        Input feature vector.
        Note, the length of the feature vector must be equal to the number of features in the training dataset ({n_features}), 
        and missing features are placed by `None`.
    Returns
    -------
    prediction : lift of float
        The predicted values.
    \"\"\"
    model = UTBoostModel()
    res = Result(model)
    for tree in model.trees:
        res.add(get_leaf_value(tree, float_features))
    return res.get()
"""


_C_TYPE_FLOAT32 = 0
_C_TYPE_FLOAT64 = 1
_C_TYPE_INT32 = 2
_C_TYPE_INT64 = 3


def _c_array_from_np1d(np1d, expect_type="float"):
    """ Convert a numpy 1d-array to c array. """
    check_numpy_nd_array(np1d, 1)
    assert expect_type in ("float", "int")

    if expect_type == "float" and np1d.dtype not in (np.float32, np.float64):
        np1d = np.array(np1d, copy=True, dtype=np.float32)

    if expect_type == "int" and np1d.dtype not in (np.int32, np.int64):
        np1d = np.array(np1d, copy=True, dtype=np.int32)

    if np1d.dtype == np.float32:
        data_ptr = np1d.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        data_type = _C_TYPE_FLOAT32
    elif np1d.dtype == np.float64:
        data_ptr = np1d.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        data_type = _C_TYPE_FLOAT64
    elif np1d.dtype == np.int32:
        data_ptr = np1d.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        data_type = _C_TYPE_INT32
    elif np1d.dtype == np.int64:
        data_ptr = np1d.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
        data_type = _C_TYPE_INT64
    else:
        raise ValueError
    return (data_ptr, data_type, np1d)


def _extract_file_pattern(path) -> Tuple[str, str]:
    """ Compile a regular expression pattern that captures the file type and file path
        from a string formatted as 'file_type://file_path'.
    """
    pattern = re.compile(r'^(.*?):\/\/(.*?)$')
    match = pattern.match(path)
    if match:
        return match.group(1).lower(), match.group(2)
    else:
        # If the pattern does not match, raise a ValueError indicating the format is incorrect.
        raise ValueError(f"The string '{path}' does not match the required format: 'file_type://file_path'.")


class Dataset:
    """
    Basic dataset class for managing and storing data in a format suitable for model training.

    This class is designed to hold data in a structured form, including features, treatment indicators,
    and target variables. It can be used to store either training or validation datasets, with the
    ability to reference another dataset. Using this container while the training data is fixed avoids
    the preprocessing of the raw data each time the model is trained, thereby optimizing the data handling
    process.
    """

    _supported_type = (np.ndarray, sp.csr_matrix, str)

    def __init__(
            self,
            data: Union[np.ndarray, str, sp.spmatrix],
            treatment: Optional[np.ndarray] = None,
            label: Optional[np.ndarray] = None,
            reference: Optional["Dataset"] = None,
    ):
        """
        Initializes a Dataset object.

        It is important to note that not specifying `treatment` and `label` may result in the dataset
        not being suitable for training purposes. If this instance is meant to be a validation dataset,
        a training dataset should be used as a reference.

        Parameters
        ----------
        data : array-like, str, or csr_matrix
            The feature matrix with shape (n_samples, n_features). If a string is provided, it is assumed
            to be the path to the data file. If a sparse matrix is provided, it should be in CSR format.
        treatment : array-like, optional
            A 1D array of treatment indicators with shape (n_samples,). Represents whether each sample
            is part of the treatment group or not.
        label : array-like, optional
            A 1D array of target variables with shape (n_samples,). These are the dependent variables
            that the model aims to predict.
        reference : Dataset, optional
            A reference to another Dataset object. This can be used to ensure consistency between training
            and validation datasets.

        Raises
        ------
        TypeError
            If the type of `data` is not one of the supported types.
        """
        if not isinstance(data, self._supported_type):
            supported_types = ", ".join([stp.__name__ for stp in self._supported_type])
            raise TypeError(f"Wrong type({type(data).__name__}) for data. Supported types: [{supported_types}].")

        self.data = data
        self.meta = dict()

        for name, value in [("label", label), ("treatment", treatment)]:
            if value is not None:
                check_numpy_nd_array(value, 1)
                if name == "treatment":
                    check_incremental_group(treatment)
                if hasattr(self.data, "shape"):
                    check_consistent_length(self.data, value)
                self.meta[name] = value

        self.reference = reference
        self.handle = None
        self.parameter_str = ""
        self.param_names = ["seed", "min_data_in_bin", "bin_construct_sample_cnt", "max_bin"]

    def __del__(self) -> None:
        self._free_raw()
        self._free_handle()

    def _free_handle(self) -> None:
        if self.handle is not None:
            _safe_call(_LIB.UTB_DatasetFree(self.handle))
            self.handle = None

    def _free_raw(self):
        self.data = None
        self.meta = None

    def _parameter_to_str(self, param: Dict[str, Union[str, float, int]]) -> str:
        """ Converts the parameter dictionary to a string representation. """
        kv_list = []
        for name in self.param_names:
            if name in param.keys():
                kv_list.append("{name}={value}".format(name=name, value=param[name]))
        return "\t".join(kv_list)

    def _set_field(self):
        """ Sets the fields of the Dataset object. """
        for name, value in self.meta.items():
            if name == "treatment":
                data = np.array(value.reshape(value.size), dtype=np.int32)
                data_ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
            else:
                data = np.array(value.reshape(value.size), dtype=np.float32)
                data_ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            _safe_call(_LIB.UTB_DatasetSetMeta(
                self.handle,
                _c_str(name),
                data_ptr,
                ctypes.c_int32(value.shape[0]),
                _c_str(self.parameter_str))
            )

    def save_binmapper(self, filename: str):
        """ Saves the bin mapper info to a file. """
        if self.handle is not None:
            _safe_call(_LIB.UTB_DatasetDumpMapper(
                self.handle,
                _c_str(filename))
            )

    def _create_from_numpy(self, ref_handle):
        """ Create dataset from a numpy.ndarray. """
        check_numpy_nd_array(self.data, 2)
        if self.data.dtype in (np.float32, np.float64):
            data = np.array(self.data.reshape(self.data.size), dtype=self.data.dtype, copy=False)
        else:
            data = np.array(self.data.reshape(self.data.size), dtype=np.float32, copy=True)

        data_ptr, data_type, _ = _c_array_from_np1d(data, expect_type="float")

        _safe_call(_LIB.UTB_CreateDataset(
            data_ptr,
            ctypes.c_int32(self.data.shape[0]),
            ctypes.c_int32(self.data.shape[1]),
            ctypes.c_int32(data_type),
            ref_handle,
            ctypes.byref(self.handle),
            _c_str(self.parameter_str))
        )

    def _create_from_csr(self, ref_handle):
        """ Create dataset from a csr matrix. """
        indptr_data, indptr_type, _ = _c_array_from_np1d(self.data.indptr, expect_type="int")
        ptr_data, ptr_type, __ = _c_array_from_np1d(self.data.data, expect_type="float")
        csr_indices = self.data.indices.astype(np.int32, copy=False)

        _safe_call(_LIB.UTB_CreateDatasetFromCSR(
            indptr_data,
            ctypes.c_int32(indptr_type),
            csr_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ptr_data,
            ctypes.c_int32(ptr_type),
            ctypes.c_int64(len(self.data.indptr)),
            ctypes.c_int32(self.data.shape[1]),
            ref_handle,
            ctypes.byref(self.handle),
            _c_str(self.parameter_str))
        )

    def _create_from_file(self, ref_handle):
        """ Create dataset from a file. """
        file_type, path = _extract_file_pattern(self.data)
        if file_type == "libsvm":
            label_idx, treatment_idx = check_libsvm_file(path)
            _safe_call(_LIB.UTB_DatasetCreateFromLibsvm(
                _c_str(path),
                ctypes.c_int32(label_idx),
                ctypes.c_int32(treatment_idx),
                _c_str(self.parameter_str),
                ref_handle,
                ctypes.byref(self.handle))
            )
        else:
            raise TypeError(f"Unknown type of file: {file_type}")

    def build(self, params: Dict[str, Union[str, float, int]]) -> "Dataset":
        """ Builds the dataset with the given parameters. """
        if self.handle is not None:
            if self.parameter_str == self._parameter_to_str(params):
                return self
            else:
                if self.data is None:
                    raise RuntimeError("The raw data has been released, please re-instantiate the object.")
                self._free_handle()

        self.parameter_str = self._parameter_to_str(params)
        self.handle = ctypes.c_void_p()
        ref_handle = self.reference.handle if self.reference else None

        if isinstance(self.data, np.ndarray):
            self._create_from_numpy(ref_handle)
        elif isinstance(self.data, sp.csr_matrix):
            self._create_from_csr(ref_handle)
        elif isinstance(self.data, str):
            self._create_from_file(ref_handle)
        else:
            raise TypeError("Unsupported feature data type: {}".format(type(self.data).__name__))

        self._set_field()
        self._free_raw()
        return self


class _ModelBase:
    """ Basic model class """

    def __init__(
            self,
            train_dataset: Optional[Dataset] = None,
            params: Optional[Dict[str, Any]] = None,
            model_file: Optional[str] = None,
    ):
        """ Initializes a ModelBase object with a train dataset, parameters, and model file. """
        self.handle = None
        self.params = dict() if params is None else deepcopy(params)
        self.params_str = self._param_to_str(self.params)
        self.train_dataset = train_dataset
        self.valid_sets: List[Dataset] = []
        self.iters = ctypes.c_int(0)
        if train_dataset is not None:
            self.handle = ctypes.c_void_p()
            train_dataset.build(self.params)
            _safe_call(_LIB.UTB_CreateBooster(
                train_dataset.handle,
                _c_str(self.params_str),
                ctypes.byref(self.handle)
            ))
        elif model_file is not None:
            self.handle = ctypes.c_void_p()
            self.load_from_file(model_file)
        else:
            raise ValueError("Since both train_dataset and model_file are None types, the model cannot be created.")

    def __del__(self) -> None:
        """ Free handle """
        if self.handle is not None:
            _safe_call(_LIB.UTB_BoosterFree(self.handle))
            self.handle = None

    def add_valid_dataset(self, valid: Dataset):
        """ Adds a validation dataset to the model. """
        if self.train_dataset is None:
            raise Exception("A valid dataset cannot be added to the model while the training dataset is of None type.")

        if valid.reference is not self.train_dataset:
            valid = deepcopy(valid)
            valid.reference = self.train_dataset
            valid.handle = None

        valid.build(self.params)

        _safe_call(_LIB.UTB_BoosterAddValidData(
            self.handle,
            valid.handle))
        self.valid_sets.append(valid)

    def _param_to_str(self, params: Optional[Dict[str, Any]]) -> str:
        """ Converts the parameter dictionary to a string representation. """
        if params is None:
            return ""
        buffer = []
        for key, value in params.items():
            if isinstance(value, list):
                buffer.append("{name}={value}".format(name=key, value=",".join(value)))
            else:
                buffer.append("{name}={value}".format(name=key, value=value))
        return "\t".join(buffer)

    def train_one_iter(self) -> bool:
        """ Trains the model for one iteration. """
        if self.train_dataset is None:
            raise Exception("The model cannot be trained if the training dataset is of None type.")
        is_finished = ctypes.c_int(0)
        _safe_call(_LIB.UTB_BoosterUpdateOneIter(
            self.handle,
            ctypes.byref(is_finished)
        ))
        self.iters.value += 1
        return is_finished.value == 1

    def eval(self) -> List[_EvalRetType]:
        """ Evaluates the model. """
        if "metric" not in self.params.keys():
            return []
        buffer = []
        n_metric = len(self.params["metric"])
        if self.train_dataset is not None:
            result = np.empty(n_metric, dtype=np.float64)
            out_len = ctypes.c_int(0)
            _safe_call(_LIB.UTB_BoosterGetEval(
                self.handle,
                ctypes.c_int(0),
                ctypes.byref(out_len),
                result.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            ))
            if out_len.value != n_metric:
                raise ValueError("The evaluation results do not match the metric size.")
            for i in range(n_metric):
                buffer.append(("train", self.params["metric"][i], result[i]))

        for idx in range(len(self.valid_sets)):
            result = np.empty(n_metric, dtype=np.float64)
            out_len = ctypes.c_int(0)
            _safe_call(_LIB.UTB_BoosterGetEval(
                self.handle,
                ctypes.c_int(idx + 1),
                ctypes.byref(out_len),
                result.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            ))
            if out_len.value != n_metric:
                raise ValueError("The evaluation results do not match the metric size.")
            for i in range(n_metric):
                buffer.append(("valid-{:d}".format(idx), self.params["metric"][i], result[i]))

        return buffer

    def rollback(self, iters: int) -> bool:
        """ Rolls back the model to a previous iteration. """
        if iters > self.iters.value:
            raise ValueError("Rollback iters {:d} exceed the number of trees {:d}.".format(iters, self.iters.value))
        for _ in range(self.iters.value - iters):
            _safe_call(_LIB.UTB_BoosterRollbackOneIter(self.handle))
            self.iters.value -= 1
        return True

    @property
    def num_feature(self) -> int:
        """ Returns the number of features in the model. """
        num_feature = ctypes.c_int(0)
        _safe_call(_LIB.UTB_BoosterGetNumFeature(
            self.handle,
            ctypes.byref(num_feature)
        ))
        return num_feature.value

    @property
    def num_treatment(self) -> int:
        """ Returns the number of treatments in the model. """
        num_treatment = ctypes.c_int(0)
        _safe_call(_LIB.UTB_BoosterGetNumTreatment(
            self.handle,
            ctypes.byref(num_treatment)
        ))
        return num_treatment.value

    def save(self,
             filename: str,
             start_iteration: int = 0,
             num_iteration: Optional[int] = None,
             format="utm",
             importance_type: str = "split"):
        """ Saves the model to a file in the specified format. """
        num_iteration = -1 if num_iteration is None else num_iteration
        imp_type = 0 if importance_type == "split" else 1
        if format == "utm":
            _safe_call(_LIB.UTB_BoosterSaveModel(
                self.handle,
                ctypes.c_int(start_iteration),
                ctypes.c_int(num_iteration),
                ctypes.c_int(imp_type),
                _c_str(str(filename))
            ))
        elif format == "json":
            _safe_call(_LIB.UTB_BoosterDumpModelToFile(
                self.handle,
                ctypes.c_int(start_iteration),
                ctypes.c_int(num_iteration),
                _c_str(str(filename))
            ))
        elif format == "py":
            model = self.to_json(start_iteration, num_iteration)
            # using code template
            model_str = _model_py.format(
                n_trees=len(model["tree_info"]),
                n_treatments=model["num_treat"],
                max_fidx=model["max_feature_idx"],
                n_features=model["max_feature_idx"] + 1,
                sigmoid=1 if model["objective"] == "logloss" else 0,
                avg=int(model["average_output"]),
                trees=",\n".join([str(tree["tree_structure"]) for tree in model["tree_info"]])
            )
            with open(filename, "w", encoding="utf-8") as f:
                f.write(model_str)
        else:
            raise ValueError("format {} not in [\"utm\", \"json\", \"py\"]".format(format))

    def to_json(self,
                start_iteration: int = 0,
                num_iteration: Optional[int] = None
                ) -> Dict[str, Any]:
        """ Converts the model to a JSON representation. """
        num_iteration = -1 if num_iteration is None else num_iteration
        buffer_len = 1 << 20
        tmp_out_len = ctypes.c_int64(0)
        string_buffer = ctypes.create_string_buffer(buffer_len)
        ptr_string_buffer = ctypes.c_char_p(*[ctypes.addressof(string_buffer)])
        _safe_call(_LIB.UTB_BoosterDumpModel(
            self.handle,
            ctypes.c_int(start_iteration),
            ctypes.c_int(num_iteration),
            ctypes.c_int64(buffer_len),
            ctypes.byref(tmp_out_len),
            ptr_string_buffer))
        actual_len = tmp_out_len.value
        # if buffer length is not long enough, reallocate a buffer
        if actual_len > buffer_len:
            string_buffer = ctypes.create_string_buffer(actual_len)
            ptr_string_buffer = ctypes.c_char_p(*[ctypes.addressof(string_buffer)])
            _safe_call(_LIB.UTB_BoosterDumpModel(
                self.handle,
                ctypes.c_int(start_iteration),
                ctypes.c_int(num_iteration),
                ctypes.c_int64(actual_len),
                ctypes.byref(tmp_out_len),
                ptr_string_buffer))
        return json.loads(string_buffer.value.decode("utf-8"))

    def load_from_file(self, filename: str):
        """ Loads the model from a file. """
        if self.handle is not None:
            _safe_call(_LIB.UTB_BoosterFree(self.handle))

        self.handle = ctypes.c_void_p()
        self.iters = ctypes.c_int(0)
        out_num_treat = ctypes.c_int(0)
        _safe_call(_LIB.UTB_BoosterCreateFromModelfile(
            _c_str(filename),
            ctypes.byref(self.iters),
            ctypes.byref(out_num_treat),
            ctypes.byref(self.handle)
        ))

    def predict(self, data: np.ndarray, start_iteration: int = 0, num_iteration: Optional[int] = None) -> np.ndarray:
        """ Makes predictions using the model. """
        check_numpy_nd_array(data=data, nd=2)
        assert data.shape[1] == self.num_feature

        num_iteration = -1 if (num_iteration is None) or (num_iteration > self.iters.value) else num_iteration
        n_preds = self.num_treatment * data.shape[0]
        out_num_preds = ctypes.c_int64(0)

        if data.dtype in (np.float32, np.float64):
            mat = np.array(data.reshape(data.size), dtype=data.dtype, copy=False)
        else:
            mat = np.array(data.reshape(data.size), dtype=np.float32, copy=True)

        mat_ptr, mat_type, _ = _c_array_from_np1d(mat, expect_type="float")

        out_preds = np.empty(n_preds, dtype=np.float64)
        _safe_call(_LIB.UTB_BoosterPredictForMat(
            self.handle,
            mat_ptr,
            ctypes.c_int32(data.shape[0]),
            ctypes.c_int32(data.shape[1]),
            ctypes.c_int32(mat_type),
            ctypes.c_int32(start_iteration),
            ctypes.c_int32(num_iteration),
            _c_str(self.params_str),
            ctypes.byref(out_num_preds),
            out_preds.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        ))
        if n_preds != out_num_preds.value:
            raise ValueError("The size of the prediction does not match, "
                             "from {:d} to {:d}.".format(n_preds, out_num_preds.value))
        return out_preds.reshape((data.shape[0], self.num_treatment))

    def feature_importance(self,
                           importance_type: str = "split",
                           iteration: Optional[int] = None
                           ) -> np.ndarray:
        """ Computes the feature importance of the model. """
        if iteration is None:
            iteration = self.iters.value
        imp_type = 0 if importance_type == "split" else 1
        result = np.empty(self.num_feature, dtype=np.float64)
        _safe_call(_LIB.UTB_BoosterFeatureImportance(
            self.handle,
            ctypes.c_int(iteration),
            ctypes.c_int(imp_type),
            result.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        ))
        if importance_type == "split":
            return result.astype(np.int32)
        else:
            return result
