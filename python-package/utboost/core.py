# -*- coding:utf-8 -*-
import numpy as np

from typing import Dict, Any, List, Optional, Union, Tuple

from .basic import Dataset, _ModelBase, Logger, TrainingStopException
from .callback import AbstractCallback, EvalCallback, EarlyStopCallback, _EvalLogsType
from .validation import check_binary, check_consistent_length, check_incremental_group


def _auto_objective(obj: str, csplit: str) -> str:
    if csplit in ("ed", "kl", "chi", "ddp"):
        if obj != "default":
            Logger.warn("The criterion {} is only compatible with the default objective. "
                        "{} does not work.".format(csplit, obj))
        return "default"
    elif csplit == "gbm":
        if obj not in ("logloss", "mse"):
            raise ValueError("The criterion gbm is compatible with both logloss and mse. "
                             "However, {} is provided.".format(obj))
        return obj
    else:
        raise ValueError("Unknown split criterion {}".format(csplit))


class UTBoostModel:

    def __init__(
            self,
            ensemble_type: str = "boosting",
            criterion: str = "gbm",
            num_leaves: int = 31,
            max_depth: int = 5,
            learning_rate: float = 0.1,
            iterations: int = 100,
            subsample: float = 1.,
            subsample_freq: int = 0,
            colsample: float = 1.,
            use_honesty: bool = False,
            init_from_average: bool = False,
            min_data_leaf: int = 20,
            max_bin: int = 256,
            gbm_gain_type: str = "global",
            scale_treat_weights: Optional[List[float]] = None,
            effect_constrains: Optional[List[int]] = None,
            auto_balance: bool = False,
            subsample_for_binmapper: int = 200000,
            use_best_model: bool = False,
            eval_metric: Optional[List[str]] = None,
            early_stopping_rounds: int = -1,
            eval_period: int = 1,
            seed: int = 618,
            n_threads: int = -1,
            **kwargs
    ):
        """
        Initialize model

        Parameters
        ----------
        ensemble_type : str, optional (default="boosting")
            Ensemble method of trees. Choose from one of the types: "boosting", "bagging".
        criterion : str, optional (default="gbm")
            Evaluation criterion to find optimal split point.
            "gbm", Gradient Boosting Decision Tree.
            "ddp", The difference of uplift between two leaves.
            "ed", Euclidean Distance.
            "kl", KL Divergence.
            "chi", Chi-Square statistic.
        num_leaves : int, optional (default=31)
            The maximum number of leaves.
        max_depth : int, optional (default=5)
            The maximum depth of the tree.
        learning_rate : float, optional (default=0.1)
            Boosting learning rate.
        iterations : int, optional (default=100)
            The maximum number of trees.
        subsample : float, optional (default=1.0)
            Subsample ratio of the training instance.
        subsample_freq : int, optional (default=0)
            Frequency of subsample, <=0 means no enable.
        colsample : float, optional (default=1.0)
            Subsample ratio of columns when constructing each tree.
        use_honesty : bool, optional (default=False)
            Whether to use different data to generate the tree and estimate the uplift value inside the leaves.
        init_from_average : bool, optional (default=False)
            Whether to initialize approximate values by best constant value for the specified criterion.
        min_data_leaf : int, optional (default=20)
            The minimum number of samples required to be split at a leaf node.
        max_bin : int, optional (default=255)
            The maximum number of bins per feature.
        gbm_gain_type : str, optional (default="global")
            How gain is computed in the gbm algorithm, Choose one from:
            "global", Gain is computed from the full instance.
            "local", Gain is computed from the treated instance.
            "tau", Gain is computed from the ite-related parts.
        scale_treat_weights : list of float, optional (default=None)
            Weights associated with control and each treated group. Works only when criterion="gbm".
            If None, all instances are supposed to have weight one.
        effect_constrains : list of int, optional (default=None)
            Impose monotonic constraints on causal effects.
            Possible values:
            `1` — Increasing constraint on the causal effect. The algorithm forces the model to be a non-decreasing function of this causal effect.
            `-1` — Decreasing constraint on the causal effect. The algorithm forces the model to be a non-increasing function of this causal effect.
            `0` — Constraints are disabled.
            Note, the length of effect_constrains must be equal to the number of treated group, if effect_constrains is not `None`.
        auto_balance : bool, optional (default=False)
            Whether to automatically balance the weights of classes.
            Use this parameter only for the case of gbm splitting criterion and binary classification.
            If True, the model will try to automatically balance the weight of the dominated label with pos/neg or
            neg/pos fraction in train set. Note, it will result in poor estimates of the individual class probabilities.
            If False, all classes are supposed to have weight one.
        subsample_for_binmapper : int, optional (default=200000)
            Number of samples for constructing bins.
        use_best_model : bool, optional (default=False)
            If this parameter is True, the optimal number of trees is maintained based on the metric value.
        eval_metric : list of strings, optional (default=None)
            Metrics used for early stop (if enabled) and best model selection (if enabled).
            It should be a built-in evaluation metric as follows:
            "logloss", Applicable to binary outcome.
            "auc", Applicable to binary outcome.
            "rmse", Applicable to continuous outcome.
            "qini_coff", Normalized qini area, applicable to binary outcome.
            "qini_area", Applicable to continuous & binary outcome.
        early_stopping_rounds : int, optional (default=-1)
            Validation metric needs to improve at least once in every **early_stopping_rounds** round(s) to continue training.
            This parameter only takes effect when the value is greater than 0 and at least one metric is provided.
            If additional validation sets is provided, evaluation is based on the first one, otherwise it is based on the training set
        eval_period : int, optional (default=1)
            The frequency of iterations to calculate the values of metrics.
            Note, if eval_period is less than 1, early-stop will also not work.
        seed : int, optional (default=618)
            Random number seed.
        n_threads : int, optional (default=-1)
            Number of parallel threads to use for training. (<=0 means using all threads)
        **kwargs
            Other parameters for the model.
        """
        self.ensemble_type = ensemble_type
        self.criterion = criterion
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.subsample = subsample
        self.subsample_freq = subsample_freq
        self.colsample = colsample
        self.use_honesty = use_honesty
        self.init_from_average = init_from_average
        self.max_bin = max_bin
        self.min_data_leaf = min_data_leaf
        self.subsample_for_binmapper = subsample_for_binmapper
        self.use_best_model = use_best_model
        self.eval_metric = eval_metric
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_period = eval_period
        self.seed = seed
        self.n_threads = n_threads
        self.scale_treat_weights = scale_treat_weights
        self.effect_constrains = effect_constrains
        self.auto_balance = auto_balance
        self.gbm_gain_type = gbm_gain_type
        self._eval_logs: _EvalLogsType = {}
        self._model: Optional[_ModelBase] = None
        self._is_fitted = False
        self._params: Dict[str, Any] = {}
        self._feature_names = None
        self.set_params(**kwargs)
        self._process_params()

    def set_params(self, **params: Any):
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params
            Parameter names with their new values.

        Returns
        -------
        self : object
            Returns self.
        """
        for key, value in params.items():
            setattr(self, key, value)

    def _process_params(self):
        # ensemble
        supported = ("boosting", "bagging")
        if self.ensemble_type in supported:
            self._params["ensemble"] = "boost" if self.ensemble_type == "boosting" else "rf"
        else:
            raise ValueError(
                "The ensemble_type compatible with {}, however, {} is provided.".format(supported, self.ensemble_type))

        # criterion
        self._params["split_criteria"] = self.criterion

        # max_depth, num_leaves
        if self.max_depth > 0:
            max_num_leaves = pow(2, self.max_depth)
            if max_num_leaves < self.num_leaves:
                self.num_leaves = max_num_leaves
        self._params["max_depth"] = self.max_depth
        self._params["num_leaves"] = self.num_leaves

        # learning_rate
        if (self.learning_rate > 0) and (self.learning_rate <= 1.0):
            self._params["learning_rate"] = self.learning_rate
        else:
            raise ValueError("The value of learning_rate should be in the range 0 to 1.")

        # subsample
        if (self.subsample > 0) and (self.subsample <= 1.0):
            self._params["bagging_fraction"] = self.subsample
            self._params["bagging_freq"] = self.subsample_freq
            if self.subsample < 1.0 and self.subsample_freq < 1:
                Logger.warn("Since subsample_freq is less than 1, subsample does not work")
        else:
            raise ValueError("The value of subsample should be in the range 0 to 1.")

        # colsample
        if (self.colsample > 0) and (self.colsample <= 1.0):
            self._params["feature_fraction"] = self.colsample
        else:
            raise ValueError("The value of colsample should be in the range 0 to 1.")

        # use_honesty
        self._params["use_honesty"] = int(self.use_honesty)
        # init_from_average
        self._params["boost_from_average"] = int(self.init_from_average)

        # max_bin
        if self.max_bin > 2:
            self._params["max_bin"] = self.max_bin
        else:
            raise ValueError("Ensure that the value of max_bin is greater than 2.")

        # min_data_leaf
        if self.min_data_leaf > 0:
            self._params["min_data_leaf"] = self.min_data_leaf
        else:
            raise ValueError("Ensure that the value of max_bin is greater than zero.")

        # subsample_for_binmapper
        if self.subsample_for_binmapper > 10000:
            self._params["bin_construct_sample_cnt"] = self.subsample_for_binmapper
        else:
            raise ValueError("The value of subsample_for_binmapper must be greater than 10000.")
        # seed
        self._params["seed"] = self.seed
        self._params["bagging_seed"] = self.seed + 1
        self._params["feature_fraction_seed"] = self.seed + 2
        # n_threads
        self._params["num_threads"] = self.n_threads

        if self.eval_metric is not None:
            self._params["metric"] = self.eval_metric

        # conflict check
        if (self.criterion in ("ed", "kl", "chi")) and (self.ensemble_type != "bagging"):
            raise ValueError("The criterion ({}) is unavailable when the ensemble_type "
                             "is set to boosting.".format(self.criterion))

        if self.ensemble_type == "bagging":
            if not ((self.subsample_freq > 0 and 0.0 < self.subsample < 1.0) or (0.0 < self.colsample < 1.0)):
                raise ValueError("When using ensemble_type=bagging, ensure that "
                                 "either subsample or colsample (or both) are set to a value between 0 and 1.")

        if self.use_honesty and self.subsample > 0.9 and self.subsample_freq < 1:
            raise ValueError("To set use_honesty=True, "
                             "the subsample must be less than 0.9 and the subsample_freq must be greater than 0.")

        if self.scale_treat_weights is not None:
            for weight in self.scale_treat_weights:
                if weight < 0.0:
                    raise ValueError("The value of scale_treat_weights must be greater than 0.")
            self._params["scale_treat_weights"] = ",".join([str(i) for i in self.scale_treat_weights])
            if self.criterion != "gbm":
                Logger.warn("scale_treat_weights is not available when criterion != gbm")

        if self.effect_constrains is not None:
            for weight in self.effect_constrains:
                if weight not in (-1, 0, 1):
                    raise ValueError("The effect_constrains should be limited to the values of -1, 0, or 1.")
            self._params["effect_constrains"] = ",".join([str(i) for i in self.effect_constrains])

        if self.gbm_gain_type == "global":
            self._params["gbm_gain_type"] = 0
        elif self.gbm_gain_type == "local":
            self._params["gbm_gain_type"] = 1
        elif self.gbm_gain_type == "tau":
            self._params["gbm_gain_type"] = 2
        else:
            raise ValueError("The possible values for gbm_gain_type are `global`, `local`, and `tau`.")

    def _check_data(
            self,
            X: Union[np.ndarray],
            ti: Optional[np.ndarray] = None,
            y: Optional[np.ndarray] = None
    ):
        """ Check if the data meets the training requirements. """
        if (ti is None) or (y is None):
            raise ValueError("The input treatment or label is None.")

        if X.shape[0] < 10:
            raise ValueError("Sample size less than 10.")

        check_consistent_length(X, ti, y)
        # check treatment indicator
        t_set = check_incremental_group(ti)

        if len(t_set) == 1:
            Logger.warn("The model is not capable of learning causal effects with only one treatment indicator.")

        uniques = np.unique(y)
        if len(uniques) < 2:
            raise ValueError("All labels have identical values.")

        if self.effect_constrains is not None:
            if len(self.effect_constrains) != len(t_set) - 1:
                raise ValueError("The size of the effect_constraints ({:d}) "
                                 "does not match the number of the treated group "
                                 "({:d})".format(len(self.effect_constrains), len(t_set) - 1))

    def fit(
            self,
            X: Union[np.ndarray, Dataset],
            ti: Optional[np.ndarray] = None,
            y: Optional[np.ndarray] = None,
            eval_sets: Optional[List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = None,
            feature_names: Optional[List[str]] = None
    ) -> "UTBoostModel":
        """
        Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like matrix of shape = [n_samples, n_features] or Dataset
            Input feature matrix.
        ti : array-like of shape = [n_samples], optional
            The treatment indicators for the training dataset.
        y : array-like of shape = [n_samples], optional
            The target variables for the training dataset.
        eval_sets : list or None, optional (default=None)
            A list of (X, T, y) tuple pairs to use as validation sets.
        feature_names : list of strings or None, optional (default=None)
            Feature names.
        Returns
        -------
        self : object
            Returns self.
        """
        if isinstance(X, np.ndarray):
            self._check_data(X, ti, y)
            train = Dataset(data=X, label=y, treatment=ti)
        elif isinstance(X, Dataset):
            train = X
        else:
            raise ValueError("Input X should be either ndarray or dataset.")

        evals = []
        if eval_sets is not None:
            for eval_x, eval_t, eval_y in eval_sets:
                evals.append(Dataset(data=eval_x, label=eval_y, treatment=eval_t, reference=train))

        if feature_names:
            assert len(feature_names) == X.shape[1], "Input feature names with inconsistent numbers of features."
            self._feature_names = feature_names

        cbs: List[AbstractCallback] = []
        if (self.eval_metric is not None) and (self.eval_period > 0):
            cbs.append(
                EvalCallback(eval_period=self.eval_period)
            )
            if self.early_stopping_rounds > 0:
                best_data_name = "train" if len(evals) == 0 else "valid-0"
                cbs.append(
                    EarlyStopCallback(stop_round=self.early_stopping_rounds,
                                      data_name=best_data_name, metric_name=self.eval_metric[0],
                                      use_best=self.use_best_model)
                )

        self._model = _ModelBase(train_dataset=train, params=self._params)

        for eval in evals:
            self._model.add_valid_dataset(eval)

        for cb in cbs:
            cb.before_train(self._model)

        for i in range(self.iterations):
            try:
                for cb in cbs:
                    cb.before_iter(i, self._model, self._eval_logs)
            except TrainingStopException:
                break

            self._model.train_one_iter()

            try:
                for cb in cbs:
                    cb.after_iter(i, self._model, self._eval_logs)
            except TrainingStopException:
                break

        for cb in cbs:
            cb.after_train(self._model)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the X according to the fitted model.

        Parameters
        ----------
        X : array-like matrix of shape = [n_samples, n_features]
            Input feature matrix.
        Returns
        -------
        results : array-like of shape = [n_samples, n_treatments]
            The predicted values.
        """
        if self._model is not None:
            return self._model.predict(X)
        else:
            raise Exception("The model must be fitted or loaded from a file before proceeding.")

    def feature_importance(self, importance_type: str = "split") -> List[Tuple[str, Any]]:
        """
        Get features importance from the fitted trees.

        Parameters
        ----------
        importance_type : string, optional (default="split")
            If "split", result is the total times the feature is used in model.
            If "gain", result is the total gain the feature is used in model.
        Returns
        -------
        features_importance : list of tuples = [(feature_name, importance), ...]
            Importance of each feature, sorted by imp value.
        """
        if self._model is not None:
            imp = self._model.feature_importance(importance_type)
            feature_names = ["f" + str(i) for i in
                             range(imp.shape[0])] if self._feature_names is None else self._feature_names
            ret = []
            for index, name in enumerate(feature_names):
                ret.append((name, imp[index]))
            ret.sort(key=lambda x: x[1], reverse=True)
            return ret
        else:
            raise Exception("The model should be fitted first.")

    def load_model(self, model_path) -> "UTBoostModel":
        """
        Load and initialize the model from file

        Parameters
        ----------
        model_path : str
            model file path
        Returns
        -------
        self : object
            Returns self.
        """
        self._model = _ModelBase(model_file=model_path)
        return self

    def save_model(self, model_path) -> "UTBoostModel":
        """
        Save the model to file.

        Parameters
        ----------
        model_path : str
            The path to the output model.
        Returns
        -------
        self : object
            Returns self.
        """
        if self._model is not None:
            self._model.save(model_path, format="utm")
        else:
            raise Exception("The model should be fitted first.")
        return self

    def to_json(self, json_file) -> "UTBoostModel":
        """
        Dump the model to json format.

        Parameters
        ----------
        json_file : str
            The path to the output model.
        Returns
        -------
        self : object
            Returns self.
        """
        if self._model is not None:
            self._model.save(json_file, format="json")
        else:
            raise Exception("The model should be fitted first.")
        return self

    def to_python(self, python_file) -> "UTBoostModel":
        """
        Export the model to python code.

        Parameters
        ----------
        python_file : str
            The path to the output model.
        Returns
        -------
        self : object
            Returns self.
        """
        if self._model is not None:
            self._model.save(python_file, format="py")
        else:
            raise Exception("The model should be fitted first.")
        return self

    def get_logs(self) -> _EvalLogsType:
        """ Get history training logs """
        return self._eval_logs.copy()

    @property
    def n_iters(self) -> int:
        """ number of rounds """
        return self._model.iters.value


class UTBClassifier(UTBoostModel):

    def _process_params(self):
        super()._process_params()
        available = ("qini_area", "qini_coff", "logloss", "auc")
        if "metric" in self._params.keys():
            for metric in self._params["metric"]:
                if metric not in available:
                    raise ValueError(
                        "Only {} is available for the regressor. "
                        "The metric {} does not work.".format(available, metric))

        if (self.criterion in ("ed", "kl", "chi")) and (self.ensemble_type != "bagging"):
            raise ValueError("The criterion {} is unavailable when the ensemble_type "
                             "is set to bagging.".format(self.criterion))

        if self.criterion in ("ed", "kl", "chi", "ddp"):
            self._params["objective"] = "default"
        elif self.criterion == "gbm":
            self._params["objective"] = "logloss"
        else:
            raise ValueError("The criterion ({}) is not included in the following list: "
                             "\"ed\", \"kl\", \"chi\", \"ddp\", or \"gbm\".".format(self.criterion))

        self._params["auto_balance"] = int(self.auto_balance)

    def _check_data(
            self,
            X: Union[np.ndarray],
            ti: Optional[np.ndarray] = None,
            y: Optional[np.ndarray] = None
    ):
        super()._check_data(X, ti, y)
        check_binary(y)


class UTBRegressor(UTBoostModel):
    """ Used for regression task """

    def _process_params(self):
        super()._process_params()
        available = ("qini_area", "rmse", "l2")
        if "metric" in self._params.keys():
            for metric in self._params["metric"]:
                if metric not in available:
                    raise ValueError(
                        "Only {} is available for the regressor. "
                        "The metric {} does not work.".format(available, metric))

        if (self.criterion in ("ed", "kl", "chi")) and (self.ensemble_type != "bagging"):
            raise ValueError("The criterion {} is unavailable when the ensemble_type "
                             "is set to boosting.".format(self.criterion))

        if self.criterion in ("ed", "chi", "ddp"):
            self._params["objective"] = "default"
        elif self.criterion == "gbm":
            self._params["objective"] = "mse"
        else:
            raise ValueError("The criterion ({}) is not included in the following list: "
                             "\"ed\", \"chi\", \"ddp\", or \"gbm\".".format(self.criterion))
