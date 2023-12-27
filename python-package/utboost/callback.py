# -*- coding:utf-8 -*-
import time
from typing import Dict, List
from .basic import _ModelBase, Logger, TrainingStopException


_EvalLogsType = Dict[str, Dict[str, List[float]]]


class EarlyStopException(TrainingStopException):
    pass


class AbstractCallback:

    def __init__(self):
        pass

    def before_train(self, model: _ModelBase) -> _ModelBase:
        """Called before training starts. Returns the model."""
        return model

    def after_train(self, model: _ModelBase) -> _ModelBase:
        """Called after training ends. Returns the model."""
        return model

    def before_iter(self, n_iter: int, model: _ModelBase, logs: _EvalLogsType):
        """Called before each iteration."""
        pass

    def after_iter(self, n_iter: int, model: _ModelBase, logs: _EvalLogsType):
        """Called after each iteration."""
        pass


class EvalCallback(AbstractCallback):

    def __init__(self, eval_period: int):
        """
        Initializes the EvalCallback class.

        Parameters
        ----------
        eval_period : int
            the period for evaluation
        """
        self.period = eval_period
        self.start_time = 0
        super().__init__()

    def before_train(self, model: _ModelBase) -> _ModelBase:
        self.start_time = time.time()
        return model

    def after_train(self, model: _ModelBase) -> _ModelBase:
        self.start_time = 0
        return model

    def after_iter(self, n_iter: int, model: _ModelBase, logs: _EvalLogsType):
        if not isinstance(logs, dict):
            return
        if (n_iter % self.period) == 0:
            eval_list = ["{iter:4d}:".format(iter=n_iter)]
            tmp_eval_rets = model.eval()
            for data, metric, score in tmp_eval_rets:
                eval_list.append(
                    "{data}-{metric}: {score:.6f}".format(
                        data=data,
                        metric=metric,
                        score=score)
                )
                if data not in logs.keys():
                    logs[data] = dict()
                if metric not in logs[data].keys():
                    logs[data][metric] = list()
                logs[data][metric].append(score)

            Logger.info("\t".join(eval_list) + "\ttotal: {:.2f}s".format(time.time() - self.start_time))


class EarlyStopCallback(AbstractCallback):

    _bigger_better = {
        "auc": True,
        "qini_coff": True,
        "qini_area": True,
        "logloss": False,
        "rmse": False,
        "l2": False
    }

    def __init__(self, stop_round: int, data_name: str, metric_name: str, use_best: bool):
        """
        Initializes the EarlyStopCallback class.

        Parameters
        ----------
        stop_round : int
            round to stop training
        data_name : string
            dataset name
        metric_name : string
            metric name
        use_best : boolean
            whether to use the best model or not
        """
        self.round = stop_round
        self.name = data_name
        self.metric = metric_name
        self.use_best = use_best
        self.best_round = 0
        self.best_iter = 0
        if metric_name not in self._bigger_better.keys():
            raise ValueError("{metric} not in [{metric_list}]".format(
                metric=metric_name, metric_list=", ".join(self._bigger_better.keys()))
            )
        self.best_score = -float("inf") if self._bigger_better[self.metric] else float("inf")
        super().__init__()

    def after_iter(self, n_iter: int, model: _ModelBase, logs: _EvalLogsType):
        n_iter += 1
        if not logs:
            return
        if self.name not in logs.keys():
            return
        if self.metric not in logs[self.name].keys():
            return

        score_list = logs[self.name][self.metric]
        if len(score_list) == 0:
            return
        if self._bigger_better[self.metric]:
            if self.best_score <= score_list[-1]:
                self.best_iter = n_iter
                self.best_score = score_list[-1]
                self.best_round = len(score_list)
        else:
            if self.best_score >= score_list[-1]:
                self.best_iter = n_iter
                self.best_score = score_list[-1]
                self.best_round = len(score_list)

        if (len(score_list) - self.round) >= self.best_round:
            Logger.info("\nbestScore = {:.6f}".format(self.best_score))
            Logger.info("bestIteration = {iter:d}".format(iter=self.best_iter - 1))
            raise TrainingStopException()

    def after_train(self, model: _ModelBase) -> _ModelBase:
        if self.use_best:  # roll back model
            if model.rollback(self.best_iter):
                Logger.info("Rollback model to iteration: {iter:d}.".format(iter=self.best_iter - 1))
        return model
