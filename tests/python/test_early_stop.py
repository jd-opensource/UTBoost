# -*- coding:utf-8 -*-
import numpy as np
from sklearn.model_selection import train_test_split

import utboost as utb

from .utils import segment_metric, make_uplift_problem


def test_early_stop():
    # noisy dataset
    dataset, x_names = make_uplift_problem(
        n_samples_per_group=1000, n_treatments=1, delta_uplift=0.1,
        n_features=20, n_informative=1, n_uplift_mix_informative=1, n_redundant=0, n_repeated=0, flip_y=0.2,
        pos_weight=0.6, random_seed=618
    )

    # Split data into train and validation
    train, valid = train_test_split(dataset, random_state=42)

    # values extraction
    X_train, y_train, T_train = train.loc[:, x_names].values, train.loc[:, 'label'].values, train.loc[:, 'treatment'].values
    X_valid, y_valid, T_valid = valid.loc[:, x_names].values, valid.loc[:, 'label'].values, valid.loc[:, 'treatment'].values

    # over fitting
    clf1 = utb.UTBClassifier(
        ensemble_type='boosting',
        criterion='gbm',
        iterations=100,
        max_depth=4
    ).fit(X_train, T_train, y_train)

    pred1 = clf1.predict(X_valid)
    loss1 = segment_metric(pred1, y_valid, T_valid)

    # early stopping
    clf2 = utb.UTBClassifier(
        ensemble_type='boosting',
        criterion='gbm',
        iterations=100,
        max_depth=4,
        eval_metric=['logloss'],
        early_stopping_rounds=5,
        use_best_model=True,
    ).fit(X_train, T_train, y_train, eval_sets=[(X_valid, T_valid, y_valid)])
    pred2 = clf2.predict(X_valid)
    loss2 = segment_metric(pred2, y_valid, T_valid)

    logs = clf2.get_logs()

    assert loss1 >= loss2
    # should be the same
    assert np.abs(np.min(logs['valid-0']['logloss']) - loss2) < 1e-5


