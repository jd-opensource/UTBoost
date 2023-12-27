# -*- coding:utf-8 -*-
import tempfile
import pytest
import os
import numpy as np
from sklearn.linear_model import LogisticRegression

import utboost as utb

from .utils import segment_metric, make_uplift_problem, logloss


class TestModels:
    """ Test basic features. """

    dataset, x_names = make_uplift_problem(
        n_samples_per_group=2000, n_treatments=1, delta_uplift=0.1,
        n_features=20, n_informative=5, n_uplift_mix_informative=2, n_redundant=2, n_repeated=2,
        pos_weight=0.6, random_seed=618
    )

    X_train = dataset.loc[:, x_names].values
    y_train = dataset.loc[:, 'label'].values
    tid_train = dataset.loc[:, 'treatment'].values

    lr_preds = LogisticRegression(max_iter=200).fit(X_train, y_train).predict_proba(X_train)[:, 1]

    def check_binary_preds(self, preds, labels, tid, mean_tolerance=0.0, check_loss=True):
        """ Check whether the model prediction performance meets expectations. """
        tid = tid.flatten().astype(np.int32)
        if mean_tolerance > 0.0:
            for t in sorted(set(tid)):
                idx = tid == t
                expect = np.mean(labels[idx])
                actual = np.mean(preds[idx, t])
                assert abs(expect - actual) < expect * mean_tolerance, \
                    "The prediction mean ({:.2f}) " \
                    "and the true label mean ({:.2f}) fail to test " \
                    "under {:.1%} tolerance".format(actual, expect, mean_tolerance)
        if check_loss:
            expect = logloss(self.lr_preds, labels)
            actual = segment_metric(preds, labels, tid, metric="logloss")
            assert actual < expect, "UTBClassifier has higher logloss " \
                                    "than logistic regression ({:.2f} vs. {:.2f}).".format(actual, expect)

    def check_model_io(self, utb_model: utb.UTBClassifier):
        """ Check the consistency of model io """
        preds = utb_model.predict(self.X_train)
        with tempfile.TemporaryDirectory() as tempdir:
            path_utm = os.path.join(tempdir, 'model.utm')
            path_py = os.path.join(tempdir, 'model.py')
            utb_model.save_model(path_utm)
            utb_model.to_python(path_py)

            # check loaded model
            new_model = utb.UTBClassifier().load_model(path_utm)
            preds_utm = new_model.predict(self.X_train)
            np.testing.assert_allclose(preds_utm, preds)

            # check apply
            exec(open(path_py, "rb").read(), globals())
            func = globals()['apply_model']
            actual = []
            n_items = 20
            for i in range(n_items):
                case = []
                for value in self.X_train[i, :]:
                    case.append(None if np.isnan(value) else value)
                actual.append(func(case))
            np.testing.assert_allclose(np.array(actual), preds[: n_items])

    def check_feature_imp(self, model):
        """ Check if relevant features are more important than irrelevant features. """
        irr_features = ['f' + str(idx) for idx, name in enumerate(self.x_names) if name.find('_irrelevant') != -1]
        irr_features += [i for i in self.x_names if i.find('_irrelevant') != -1]
        for imp_type in ('split', 'ganin'):
            imp = model.feature_importance(importance_type=imp_type)
            irr_imp, other_imp = [], []
            for name, value in imp:
                if name in irr_features:
                    irr_imp.append(value)
                else:
                    other_imp.append(value)
            assert np.mean(irr_imp) < np.mean(other_imp), \
                "The average importance of relevant features ({:.2f}) " \
                "is lower than that of irrelevant features ({:.2f}).".format(np.mean(other_imp), np.mean(irr_imp))

    def test_causalgbm(self):
        model = utb.UTBClassifier(
            ensemble_type='boosting',
            criterion='gbm',
            iterations=50,
            max_depth=3
        )
        model.fit(X=self.X_train, ti=self.tid_train, y=self.y_train, feature_names=self.x_names)
        preds = model.predict(self.X_train)
        self.check_binary_preds(preds, self.y_train, self.tid_train, mean_tolerance=0.05, check_loss=True)
        self.check_model_io(model)
        self.check_feature_imp(model)

    def test_tddp(self):
        model = utb.UTBClassifier(
            ensemble_type='boosting',
            criterion='ddp',
            iterations=50,
            max_depth=3
        )
        model.fit(X=self.X_train, ti=self.tid_train, y=self.y_train)
        preds = model.predict(self.X_train)
        self.check_binary_preds(preds, self.y_train, self.tid_train, mean_tolerance=0.2, check_loss=False)
        self.check_model_io(model)
        self.check_feature_imp(model)

    def test_bagging_models(self):
        for criterion in ('ddp', 'gbm', 'kl', 'ed', 'chi'):
            model = utb.UTBClassifier(
                ensemble_type='bagging',
                criterion=criterion,
                iterations=30,
                colsample=0.5,
                max_depth=3
            )
            model.fit(X=self.X_train, ti=self.tid_train, y=self.y_train)
            preds = model.predict(self.X_train)
            self.check_binary_preds(preds, self.y_train, self.tid_train, mean_tolerance=0.2, check_loss=False)
            self.check_model_io(model)
            self.check_feature_imp(model)


class TestMultiTreatment(TestModels):
    """ Test all cases in multi-treatment context """

    dataset, x_names = make_uplift_problem(
        n_samples_per_group=2000, n_treatments=3, delta_uplift=0.1,
        n_features=20, n_informative=5, n_uplift_mix_informative=2, n_redundant=2, n_repeated=2,
        pos_weight=0.6, random_seed=618
    )

    X_train = dataset.loc[:, x_names].values
    y_train = dataset.loc[:, 'label'].values
    tid_train = dataset.loc[:, 'treatment'].values

    lr_preds = LogisticRegression(max_iter=200).fit(X_train, y_train).predict_proba(X_train)[:, 1]
