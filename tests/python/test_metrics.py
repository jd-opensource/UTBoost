# -*- coding:utf-8 -*-
import pytest
import numpy as np

try:
    from sklift.metrics import qini_auc_score, uplift_auc_score
    sklift_installed = True
except:
    sklift_installed = False

import utboost
from utboost.metric import *


class TestMetrics:

    np.random.seed(618)
    n = 1000

    binary_treatment = np.random.choice([0, 1], size=(n, ), replace=True)
    multi_treatment = np.random.choice([0, 1, 2], size=(n, ), replace=True)

    binary_uplift = np.random.uniform(0.0, 1.0, size=(n, ))
    multi_uplift = np.random.uniform(0.0, 1.0, size=(n, np.max(multi_treatment)))

    outcomes = np.random.choice([0, 1], size=(n, ), replace=True)

    @pytest.mark.skipif(not sklift_installed, reason="sklift is not installed.")
    def test_binary_qini(self):
        expect = qini_auc_score(self.outcomes, self.binary_uplift, self.binary_treatment, negative_effect=True)
        actual = qini_score(self.outcomes, self.binary_uplift, self.binary_treatment, negative_effect=True)
        assert np.abs(expect - actual) < 1e-4

        expect = qini_auc_score(self.outcomes, self.binary_uplift, self.binary_treatment, negative_effect=False)
        actual = qini_score(self.outcomes, self.binary_uplift, self.binary_treatment, negative_effect=False)
        assert np.abs(expect - actual) < 1e-4

    @pytest.mark.skipif(not sklift_installed, reason="sklift is not installed.")
    def test_multi_qini(self):
        actual_outcome, uplift_score, treatment_flag = utboost.metric._multi_treatment_to_binary(
            self.outcomes, self.multi_uplift, self.multi_treatment
        )
        expect = qini_auc_score(actual_outcome, uplift_score, treatment_flag, negative_effect=True)
        actual = qini_score(self.outcomes, self.multi_uplift, self.multi_treatment, negative_effect=True)
        assert np.abs(expect - actual) < 1e-4

    @pytest.mark.skipif(not sklift_installed, reason="sklift is not installed.")
    def test_binary_auuc(self):
        expect = uplift_auc_score(self.outcomes, self.binary_uplift, self.binary_treatment)
        actual = auuc_score(self.outcomes, self.binary_uplift, self.binary_treatment)
        assert np.abs(expect - actual) < 1e-4

        expect = uplift_auc_score(self.outcomes, self.binary_uplift, self.binary_treatment)
        actual = auuc_score(self.outcomes, self.binary_uplift, self.binary_treatment)
        assert np.abs(expect - actual) < 1e-4

    @pytest.mark.skipif(not sklift_installed, reason="sklift is not installed.")
    def test_multi_auuc(self):
        actual_outcome, uplift_score, treatment_flag = utboost.metric._multi_treatment_to_binary(
            self.outcomes, self.multi_uplift, self.multi_treatment
        )
        expect = uplift_auc_score(actual_outcome, uplift_score, treatment_flag)
        actual = auuc_score(self.outcomes, self.multi_uplift, self.multi_treatment)
        assert np.abs(expect - actual) < 1e-4

    def test_cumulative_uplift_top_k(self):
        expect = np.mean(self.outcomes[self.binary_treatment == 1]) - \
                 np.mean(self.outcomes[self.binary_treatment == 0])
        actual = cumulative_uplift_top_k(self.outcomes, self.binary_uplift, self.binary_treatment, k=1.0)
        assert np.abs(expect - actual) < 1e-4

    def test_uplift_curve_by_percentile(self):
        expect = np.mean(self.outcomes[self.binary_treatment == 1]) - \
                 np.mean(self.outcomes[self.binary_treatment == 0])
        actual = uplift_curve_by_percentile(self.outcomes, self.binary_uplift, self.binary_treatment, n_bins=10)

        assert actual.shape[0] == 10
        assert np.abs(expect - actual["cumulative_uplift"].values[-1]) < 1e-4
        assert np.sum(actual[["n_control", "n_treatment"]].values) == self.n

    def test_treatment_balance_score(self):
        bout_psi, bout_df = treatment_balance_score(self.binary_uplift, self.binary_treatment, n_bins=4)
        assert bout_psi < 0.1
        assert bout_df.shape[0] == 4
        mout_psi, mout_df = treatment_balance_score(
            np.c_[self.binary_uplift, self.binary_uplift], self.binary_treatment, n_bins=4
        )
        assert len(mout_psi) == 2
        assert mout_df.shape[0] == 8
        assert np.max(np.abs(mout_psi - bout_psi)) < 1e-5
