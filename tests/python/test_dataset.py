import tempfile
import pytest
import os
import numpy as np
import scipy.sparse as sp
import utboost as utb

from .utils import make_uplift_problem


class TestDataset:

    dataset, x_names = make_uplift_problem(
        n_samples_per_group=500, n_treatments=1, delta_uplift=0.1,
        n_features=20, n_informative=5, n_uplift_mix_informative=2, n_redundant=2, n_repeated=2,
        pos_weight=0.6, random_seed=618
    )

    params = {
        "ensemble_type": 'boosting',
        "criterion": 'gbm',
        "iterations": 30,
        "max_depth": 3
    }

    X_train = dataset.loc[:, x_names].values
    total_elements = X_train.size
    nan_elements = int(total_elements * 0.2)
    flat_indices = np.random.choice(total_elements, nan_elements, replace=False)
    np.put(X_train, flat_indices, np.nan)
    flat_indices = np.random.choice(total_elements, nan_elements, replace=False)
    np.put(X_train, flat_indices, 0.0)

    y_train = dataset.loc[:, 'label'].values
    tid_train = dataset.loc[:, 'treatment'].values

    model = utb.UTBClassifier(**params)
    model.fit(X=X_train, ti=tid_train, y=y_train)

    expect = model.predict(X_train)

    def test_dataset_from_numpy(self):
        dataset = utb.Dataset(data=self.X_train, treatment=self.tid_train, label=self.y_train)
        model = utb.UTBClassifier(**self.params)
        model.fit(X=dataset)
        actual = model.predict(self.X_train)
        assert actual.shape == self.expect.shape
        assert np.max(np.abs(actual - self.expect)) < 1e-7

    def test_dataset_from_libsvm(self):
        with tempfile.TemporaryDirectory() as tempdir:
            file_path = os.path.join(tempdir, 'data.libsvm')
            # write file
            with open(file_path, 'w') as file:
                for i in range(self.X_train.shape[0]):
                    label = str(int(self.y_train[i]))
                    treat = str(int(self.tid_train[i]))
                    features_str = " ".join(f"{index + 1}:{value:.8f}" for index, value in enumerate(self.X_train[i]) if not np.isnan(value))
                    file.write(f"{label} {treat} {features_str}\n")

            dataset = utb.Dataset(data="libsvm://" + file_path)
            model = utb.UTBClassifier(**self.params)
            model.fit(X=dataset)
            actual = model.predict(self.X_train)
            assert actual.shape == self.expect.shape
            assert np.max(np.abs(actual - self.expect)) < 1e-7

    def test_dataset_from_csr(self):
        csr_train = sp.csr_matrix(self.X_train, copy=True)
        dataset = utb.Dataset(data=csr_train, treatment=self.tid_train, label=self.y_train)
        model = utb.UTBClassifier(**self.params)
        model.fit(X=dataset)
        actual = model.predict(self.X_train)
        assert actual.shape == self.expect.shape
        assert np.max(np.abs(actual - self.expect)) < 1e-7
