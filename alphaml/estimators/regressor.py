import numpy as np
import pandas as pd
from sklearn.utils.multiclass import type_of_target
from sklearn.metrics import mean_squared_error
from alphaml.estimators.base_estimator import BaseEstimator
from alphaml.engine.automl import AutoMLRegressor
from alphaml.engine.components.data_manager import DataManager
from alphaml.engine.components.pipeline.data_preprocessing_pipeline import DP_Pipeline
from alphaml.utils.metrics_util import get_metric


class Regressor(BaseEstimator):
    """This class implements the regression task. """

    def fit(self, data, **kwargs):
        """
        Fit the regressor to given training data.
        :param data: instance of DataManager
        :return: self
        """
        metric = mean_squared_error if 'metric' not in kwargs else kwargs['metric']

        # TODO:Automated feature engineering
        if isinstance(data, pd.DataFrame):
            self.pre_pipeline = DP_Pipeline(None)
            data = self.pre_pipeline.execute(data, phase='train', stratify=False)
        # Check the task type: {continuous}
        task_type = type_of_target(data.train_y)
        if task_type != 'continuous':
            raise ValueError("UNSUPPORTED TASK TYPE: %s!" % task_type)
        self.task_type = task_type
        kwargs['task_type'] = task_type

        metric = get_metric(metric)
        kwargs['metric'] = metric

        super().fit(data, **kwargs)

        return self

    def predict(self, X, batch_size=None, n_jobs=1):
        """
        Make predictions for X.
        :param X: array-like or sparse matrix of shape = [n_samples, n_features]
        :param batch_size: int
        :param n_jobs: int
        :return: y : array of shape = [n_samples] or [n_samples, n_labels]
            The predicted classes.
        """
        if isinstance(X, pd.DataFrame):
            if not isinstance(self.pre_pipeline, DP_Pipeline):
                raise ValueError("The preprocessing pipeline is empty. Use DataFrame as the input of function fit.")
            dm = self.pre_pipeline.execute(X, phase='test')
            X = dm.test_X
        return super().predict(X, batch_size=batch_size, n_jobs=n_jobs)

    def get_automl(self):
        return AutoMLRegressor
