from ConfigSpace.configuration_space import ConfigurationSpace
from alphaml.utils.constants import *
from alphaml.engine.components.models.base_model import BaseRegressionModel


class LinearRegressor(BaseRegressionModel):
    def __init__(self, random_state=None):
        self.random_state = random_state
        self.estimator = None

    def fit(self, X, Y):
        from sklearn.linear_model import LinearRegression
        self.estimator = LinearRegression(fit_intercept=False,
                                          n_jobs=-1)
        self.estimator.fit(X, Y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'Linear-Regression',
                'name': 'Linear Regression',
                'handles_regression': True,
                'handles_classification': False,
                'handles_multiclass': False,
                'handles_multilabel': False,
                'is_deterministic': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None, optimizer='smac'):
        if optimizer == 'smac':
            cs = ConfigurationSpace()
            return cs
