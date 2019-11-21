import numpy as np
from hyperopt import hp
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, UnParametrizedHyperparameter

from alphaml.utils.constants import *
from alphaml.utils.model_util import softmax
from alphaml.utils.common import check_none, check_for_bool
from alphaml.engine.components.models.base_model import BaseRegressionModel, IterativeComponentWithSampleWeight

import time


class PassiveAggressive(
    IterativeComponentWithSampleWeight,
    BaseRegressionModel,
):
    def __init__(self, C, fit_intercept, tol, loss, average, random_state=None):
        self.C = C
        self.fit_intercept = fit_intercept
        self.average = average
        self.tol = tol
        self.loss = loss
        self.random_state = random_state
        self.estimator = None
        self.start_time = time.time()
        self.time_limit = None

    def iterative_fit(self, X, y, n_iter=2, refit=False, sample_weight=None):
        from sklearn.linear_model.passive_aggressive import \
            PassiveAggressiveRegressor

        # Need to fit at least two iterations, otherwise early stopping will not
        # work because we cannot determine whether the algorithm actually
        # converged. The only way of finding this out is if the sgd spends less
        # iterations than max_iter. If max_iter == 1, it has to spend at least
        # one iteration and will always spend at least one iteration, so we
        # cannot know about convergence.

        if refit:
            self.estimator = None

        if self.estimator is None:
            self.fully_fit_ = False

            self.average = check_for_bool(self.average)
            self.fit_intercept = check_for_bool(self.fit_intercept)
            self.tol = float(self.tol)
            self.C = float(self.C)

            call_fit = True
            self.estimator = PassiveAggressiveRegressor(
                C=self.C,
                fit_intercept=self.fit_intercept,
                max_iter=n_iter,
                tol=self.tol,
                loss=self.loss,
                shuffle=True,
                random_state=self.random_state,
                warm_start=True,
                average=self.average,
            )
        else:
            call_fit = False

        if call_fit:
            self.estimator.fit(X, y)
        else:
            self.estimator.max_iter += n_iter
            self.estimator.max_iter = min(self.estimator.max_iter,
                                          1000)
            self.estimator._validate_params()
            lr = "pa1" if self.estimator.loss == "epsilon_insensitive" else "pa2"
            self.estimator._partial_fit(
                X, y,
                alpha=1.0,
                C=self.estimator.C,
                loss="epsilon_insensitive",
                learning_rate=lr,
                max_iter=n_iter,
                sample_weight=sample_weight,
                coef_init=None,
                intercept_init=None
            )
            if self.estimator.max_iter >= 1000 or n_iter > self.estimator.n_iter_:
                self.fully_fit_ = True

        return self

    def configuration_fully_fitted(self):
        if self.estimator is None:
            return False
        elif not hasattr(self, 'fully_fit_'):
            return False
        else:
            return self.fully_fit_

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'PassiveAggressive Regressor',
                'name': 'Passive Aggressive Regressor',
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
            C = UniformFloatHyperparameter("C", 1e-5, 10, 1.0, log=True)
            fit_intercept = UnParametrizedHyperparameter("fit_intercept", "True")
            loss = CategoricalHyperparameter(
                "loss", ["epsilon_insensitive", "squared_epsilon_insensitive"], default_value="squared_epsilon_insensitive"
            )

            tol = UniformFloatHyperparameter("tol", 1e-5, 1e-1, default_value=1e-4,
                                             log=True)
            # Note: Average could also be an Integer if > 1
            average = CategoricalHyperparameter('average', ['False', 'True'],
                                                default_value='False')

            cs = ConfigurationSpace()
            cs.add_hyperparameters([loss, fit_intercept, tol, C, average])
            return cs
        elif optimizer == 'tpe':
            space = {'C': hp.loguniform("pa_C", np.log(1e-5), np.log(10)),
                     'fit_intercept': hp.choice('pa_fit_intercept', ["True"]),
                     'loss': hp.choice('pr_loss', ["epsilon_insensitive", "squared_epsilon_insensitive"]),
                     'tol': hp.loguniform('pr_tol', np.log(1e-5), np.log(1e-1)),
                     'average': hp.choice('pr_average', ["False", "True"])}

            init_trial = {'C': 1, 'fit_intercept': "True", 'loss': "squared_epsilon_insensitive",
                          'tol': 1e-4, 'average': "False"}

            return space