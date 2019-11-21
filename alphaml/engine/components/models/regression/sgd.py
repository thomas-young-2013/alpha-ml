import numpy as np
import time
from hyperopt import hp
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    CategoricalHyperparameter, UnParametrizedHyperparameter
from ConfigSpace.conditions import EqualsCondition, InCondition

from alphaml.engine.components.models.base_model import BaseRegressionModel, IterativeComponentWithSampleWeight
from alphaml.utils.constants import *
from alphaml.utils.common import check_none, check_for_bool


class SGD(
    IterativeComponentWithSampleWeight,
    BaseRegressionModel,
):
    def __init__(self, loss, penalty, alpha, fit_intercept, tol,
                 learning_rate, epsilon_insensitive, l1_ratio=0.15, epsilon_huber=0.1,
                 eta0=0.01, power_t=0.5, average=False, random_state=None):
        self.loss = loss
        self.penalty = penalty
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.learning_rate = learning_rate
        self.l1_ratio = l1_ratio
        self.epsilon_huber = epsilon_huber
        self.epsilon_insensitive = epsilon_insensitive
        self.eta0 = eta0
        self.power_t = power_t
        self.random_state = random_state
        self.average = average
        self.estimator = None
        self.start_time = time.time()
        self.time_limit = None

    def iterative_fit(self, X, y, n_iter=2, refit=False, sample_weight=None):
        from sklearn.linear_model.stochastic_gradient import SGDRegressor

        # Need to fit at least two iterations, otherwise early stopping will not
        # work because we cannot determine whether the algorithm actually
        # converged. The only way of finding this out is if the sgd spends less
        # iterations than max_iter. If max_iter == 1, it has to spend at least
        # one iteration and will always spend at least one iteration, so we
        # cannot know about convergence.

        if refit:
            self.estimator = None

        if isinstance(self.loss, tuple):
            nested_loss = self.loss
            self.loss = nested_loss[0]
            if self.loss == 'huber':
                self.epsilon_huber = nested_loss[1]['epsilon_huber']
            elif self.loss in ["epsilon_insensitive", "squared_epsilon_insensitive"]:
                self.epsilon_insensitive = nested_loss[1]['epsilon_insensitive']

        if isinstance(self.penalty, tuple):
            nested_penalty = self.penalty
            self.penalty = nested_penalty[0]
            if self.penalty == "elasticnet":
                self.l1_ratio = nested_penalty[1]['l1_ratio']

        if isinstance(self.learning_rate, tuple):
            nested_learning_rate = self.learning_rate
            self.learning_rate = nested_learning_rate[0]
            if self.learning_rate == 'invscaling':
                self.eta0 = nested_learning_rate[1]['eta0']
                self.power_t = nested_learning_rate[1]['power_t']
            elif self.learning_rate == 'constant':
                self.eta0 = nested_learning_rate[1]['eta0']
            self.fully_fit_ = False

        if self.estimator is None:
            self.fully_fit_ = False

            self.alpha = float(self.alpha)
            if not check_none(self.epsilon_insensitive):
                self.epsilon_insensitive = float(self.epsilon_insensitive)
            self.l1_ratio = float(self.l1_ratio) if self.l1_ratio is not None \
                else 0.15
            self.epsilon_huber = float(self.epsilon_huber) if self.epsilon_huber is not None \
                else 0.1
            self.eta0 = float(self.eta0) if self.eta0 is not None else 0.01
            self.power_t = float(self.power_t) if self.power_t is not None \
                else 0.5
            self.average = check_for_bool(self.average)
            self.fit_intercept = check_for_bool(self.fit_intercept)
            self.tol = float(self.tol)
            if self.loss == "huber":
                epsilon = self.epsilon_huber
            elif self.loss in ["epsilon_insensitive", "squared_epsilon_insensitive"]:
                epsilon = self.epsilon_insensitive
            else:
                epsilon = None
            self.estimator = SGDRegressor(loss=self.loss,
                                          penalty=self.penalty,
                                          alpha=self.alpha,
                                          fit_intercept=self.fit_intercept,
                                          max_iter=n_iter,
                                          tol=self.tol,
                                          learning_rate=self.learning_rate,
                                          l1_ratio=self.l1_ratio,
                                          epsilon=epsilon,
                                          eta0=self.eta0,
                                          power_t=self.power_t,
                                          shuffle=True,
                                          average=self.average,
                                          random_state=self.random_state,
                                          warm_start=True)
            self.estimator.fit(X, y, sample_weight=sample_weight)
        else:
            self.estimator.max_iter += n_iter
            self.estimator.max_iter = min(self.estimator.max_iter, 512)
            self.estimator._validate_params()
            self.estimator._partial_fit(
                X, y,
                alpha=self.estimator.alpha,
                C=1.0,
                loss=self.estimator.loss,
                learning_rate=self.estimator.learning_rate,
                max_iter=n_iter,
                sample_weight=sample_weight,
                coef_init=None,
                intercept_init=None
            )

        if self.estimator.max_iter >= 512 or n_iter > self.estimator.n_iter_:
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
        return {'shortname': 'SGD Regressor',
                'name': 'Stochastic Gradient Descent Regressor',
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

            loss = CategoricalHyperparameter("loss",
                                             ["squared_loss", "huber", "epsilon_insensitive",
                                              "squared_epsilon_insensitive"],
                                             default_value="squared_loss")
            penalty = CategoricalHyperparameter(
                "penalty", ["l1", "l2", "elasticnet"], default_value="l2")
            alpha = UniformFloatHyperparameter(
                "alpha", 1e-7, 1e-1, log=True, default_value=0.0001)
            l1_ratio = UniformFloatHyperparameter(
                "l1_ratio", 1e-9, 1, log=True, default_value=0.15)
            fit_intercept = UnParametrizedHyperparameter("fit_intercept", "True")
            tol = UniformFloatHyperparameter("tol", 1e-5, 1e-1, log=True,
                                             default_value=1e-4)
            epsilon_huber = UniformFloatHyperparameter(
                "epsilon_huber", 1e-5, 1e-1, default_value=1e-4, log=True)
            epsilon_insensitive = UniformFloatHyperparameter(
                "epsilon_insensitive", 1e-5, 1e-1, default_value=1e-4, log=True)
            learning_rate = CategoricalHyperparameter(
                "learning_rate", ["optimal", "invscaling", "constant"],
                default_value="invscaling")
            eta0 = UniformFloatHyperparameter(
                "eta0", 1e-7, 1e-1, default_value=0.01, log=True)
            power_t = UniformFloatHyperparameter("power_t", 1e-5, 1, log=True,
                                                 default_value=0.5)
            average = CategoricalHyperparameter(
                "average", ["False", "True"], default_value="False")
            cs.add_hyperparameters([loss, penalty, alpha, l1_ratio, fit_intercept,
                                    tol, epsilon_huber, epsilon_insensitive, learning_rate, eta0, power_t,
                                    average])

            elasticnet = EqualsCondition(l1_ratio, penalty, "elasticnet")
            epsilon_huber_condition = EqualsCondition(epsilon_huber, loss, "huber")
            epsilon_insensitive_condition = InCondition(epsilon_insensitive, loss,
                                                        ["epsilon_insensitive", "squared_epsilon_insensitive"])
            power_t_condition = EqualsCondition(power_t, learning_rate,
                                                "invscaling")

            # eta0 is only relevant if learning_rate!='optimal' according to code
            # https://github.com/scikit-learn/scikit-learn/blob/0.19.X/sklearn/
            # linear_model/sgd_fast.pyx#L603
            eta0_in_inv_con = InCondition(eta0, learning_rate, ["invscaling",
                                                                "constant"])
            cs.add_conditions([elasticnet, epsilon_huber_condition, epsilon_insensitive_condition,
                               power_t_condition, eta0_in_inv_con])

            return cs
        elif optimizer == 'tpe':
            eta0 = hp.loguniform('sgd_eta0', np.log(1e-7), np.log(1e-1))
            epsilon_insensitive = hp.loguniform('sgd_epsilon_insensitive', np.log(1e-5), np.log(1e-1))
            space = {
                'loss': hp.choice('sgd_loss', [
                    ("huber", {'epsilon_huber': hp.loguniform('sgd_epsilon_huber', np.log(1e-5), np.log(1e-1))}),
                    ("squared_loss", {}),
                    ("epsilon_insensitive", {'epsilon_insensitive': epsilon_insensitive}),
                    ("squared_epsilon_insensitive", {'epsilon_insensitive': epsilon_insensitive})]),
                'penalty': hp.choice('sgd_penalty',
                                     [("elasticnet",
                                       {'l1_ratio': hp.loguniform('sgd_l1_ratio', np.log(1e-9), np.log(1))}),
                                      ("l1", None),
                                      ("l2", None)]),
                'alpha': hp.loguniform('sgd_alpha', np.log(1e-7), np.log(1e-1)),
                'fit_intercept': hp.choice('sgd_fit_intercept', ["True"]),
                'tol': hp.loguniform('sgd_tol', np.log(1e-5), np.log(1e-1)),
                'learning_rate': hp.choice('sgd_learning_rate', [("optimal", {}),
                                                                 ("invscaling",
                                                                  {'power_t': hp.loguniform('sgd_power_t', np.log(1e-5),
                                                                                            np.log(1)),
                                                                   'eta0': eta0}),
                                                                 ("constant", {'eta0': eta0})]),

                'average': hp.choice('sgd_average', ["True", "False"])}

            init_trial = {'loss': ("squared_loss", {}),
                          'penalty': ("l2", {}),
                          'alpha': 1e-4,
                          'fit_intercept': "True",
                          'tol': 1e-4,
                          'learning_rate': ("invscaling", {'power_t': 0.5, 'eta0': 0.01}),
                          'average': "False"}

            return space
