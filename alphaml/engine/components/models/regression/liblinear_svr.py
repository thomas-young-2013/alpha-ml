from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    CategoricalHyperparameter, Constant
from ConfigSpace.forbidden import ForbiddenEqualsClause, \
    ForbiddenAndConjunction

from alphaml.utils.constants import *
from alphaml.utils.model_util import softmax
from alphaml.utils.common import check_none, check_for_bool
from alphaml.engine.components.models.base_model import BaseRegressionModel


class LibLinear_SVR(BaseRegressionModel):
    # Liblinear is not deterministic as it uses a RNG inside
    def __init__(self, loss, dual, tol, C,
                 fit_intercept, intercept_scaling,
                 random_state=None):
        self.loss = loss
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.random_state = random_state
        self.estimator = None

    def fit(self, X, Y):
        from sklearn.svm import LinearSVR

        self.C = float(self.C)
        self.tol = float(self.tol)

        self.dual = check_for_bool(self.dual)

        self.fit_intercept = check_for_bool(self.fit_intercept)

        self.intercept_scaling = float(self.intercept_scaling)

        self.estimator = LinearSVR(loss=self.loss,
                                   dual=self.dual,
                                   tol=self.tol,
                                   C=self.C,
                                   fit_intercept=self.fit_intercept,
                                   intercept_scaling=self.intercept_scaling,
                                   random_state=self.random_state)
        self.estimator.fit(X, Y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'Liblinear-SVR',
                'name': 'Liblinear Support Vector Regression',
                'handles_regression': True,
                'handles_classification': False,
                'handles_multiclass': False,
                'handles_multilabel': False,
                'is_deterministic': False,
                'input': (SPARSE, DENSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None, optimizer='smac'):
        if optimizer == 'smac':
            cs = ConfigurationSpace()

            loss = CategoricalHyperparameter(
                "loss", ["epsilon_insensitive", "squared_epsilon_insensitive"], default_value="epsilon_insensitive")
            dual = CategoricalHyperparameter("dual", ['True', 'False'], default_value='True')
            # This is set ad-hoc
            tol = UniformFloatHyperparameter(
                "tol", 1e-5, 1e-1, default_value=1e-4, log=True)
            C = UniformFloatHyperparameter(
                "C", 0.03125, 32768, log=True, default_value=1.0)
            fit_intercept = Constant("fit_intercept", "True")
            intercept_scaling = Constant("intercept_scaling", 1)
            cs.add_hyperparameters([loss, dual, tol, C,
                                    fit_intercept, intercept_scaling])

            dual_and_loss = ForbiddenAndConjunction(
                ForbiddenEqualsClause(dual, "False"),
                ForbiddenEqualsClause(loss, "epsilon_insensitive")
            )
            cs.add_forbidden_clause(dual_and_loss)
            return cs
