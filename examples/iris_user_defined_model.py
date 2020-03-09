import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics.scorer import make_scorer

from alphaml.estimators.classifier import Classifier
from alphaml.engine.components.data_manager import DataManager
from alphaml.datasets.cls_dataset.dataset_loader import load_data
from alphaml.engine.components.models.classification import add_classifier

import numpy as np
from hyperopt import hp
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, \
    UnParametrizedHyperparameter, Constant

from alphaml.engine.components.models.base_model import BaseClassificationModel
from alphaml.utils.constants import *
from alphaml.utils.common import check_none


class UserDefinedDecisionTree(BaseClassificationModel):
    def __init__(self, criterion, max_features, max_depth_factor,
                 min_samples_split, min_samples_leaf, min_weight_fraction_leaf,
                 max_leaf_nodes, min_impurity_decrease, class_weight=None,
                 random_state=None):
        self.criterion = criterion
        self.max_features = max_features
        self.max_depth_factor = max_depth_factor
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.class_weight = class_weight
        self.estimator = None
        self.time_limit = None

    def fit(self, X, y, sample_weight=None):
        from sklearn.tree import DecisionTreeClassifier

        self.max_features = float(self.max_features)
        # Heuristic to set the tree depth
        if check_none(self.max_depth_factor):
            max_depth_factor = self.max_depth_factor = None
        else:
            num_features = X.shape[1]
            self.max_depth_factor = int(self.max_depth_factor)
            max_depth_factor = max(
                1,
                int(np.round(self.max_depth_factor * num_features, 0)))
        self.min_samples_split = int(self.min_samples_split)
        self.min_samples_leaf = int(self.min_samples_leaf)
        if check_none(self.max_leaf_nodes):
            self.max_leaf_nodes = None
        else:
            self.max_leaf_nodes = int(self.max_leaf_nodes)
        self.min_weight_fraction_leaf = float(self.min_weight_fraction_leaf)
        self.min_impurity_decrease = float(self.min_impurity_decrease)

        self.estimator = DecisionTreeClassifier(
            criterion=self.criterion,
            max_depth=max_depth_factor,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_leaf_nodes=self.max_leaf_nodes,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            class_weight=self.class_weight,
            random_state=self.random_state)
        self.estimator.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        probas = self.estimator.predict_proba(X)
        return probas

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'DT',
                'name': 'Decision Tree Classifier',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None, optimizer='smac'):
        if optimizer == 'smac':
            cs = ConfigurationSpace()
            criterion = CategoricalHyperparameter(
                "criterion", ["gini", "entropy"], default_value="gini")
            max_depth_factor = UniformFloatHyperparameter(
                'max_depth_factor', 0., 2., default_value=0.5)
            min_samples_split = UniformIntegerHyperparameter(
                "min_samples_split", 2, 20, default_value=2)
            min_samples_leaf = UniformIntegerHyperparameter(
                "min_samples_leaf", 1, 20, default_value=1)
            min_weight_fraction_leaf = Constant("min_weight_fraction_leaf", 0.0)
            max_features = UnParametrizedHyperparameter('max_features', 1.0)
            max_leaf_nodes = UnParametrizedHyperparameter("max_leaf_nodes", "None")
            min_impurity_decrease = UnParametrizedHyperparameter('min_impurity_decrease', 0.0)

            cs.add_hyperparameters([criterion, max_features, max_depth_factor,
                                    min_samples_split, min_samples_leaf,
                                    min_weight_fraction_leaf, max_leaf_nodes,
                                    min_impurity_decrease])
            return cs
        elif optimizer == 'tpe':
            space = {'criterion': hp.choice('dt_criterion', ["gini", "entropy"]),
                     'max_depth_factor': hp.uniform('dt_max_depth_factor', 0, 2),
                     'min_samples_split': hp.randint('dt_min_samples_split', 19) + 2,
                     'min_samples_leaf': hp.randint('dt_min_samples_leaf', 20) + 1,
                     'min_weight_fraction_leaf': hp.choice('dt_min_weight_fraction_leaf', [0]),
                     'max_features': hp.choice('dt_max_features', [1.0]),
                     'max_leaf_nodes': hp.choice('dt_max_leaf_nodes', [None]),
                     'min_impurity_decrease': hp.choice('dt_min_impurity_decrease', [0.0])}

            init_trial = {'criterion': "gini",
                          'max_depth_factor': 0.5,
                          'min_samples_split': 2,
                          'min_samples_leaf': 1,
                          'min_weight_fraction_leaf': 0,
                          'max_features': 1,
                          'max_leaf_nodes': None,
                          'min_impurity_decrease': 0}
            return space



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", type=str, default='smac', help="The optimizer in Alpha-ML.")
    parser.add_argument("--run_count", type=int, default=200, help="The number of trials to in Alpha-ML.")
    parser.add_argument("--ensemble_size", type=int, default=12,
                        help="The number of base models to ensemble in Alpha-ML.")
    parser.add_argument("--k_fold", type=int, default=3, help="Folds for cross validation in Alpha-ML.")
    args = parser.parse_args()

    x, y, _ = load_data("iris")
    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2)
    dm = DataManager(x_train, y_train)
    add_classifier(UserDefinedDecisionTree)

    clf = Classifier(optimizer=args.optimizer,
                     k_fold=args.k_fold,
                     include_models=['UserDefinedDecisionTree'],
                     ensemble_method='bagging',
                     ensemble_size=args.ensemble_size,
                     save_dir='data/save_models')

    # clf.fit(dm, metric='acc', runcount=args.run_count)
    # Or we can use a user-defined scorer as metric input
    clf.fit(dm, metric='acc', runcount=args.run_count)

    print("The accuracy score is: ", clf.score(x_test, y_test))
