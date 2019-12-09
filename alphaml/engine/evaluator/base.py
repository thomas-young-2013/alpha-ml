import time
import logging
import multiprocessing
import pickle as pkl
import os
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

from alphaml.engine.components.models.classification import _classifiers
from alphaml.engine.components.models.regression import _regressors
from alphaml.utils.save_ease import save_ease
from alphaml.utils.constants import *


def get_smac_config(config):
    """
    Convert a configuration for SMAC into dictionary.
    :param config: A configuration in hyper-parameter space for SMAC
    :return: config_dict: dictionary
    """
    config_dict = {}
    for param in config:
        if param == 'estimator':
            continue
        if param.find(":") != -1:
            value = config[param]
            new_name = param.split(':')[-1]
            config_dict[new_name] = value
        else:
            config_dict[param] = config[param]
    return config_dict


def get_tpe_config(config):
    """
    Convert a configuration for TPE into dictionary.
    :param config: A configuration in hyper-parameter space for TPE
    :return: str, dictionary
    """
    assert isinstance(config, dict)
    classifier_type = config['estimator'][0]
    space = config['estimator'][1]
    return classifier_type, space


class BaseClassificationEvaluator(object):
    """ A class to evaluate configurations for classification"""

    def __init__(self, optimizer='smac', val_size=0.33, kfold=None, save_dir='./data/save_models', random_state=None):
        """
        :param optimizer: Algorithm for hyper-parameter tuning
        :param val_size: float from (0,1), used if kfold is None
        :param kfold: int larger than 2
        :param save_dir: str, path to save and load models
        :param random_state: int
        """
        self.optimizer = optimizer
        self.val_size = val_size
        self.kfold = kfold
        self.data_manager = None
        self.metric_func = None
        self.save_dir = save_dir
        self.seed = random_state
        self.logger = logging.getLogger(__name__)

    @save_ease(None)
    def __call__(self, config, **kwargs):
        """
        Get the performance of a given configuration
        :param config: A configuration in hyper-parameter space for SMAC or TPE
        :return: performance: float
        """
        # Build the corresponding estimator.
        classifier_type, estimator = self.set_config(config, self.optimizer)

        save_path = os.path.join(self.save_dir, kwargs['save_path'])
        # TODO: how to parallelize.
        if hasattr(estimator, 'n_jobs'):
            setattr(estimator, 'n_jobs', multiprocessing.cpu_count() - 1)
        start_time = time.time()
        self.logger.info('<START TO FIT> %s' % classifier_type)
        if self.optimizer == 'smac':
            self.logger.info('<CONFIG> %s' % config.get_dictionary())
        elif self.optimizer == 'tpe':
            self.logger.info('<CONFIG> %s' % config)
        if self.kfold:
            if not isinstance(self.kfold, int) or self.kfold < 2:
                raise ValueError("Kfold must be an integer larger than 2!")
        data_X, data_y = self.data_manager.train_X, self.data_manager.train_y
        encoder = OneHotEncoder()
        if len(data_y.shape) == 1:
            reshape_y = np.reshape(data_y, (len(data_y), 1))
            encoder.fit(reshape_y)
        if not self.kfold:
            # Split data
            # TODO: Specify random_state
            train_X, val_X, train_y, val_y = train_test_split(data_X, data_y,
                                                              test_size=self.val_size,
                                                              stratify=data_y,
                                                              random_state=self.seed)

            # Fit the estimator on the training data.
            estimator.fit(train_X, train_y)
            self.logger.info('<FIT MODEL> finished!')
            with open(save_path, 'wb') as f:
                pkl.dump(estimator, f)
                self.logger.info('<MODEL SAVED IN %s>' % save_path)

            # In case of failed estimator
            try:
                # Validate it on val data.
                if self.metric_func == roc_auc_score:
                    y_pred = estimator.predict_proba(val_X)
                    if len(val_y.shape) == 1:
                        val_y = encoder.transform(np.reshape(val_y, (len(val_y), 1))).toarray()
                else:
                    y_pred = estimator.predict(val_X)
                metric = self.metric_func(val_y, y_pred)
            except ValueError:
                self.logger.info("<Fit Model> failed!")
                return FAILED
        else:
            kfold = StratifiedKFold(n_splits=self.kfold, shuffle=True)
            metric = 0
            for i, (train_index, valid_index) in enumerate(kfold.split(data_X, data_y)):
                train_X = self.data_manager.train_X[train_index]
                val_X = self.data_manager.train_X[valid_index]
                train_y = self.data_manager.train_y[train_index]
                val_y = self.data_manager.train_y[valid_index]

                # Fit the estimator on the training data.
                estimator.fit(train_X, train_y)
                self.logger.info('<FIT MODEL> %d/%d finished!' % (i + 1, self.kfold))
                with open(save_path, 'wb') as f:
                    pkl.dump(estimator, f)
                    self.logger.info('<MODEL SAVED IN %s>' % save_path)

                # In case of failed estimator
                try:
                    # Validate it on val data.
                    if self.metric_func == roc_auc_score:
                        y_pred = estimator.predict_proba(val_X)
                        if len(val_y.shape) == 1:
                            val_y = encoder.transform(np.reshape(val_y, (len(val_y), 1))).toarray()
                        metric += self.metric_func(val_y, y_pred) / self.kfold
                    else:
                        y_pred = estimator.predict(val_X)
                        metric += self.metric_func(val_y, y_pred) / self.kfold
                except ValueError:
                    self.logger.info("<Fit Model> failed!")
                    return FAILED

        self.logger.info(
            '<EVALUATE %s-%.2f TAKES %.2f SECONDS>' % (classifier_type, metric, time.time() - start_time))
        # Turn it to a minimization problem.
        return - metric

    def set_config(self, config, optimizer):
        """
        Build an sklearn classifier
        :param config: A configuration in hyper-parameter space for SMAC or TPE
        :param optimizer: Algorithm for hyper-parameter tuning
        :return: str, sklearn classifier
        """
        if optimizer == 'smac':
            if not hasattr(self, 'estimator'):
                # Build the corresponding estimator.
                params_num = len(config.get_dictionary().keys()) - 1
                classifier_type = config['estimator']
                estimator = _classifiers[classifier_type](*[None] * params_num)
            else:
                estimator = self.estimator
                classifier_type = None
            config = get_smac_config(config)
            estimator.set_hyperparameters(config)
            return classifier_type, estimator
        elif optimizer == 'tpe':
            assert isinstance(config, dict)
            classifier_type, config = get_tpe_config(config)
            if not hasattr(self, 'estimator'):
                # Build the corresponding estimator.
                params_num = len(config.keys())
                estimator = _classifiers[classifier_type](*[None] * params_num)
            else:
                estimator = self.estimator
            estimator.set_hyperparameters(config)
            return classifier_type, estimator

    @save_ease()
    def fit(self, config, **kwargs):
        """
        Build and fit an sklearn classifier
        :param config: A configuration in hyper-parameter space for SMAC or TPE
        :return: self
        """
        # Build the corresponding estimator.
        save_path = os.path.join(self.save_dir, kwargs['save_path'])
        _, estimator = self.set_config(config, self.optimizer)
        # Fit the estimator on the training data.
        estimator.fit(self.data_manager.train_X, self.data_manager.train_y)
        with open(save_path, 'wb') as f:
            pkl.dump(estimator, f)
            self.logger.info("Estimator retrained!")
        return self

    # Do not remove config
    @save_ease(None)
    def predict(self, config, test_X=None, **kwargs):
        """
        Load an sklearn classifier and predict classes for X.
        :param config: A configuration in hyper-parameter space for SMAC or TPE
        :param test_X: Array-like or sparse matrix of shape = [n_samples, n_features]
        :return: y_pred: Array of shape = [n_samples]
        """
        save_path = os.path.join(self.save_dir, kwargs['save_path'])
        assert os.path.exists(save_path)
        with open(save_path, 'rb') as f:
            estimator = pkl.load(f)
            self.logger.info("Estimator loaded from " + save_path)

        # Inference.
        if test_X is None:
            test_X = self.data_manager.test_X

        y_pred = estimator.predict(test_X)
        return y_pred

    @save_ease(None)
    def predict_proba(self, config, test_X=None, **kwargs):
        """
        Load an sklearn classifier and predict probabilities of classes for all samples X.
        :param config: A configuration in hyper-parameter space for SMAC or TPE
        :param test_X: Array-like or sparse matrix of shape = [n_samples, n_features]
        :return: y_pred : Array of shape = [n_samples, n_classes]
        """
        save_path = os.path.join(self.save_dir, kwargs['save_path'])
        assert os.path.exists(save_path)
        with open(save_path, 'rb') as f:
            estimator = pkl.load(f)
            self.logger.info("Estimator loaded from " + save_path)

        # Inference.
        if test_X is None:
            test_X = self.data_manager.test_X

        y_pred = estimator.predict_proba(test_X)
        return y_pred


class BaseRegressionEvaluator(object):
    """ A class to evaluate configurations for classification"""

    def __init__(self, optimizer='smac', val_size=0.33, kfold=None, save_dir='./data/save_models', random_state=None):
        """
        :param optimizer: algorithm for hyper-parameter tuning
        :param val_size: float from (0,1), used if kfold is None
        :param kfold: int larger than 2
        :param save_dir: str, path to save and load models
        :param random_state: int
        """
        self.optimizer = optimizer
        self.val_size = val_size
        self.kfold = kfold
        self.data_manager = None
        self.metric_func = None
        self.save_dir = save_dir
        self.seed = random_state
        self.logger = logging.getLogger(__name__)

    @save_ease(None)
    def __call__(self, config, **kwargs):
        """
        Get the performance of a given configuration
        :param config: A configuration in hyper-parameter space for SMAC or TPE
        :return: performance: float
        """
        # Build the corresponding estimator.
        regressor_type, estimator = self.set_config(config, self.optimizer)
        save_path = os.path.join(self.save_dir, kwargs['save_path'])
        # TODO: how to parallelize.
        if hasattr(estimator, 'n_jobs'):
            setattr(estimator, 'n_jobs', multiprocessing.cpu_count() - 1)
        start_time = time.time()
        self.logger.info('<START TO FIT> %s' % regressor_type)
        if self.optimizer == 'smac':
            self.logger.info('<CONFIG> %s' % config.get_dictionary())
        elif self.optimizer == 'tpe':
            self.logger.info('<CONFIG> %s' % config)
        if self.kfold:
            if not isinstance(self.kfold, int) or self.kfold < 2:
                raise ValueError("Kfold must be an integer larger than 2!")

        data_X, data_y = self.data_manager.train_X, self.data_manager.train_y
        if not self.kfold:
            # Split data
            # TODO: Specify random_state
            train_X, val_X, train_y, val_y = train_test_split(data_X, data_y, test_size=self.val_size,
                                                              random_state=self.seed)

            # Fit the estimator on the training data.
            estimator.fit(train_X, train_y)
            self.logger.info('<FIT MODEL> finished!')
            with open(save_path, 'wb') as f:
                pkl.dump(estimator, f)
                self.logger.info('<MODEL SAVED IN %s>' % save_path)

            # In case of failed estimator
            try:
                # Validate it on val data.
                y_pred = estimator.predict(val_X)
                metric = self.metric_func(val_y, y_pred)
            except ValueError:
                self.logger.info("<Fit Model> failed!")
                return FAILED
        else:
            kfold = KFold(n_splits=self.kfold, shuffle=True)
            metric = 0
            for i, (train_index, valid_index) in enumerate(kfold.split(data_X, data_y)):
                train_X = self.data_manager.train_X[train_index]
                val_X = self.data_manager.train_X[valid_index]
                train_y = self.data_manager.train_y[train_index]
                val_y = self.data_manager.train_y[valid_index]

                # Fit the estimator on the training data.
                estimator.fit(train_X, train_y)
                self.logger.info('<FIT MODEL> %d/%d finished!' % (i + 1, self.kfold))
                with open(save_path, 'wb') as f:
                    pkl.dump(estimator, f)
                    self.logger.info('<MODEL SAVED IN %s>' % save_path)

                # In case of failed estimator
                try:
                    # Validate it on val data.
                    y_pred = estimator.predict(val_X)
                    metric += self.metric_func(val_y, y_pred) / self.kfold
                except ValueError:
                    self.logger.info("<Fit Model> failed!")
                    return FAILED

        self.logger.info(
            '<EVALUATE %s-%.2f TAKES %.2f SECONDS>' % (regressor_type, metric, time.time() - start_time))
        return metric

    def set_config(self, config, optimizer):
        """
        Build an sklearn regressor
        :param config: A configuration in hyper-parameter space for SMAC or TPE
        :param optimizer: Algorithm for hyper-parameter tuning
        :return: str, sklearn regressor
        """
        if optimizer == 'smac':
            if not hasattr(self, 'estimator'):
                # Build the corresponding estimator.
                params_num = len(config.get_dictionary().keys()) - 1
                regressor_type = config['estimator']
                estimator = _regressors[regressor_type](*[None] * params_num)
            else:
                estimator = self.estimator
                regressor_type = None
            config = get_smac_config(config)
            estimator.set_hyperparameters(config)
            return regressor_type, estimator
        elif optimizer == 'tpe':
            assert isinstance(config, dict)
            print(config)
            regressor_type, config = get_tpe_config(config)
            if not hasattr(self, 'estimator'):
                # Build the corresponding estimator.
                params_num = len(config.keys())
                estimator = _regressors[regressor_type](*[None] * params_num)
            else:
                estimator = self.estimator
            estimator.set_hyperparameters(config)
            return regressor_type, estimator

    @save_ease(None)
    def fit(self, config, **kwargs):
        """
        Build and fit an sklearn regressor
        :param config: A configuration in hyper-parameter space for SMAC or TPE
        :return: self
        """
        # Build the corresponding estimator.
        save_path = os.path.join(self.save_dir, kwargs['save_path'])
        _, estimator = self.set_config(config, self.optimizer)
        # Fit the estimator on the training data.
        estimator.fit(self.data_manager.train_X, self.data_manager.train_y)
        with open(save_path, 'wb') as f:
            pkl.dump(estimator, f)
            self.logger.info("Estimator retrained!")
        return self

    # Do not remove config
    @save_ease(None)
    def predict(self, config, test_X=None, **kwargs):
        """
        Load an sklearn regressor and make predictions for X.
        :param config: A configuration in hyper-parameter space for SMAC or TPE
        :param test_X: Array-like or sparse matrix of shape = [n_samples, n_features]
        :return: y_pred: Array of shape = [n_samples]
        """
        save_path = os.path.join(self.save_dir, kwargs['save_path'])
        assert os.path.exists(save_path)
        with open(save_path, 'rb') as f:
            estimator = pkl.load(f)
            print("Estimator loaded from", save_path)

        # Inference.
        if test_X is None:
            test_X = self.data_manager.test_X

        y_pred = estimator.predict(test_X)
        return y_pred
