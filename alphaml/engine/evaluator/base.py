import time
import logging
import multiprocessing
import pickle as pkl
import os
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold

from alphaml.engine.components.models.classification import _classifiers
from alphaml.engine.components.models.regression import _regressors
from alphaml.utils.save_ease import save_ease
from alphaml.utils.constants import FAILED


def update_config(config):
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


class BaseClassificationEvaluator(object):
    def __init__(self, val_size=0.33, kfold=None):
        self.val_size = val_size
        self.kfold = kfold
        self.data_manager = None
        self.metric_func = None
        self.logger = logging.getLogger(__name__)

    @save_ease(save_dir='./data/save_models')
    def __call__(self, config, **kwargs):
        # Build the corresponding estimator.
        classifier_type, estimator = self.set_config(config)
        save_path = kwargs['save_path']
        # TODO: how to parallelize.
        if hasattr(estimator, 'n_jobs'):
            setattr(estimator, 'n_jobs', multiprocessing.cpu_count() - 1)
        start_time = time.time()
        self.logger.info('<START TO FIT> %s' % classifier_type)
        self.logger.info('<CONFIG> %s' % config.get_dictionary())
        if self.kfold:
            if not isinstance(self.kfold, int) or self.kfold < 2:
                raise ValueError("Kfold must be an integer larger than 2!")

        if not self.kfold:
            # Split data
            data_X, data_y = self.data_manager.train_X, self.data_manager.train_y
            # TODO: Specify random_state
            train_X, val_X, train_y, val_y = train_test_split(data_X, data_y,
                                                              test_size=self.val_size,
                                                              stratify=data_y,
                                                              random_state=42, )

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
                    y_pred = estimator.predict_proba(val_X)[:, 1]
                    metric = self.metric_func(val_y, y_pred)
                else:
                    y_pred = estimator.predict(val_X)
                    metric = self.metric_func(val_y, y_pred)
            except ValueError:
                return -FAILED

            self.logger.info(
                '<EVALUATE %s-%.2f TAKES %.2f SECONDS>' % (classifier_type, 1 - metric, time.time() - start_time))
            # Turn it to a minimization problem.
            return 1 - metric

        else:
            kfold = StratifiedKFold(n_splits=self.kfold, shuffle=True)
            metric = 0
            for i, (train_index, valid_index) in enumerate(
                    kfold.split(self.data_manager.train_X, self.data_manager.train_y)):
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
                        y_pred = estimator.predict_proba(val_X)[:, 1]
                        metric += self.metric_func(val_y, y_pred) / self.kfold
                    else:
                        y_pred = estimator.predict(val_X)
                        metric += self.metric_func(val_y, y_pred) / self.kfold
                except ValueError:
                    return -FAILED

            self.logger.info('<FIT MODEL> finished!')
            self.logger.info(
                '<EVALUATE %s-%.2f TAKES %.2f SECONDS>' % (classifier_type, 1 - metric, time.time() - start_time))
            # Turn it to a minimization problem.
            return 1 - metric

    def set_config(self, config):
        if not hasattr(self, 'estimator'):
            # Build the corresponding estimator.
            params_num = len(config.get_dictionary().keys()) - 1
            classifier_type = config['estimator']
            estimator = _classifiers[classifier_type](*[None] * params_num)
        else:
            estimator = self.estimator
        config = update_config(config)
        estimator.set_hyperparameters(config)
        return classifier_type, estimator

    @save_ease(save_dir='data/save_models')
    def fit(self, config, **kwargs):
        # Build the corresponding estimator.
        save_path = kwargs['save_path']
        _, estimator = self.set_config(config)
        # Fit the estimator on the training data.
        estimator.fit(self.data_manager.train_X, self.data_manager.train_y)
        with open(save_path, 'wb') as f:
            pkl.dump(estimator, f)
            self.logger.info("Estimator retrained!")

    # Do not remove config
    @save_ease(save_dir='data/save_models')
    def predict(self, config, test_X=None, **kwargs):
        save_path = kwargs['save_path']
        assert os.path.exists(save_path)
        with open(save_path, 'rb') as f:
            estimator = pkl.load(f)
            self.logger.info("Estimator loaded from " + save_path)

        # Inference.
        if test_X is None:
            test_X = self.data_manager.test_X

        y_pred = estimator.predict(test_X)
        return y_pred

    @save_ease(save_dir='data/save_models')
    def predict_proba(self, config, test_X=None, **kwargs):
        save_path = kwargs['save_path']
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
    def __init__(self, val_size=0.33, kfold=None):
        self.val_size = val_size
        self.kfold = kfold
        self.data_manager = None
        self.metric_func = None
        self.logger = logging.getLogger(__name__)

    @save_ease(save_dir='./data/save_models')
    def __call__(self, config, **kwargs):
        # Build the corresponding estimator.
        regressor_type, estimator = self.set_config(config)
        save_path = kwargs['save_path']
        # TODO: how to parallelize.
        if hasattr(estimator, 'n_jobs'):
            setattr(estimator, 'n_jobs', multiprocessing.cpu_count() - 1)
        start_time = time.time()
        self.logger.info('<START TO FIT> %s' % regressor_type)
        self.logger.info('<CONFIG> %s' % config.get_dictionary())
        if self.kfold:
            if not isinstance(self.kfold, int) or self.kfold < 2:
                raise ValueError("Kfold must be an integer larger than 2!")

        if not self.kfold:
            # Split data
            data_X, data_y = self.data_manager.train_X, self.data_manager.train_y
            # TODO: Specify random_state
            train_X, val_X, train_y, val_y = train_test_split(data_X, data_y, test_size=self.val_size, random_state=42)

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
                return -FAILED

            self.logger.info(
                '<EVALUATE %s-%.2f TAKES %.2f SECONDS>' % (regressor_type, metric, time.time() - start_time))
            return metric
        else:
            kfold = KFold(n_splits=self.kfold, shuffle=True)
            metric = 0
            for i, (train_index, valid_index) in enumerate(
                    kfold.split(self.data_manager.train_X, self.data_manager.train_y)):
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
                    return -FAILED

            self.logger.info('<FIT MODEL> finished!')
            self.logger.info(
                '<EVALUATE %s-%.2f TAKES %.2f SECONDS>' % (regressor_type, metric, time.time() - start_time))
            return metric

    def set_config(self, config):
        if not hasattr(self, 'estimator'):
            # Build the corresponding estimator.
            params_num = len(config.get_dictionary().keys()) - 1
            regressor_type = config['estimator']
            estimator = _regressors[regressor_type](*[None] * params_num)
        else:
            estimator = self.estimator
        config = update_config(config)
        estimator.set_hyperparameters(config)
        return regressor_type, estimator

    @save_ease(save_dir='data/save_models')
    def fit(self, config, **kwargs):
        # Build the corresponding estimator.
        save_path = kwargs['save_path']
        _, estimator = self.set_config(config)
        # Fit the estimator on the training data.
        estimator.fit(self.data_manager.train_X, self.data_manager.train_y)
        with open(save_path, 'wb') as f:
            pkl.dump(estimator, f)
            self.logger.info("Estimator retrained!")

    # Do not remove config
    @save_ease(save_dir='data/save_models')
    def predict(self, config, test_X=None, **kwargs):
        save_path = kwargs['save_path']
        assert os.path.exists(save_path)
        with open(save_path, 'rb') as f:
            estimator = pkl.load(f)
            print("Estimator loaded from", save_path)

        # Inference.
        if test_X is None:
            test_X = self.data_manager.test_X

        y_pred = estimator.predict(test_X)
        return y_pred
