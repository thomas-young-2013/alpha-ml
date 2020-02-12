from alphaml.utils.constants import *
from alphaml.utils.save_ease import save_ease

import os
import pickle as pkl
import functools
import math

from alphaml.utils.logging_utils import get_logger


class BaseEnsembleModel(object):
    """Base class for model ensemble"""

    def __init__(self, model_info, ensemble_size, task_type, metric, evaluator, model_type='ml', threshold=0.2,
                 save_dir=None, random_state=None):
        """

        :param model_info: tuple of lists recording configurations and their performance
        :param ensemble_size: int, number of models participating model ensemble
        :param task_type: str
        :param metric: function,
        :param evaluator: Evaluator
        :param model_type: str
        :param threshold: float, threshold to obsolete candidates
        :param save_dir: str, path to load models
        :param if_show: bool, print ensemble candidates
        :param random_state: int
        """
        self.model_info = model_info
        self.model_type = model_type
        self.metric = metric
        self.evaluator = evaluator
        self.ensemble_models = list()
        self.threshold = threshold
        self.logger = get_logger(__name__)
        self.save_dir = save_dir
        self.seed = random_state

        if task_type in ['binary', 'multiclass', 'img_binary', 'img_multiclass', 'img_multilabel-indicator']:
            self.task_type = CLASSIFICATION
        elif task_type in ['continuous']:
            self.task_type = REGRESSION
        else:
            raise ValueError('Undefined Task Type: %s' % task_type)

        if len(model_info[0]) < ensemble_size:
            self.ensemble_size = len(model_info[0])
        else:
            self.ensemble_size = ensemble_size

        # Determine the best basic models (the best for each algorithm) from models_infos.
        index_list = []
        model_len = len(self.model_info[1])

        def cmp(x, y):
            if self.model_info[1][x] > self.model_info[1][y]:
                return -1
            elif self.model_info[1][x] == self.model_info[1][y]:
                return 0
            else:
                return 1

        # Get the top-k models for each algorithm
        best_performance = float('-INF')
        try:
            # SMAC
            estimator_set = set([self.model_info[0][i]['estimator'] for i in range(model_len)])
            top_k = math.ceil(ensemble_size / len(estimator_set))
            # Get the estimator with the best performance for each algorithm
            for estimator in estimator_set:
                id_list = []
                for i in range(model_len):
                    if self.model_info[0][i]['estimator'] == estimator:
                        if self.model_info[1][i] != -FAILED:
                            if best_performance < self.model_info[1][i]:
                                best_performance = self.model_info[1][i]
                            id_list.append(i)
                sort_list = sorted(id_list, key=functools.cmp_to_key(cmp))
                index_list.extend(sort_list[:top_k])

        except:
            # Hyperopt
            estimator_set = set(self.model_info[0][i]['estimator'][0] for i in range(model_len))
            top_k = math.ceil(ensemble_size / len(estimator_set))
            for estimator in estimator_set:
                id_list = []
                for i in range(model_len):
                    if self.model_info[0][i]['estimator'][0] == estimator:
                        if self.model_info[1][i] != -FAILED:
                            if best_performance < self.model_info[1][i]:
                                best_performance = self.model_info[1][i]
                            id_list.append(i)
                sort_list = sorted(id_list, key=functools.cmp_to_key(cmp))
                index_list.extend(sort_list[:top_k])

        # Obsolete models which perform badly compared to the best model
        self.config_list = []
        for i in index_list:
            if abs((best_performance - self.model_info[1][i]) / best_performance) < self.threshold:
                self.config_list.append(i)

    def fit(self, dm):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def predict_each(self, X):
        raise NotImplementedError

    @save_ease(None)
    def get_estimator(self, config, x, y, config_idx, if_load=False, if_show=False, **kwargs):
        """
        Build a sklearn estimator and fit it with training data
        :param config_idx: int, configuration index
        :param if_show: bool
        :param config: configuration
        :param x: Array-like or sparse matrix of shape = [n_samples, n_features]
        :param y: Array of shape = [n_samples] or [n_samples, n_classes]
        :param if_load: bool
        :return: sklearn model
        """
        save_path = os.path.join(self.save_dir, kwargs['save_path'])
        if if_load and os.path.exists(save_path):
            with open(save_path, 'rb') as f:
                estimator = pkl.load(f)
                # self.logger.info("Estimator loaded from " + save_path)

        else:
            _, estimator = self.evaluator.set_config(config, self.evaluator.optimizer)
            estimator.fit(x, y)
            with open(save_path, 'wb') as f:
                pkl.dump(estimator, f)
            if if_show:
                self.logger.info('--------Base Model Info Start---------')
                self.logger.info(str(config))
                self.logger.info(
                    "Validation performance (Negative if minimize): " + str(self.model_info[1][config_idx]))
                self.logger.info("Estimator retrained and saved in " + save_path)
                self.logger.info('--------Base Model Info End----------')
        return estimator

    def get_proba_predictions(self, estimator, X):
        """
        Predict probabilities of classes for all samples X.
        :param estimator: sklearn model
        :param X: Array-like or sparse matrix of shape = [n_samples, n_features]
        :return: Array of shape = [n_samples, n_classes]
        """
        if self.task_type == CLASSIFICATION:
            return estimator.predict_proba(X)
        elif self.task_type == REGRESSION:
            pred = estimator.predict(X)
            shape = pred.shape
            if len(shape) == 1:
                pred = pred.reshape((shape[0], 1))
            return pred

    def get_base_config(self):
        return self.config_list
