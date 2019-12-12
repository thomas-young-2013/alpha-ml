import pandas as pd
import os
from alphaml.engine.components.data_manager import DataManager


class BaseEstimator(object):
    """Base class for all estimators in alpha-ml. """

    def __init__(
            self,
            optimizer='mono_smbo',
            time_budget=3600,
            each_run_budget=360,
            ensemble_method='none',
            ensemble_size=50,
            cross_valid=True,
            k_fold=3,
            memory_limit=1024,
            seed=42,
            include_models=None,
            exclude_models=None,
            save_dir='./data/save_models',
            output_dir=None):
        """

        :param optimizer: str, algorithm hyper-parameter optimization
        :param time_budget: int, total time limit
        :param each_run_budget: int, time limit for each model
        :param ensemble_method: str, algorithm for model ensemble
        :param ensemble_size: int, number of models participating model ensemble
        :param cross_valid: bool
        :param k_fold: int
        :param memory_limit: int
        :param seed: int, random seed
        :param include_models: list, names of models included.
        :param exclude_models: list, names of models excluded.
        :param save_dir: str, path to save models
        :param output_dir: str
        """
        self.optimizer_type = optimizer
        self.time_budget = time_budget
        self.each_run_budget = each_run_budget
        self.ensemble_method = ensemble_method
        self.ensemble_size = ensemble_size
        self.memory_limit = memory_limit
        self.include_models = include_models
        self.exclude_models = exclude_models
        self.cross_valid = cross_valid
        self.k_fold = k_fold
        self.seed = seed
        self.save_dir = save_dir
        self.output_dir = output_dir
        self.pre_pipeline = None
        self._ml_engine = None

        if self.ensemble_size == 'ensemble_selection' and self.cross_valid is True:
            raise ValueError("Ensemble selection can not work with cv.")
        # Delete the temporary model files.
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.save_dir = save_dir
        ls = os.listdir(self.save_dir)
        for item in ls:
            c_path = os.path.join(self.save_dir, item)
            if os.path.isfile(c_path):
                os.remove(c_path)

    def build_engine(self):
        """Build AutoML controller"""
        engine = self.get_automl()(
            time_budget=self.time_budget,
            each_run_budget=self.each_run_budget,
            ensemble_method=self.ensemble_method,
            ensemble_size=self.ensemble_size,
            memory_limit=self.memory_limit,
            include_models=self.include_models,
            exclude_models=self.exclude_models,
            optimizer_type=self.optimizer_type,
            cross_valid=self.cross_valid,
            k_fold=self.k_fold,
            save_dir=self.save_dir,
            seed=self.seed
        )
        return engine

    def fit(self, data, **kwargs):
        assert data is not None and isinstance(data, (DataManager, pd.DataFrame))
        self._ml_engine = self.build_engine()
        self._ml_engine.fit(data, **kwargs)
        return self

    def predict(self, X, batch_size=None, n_jobs=1):
        return self._ml_engine.predict(X, batch_size=batch_size, n_jobs=n_jobs)

    def score(self, X, y):
        return self._ml_engine.score(X, y)

    def predict_proba(self, X, batch_size=None, n_jobs=1):
        return self._ml_engine.predict_proba(X, batch_size=None, n_jobs=n_jobs)

    def get_automl(self):
        raise NotImplementedError()

    def show_info(self):
        self._ml_engine.show_info()