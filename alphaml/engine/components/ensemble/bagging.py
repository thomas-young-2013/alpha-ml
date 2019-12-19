from alphaml.engine.components.ensemble.base_ensemble import *
from alphaml.engine.components.data_manager import DataManager
import numpy as np
from functools import reduce


class Bagging(BaseEnsembleModel):
    def __init__(self, model_info, ensemble_size, task_type, metric, evaluator, model_type='ml', save_dir=None,
                 random_state=None):
        super().__init__(model_info=model_info,
                         ensemble_size=ensemble_size,
                         task_type=task_type,
                         metric=metric,
                         evaluator=evaluator,
                         model_type=model_type,
                         save_dir=save_dir,
                         random_state=random_state)

    def fit(self, dm: DataManager):
        # Train the basic models on this training set.
        if self.model_type == 'ml':
            for config in self.config_list:
                estimator = self.get_estimator(config, dm.train_X, dm.train_y, if_show=True)
                self.ensemble_models.append(estimator)
        elif self.model_type == 'dl':
            pass
        return self

    def predict(self, X):
        model_pred_list = []
        final_pred = []
        # Get predictions from each model
        for model in self.ensemble_models:
            pred = self.get_proba_predictions(model, X)
            num_outputs = len(pred)
            model_pred_list.append(pred)

        if self.task_type == CLASSIFICATION:
            # Calculate the average of predictions
            for i in range(num_outputs):
                sample_pred_list = [model_pred[i] for model_pred in model_pred_list]
                pred_average = reduce(lambda x, y: x + y, sample_pred_list) / len(sample_pred_list)
                final_pred.append(pred_average)
            final_pred = np.argmax(final_pred, axis=-1)

        elif self.task_type == REGRESSION:
            # Calculate the average of predictions
            for i in range(num_outputs):
                sample_pred_list = [model_pred[i] for model_pred in model_pred_list]
                pred_average = reduce(lambda x, y: x + y, sample_pred_list) / len(sample_pred_list)
                final_pred.append(pred_average)

        return np.array(final_pred)

    def predict_proba(self, X):
        # Predict the labels via voting results from the basic models.
        model_pred_list = []
        final_pred = []
        num_outputs = None
        # Get predictions from each model
        for model in self.ensemble_models:
            pred = self.get_proba_predictions(model, X)
            if num_outputs is None:
                num_outputs = len(pred)
            model_pred_list.append(pred)

        assert num_outputs is not None  # Ensemble model contains at least one base model
        # Calculate the average of predictions
        for i in range(num_outputs):
            sample_pred_list = [model_pred[i] for model_pred in model_pred_list]
            pred_average = reduce(lambda x, y: x + y, sample_pred_list) / len(sample_pred_list)
            final_pred.append(pred_average)

        return np.array(final_pred)
