from alphaml.engine.components.ensemble.base_ensemble import *
from alphaml.engine.components.data_manager import DataManager
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold


class Stacking(BaseEnsembleModel):
    def __init__(self, model_info, ensemble_size, task_type, metric, evaluator, model_type='ml', meta_learner='xgboost',
                 kfold=3):
        super().__init__(model_info, ensemble_size, task_type, metric, evaluator, model_type)
        self.kfold = kfold
        # We use Xgboost as default meta-learner
        if self.task_type == CLASSIFICATION:
            if meta_learner == 'logistic':
                from sklearn.linear_model.logistic import LogisticRegression
                self.meta_learner = LogisticRegression(max_iter=1000)
            elif meta_learner == 'gb':
                from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
                self.meta_learner = GradientBoostingClassifier(learning_rate=0.05, subsample=0.7, max_depth=4,
                                                               n_estimators=250)
            elif meta_learner == 'xgboost':
                from xgboost import XGBClassifier
                self.meta_learner = XGBClassifier(max_depth=4, learning_rate=0.05, n_estimators=150)
        elif self.task_type == REGRESSION:
            if meta_learner == 'linear':
                from sklearn.linear_model import LinearRegression
                self.meta_learner = LinearRegression()
            elif meta_learner == 'xgboost':
                from xgboost import XGBRegressor
                self.meta_learner = XGBRegressor(max_depth=4, learning_rate=0.05, n_estimators=70)

    def fit(self, dm: DataManager):
        # Split training data for phase 1 and phase 2
        if self.task_type == CLASSIFICATION:
            kf = StratifiedKFold(n_splits=self.kfold)
        elif self.task_type == REGRESSION:
            kf = KFold(n_splits=self.kfold)
        feature_p2 = None
        if self.model_type == 'ml':
            # Train basic models using a part of training data
            for i, config in enumerate(self.config_list):
                for j, (train, test) in enumerate(kf.split(dm.train_X, dm.train_y)):
                    x_p1, x_p2, y_p1, _ = dm.train_X[train], dm.train_X[test], dm.train_y[train], dm.train_y[test]
                    estimator = self.get_estimator(config, x_p1, y_p1)
                    # The final list will contain self.kfold * self.ensemble_size models
                    self.ensemble_models.append(estimator)
                    pred = self.get_proba_predictions(estimator, x_p2)
                    if self.task_type == CLASSIFICATION:
                        n_dim = np.array(pred).shape[1]
                        if n_dim == 2:
                            # Binary classificaion
                            n_dim = 1
                        # Initialize training matrix for phase 2
                        if feature_p2 is None:
                            num_samples = len(train) + len(test)
                            feature_p2 = np.zeros((num_samples, self.ensemble_size * n_dim))
                        if n_dim == 1:
                            feature_p2[test, i * n_dim:(i + 1) * n_dim] = pred[:, 1:2]
                        else:
                            feature_p2[test, i * n_dim:(i + 1) * n_dim] = pred
                    elif self.task_type == REGRESSION:
                        shape = np.array(pred).shape
                        n_dim = shape[1]
                        # Initialize training matrix for phase 2
                        if feature_p2 is None:
                            num_samples = len(train) + len(test)
                            feature_p2 = np.zeros((num_samples, self.ensemble_size * n_dim))
                        feature_p2[test, i * n_dim:(i + 1) * n_dim] = pred
            # Train model for stacking using the other part of training data
            self.meta_learner.fit(feature_p2, dm.train_y)
        elif self.model_type == 'dl':
            pass
        return self

    def get_feature(self, X):
        # Predict the labels via stacking
        feature_p2 = None
        for i, model in enumerate(self.ensemble_models):
            pred = self.get_proba_predictions(model, X)
            if self.task_type == CLASSIFICATION:
                n_dim = np.array(pred).shape[1]
                if n_dim == 2:
                    n_dim = 1
                if feature_p2 is None:
                    num_samples = len(X)
                    feature_p2 = np.zeros((num_samples, self.ensemble_size * n_dim))
                index = i % self.kfold
                # Get average predictions
                if n_dim == 1:
                    feature_p2[:, index * n_dim:(index + 1) * n_dim] = feature_p2[:,
                                                                       index * n_dim:(index + 1) * n_dim] + \
                                                                       pred[:, 1:2] / self.kfold
                else:
                    feature_p2[:, index * n_dim:(index + 1) * n_dim] = feature_p2[:,
                                                                       index * n_dim:(index + 1) * n_dim] + \
                                                                       pred / self.kfold
            elif self.task_type == REGRESSION:
                shape = np.array(pred).shape
                n_dim = shape[1]
                # Initialize training matrix for phase 2
                if feature_p2 is None:
                    num_samples = len(X)
                    feature_p2 = np.zeros((num_samples, self.ensemble_size * n_dim))
                index = i % self.kfold
                # Get average predictions
                feature_p2[:, index * n_dim:(index + 1) * n_dim] = feature_p2[:,
                                                                   index * n_dim:(index + 1) * n_dim] + \
                                                                   pred / self.kfold
        return feature_p2

    def predict(self, X):
        feature_p2 = self.get_feature(X)
        # Get predictions from meta-learner
        final_pred = self.meta_learner.predict(feature_p2)
        return final_pred

    def predict_proba(self, X):
        feature_p2 = self.get_feature(X)
        # Get predictions from meta-learner
        final_pred = self.meta_learner.predict_proba(feature_p2)
        return final_pred
