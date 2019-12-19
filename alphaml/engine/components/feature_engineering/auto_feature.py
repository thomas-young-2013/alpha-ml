import numpy as np

from alphaml.engine.components.feature_engineering.auto_cross import AutoCross
from alphaml.engine.components.feature_engineering.selector import RandomForestSelector
from alphaml.utils.metrics_util import get_metric

from sklearn.linear_model import LogisticRegression


class AutoFeature:

    def __init__(self, metrics, strategy="auto_cross"):
        self.metrics = metrics
        self.strategy = strategy
        self.solver = None

    def fit(self, dm, generated_num=50):
        if self.strategy == "auto_cross":
            self.solver = AutoCross(generated_num, self.metrics)
        elif self.strategy == "auto_deep":
            self.solver = None
        elif self.strategy == "auto_learn":
            self.solver = None
        else:
            raise ValueError("AutoFeature supports only strategies in ['autocross', 'auto_deep', 'auto_learn', "
                             "got", self.strategy, ".")
        self.solver.fit(dm.train_X, dm.val_X, dm.train_y, dm.val_y, dm.test_X)

    def transform(self, dm):
        if self.solver is None:
            raise ValueError("The AutoFeature has not been fitted!!!")
        dm.train_X = self.solver.transform(dm.train_X)
        dm.val_X = self.solver.transform(dm.val_X)
        if dm.test_X is not None:
            dm.test_X = self.solver.transform(dm.test_X)
        return dm

    def _selection_process(self, dm):
        generated_train_data, generated_valid_data, generated_test_data = self.solver.transform()
        features = np.load("features_27.npz")
        generated_train_data, generated_valid_data, generated_test_data = features["train"], features["valid"], None
        feature_num = dm.train_X.shape[1]

        if feature_num < 20:
            dm.train_X = generated_train_data
            dm.val_X = generated_valid_data
            dm.test_X = generated_test_data
        else:
            print("start selection process...............")
            selector = RandomForestSelector()
            selector.fit(dm.train_X, dm.train_y)

            lr = LogisticRegression()

            best_perf = get_metric(self.metrics)
            best_k = 0
            for percentile in range(1, 10):
                k = int((percentile / 10.0) * feature_num)
                selected_train_data = selector.transform(dm.train_X, k)
                selected_valid_data = selector.transform(dm.val_X, k)

                perf = get_metric()
                if perf <= best_perf:
                    break
                else:
                    print("selected pertentile:", percentile, "perf:", best_perf, flush=True)
                    best_perf = perf
                    best_k = k
            if best_k != 0:
                dm.train_X = np.hstack((generated_train_data, selector.transform(dm.train_X, best_k)))
                dm.val_X = np.hstack((generated_valid_data, selector.transform(dm.val_X, best_k)))
                if dm.test_X is not None:
                    dm.test_X = np.hstack((generated_test_data, selector.transform(dm.test_X, best_k)))

            else:
                dm.train_X = generated_train_data
                dm.val_X = generated_valid_data
                dm.test_X = generated_test_data
