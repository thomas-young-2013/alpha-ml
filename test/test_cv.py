import warnings
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--run_count', type=int, default=300)
parser.add_argument('--ensemble_size', type=int, default=12)
args = parser.parse_args()

warnings.filterwarnings("ignore")
sys.path.append("/home/daim_gpu/sy/AlphaML")

'''
Available models:
adaboost, decision_tree, extra_trees, gaussian_nb, gradient_boosting, k_nearest_neighbors, lda, liblinear_svc,
libsvm_svc, logistic_regression, mlp, passive_aggressive, qda, random_forest, sgd, xgboost
'''


def test_cash_module():
    from alphaml.engine.components.data_manager import DataManager
    from alphaml.estimators.classifier import Classifier
    from sklearn.datasets import load_iris
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score
    import random

    data_x, data_y = load_iris(return_X_y=True)
    kfold = StratifiedKFold(n_splits=5, shuffle=True)
    result = []
    for train_index, valid_index in kfold.split(data_x, data_y):
        train_X = data_x[train_index]
        val_X = data_x[valid_index]
        train_y = data_y[train_index]
        val_y = data_y[valid_index]
        dm = DataManager(train_X, train_y)
        cls = Classifier(
            include_models=['liblinear_svc', 'libsvm_svc', 'random_forest', 'logistic_regression', 'mlp', 'xgboost'],
            optimizer='smbo',
            cross_valid=False,
            ensemble_method='bagging',
            ensemble_size=args.ensemble_size,
            save_dir='data/save_models'
        )
        cls.fit(dm, metric='acc', runcount=args.run_count)
        pred_y = cls.predict(val_X)
        result.append(accuracy_score(val_y, pred_y))

    print(result)


if __name__ == "__main__":
    test_cash_module()
