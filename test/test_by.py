import warnings
import numpy as np
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
    import random
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder

    result = []
    for i in range(1):
        import xlrd
        sheet = xlrd.open_workbook("ybai_Keratoconus_TJ_20190425.xlsx")
        sheet = sheet.sheet_by_index(0)
        nrows = sheet.nrows
        X_train = []
        y_train = []

        for i in range(1, nrows):
            X_train.append(sheet.row_values(i, start_colx=1))
            y_train.append(int(sheet.cell_value(i, 0)))

        encoder = OneHotEncoder()
        encoder.fit(np.reshape(y_train, (len(y_train), 1)))
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train)

        dm = DataManager(X_train, y_train)
        cls = Classifier(
            # include_models=['liblinear_svc', 'libsvm_svc', 'xgboost', 'random_forest', 'logistic_regression', 'mlp'],
            optimizer='smbo',
            ensemble_method='bagging',
            ensemble_size=args.ensemble_size,
        )
        cls.fit(dm, metric='auc', runcount=args.run_count)

        pred = cls.predict_proba(X_test)
        print(pred)
        y_test = encoder.transform(np.reshape(y_test, (len(y_test), 1))).toarray()
        result.append(roc_auc_score(y_test, pred))
        print(result)

        import pickle
        with open('result.pkl', 'wb') as f:
            pickle.dump(result, f)


if __name__ == "__main__":
    test_cash_module()
