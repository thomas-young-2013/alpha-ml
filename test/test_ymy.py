import warnings
from sklearn.preprocessing import LabelEncoder
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
    result = []
    for i in range(1):
        import xlrd
        sheet = xlrd.open_workbook("lyqdata.xlsx")
        sheet = sheet.sheet_by_index(0)
        nrows = sheet.nrows
        X_train = []
        y_train = []
        for i in range(2, nrows):
            X_train.append(sheet.row_values(i, start_colx=1))
            y_train.append(int(sheet.cell_value(i, 0)))

        dm = DataManager(X_train, y_train)
        cls = Classifier(
            include_models=['liblinear_svc', 'libsvm_svc', 'random_forest', 'logistic_regression', 'mlp'],
            optimizer='smbo',
            cross_valid=False,
            ensemble_method='bagging',
            ensemble_size=args.ensemble_size,
            save_dir='data/save_models'
        )
        cls.fit(dm, metric='auc', runcount=args.run_count)

        sheet = xlrd.open_workbook("lyqtestdata.xlsx")
        sheet = sheet.sheet_by_index(0)
        nrows = sheet.nrows
        X_test = []
        y_test = []
        for i in range(1, nrows):
            X_test.append(sheet.row_values(i, start_colx=1))
            y_test.append(int(sheet.cell_value(i, 0)))

        pred = cls.predict_proba(X_test)
        result.append(roc_auc_score(y_test, pred[:, 1:2]))
        print(result)

    import pickle
    with open('result.pkl', 'wb') as f:
        pickle.dump(result, f)


if __name__ == "__main__":
    test_cash_module()
