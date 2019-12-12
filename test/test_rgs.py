import sys
import os
import argparse
import warnings

alphaml_path = os.getcwd()
sys.path.append(alphaml_path)

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from alphaml.engine.components.data_manager import DataManager
from alphaml.estimators.regressor import Regressor

parser = argparse.ArgumentParser()
parser.add_argument('--run_count', type=int, default=30)
parser.add_argument('--ensemble_size', type=int, default=30)
args = parser.parse_args()
warnings.filterwarnings("ignore")


def test_best():
    X, y = load_boston(return_X_y=True)
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.8)
    dm = DataManager(train_x, train_y)

    clf = Regressor(
        optimizer='smac',
        cross_valid=False,
        save_dir='data/save_models'
    )
    clf.fit(dm, metric='mse', runcount=args.run_count)

    pred = clf.predict(test_x)
    print("Test best: %f" % mean_squared_error(test_y, pred))


def test_cv():
    X, y = load_boston(return_X_y=True)
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.8)
    dm = DataManager(train_x, train_y)

    clf = Regressor(
        optimizer='smac',
        cross_valid=True,
        k_fold=5,
        save_dir='data/save_models'
    )
    clf.fit(dm, metric='mse', runcount=args.run_count)

    pred = clf.predict(test_x)
    print("Test best: %f" % mean_squared_error(test_y, pred))


if __name__ == '__main__':
    test_best()
    test_cv()
