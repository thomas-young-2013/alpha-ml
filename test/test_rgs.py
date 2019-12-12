import sys
import os
import warnings

alphaml_path = os.getcwd()
sys.path.append(alphaml_path)

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from alphaml.engine.components.data_manager import DataManager
from alphaml.estimators.regressor import Regressor

warnings.filterwarnings("ignore")


def test_best():
    X, y = load_boston(return_X_y=True)
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.8)
    dm = DataManager(train_x, train_y)

    clf = Regressor(
        optimizer='tpe',
        cross_valid=False,
        save_dir='data/save_models'
    )
    clf.fit(dm, metric='mse', runcount=30)

    pred = clf.predict(test_x)
    print("Test best: %f" % mean_squared_error(test_y, pred))


def test_cv():
    X, y = load_boston(return_X_y=True)
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.8)
    dm = DataManager(train_x, train_y)

    clf = Regressor(
        optimizer='tpe',
        cross_valid=True,
        k_fold=3,
        save_dir='data/save_models'
    )
    clf.fit(dm, metric='mse', runcount=30)

    pred = clf.predict(test_x)
    print("Test best: %f" % mean_squared_error(test_y, pred))
