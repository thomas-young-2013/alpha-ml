import sys
import os
import argparse
import warnings

alphaml_path = os.getcwd()
sys.path.append(alphaml_path)

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from alphaml.engine.components.data_manager import DataManager
from alphaml.estimators.classifier import Classifier

parser = argparse.ArgumentParser()
parser.add_argument('--run_count', type=int, default=30)
parser.add_argument('--ensemble_size', type=int, default=30)
args = parser.parse_args()
warnings.filterwarnings("ignore")


def test_best():
    X, y = load_breast_cancer(return_X_y=True)
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.8, stratify=y)
    dm = DataManager(train_x, train_y)

    clf = Classifier(
        optimizer='smac',
        cross_valid=False,
        save_dir='data/save_models'
    )
    clf.fit(dm, metric='acc', runcount=args.run_count)

    pred = clf.predict(test_x)
    print("Test best: %f" % accuracy_score(test_y, pred))
    clf.show_info()


def test_bagging():
    X, y = load_breast_cancer(return_X_y=True)
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.8, stratify=y)
    dm = DataManager(train_x, train_y)

    clf = Classifier(
        optimizer='smac',
        cross_valid=False,
        ensemble_method='bagging',
        ensemble_size=args.ensemble_size,
        save_dir='data/save_models'
    )
    clf.fit(dm, metric='acc', runcount=args.run_count)

    pred = clf.predict(test_x)
    print("Test bagging: %f" % accuracy_score(test_y, pred))
    clf.show_info()


def test_stacking():
    X, y = load_breast_cancer(return_X_y=True)
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.8, stratify=y)
    dm = DataManager(train_x, train_y)

    clf = Classifier(
        optimizer='smac',
        cross_valid=False,
        ensemble_method='stacking',
        ensemble_size=args.ensemble_size,
        save_dir='data/save_models'
    )
    clf.fit(dm, metric='acc', runcount=args.run_count)

    pred = clf.predict(test_x)
    print("Test stacking: %f" % accuracy_score(test_y, pred))
    clf.show_info()


def test_ensemble_selection():
    X, y = load_breast_cancer(return_X_y=True)
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.8, stratify=y)
    dm = DataManager(train_x, train_y)

    clf = Classifier(
        optimizer='smac',
        cross_valid=False,
        ensemble_method='ensemble_selection',
        ensemble_size=args.ensemble_size,
        save_dir='data/save_models'
    )
    clf.fit(dm, metric='acc', runcount=args.run_count)

    pred = clf.predict(test_x)
    print("Test bagging: %f" % accuracy_score(test_y, pred))
    clf.show_info()


def test_tpe():
    X, y = load_breast_cancer(return_X_y=True)
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.8, stratify=y)
    dm = DataManager(train_x, train_y)

    clf = Classifier(
        optimizer='tpe',
        cross_valid=False,
        save_dir='data/save_models'
    )
    clf.fit(dm, metric='acc', runcount=args.run_count)

    pred = clf.predict(test_x)
    print("Test best: %f" % accuracy_score(test_y, pred))
    clf.show_info()


def test_cv():
    X, y = load_breast_cancer(return_X_y=True)
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.8, stratify=y)
    dm = DataManager(train_x, train_y)

    clf = Classifier(
        optimizer='tpe',
        cross_valid=True,
        k_fold=3,
        save_dir='data/save_models'
    )
    clf.fit(dm, metric='acc', runcount=args.run_count)

    pred = clf.predict(test_x)
    print("Test best: %f" % accuracy_score(test_y, pred))
    clf.show_info()


if __name__ == '__main__':
    # test_best()
    # test_bagging()
    # test_stacking()
    # test_tpe()
    # test_cv()
    test_ensemble_selection()
