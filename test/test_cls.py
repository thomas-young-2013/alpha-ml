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

warnings.filterwarnings("ignore")


def test_best():
    X, y = load_breast_cancer(return_X_y=True)
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.8, stratify=y)
    dm = DataManager(train_x, train_y)

    clf = Classifier(
        optimizer='tpe',
        cross_valid=False,
        save_dir='data/save_models'
    )
    clf.fit(dm, metric='acc', runcount=30)

    pred = clf.predict(test_x)
    print("Test best: %f" % accuracy_score(test_y, pred))
    clf.show_info()


def test_bagging():
    X, y = load_breast_cancer(return_X_y=True)
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.8, stratify=y)
    dm = DataManager(train_x, train_y)

    clf = Classifier(
        optimizer='tpe',
        cross_valid=False,
        ensemble_method='bagging',
        ensemble_size=30,
        save_dir='data/save_models'
    )
    clf.fit(dm, metric='acc', runcount=30)

    pred = clf.predict(test_x)
    print("Test bagging: %f" % accuracy_score(test_y, pred))
    clf.show_info()


def test_stacking():
    X, y = load_breast_cancer(return_X_y=True)
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.8, stratify=y)
    dm = DataManager(train_x, train_y)

    clf = Classifier(
        optimizer='tpe',
        cross_valid=False,
        ensemble_method='stacking',
        ensemble_size=30,
        save_dir='data/save_models'
    )
    clf.fit(dm, metric='acc', runcount=30)

    pred = clf.predict(test_x)
    print("Test stacking: %f" % accuracy_score(test_y, pred))
    clf.show_info()


def test_ensemble_selection():
    X, y = load_breast_cancer(return_X_y=True)
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.8, stratify=y)
    dm = DataManager(train_x, train_y)

    clf = Classifier(
        optimizer='tpe',
        cross_valid=False,
        ensemble_method='ensemble_selection',
        ensemble_size=30,
        save_dir='data/save_models'
    )
    clf.fit(dm, metric='acc', runcount=30)

    pred = clf.predict(test_x)
    print("Test bagging: %f" % accuracy_score(test_y, pred))
    clf.show_info()


# def test_smac():
#     X, y = load_breast_cancer(return_X_y=True)
#     train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.8, stratify=y)
#     dm = DataManager(train_x, train_y)
#
#     clf = Classifier(
#         optimizer='smac',
#         cross_valid=False,
#         save_dir='data/save_models'
#     )
#     clf.fit(dm, metric='acc', runcount=30)
#
#     pred = clf.predict(test_x)
#     print("Test best: %f" % accuracy_score(test_y, pred))
#     clf.show_info()


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
    clf.fit(dm, metric='acc', runcount=30)

    pred = clf.predict(test_x)
    print("Test best: %f" % accuracy_score(test_y, pred))
    clf.show_info()
