import argparse
from sklearn.model_selection import train_test_split

from alphaml.estimators.classifier import Classifier
from alphaml.engine.components.data_manager import DataManager
from alphaml.datasets.cls_dataset.dataset_loader import load_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", type=str, default='smac_mono', help="The optimizer in Alpha-ML.")
    parser.add_argument("--run_count", type=int, default=200, help="The number of trials to in Alpha-ML.")
    parser.add_argument("--ensemble_size", type=int, default=12,
                        help="The number of base models to ensemble in Alpha-ML.")
    parser.add_argument("--k_fold", type=int, default=3, help="Folds for cross validation in Alpha-ML.")
    args = parser.parse_args()

    x, y, _ = load_data("iris")
    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2)
    dm = DataManager(x_train, y_train)

    clf = Classifier(optimizer=args.optimizer,
                     k_fold=args.k_fold,
                     ensemble_method='bagging',
                     ensemble_size=args.ensemble_size,
                     save_dir='data/save_models')
    clf.fit(dm, metric='acc', runcount=args.run_count)
    print("The accuracy score is: ", clf.score(x_test, y_test))
