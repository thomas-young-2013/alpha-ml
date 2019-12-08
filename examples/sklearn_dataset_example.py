import argparse

from enum import Enum
from sklearn.datasets import *
from sklearn.model_selection import train_test_split

from alphaml.estimators.classifier import Classifier
from alphaml.estimators.regressor import Regressor
from alphaml.engine.components.data_manager import DataManager


class TaskType(Enum):
    classification = 0
    regression = 1


SKLEARN_TOY_DATASETS = {"boston": (load_boston, TaskType.regression),
                        "iris": (load_iris, TaskType.classification),
                        "diabetes": load_diabetes,
                        "digits": (load_digits, TaskType.classification),
                        "linnerud": load_linnerud}
SKLEARN_REALWORLD_DATASETS = {}


def evaluate_all_datasets(arg):
    for dataset in list(SKLEARN_TOY_DATASETS.keys()):
        arg.dataset = dataset
        evaluate_single_dataset(arg)
    for dataset in list(SKLEARN_REALWORLD_DATASETS.keys()):
        arg.dataset = dataset
        evaluate_single_dataset(arg)


def evaluate_single_dataset(arg):
    if SKLEARN_TOY_DATASETS.get(arg.dataset_name) is not None:
        load_data, task_type = SKLEARN_TOY_DATASETS[arg.dataset_name]
    else:
        load_data, task_type = SKLEARN_REALWORLD_DATASETS[arg.dataset_name]

    x, y = load_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2)
    dm = DataManager(x_train, y_train)

    if task_type == TaskType.classification:
        clf = Classifier(optimizer=args.optimizer,
                         seed=args.seed,
                         )
        clf.fit(dm,
                runcount=args.runcount)
        print("The Alpha-ML score is: ", clf.score(x_test, y_test))
    else:
        reg = Regressor()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="The sklearn dataset name and "
                                                    "`all` means test all datasets which may be slow.")
    parser.add_argument("--optimizer", type=str, help="The optimizer in Alpha-ML.")
    parser.add_argument("--runcount", type=str, help="The number of trials to in Alpha-ML.")
    parser.add_argument("--seed", type=str, help="The seed in the experiment.")

    args = parser.parse_args()

    if args.dataset == "all":
        evaluate_all_datasets()
    elif args.dataset in SKLEARN_TOY_DATASETS.keys() or args.dataset in SKLEARN_REALWORLD_DATASETS:
        evaluate_single_dataset(args.dataset)
    else:
        raise ValueError("The dataset must in the sklearn toy datasets: {} or real-world datasets: {}, "
                         "got {}.".format(SKLEARN_TOY_DATASETS.keys(), SKLEARN_REALWORLD_DATASETS.keys(), args.dataset))
