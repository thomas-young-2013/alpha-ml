from sklearn.model_selection import train_test_split
from sklearn.metrics.scorer import make_scorer

from alphaml.estimators.classifier import Classifier
from alphaml.engine.components.data_manager import DataManager
from alphaml.datasets.cls_dataset.dataset_loader import load_data


def my_acc(y_true, y_pred):
    from sklearn.metrics import accuracy_score
    return accuracy_score(y_true, y_pred)


if __name__ == '__main__':
    x, y, _ = load_data("iris")
    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2)
    dm = DataManager(x_train, y_train)

    clf = Classifier(optimizer='smac',
                     k_fold=3,
                     include_models=['random_forest', 'xgboost', 'libsvm_svc'],
                     ensemble_method='bagging',
                     ensemble_size=12,
                     save_dir='data/save_models')

    # clf.fit(dm, metric='acc', runcount=args.run_count)
    # Or we can use a user-defined scorer as metric input
    clf.fit(dm, metric=make_scorer(my_acc, greater_is_better=True), runcount=200)

    print("The accuracy score is: ", clf.score(x_test, y_test))
