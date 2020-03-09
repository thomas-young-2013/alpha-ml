from sklearn.model_selection import train_test_split

from alphaml.estimators.regressor import Regressor
from alphaml.engine.components.data_manager import DataManager
from alphaml.datasets.rgs_dataset.dataset_loader import load_data

if __name__ == '__main__':
    x, y, _ = load_data("boston")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    dm = DataManager(x_train, y_train)

    clf = Regressor(optimizer='mono_tpe_smbo',
                    cross_valid=False,
                    exclude_models=['mlp'],
                    ensemble_method='bagging',
                    ensemble_size=12,
                    save_dir='data/save_models')
    clf.fit(dm, metric='mse', runcount=200)
    print("The mse score is: ", clf.score(x_test, y_test))
