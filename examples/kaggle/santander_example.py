import os
import pandas as pd
import warnings

from alphaml.engine.components.data_manager import DataManager
from alphaml.estimators.classifier import Classifier

warnings.filterwarnings("ignore")

home_path = os.path.expanduser('~')
train_path = os.path.join(home_path, "datasets/santander/train.csv")
test_path = os.path.join(home_path, "datasets/santander/test.csv")

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

df_train.drop(labels=["ID_code"], axis=1, inplace=True)
df_test.drop(labels=["ID_code"], axis=1, inplace=True)

x_train = df_train.drop(labels=["target"], axis=1).values
y_train = df_train["target"].values
x_test = df_test.values

dm = DataManager()
dm.train_X = x_train
dm.train_y = y_train


clf = Classifier()
clf.fit(dm, metric="auc", runcount=200)

submission = pd.read_csv(home_path + "/datasets/titanic/gender_submission.csv")
submission["Survived"] = clf.predict(x_test)
submission.to_csv(home_path + "/datasets/titanic/alpha-ml.csv", index=False)
