import os
import pandas as pd
import warnings

from alphaml.engine.components.data_preprocessing.imputer import impute_df
from alphaml.engine.components.data_manager import DataManager
from alphaml.estimators.classifier import Classifier
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

home_path = os.path.expanduser('~')
train_path = os.path.join(home_path, "datasets/titanic/train.csv")
test_path = os.path.join(home_path, "datasets/titanic/test.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
y_train = train_df["Survived"].values

train_df.drop(columns="Survived", axis=1, inplace=True)

train_size = train_df.shape[0]
test_size = test_df.shape[0]

df = pd.concat([train_df, test_df], ignore_index=True)

df.drop(columns=["PassengerId", "Name"], axis=1, inplace=True)

df = impute_df(df)

df["Sex"] = df["Sex"].replace(["male", "female"], [0, 1])

df.drop(columns="Ticket", axis=1, inplace=True)

for i in range(df.shape[0]):
    if df["Cabin"][i] == "C23 C25 C27":
        df["Cabin"][i] = 0
    else:
        df["Cabin"][i] = 1

df["Cabin"] = df["Cabin"].astype("float")

df = pd.get_dummies(df)

x = df.values

x_train = x[:train_size]
x_test = x[train_size:]

dm = DataManager()
dm.train_X = x_train
dm.train_y = y_train


clf = Classifier(optimizer="smbo")
clf.fit(dm, metric="accuracy", runcount=200)

submission = pd.read_csv(home_path + "/datasets/titanic/gender_submission.csv")
submission["Survived"] = clf.predict(x_test)
submission.to_csv(home_path + "/datasets/titanic/xgboost.csv", index=False)
