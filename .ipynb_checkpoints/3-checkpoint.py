import yaml
import pickle as pkl
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import os

import warnings
warnings.filterwarnings('ignore')

data_test = pd.read_csv("data/clean/data_test_prepared.csv")
data_train = pd.read_csv("data/clean/data_train_prepared.csv")
target = pd.read_csv("data/clean/target.csv")

with open("params.yaml","r") as s:
    params = yaml.safe_load(s)

data_train = data_train.drop("Unnamed: 0",axis=1)
data_test = data_test.drop("Unnamed: 0",axis=1)
target = target.drop("Unnamed: 0",axis=1)

gbc_clf = GradientBoostingClassifier(max_depth=params["train"]["max_depth"],learning_rate=params["train"]["learning_rate"])
gbc_clf.fit(data_train, target)

out = pd.DataFrame({"pred":gbc_clf.predict(data_test)})
os.makedirs("models",exist_ok=True)
pkl.dump(gbc_clf, open("models/model.pkl", 'wb'))
os.makedirs("data/out")
out.to_csv("data/out/out.csv")
os.makedirs("metrics")

with open("metrics/gbc.yaml","w") as f:
    f.write(f"score: {gbc_clf.score(data_train,target)}")