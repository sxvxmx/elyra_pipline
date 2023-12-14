import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn
import seaborn as sns
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV

import warnings
warnings.filterwarnings('ignore')

data_test = pd.read_csv("data/prep/data_test_cleaned.csv")
data_train = pd.read_csv("data/prep/data_train_cleaned.csv")

data_train = data_train.drop("Unnamed: 0",axis=1)
data_test = data_test.drop("Unnamed: 0",axis=1)

class autoLabelEncoder:
    def __init__(self) -> None:
        self.cat_encoders:dict = {}

    def fit(self, data:pd.DataFrame, categories:list[str]) -> None:
        for feat in categories:
            enc = LabelEncoder()
            self.cat_encoders[feat] = enc.fit(data.loc[data[feat].notna(), feat])

    def transform(self, data:pd.DataFrame, categories:list[str]) -> pd.DataFrame:
        for feat in categories:
            enc = self.cat_encoders[feat]
            data.loc[data[feat].notna(), feat] = enc.transform(data.loc[data[feat].notna(), feat])
        return data
    
    def get_encoder(self, category) -> LabelEncoder:
        return self.cat_encoders[category]
    
def feat(q):
    if str(q) == "nan" or q == np.nan:
        return np.nan
    else:
        return q[0]
    
data_train["Cabin"] = np.array(map(lambda q: feat(q),data_train["Cabin"]))
data_test["Cabin"] = np.array(map(lambda q: feat(q),data_test["Cabin"]))

ale = autoLabelEncoder()

cat = list(data_train.select_dtypes(include=['object']).columns)

ale.fit(data_train, cat)
data_train = ale.transform(data_train,cat)
data_test = ale.transform(data_test,cat)

corr1 = data_train.corr()

data_train["Way"] = data_train["HomePlanet"].astype(str) + data_train["Destination"].astype(str)
data_test["Way"] = data_test["HomePlanet"].astype(str) + data_test["Destination"].astype(str)


for i in range(len(data_train["Way"])):
    if "nan" in str(data_train["Way"][i]):
        data_train["Way"][i] = np.nan


for i in range(len(data_test["Way"])):
    if "nan" in str(data_test["Way"][i]):
        data_test["Way"][i] = np.nan

data_train = data_train.dropna()
data_test = data_test.dropna()

target = data_train["Transported"]
data_train = data_train.drop("Transported",axis=1)

cat1 = data_train.columns[data_train.isnull().any()].tolist()
cat2 = ['Age','RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
cat1 = [x for x in cat1 if x not in cat2]

data_train["Spend"] = data_train["RoomService"] + data_train["FoodCourt"] + data_train["ShoppingMall"] + data_train["Spa"] + data_train["VRDeck"]
data_test["Spend"] = data_test["RoomService"] + data_test["FoodCourt"] + data_test["ShoppingMall"] + data_test["Spa"] + data_test["VRDeck"]

corr2 = data_train.corr()

fig, axs = plt.subplots(ncols = 2, figsize = (20,8))

h1 = sns.heatmap(corr1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5},ax=axs[0])
h2 = sns.heatmap(corr2, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5},ax=axs[1])
os.makedirs("images2")
fig.savefig("images2/heatmap1.png")

os.makedirs("data/clean")
data_test.to_csv("data/clean/data_test_prepared.csv")
data_train.to_csv("data/clean/data_train_prepared.csv")
target.to_csv("data/clean/target.csv")