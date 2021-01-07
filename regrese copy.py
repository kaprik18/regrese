import glob
import os
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

list_of_files = glob.glob("data/prodej/*")
latest_data = max(list_of_files, key=os.path.getctime)
na_values1 = ["N/A", "nan"]

df = pd.read_csv(latest_data, na_values=na_values1, encoding="utf-8-sig")
df = df.drop_duplicates(subset=["item_no"])
df = df.dropna(subset=["cena", "cena_na_metr", "ctvrt", "metry"])

df = df[df.cena_na_metr != "cena_na_metr"]

df_dummy = pd.get_dummies(df["ctvrt"])
list_of_ctvrt = [
    "Praha 1",
    "Praha 2",
    "Praha 3",
    "Praha 4",
    "Praha 5",
    "Praha 6",
    "Praha 7",
    "Praha 8",
    "Praha 9",
    "Praha 10",
]


for item in df_dummy.columns:
    if item not in list_of_ctvrt:
        del df_dummy[item]


df_dummy2 = pd.get_dummies(df["dispozice"])
list_of_dispozice = [
    "1+1",
    "1+kk",
    "2+1",
    "2+kk",
    "3+1",
    "3+kk",
    "4+kk",
]

for item in df_dummy2.columns:
    if item not in list_of_dispozice:
        del df_dummy2[item]

df = pd.concat([df, df_dummy, df_dummy2], axis=1)

columns_to_drop = [
    "item_no",
    "url",
    "dispozice",
    "cena_na_metr",
    "realitka",
    "ulice",
    "ctvrt",
    "podctvrt",
]
df = df.drop(columns=columns_to_drop)

df = df.astype({"metry": "float64", "cena": "float64",})
df = df.sample(frac=1).reset_index(drop=True)


df["metry_normalized"] = (df["metry"] - df["metry"].mean()) / df["metry"].std()
df["cena_normalized"] = (df["cena"] - df["cena"].mean()) / df["cena"].std()

df = df.drop(columns=["cena", "metry"])

number_of_items = len(df.index)
train_coeff = 0.66
last_item_of_training_set = int(round(number_of_items * train_coeff))

train = df.iloc[:last_item_of_training_set]
test = df.iloc[last_item_of_training_set:]

X_train, X_test = (
    train.drop("cena_normalized", axis=1),
    test.drop("cena_normalized", axis=1),
)
y_train, y_test = train[["cena_normalized"]], test[["cena_normalized"]]


alphas = [
    0.005 * last_item_of_training_set,
    0.01 * last_item_of_training_set,
    0.02 * last_item_of_training_set,
    0.03 * last_item_of_training_set,
    0.04 * last_item_of_training_set,
    0.05 * last_item_of_training_set,
    0.075 * last_item_of_training_set,
    0.1 * last_item_of_training_set,
    0.2 * last_item_of_training_set,
    0.4 * last_item_of_training_set,
]

mses = []
for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    pred = ridge.predict(X_test)
    mses.append(mse(y_test, pred))
    print(mse(y_test, pred))

