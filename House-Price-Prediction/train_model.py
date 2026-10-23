import warnings

warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import pickle

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor

try:
    boston = fetch_openml(data_id=531, as_frame=False)
except Exception:
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    boston_data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    boston_target = raw_df.values[1::2, 2]
    boston_feature_names = [
        "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM",
        "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT",
    ]
    boston = type(
        "Boston",
        (),
        {
            "data": boston_data,
            "target": boston_target,
            "feature_names": boston_feature_names,
        },
    )()

df = pd.DataFrame(boston.data, columns=boston.feature_names)
df["PRICE"] = boston.target

x = df.drop("PRICE", axis=1)
y = df["PRICE"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=10
)

ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

lr = CatBoostRegressor(random_state=42, verbose=0)

param_dist = {
    "depth": [4, 6, 8],
    "learning_rate": [0.1, 0.01, 0.001],
    "l2_leaf_reg": [1, 3, 5, 7],
    "bagging_temperature": [0.5, 1, 1.5],
    "random_strength": [0.5, 1, 1.5],
    "border_count": [32, 64, 128],
    "iterations": [100, 200, 300],
}

random_search = RandomizedSearchCV(
    lr,
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    random_state=42,
)
random_search.fit(x_train, y_train)

print("Best params:", random_search.best_params_)
print("Best CV score (R2):", round(random_search.best_score_, 4))

y_pred = random_search.predict(x_test)
print("Test score (R2):", round(r2_score(y_test, y_pred), 4))

model = random_search.best_estimator_

with open("housepred.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(ss, f)

print("Saved housepred.pkl and scaler.pkl")