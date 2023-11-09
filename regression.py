import sklearn
from sklearn.datasets import fetch_california_housing

data_X, data_y = fetch_california_housing(return_X_y=True, as_frame=True)
