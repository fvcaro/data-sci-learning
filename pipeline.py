from ucimlrepo import fetch_ucirepo
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

# fetch dataset
national_poll_on_healthy_aging_npha = fetch_ucirepo(id=936)

# data (as pandas dataframes)
data_X = national_poll_on_healthy_aging_npha.data.features
data_y = national_poll_on_healthy_aging_npha.data.targets
print((data_y == 1).sum() / len(data_y))
print((data_y == 2).sum() / len(data_y))
print((data_y == 3).sum() / len(data_y))
# metadata
print(national_poll_on_healthy_aging_npha.metadata)

# variable information
print(national_poll_on_healthy_aging_npha.variables)

data_X = data_X.replace({'Trouble_Sleeping': {-1: np.nan}, 'Employment': {-1: np.nan}, 'Dental_Health': {-1: np.nan}, 'Mental_Health': {-1: np.nan}, 'Physical_Health': {-1: np.nan}, 'Age': {1: 0, 2: 1}, 'Race': {-2: np.nan, -1: np.nan}, 'Gender': {-2: np.nan, -1: np.nan, 1: 0, 2:1}, 'Prescription_Sleep_Medication': {-1: np.nan}})

# Load and split the data
data_X_trval, data_X_test, data_y_trval, data_y_test = train_test_split(data_X, data_y, test_size=0.20, shuffle=True, stratify=data_y)
data_X_train, data_X_valid, data_y_train, data_y_valid = train_test_split(data_X_trval, data_y_trval, test_size=0.25, shuffle=True, stratify=data_y_trval)
print((data_y_test == 1).sum() / len(data_y_test))
print((data_y_test == 2).sum() / len(data_y_test))
print((data_y_test == 3).sum() / len(data_y_test))

ct = ColumnTransformer([('enc_1', OneHotEncoder(sparse_output = False), ['Employment']), ('enc_2', OneHotEncoder(sparse_output = False), ['Race'])], remainder = 'passthrough')

# Define the pipeline including data preprocessing and model
pipeline = Pipeline([
    ('encoding', ct),
    ('scaling', StandardScaler())
])

pipeline.fit(data_X_train, data_y_train)

Xt = pipeline.transform(data_X_train)
Xv = pipeline.transform(data_X_valid)
Xtt = pipeline.transform(data_X_test)

print('done')