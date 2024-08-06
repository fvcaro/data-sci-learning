from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.base import clone

# fetch dataset
shoppers_purchasing_intention = fetch_ucirepo(id=468)

# data (as pandas dataframes)
data_X = shoppers_purchasing_intention.data.features
data_y = shoppers_purchasing_intention.data.targets
print((data_y == 0).sum() / len(data_y))
print((data_y == 1).sum() / len(data_y))

# metadata
print(shoppers_purchasing_intention.metadata)

# variable information
print(shoppers_purchasing_intention.variables)

# Replace missing values if encoded as specific values (example, adapt as necessary)
data_X = data_X.replace({
    'Administrative': {-1: np.nan},
    'Informational': {-1: np.nan},
    'ProductRelated': {-1: np.nan},
    'BounceRates': {-1: np.nan},
    'ExitRates': {-1: np.nan},
    'PageValues': {-1: np.nan},
    'SpecialDay': {-1: np.nan},
    'OperatingSystems': {-1: np.nan},
    'Browser': {-1: np.nan},
    'Region': {-1: np.nan},
    'TrafficType': {-1: np.nan},
    'VisitorType': {'Other': np.nan, 'New_Visitor': 0, 'Returning_Visitor': 1},
    'Weekend': {'False': 0, 'True': 1}
})

# Load and split the data
data_X_trval, data_X_test, data_y_trval, data_y_test = train_test_split(data_X, data_y, test_size=0.20, shuffle=True, stratify=data_y)
data_X_train, data_X_valid, data_y_train, data_y_valid = train_test_split(data_X_trval, data_y_trval, test_size=0.25, shuffle=True, stratify=data_y_trval)
print((data_y_test == 0).sum() / len(data_y_test))
print((data_y_test == 1).sum() / len(data_y_test))

# Define the ColumnTransformer for preprocessing
ct = ColumnTransformer([
    ('enc_1', OneHotEncoder(sparse_output=False), ['Month']),
    ('enc_2', OneHotEncoder(sparse_output=False), ['VisitorType']),
], remainder='passthrough')

param_grid = {
    'clustering__n_clusters': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
}

# Specify the number of random samples
n_samples = 10  # Adjust the number as needed
# Generate random samples
param_samples = list(ParameterSampler(param_grid, n_iter=n_samples, random_state=42))

# Define the pipeline including data preprocessing and model
pipeline = Pipeline([
    ('encoding', ct),
    ('scaling', StandardScaler()),
    ('imputer', KNNImputer()),
    ('clustering', KMeans())
])

# Lists to store performance metrics for each configuration
silhouette_scores = []

for i, params in enumerate(param_samples):
    temp_pipeline = clone(pipeline)
    temp_pipeline.set_params(**params)

    # Inside the loop where models are trained
    print(f'Training model with parameters: {params}...')

    temp_pipeline.fit(data_X_train)

    # Evaluate the temp model on the validation set
    cluster_labels = temp_pipeline.named_steps['clustering'].labels_
    silhouette_avg = silhouette_score(data_X_valid, cluster_labels)
    silhouette_scores.append(silhouette_avg)

    print(f'Model trained successfully. Silhouette Score on validation set: {silhouette_avg:.4f}')