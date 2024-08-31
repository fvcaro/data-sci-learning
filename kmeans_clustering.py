from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split, ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.base import clone
import pandas as pd

# fetch dataset
shoppers_purchasing_intention = fetch_ucirepo(id=468)

# data (as pandas dataframes)
data_X = shoppers_purchasing_intention.data.features
data_X_new = data_X['Month'].values.flatten()
data_y = shoppers_purchasing_intention.data.targets
data_y_new = data_y.values.flatten()
print('labels: ', pd.unique(data_y_new))
data_X.describe()
# data_X.isnull().sum()
# metadata
print(shoppers_purchasing_intention.metadata)

# variable information
print(shoppers_purchasing_intention.variables)

# Replace missing values if encoded as specific values (example, adapt as necessary)
data_X = data_X.replace({
    'Month': {'June': 'Jun'},
    'Weekend': {'False': 0, 'True': 1}
})

# Load and split the data
data_X_trval, data_X_test = train_test_split(data_X, test_size=0.20, shuffle=True)
data_X_train, data_X_valid = train_test_split(data_X_trval, test_size=0.25, shuffle=True)

# Define the ColumnTransformer for preprocessing
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
visitors = ['New_Visitor', 'Returning_Visitor', 'Other']
ct = ColumnTransformer([
    ('enc_1', OneHotEncoder(categories=[months], sparse_output=False), ['Month']),
    ('enc_2', OneHotEncoder(categories=[visitors], sparse_output=False), ['VisitorType']),
    ('enc_3', OneHotEncoder(handle_unknown='infrequent_if_exist', sparse_output=False), ['OperatingSystems']),
    ('enc_4', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['Browser']),
    ('enc_5', OneHotEncoder(handle_unknown='infrequent_if_exist', sparse_output=False), ['Region']),
    ('enc_6', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['TrafficType']),
], remainder='passthrough')

param_grid = {
    'clustering__n_clusters': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    'clustering__n_init': [4, 5, 6, 7, 8, 9, 10],
}

# Specify the number of random samples
n_samples = 10  # Adjust the number as needed
# Generate random samples
param_samples = list(ParameterSampler(param_grid, n_iter=n_samples, random_state=42))

# Define the pipeline including data preprocessing and model
pipeline = Pipeline([
    ('encoding', ct),
    ('scaling', StandardScaler()),
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
    # cluster_labels = temp_pipeline.named_steps['clustering'].labels_ # labels del training set
    labels_valid = temp_pipeline.predict(data_X_valid)
    data_X_transformed = temp_pipeline[:-1].transform(data_X_valid)
    silhouette_avg = silhouette_score(data_X_transformed, labels_valid)
    silhouette_scores.append(silhouette_avg)

    print(f'Model trained successfully. Silhouette Score on validation set: {silhouette_avg:.4f}')