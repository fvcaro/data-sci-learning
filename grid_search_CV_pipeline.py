from ucimlrepo import fetch_ucirepo
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.base import clone
import joblib

# Define the number of parallel jobs
n_jobs = -1  # Set to -1 to use all available CPU cores, adjust as needed

# fetch dataset
national_poll_on_healthy_aging_npha = fetch_ucirepo(id=936)

# data (as pandas dataframes)
data_X = national_poll_on_healthy_aging_npha.data.features
data_y = national_poll_on_healthy_aging_npha.data.targets

# Replace certain values with NaN
data_X = data_X.replace({'Trouble_Sleeping': {-1: np.nan}, 'Employment': {-1: np.nan}, 
                         'Dental_Health': {-1: np.nan}, 'Mental_Health': {-1: np.nan}, 
                         'Physical_Health': {-1: np.nan}, 'Age': {1: 0, 2: 1}, 
                         'Race': {-2: np.nan, -1: np.nan}, 'Gender': {-2: np.nan, -1: np.nan, 1: 0, 2: 1},
                         'Prescription_Sleep_Medication': {-1: np.nan}})

# Define column transformer for encoding categorical features
ct = ColumnTransformer([('enc_1', OneHotEncoder(sparse_output=False), ['Employment']), 
                        ('enc_2', OneHotEncoder(sparse_output=False), ['Race'])], 
                        remainder='passthrough')

# Create a StratifiedKFold object for cross-validation
skf_out = StratifiedKFold(n_splits=5)
skf_in = StratifiedKFold(n_splits=3)

# Define parameter grid for hyperparameter tuning
param_grid = {
    'imputer__n_neighbors': [5, 7, 9, 11],
    'imputer__weights': ['uniform', 'distance'],
    'imputer__add_indicator': [True, False],
    'classifier__hidden_layer_sizes': [(10,), (20,), (10, 10), (20, 10)],
    'classifier__max_iter': [750, 1250, 1500],
    'classifier__alpha': [0.0001, 0.001, 0.01, 0.1],
}

# Create a pipeline including data preprocessing and model
pipeline = Pipeline([
    ('encoding', ct),
    ('scaling', StandardScaler()),
    ('imputer', KNNImputer()),
    ('classifier', MLPClassifier())
])

# Create a GridSearchCV object
grid_search = GridSearchCV(pipeline, param_grid, cv=skf_in, scoring='balanced_accuracy', n_jobs=n_jobs)

# Define lists to store evaluation metrics for each fold
balanced_accuracy_scores = []

# Nested cross-validation loop
for i, (train_index, test_index) in enumerate(skf_out.split(data_X, data_y)):
    print(f'Fold {i+1}: ')
    print(f'  Train: index={train_index}')
    print(f'  Test:  index={test_index}')

    temp_grid_search = clone(grid_search)

    # Split the data into training and test sets
    X_train, X_test = data_X.iloc[train_index, :], data_X.iloc[test_index, :]
    y_train, y_test = data_y.iloc[train_index, 0].tolist(), data_y.iloc[test_index, 0].tolist()

    # Inner cross-validation loop for hyperparameter tuning
    with joblib.parallel_backend('threading', n_jobs=n_jobs):
        temp_grid_search.fit(X_train, y_train)

    # Get the best model from the grid search
    best_model = temp_grid_search.best_estimator_

    # Evaluate the model on the test set
    y_pred = best_model.predict(X_test)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

    # Append scores to respective lists
    balanced_accuracy_scores.append(balanced_accuracy)

    # Print evaluation metrics
    print('Evaluation Metrics: ')
    print('Balanced Accuracy: ', balanced_accuracy)

# After the loop, print average scores
print('Average Evaluation Metrics: ')
print('Balanced Accuracy: ', np.mean(balanced_accuracy_scores))
print('Training completed. Evaluating models ... ')
print('done')

# python grid_search_CV_pipeline.py > grid_search_CV_pipeline_$(date +%d-%m-%Y_%H.%M.%S).txt 2>&1 &