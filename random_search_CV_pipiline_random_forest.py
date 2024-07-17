from ucimlrepo import fetch_ucirepo
import numpy as np
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.base import clone
import joblib
from scipy.stats import loguniform

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

# Define parameter distribution for random search
param_dist = {
    # 'imputer__n_neighbors': [5, 7, 9, 11],
    # 'imputer__weights': ['uniform', 'distance'],
    # 'imputer__add_indicator': [True, False],
    'imputer__n_neighbors': [5],
    'imputer__weights': ['distance'],
    'imputer__add_indicator': [False],
    'classifier__n_estimators': [5, 10, 50],
    'classifier__max_depth': [2, 4, 8],
}

# Create a pipeline including data preprocessing and model
pipeline = Pipeline([
    ('encoding', ct),
    ('scaling', StandardScaler()),
    ('imputer', KNNImputer()),
    ('classifier', RandomForestClassifier())  # Random Forest classifier
])

# Create a RandomizedSearchCV object
# random_search = RandomizedSearchCV(pipeline, param_dist, cv=skf_in, scoring='balanced_accuracy', n_jobs=n_jobs, n_iter=50)
random_search = RandomizedSearchCV(pipeline, param_dist, cv=skf_in, scoring='balanced_accuracy', n_jobs=n_jobs, n_iter=5)

# Define lists to store evaluation metrics for each fold
balanced_accuracy_scores = []

# Nested cross-validation loop
for i, (train_index, test_index) in enumerate(skf_out.split(data_X, data_y)):
    print(f'Fold {i+1}: ')
    print(f'  Train: index={train_index}')
    print(f'  Test:  index={test_index}')

    temp_random_search = clone(random_search)

    # Split the data into training and test sets
    X_train, X_test = data_X.iloc[train_index, :], data_X.iloc[test_index, :]
    y_train, y_test = data_y.iloc[train_index, 0].tolist(), data_y.iloc[test_index, 0].tolist()

    # Inner cross-validation loop for hyperparameter tuning
    with joblib.parallel_backend('threading', n_jobs=n_jobs):
        temp_random_search.fit(X_train, y_train)

    # Get the best model from the random search
    best_model = temp_random_search.best_estimator_

    # Evaluate the model on the test set
    y_pred = best_model.predict(X_test)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

    # Evaluate the model on the train set
    y_pred_aux_ = best_model.predict(X_train)
    balanced_accuracy_aux_ = balanced_accuracy_score(y_train, y_pred_aux_)

    # Append scores to respective lists
    balanced_accuracy_scores.append(balanced_accuracy)

    # Print evaluation metrics
    print('Evaluation Metrics: ')
    print('Balanced Accuracy: ', balanced_accuracy)
    print('Balanced Accuracy: ', balanced_accuracy_aux_)

# After the loop, print average scores
print('Average Evaluation Metrics: ')
print('Balanced Accuracy: ', np.mean(balanced_accuracy_scores))
print('Training completed. Evaluating models ... ')
print('done')

# python random_search_CV_pipiline_random_forest.py > random_search_CV_pipiline_random_forest_$(date +%d-%m-%Y_%H.%M.%S).txt 2>&1 &
