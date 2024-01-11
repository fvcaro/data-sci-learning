from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

# Fetch and prepare the data
data_X, data_y = fetch_california_housing(return_X_y=True, as_frame=True)
data_all = data_X.copy()
data_all['y'] = data_y

# Create a smaller sample
n_small = 200
data_all_small = data_all.sample(n_small)
data_X_small = data_all_small.drop(labels=['y'], axis='columns')
data_y_small = data_all_small['y']

hyperparameters = {'kernel': ['rbf'], 'gamma': [1e-1, 1e0, 1e1], 'C': [1e-1, 1e0, 1e1]}

scoring = ['neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_mean_absolute_percentage_error', 'r2']

# K-Fold Cross-Validation
k_in = 5
kf_in = KFold(n_splits=k_in)

# Initialize SVR model
svr = SVR()

# Use GridSearchCV to perform cross-validated grid search
grid_search = GridSearchCV(estimator=svr, param_grid=hyperparameters, scoring=scoring, cv=kf_in, refit='neg_mean_squared_error', return_train_score=False)
grid_search.fit(data_X_small, data_y_small)

# Get the results
results = grid_search.cv_results_

# Print the results
for mean_test_score, params in zip(results['mean_test_neg_mean_squared_error'], results['params']):
    print(f'Mean MSE: {mean_test_score:.4f} for parameters: {params}')

# Access the best model and its parameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

print(f'Best model: {best_model}')