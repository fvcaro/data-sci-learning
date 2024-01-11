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
k_out = 5
kf_out = KFold(n_splits=k_out)

# K-Fold Cross-Validation
k_in = 3
kf_in = KFold(n_splits=k_in)

# Initialize SVR model
svr = SVR()

for idx_k, (train_index, test_index) in enumerate(kf_out.split(data_X_small)):
    print(f'Fold {idx_k + 1} / {k_out}')
    # Split the data into training and testing sets for this fold
    X_train, X_test = data_X_small.iloc[train_index], data_X_small.iloc[test_index]
    y_train, y_test = data_y_small.iloc[train_index], data_y_small.iloc[test_index]

    grid_search = GridSearchCV(estimator=svr, param_grid=hyperparameters, scoring=scoring, cv=kf_in, refit='neg_mean_squared_error', return_train_score=False)
    temp_svr = grid_search.fit(X_train, y_train)

    # Get the results
    results = grid_search.cv_results_

    y_pred_test = temp_svr.predict(X_test)  # Use X_test instead of X_valid for testing
    temp_rmse_test = mean_squared_error(y_test, y_pred_test)
    temp_mae_test = mean_absolute_error(y_test, y_pred_test)
    temp_r2_test = r2_score(y_test, y_pred_test)