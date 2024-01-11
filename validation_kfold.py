from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

import numpy as np
np.random.seed(42)

# Fetch and prepare the data
data_X, data_y = fetch_california_housing(return_X_y=True, as_frame=True)
data_all = data_X.copy()
data_all['y'] = data_y
# Create a smaller sample
n_small = 200
data_all_small = data_all.sample(n_small)
data_X_small = data_all_small.drop(labels=['y'], axis='columns')
data_y_small = data_all_small['y']
# K-Fold Cross-Validation
k = 5
kf = KFold(n_splits=k)

# Lists to store performance metrics
rmse_results = []
mae_results = []
r2_results = []
l_models = []

for idx_k, (train_index, test_index) in enumerate(kf.split(data_X_small)):
    print(f'Fold {idx_k + 1} / {k}')
    # Split the data into training and testing sets for this fold
    X_train, X_test = data_X_small.iloc[train_index], data_X_small.iloc[test_index]
    y_train, y_test = data_y_small.iloc[train_index], data_y_small.iloc[test_index]

    # Create and train the model
    temp_svr = svm.SVR()  # using Support Vector Regression
    try:
        temp_svr.fit(X_train, y_train)
    except ConvergenceWarning as e:
        print(f'Model for the {idx_k + 1}th-fold did not converge: {e}')
        continue  # Move to the next iteration

    l_models.append(temp_svr)

    y_pred_test = temp_svr.predict(X_test)  # Use X_test instead of X_valid for testing
    temp_rmse_test = mean_squared_error(y_test, y_pred_test)
    temp_mae_test = mean_absolute_error(y_test, y_pred_test)
    temp_r2_test = r2_score(y_test, y_pred_test)

    # Store metrics for later plotting
    rmse_results.append(temp_rmse_test)
    mae_results.append(temp_mae_test)
    r2_results.append(temp_r2_test)

# Rest of the code remains the same
# After the loop
print('Training completed. Evaluating models...')

# Print the evaluation metrics for the test folds
print(f'Performance on Test Set:')
print(f'RMSE Test: {rmse_results}')
print(f'MAE Test: {mae_results}')
print(f'R2 Score Test: {r2_results}')

print('Done!')