from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import ParameterGrid
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
np.random.seed(42)

# Load and split the data
data_X, data_y = fetch_california_housing(return_X_y=True, as_frame=True)
data_X_trval, data_X_test, data_y_trval, data_y_test = train_test_split(data_X, data_y, test_size=0.20, shuffle=True)
data_X_train, data_X_valid, data_y_train, data_y_valid = train_test_split(data_X_trval, data_y_trval, test_size=0.25, shuffle=True)

kernel = 'rbf'
param_grid = {'gamma': [1e-1, 1e0, 1e1], 'C': [1e-1, 1e0, 1e1]}

# Extract values from the param_grid
l_gammas = param_grid['gamma']
l_Cs = param_grid['C']

# Lists to store performance metrics
rmse_results = []
mae_results = []
r2_results = []
l_models = []

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Initialize variables to keep track of the best model and its performance
best_model = None
best_rmse = float('inf')  # Initialize with a large value

# Use ParameterGrid to generate all combinations of hyperparameters
for i, params in enumerate(ParameterGrid(param_grid), 1):
    gamma, C = params['gamma'], params['C']
    
    print(f'Training model {i}/{len(ParameterGrid(param_grid))} with gamma={gamma}, C={C}')
    
    temp_svr = svm.SVR(kernel=kernel, gamma=gamma, C=C)
    
    try:
        temp_svr.fit(data_X_train, data_y_train)
    except ConvergenceWarning as e:
        print(f'Model with gamma={gamma}, C={C} did not converge: {e}')
        continue  # Move to the next iteration

    l_models.append(temp_svr)

    y_pred_valid = temp_svr.predict(data_X_valid)
    temp_rmse_valid = mean_squared_error(data_y_valid, y_pred_valid)

    # Store metrics for later plotting
    rmse_results.append(temp_rmse_valid)
    mae_results.append(mean_absolute_error(data_y_valid, y_pred_valid))
    r2_results.append(r2_score(data_y_valid, y_pred_valid))

    print(f'Model trained successfully. RMSE on validation set: {temp_rmse_valid:.4f}')

    # Update the best model if the current one is better
    if temp_rmse_valid < best_rmse:
        best_rmse = temp_rmse_valid
        best_model = temp_svr

# After the loop
print('Training completed. Evaluating models...')

# Print the hyperparameters of the best model
best_gamma = best_model.get_params()['gamma']
best_C = best_model.get_params()['C']
print(f'Best hyperparameters: gamma={best_gamma}, C={best_C}')

# Plotting the results in 3D for RMSE
fig = plt.figure(figsize=(12, 10))
fig.suptitle('Model Performance on Validation Set', fontsize=16)
ax = fig.add_subplot(111, projection='3d')

# In this example, np.log10() transforms the gamma and C values to log scale for better visualization. 
# The plot_surface function is then used to create a 3D surface plot, where the x and y axes represent the 
# log-transformed gamma and C values, and the z-axis represents the RMSE values.
# Create meshgrid for 3D plot
X, Y = np.meshgrid(np.log10(l_gammas), np.log10(l_Cs))

# Plotting RMSE values in 3D
ax.plot_surface(X, Y, np.array(rmse_results).reshape(X.shape), cmap='viridis', alpha=0.8)
ax.set_xlabel('log10(Gamma)')
ax.set_ylabel('log10(C)')
ax.set_zlabel('RMSE')
ax.set_title('RMSE Performance on Validation Set')

plt.show()

# Identify the best model based on the minimum RMSE
best_model_index = np.argmin(rmse_results)
best_model = l_models[best_model_index]

# Predict on the test set with the best model
y_pred_test = best_model.predict(data_X_test)

# Evaluate the best model on the test set
rmse_test = mean_squared_error(data_y_test, y_pred_test)
mae_test = mean_absolute_error(data_y_test, y_pred_test)
r2_test = r2_score(data_y_test, y_pred_test)

# Print the evaluation metrics for the best model
print(f'Best Model Performance on Test Set:')
print(f'RMSE Test: {rmse_test:.4f}')
print(f'MAE Test: {mae_test:.4f}')
print(f'R2 Score Test: {r2_test:.4f}')

print('Done!')