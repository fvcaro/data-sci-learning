from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import ParameterSampler
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

import warnings
import numpy as np
np.random.seed(42)

# Load and split the data
data_X, data_y = fetch_california_housing(return_X_y=True, as_frame=True)
data_X_trval, data_X_test, data_y_trval, data_y_test = train_test_split(data_X, data_y, test_size=0.20, shuffle=True)
data_X_train, data_X_valid, data_y_train, data_y_valid = train_test_split(data_X_trval, data_y_trval, test_size=0.25, shuffle=True)

l_num_hidden = [(5,), (10,), (20,), (50,), (5, 2,), (10, 5,), (20, 20,)]
# Define the parameter space
param_space = {
    'hidden_layer_sizes': l_num_hidden,
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1e0],
    'max_iter': [1000],
    'solver': ['lbfgs']
}

# Specify the number of random samples
n_samples = 10  # Adjust the number as needed

# Generate random samples
param_samples = list(ParameterSampler(param_space, n_iter=n_samples, random_state=42))

l_models = []
l_perform_valid = []

# Lists to store performance metrics for each configuration
rmse_results = []
mae_results = []
mape_results = []
r2_results = []

warnings.filterwarnings("ignore", category=ConvergenceWarning)

for i, params in enumerate(param_samples):
    temp_mlp = MLPRegressor(**params)

    # Inside the loop where models are trained
    print(f'Training model with parameters: {params}...')

    try:
        temp_mlp.fit(data_X_train, data_y_train)
    except ConvergenceWarning as e:
        print(f'Model with parameters {params} did not converge: {e}')
        continue  # Move to the next iteration

    l_models.append(temp_mlp)

    y_pred_valid = temp_mlp.predict(data_X_valid)
    # Evaluate the best model on the validation set
    temp_rmse_valid = mean_squared_error(data_y_valid, y_pred_valid)
    temp_mae_valid = mean_absolute_error(data_y_valid, y_pred_valid)
    temp_mape_valid = mean_absolute_percentage_error(data_y_valid, y_pred_valid)
    temp_r2_valid = r2_score(data_y_valid, y_pred_valid)

    l_perform_valid.append(temp_rmse_valid)

    # Store metrics for later plotting
    rmse_results.append(temp_rmse_valid)
    mae_results.append(mean_absolute_error(data_y_valid, y_pred_valid))
    mape_results.append(mean_absolute_percentage_error(data_y_valid, y_pred_valid))
    r2_results.append(r2_score(data_y_valid, y_pred_valid))

    print(f'Model trained successfully. RMSE on validation set: {temp_rmse_valid:.4f}')

# After the loop
print('Training completed. Evaluating models...')

# Identify the best model based on the minimum RMSE
best_model_index = np.argmin(rmse_results)
best_model = l_models[best_model_index]

# Print hyperparameters of the best model
best_model_hyperparameters = param_samples[best_model_index]
print(f'Best Model Hyperparameters:')
print(f'Hidden Layer Sizes: {best_model_hyperparameters["hidden_layer_sizes"]}')
print(f'Alpha: {best_model_hyperparameters["alpha"]}')
print(f'Activation: {best_model_hyperparameters["activation"]}')

# Predict on the test set with the best model
y_pred_test = best_model.predict(data_X_test)

# Evaluate the best model on the test set
rmse_test = mean_squared_error(data_y_test, y_pred_test)
mae_test = mean_absolute_error(data_y_test, y_pred_test)
mape_test = mean_absolute_percentage_error(data_y_test, y_pred_test)
r2_test = r2_score(data_y_test, y_pred_test)

# Print the evaluation metrics for the best model
print(f'Best Model Performance on Test Set:')
print(f'RMSE Test: {rmse_test:.4f}')
print(f'MAE Test: {mae_test:.4f}')
print(f'MAPE Test: {mape_test:.4f}')
print(f'R2 Score Test: {r2_test:.4f}')

print('Done!')