from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

# Load and split the data
data_X, data_y = fetch_california_housing(return_X_y=True, as_frame=True)
data_X_trval, data_X_test, data_y_trval, data_y_test = train_test_split(data_X, data_y, test_size=0.20, shuffle=True)
data_X_train, data_X_valid, data_y_train, data_y_valid = train_test_split(data_X_trval, data_y_trval, test_size=0.25, shuffle=True)

l_num_hidden = [(5,), (10,), (20,), (50,), (5, 2,), (10, 5,), (20, 20,)]
l_models = []
l_perform_valid = []

# Lists to store performance metrics for each hidden layer configuration
rmse_results = []
mae_results = []
mape_results = []
r2_results = []

for i, hyper_num_hidden in enumerate(l_num_hidden):
    temp_mlp = MLPRegressor(hidden_layer_sizes=hyper_num_hidden, max_iter=500)

    # Inside the loop where models are trained
    print(f'Training model with hidden layers {hyper_num_hidden}...')

    try:
        temp_mlp.fit(data_X_train, data_y_train)
    except ConvergenceWarning as e:
        print(f'Model with hidden layers {hyper_num_hidden} did not converge: {e}')
        continue  # Move to the next iteration

    l_models.append(temp_mlp)

    y_pred_valid = temp_mlp.predict(data_X_valid)
    temp_rmse_valid = mean_squared_error(data_y_valid, y_pred_valid)

    l_perform_valid.append(temp_rmse_valid)

    # Store metrics for later plotting
    rmse_results.append(temp_rmse_valid)
    mae_results.append(mean_absolute_error(data_y_valid, y_pred_valid))
    mape_results.append(mean_absolute_percentage_error(data_y_valid, y_pred_valid))
    r2_results.append(r2_score(data_y_valid, y_pred_valid))

# After the loop
print('Training completed. Evaluating models...')

# Plotting the results
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Model Performance on Validation Set', fontsize=16)

metrics = {'RMSE': rmse_results, 'MAE': mae_results, 'MAPE': mape_results, 'R2 Score': r2_results}

for i, (metric, values) in enumerate(metrics.items()):
    ax = axes[i // 2, i % 2]
    ax.bar(range(len(l_num_hidden)), values, color='skyblue')
    ax.set_xticks(range(len(l_num_hidden)))
    ax.set_xticklabels([str(hidden) for hidden in l_num_hidden], rotation=45, ha='right')
    ax.set_title(metric)
    ax.set_ylabel(metric)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Identify the best model based on the minimum RMSE
best_model_index = np.argmin(rmse_results)
best_model = l_models[best_model_index]

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