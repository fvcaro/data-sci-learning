from sklearn.datasets import fetch_california_housing

from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

data_X, data_y = fetch_california_housing(return_X_y=True, as_frame=True)

data_X_trval, data_X_test, data_y_trval, data_y_test = train_test_split(data_X, data_y, test_size=0.20, shuffle=True)
data_X_train, data_X_valid, data_y_train, data_y_valid = train_test_split(data_X_trval, data_y_trval, test_size=0.25, shuffle=True)

# my_mlp = MLPRegressor(hidden_layer_sizes=(10,))
# my_mlp.fit(data_X_trval, data_y_trval)
#
# y_pred_trval = my_mlp.predict(data_X_trval)
# y_pred_test = my_mlp.predict(data_X_test)
#
# rmse_trval = mean_squared_error(data_y_trval, y_pred_trval)
# mae_trval = mean_absolute_error(data_y_trval, y_pred_trval)
# mape_trval = mean_absolute_percentage_error(data_y_trval, y_pred_trval)
# r2_trval = r2_score(data_y_trval, y_pred_trval)
#
# rmse_test = mean_squared_error(data_y_test, y_pred_test)
# mae_test = mean_absolute_error(data_y_test, y_pred_test)
# mape_test = mean_absolute_percentage_error(data_y_test, y_pred_test)
# r2_test = r2_score(data_y_test, y_pred_test)


l_num_hidden = [(5,), (10,), (20,), (50,), (5, 2,), (10, 5,), (20, 20,)]
l_models = []
l_perform_valid = []
for hyper_num_hidden in l_num_hidden:
    # temp_mlp = MLPRegressor()
    # temp_mlp.set_params(**{'hidden_layer_sizes': hyper_num_hidden})
    temp_mlp = MLPRegressor(hidden_layer_sizes=hyper_num_hidden)

    temp_mlp.fit(data_X_train, data_y_train)
    l_models.append(temp_mlp)

    y_pred_valid = temp_mlp.predict(data_X_valid)
    temp_rmse_valid = mean_squared_error(data_y_valid, y_pred_valid)

    l_perform_valid.append(temp_rmse_valid)

l_perform_best = min(l_perform_valid)
idx_best_model = l_perform_valid.index(l_perform_best)
mlp_best_model = l_models[idx_best_model]

y_pred_test = mlp_best_model.predict(data_X_test)
rmse_test = mean_squared_error(data_y_test, y_pred_test)

print('Done!')
