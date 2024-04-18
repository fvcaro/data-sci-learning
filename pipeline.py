from ucimlrepo import fetch_ucirepo
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import ParameterSampler
from sklearn.base import clone


# fetch dataset
national_poll_on_healthy_aging_npha = fetch_ucirepo(id=936)

# data (as pandas dataframes)
data_X = national_poll_on_healthy_aging_npha.data.features
data_y = national_poll_on_healthy_aging_npha.data.targets
print((data_y == 1).sum() / len(data_y))
print((data_y == 2).sum() / len(data_y))
print((data_y == 3).sum() / len(data_y))
# metadata
print(national_poll_on_healthy_aging_npha.metadata)

# variable information
print(national_poll_on_healthy_aging_npha.variables)

data_X = data_X.replace({'Trouble_Sleeping': {-1: np.nan}, 'Employment': {-1: np.nan}, 'Dental_Health': {-1: np.nan}, 'Mental_Health': {-1: np.nan}, 'Physical_Health': {-1: np.nan}, 'Age': {1: 0, 2: 1}, 'Race': {-2: np.nan, -1: np.nan}, 'Gender': {-2: np.nan, -1: np.nan, 1: 0, 2:1}, 'Prescription_Sleep_Medication': {-1: np.nan}})

# Load and split the data
data_X_trval, data_X_test, data_y_trval, data_y_test = train_test_split(data_X, data_y, test_size=0.20, shuffle=True, stratify=data_y)
data_X_train, data_X_valid, data_y_train, data_y_valid = train_test_split(data_X_trval, data_y_trval, test_size=0.25, shuffle=True, stratify=data_y_trval)
print((data_y_test == 1).sum() / len(data_y_test))
print((data_y_test == 2).sum() / len(data_y_test))
print((data_y_test == 3).sum() / len(data_y_test))

ct = ColumnTransformer([('enc_1', OneHotEncoder(sparse_output = False), ['Employment']), ('enc_2', OneHotEncoder(sparse_output = False), ['Race'])], remainder = 'passthrough')

param_grid = {
    'imputer__n_neighbors': [5, 7, 9, 11],
    'imputer__weights': ['uniform', 'distance'],
    'imputer__add_indicator': [True, False],
    'classifier__hidden_layer_sizes': [(10,), (20,), (10, 10), (20, 10)],
    # 'classifier__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'classifier__max_iter': [750, 1250, 1500],
    'classifier__alpha': [0.0001, 0.001, 0.01, 0.1],
}

# Specify the number of random samples
n_samples = 50  # Adjust the number as needed

# Generate random samples
param_samples = list(ParameterSampler(param_grid, n_iter=n_samples, random_state=42))

# Define the pipeline including data preprocessing and model
pipeline = Pipeline([
    ('encoding', ct),
    ('scaling', StandardScaler()),
    ('imputer', KNNImputer()),
    ('classifier', MLPClassifier())
])

l_models = []

# Lists to store performance metrics for each configuration
bas_results = []

for i, params in enumerate(param_samples):
    temp_pipeline = clone(pipeline)
    temp_pipeline.set_params(**params)

    # Inside the loop where models are trained
    print(f'Training model with parameters: {params}...')

    temp_pipeline.fit(data_X_train, data_y_train.squeeze())

    l_models.append(temp_pipeline)

    y_pred_valid = temp_pipeline.predict(data_X_valid)
    # Evaluate the temp model on the validation set
    temp_bas_v = balanced_accuracy_score(data_y_valid, y_pred_valid)

    bas_results.append(temp_bas_v)

    print(f'Model trained successfully. balanced accuracy on validation set: {temp_bas_v:.4f}')

# After the loop
print('Training completed. Evaluating models...')

# Identify the best model based on the balanced accuracy
best_model_index = np.argmax(bas_results)
best_model = l_models[best_model_index]

# Xt = pipeline.transform(data_X_train)
# Xv = pipeline.transform(data_X_valid)
# Xtt = pipeline.transform(data_X_test)

Yt = best_model.predict(data_X_train)
Yv = best_model.predict(data_X_valid)
Ytt = best_model.predict(data_X_test)

Yt_ = best_model.predict_proba(data_X_train)
Yv_ = best_model.predict_proba(data_X_valid)
Ytt_ = best_model.predict_proba(data_X_test)

cm_t = confusion_matrix(data_y_train, Yt)
cm_v = confusion_matrix(data_y_valid, Yv)
cm_tt = confusion_matrix(data_y_test, Ytt)

bas_t = balanced_accuracy_score(data_y_train, Yt)
bas_v = balanced_accuracy_score(data_y_valid, Yv)
bas_tt = balanced_accuracy_score(data_y_test, Ytt)
# best_model[:-1].transform(data_X_train)
print('done')