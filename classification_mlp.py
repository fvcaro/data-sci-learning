from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, precision_score, roc_auc_score
from sklearn.metrics import RocCurveDisplay, roc_curve, auc
import matplotlib.pyplot as plt

# Load the Breast Cancer dataset
data = load_breast_cancer()
dataX = data.data
datay = data.target

# Create a StratifiedKFold object for cross-validation
skf_in = StratifiedKFold(n_splits=2)
skf_out = StratifiedKFold(n_splits=3)

# Define the hyperparameter space
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'max_iter': [500, 1000, 1500],
    'alpha': [0.0001, 0.001, 0.01],
}

# Create an MLPClassifier
model = MLPClassifier(random_state=42)

# Create a GridSearchCV object
grid_search = GridSearchCV(model, param_grid, cv=skf_in, scoring='roc_auc')

# Loop through the folds
for i, (train_index, test_index) in enumerate(skf_out.split(dataX, datay)):
    print(f"Fold {i+1}:")

    # Split the data into training and test sets
    X_train, X_test = dataX[train_index, :], dataX[test_index, :]
    y_train, y_test = datay[train_index], datay[test_index]

    # Fit the GridSearchCV object on the training data
    grid_search.fit(X_train, y_train)

    # Get the best model from the grid search
    best_model = grid_search.best_estimator_
    # Print the best hyperparameters found by GridSearchCV
    print("Best Hyperparameters:", grid_search.best_params_)

    # Make predictions on the test set
    y_pred = best_model.predict(X_test)
    y_pred_prob = best_model.predict_proba(X_test)[:, 1]  # Probability estimates for the positive class

    # Calculate and print ROC AUC Score
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    print(f"ROC AUC Score: {roc_auc}")

    # Calculate ROC curve and plot
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc_plot = auc(fpr, tpr)
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc_plot, estimator_name='Best MLP Classifier')
    display.plot()
    plt.show()

print('Done')