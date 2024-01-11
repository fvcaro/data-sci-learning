from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt

skf = StratifiedKFold(n_splits=3)
data = load_breast_cancer()
dataX = data.data
datay = data.target

for i, (train_index, test_index) in enumerate(skf.split(dataX, datay)):
    print(f"Fold {i}:")
    X_train, X_test = dataX[train_index, :], dataX[test_index, :]
    y_train, y_test = datay[train_index], datay[test_index]

    # Replace this with your own hyperparameter settings
    model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    # model = MLPClassifier()
    # Training the MLP model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]  # Probability estimates for the positive class

    # Calculate and print accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    # Calculate and print sensitivity (recall)
    sensitivity = recall_score(y_test, y_pred)
    print(f"Sensitivity (Recall): {sensitivity}")

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Calculate and print specificity
    specificity = tn / (tn + fp)
    print(f"Specificity: {specificity}")

    # Calculate and print precision
    precision = precision_score(y_test, y_pred)
    print(f"Precision: {precision}")

    # Calculate and print roc_auc_score
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    print(f"ROC AUC Score: {roc_auc}")

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc_plot = auc(fpr, tpr)
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc_plot, estimator_name = 'example estimator')
    display.plot()
    plt.show()

print('Done')
