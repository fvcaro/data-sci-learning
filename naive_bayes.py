import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the scikit-learn Gaussian Naive Bayes classifier
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_sklearn = gnb.predict(X_test)
accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)

# Manual implementation of Gaussian Naive Bayes
class ManualGaussianNB:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.means = {}
        self.variances = {}
        self.priors = {}
        
        for c in self.classes:
            X_c = X[y == c]
            self.means[c] = np.mean(X_c, axis=0)
            self.variances[c] = np.var(X_c, axis=0)
            self.priors[c] = X_c.shape[0] / X.shape[0]
    
    def gaussian_probability(self, x, mean, var):
        coefficient = 1 / np.sqrt(2 * np.pi * var)
        exponent = np.exp(-((x - mean) ** 2) / (2 * var))
        return coefficient * exponent
    
    def predict(self, X):
        predictions = []
        for x in X:
            posteriors = []
            for c in self.classes:
                prior = np.log(self.priors[c])
                conditional = np.sum(np.log(self.gaussian_probability(x, self.means[c], self.variances[c])))
                posterior = prior + conditional
                posteriors.append(posterior)
            predictions.append(np.argmax(posteriors))
        return predictions

# Train and predict using the manual Gaussian Naive Bayes classifier
manual_gnb = ManualGaussianNB()
manual_gnb.fit(X_train, y_train)
y_pred_manual = manual_gnb.predict(X_test)
accuracy_manual = accuracy_score(y_test, y_pred_manual)

# Print the results
print(f"Accuracy (scikit-learn): {accuracy_sklearn}")
print(f"Accuracy (manual): {accuracy_manual}")

# Create a DataFrame to compare actual vs. predicted values
comparison = pd.DataFrame({
    'Actual': y_test,
    'Predicted_sklearn': y_pred_sklearn,
    'Predicted_manual': y_pred_manual
})

print(comparison)