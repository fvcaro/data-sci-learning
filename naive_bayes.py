from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

x_test = [[6.1, 2.8, 4.7, 1.2]]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the scikit-learn Gaussian Naive Bayes classifier
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_sklearn = gnb.predict(X_test)

print(gnb.predict_proba(x_test))

# Calculate accuracy
accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
print('\nAccuracy of sklearn GaussianNB: ', accuracy_sklearn)

# python naive_bayes.py > log_naive_bayes_$(date +%d-%m-%Y_%H.%M.%S).txt 2>&1 &