import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Calculate prior probabilities
# np.unique( ): Finds unique classes in the target labels
classes, counts = np.unique(y_train, return_counts=True)
# print('y_train.shape[0]: ', y_train.shape[0])
# Prior is the probability of class c occurring in the training data.
priors = counts / y_train.shape[0]
print('Classes: ', classes)
print('Counts: ', counts)
print('Priors: ', priors)

# Calculate means and variances
means = {}
variances = {}

for c_class in classes:
    X_c = X_train[y_train == c_class]
    print(f'Class {c_class}: ')
    print(f'X_c:\n {X_c} ')
    means[c_class] = np.mean(X_c, axis=0)
    variances[c_class] = np.var(X_c, axis=0)
    print('Means: ', means[c_class])
    print('Variances: ', variances[c_class])

print('\n ...means and variances ...\n')

# Define the Gaussian probability function
def gaussian_probability(x, mean, var):
    coefficient = 1 / np.sqrt(2 * np.pi * var)
    print(f'coefficient:\n {coefficient}: ')
    exponent = np.exp(-((x - mean) ** 2) / (2 * var))
    print(f'exponent:\n {exponent}: ')
    return coefficient * exponent

# Calculate the posterior probability for each class
def calculate_posterior(x):
    posteriors = []
    for y_class in classes:
        print(f'Class {y_class}: ')
        prior = np.log(priors[y_class])
        print(f'x:\n {x}: ')
        print(f'means[y_class]:\n {means[y_class]}: ')
        print(f'variances[y_class]:\n {variances[y_class]}: ')
        likelihood = gaussian_probability(x, means[y_class], variances[y_class])
        print(f'likelihood:\n {likelihood}: ')
        conditional = np.sum(np.log(likelihood))
        posterior = prior + conditional
        posteriors.append(posterior)
        # print(f'\nPosterior for class {y_class} for x = {x}: {posterior}')
    return np.argmax(posteriors)

# Predict for the entire test set
y_pred_manual = [calculate_posterior(x) for x in X_test]

# Calculate accuracy
accuracy_manual = accuracy_score(y_test, y_pred_manual)
print('\nAccuracy of manual GaussianNB: ', accuracy_manual)

# python naive_bayes_manual.py > log_naive_bayes_manual_$(date +%d-%m-%Y_%H.%M.%S).txt 2>&1 &