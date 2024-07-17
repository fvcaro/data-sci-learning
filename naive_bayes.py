# Import necessary libraries
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

# Example dataset (replace this with your actual dataset)
data = {
    'Fever': [37.5, 39.0, 40.0, 37.0, 38.5, 41.0],
    'Cough_Severity': [2, 3, 5, 1, 4, 5],
    'Breathing_Difficulty': [1, 3, 5, 0, 4, 5],
    'Fatigue_Level': [1, 3, 4, 1, 3, 5],
    'Blood_Pressure': ['120/80', '130/85', '140/90', '115/75', '135/88', '150/95'],
    'Oxygen_Saturation': [98, 94, 90, 99, 92, 88],
    'Classification': ['Soft', 'Moderate', 'Grave', 'Soft', 'Moderate', 'Grave']
}

# Create a DataFrame
df = pd.DataFrame(data)

# Assume you need to convert 'Blood_Pressure' into numerical feature(s), you'd typically do this:
df['Blood_Pressure'] = df['Blood_Pressure'].apply(lambda x: float(x.split('/')[0]))  # Example: Taking only systolic value

# Separate features and target variable
X = df.drop('Classification', axis=1)
y = df['Classification']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Gaussian Naive Bayes classifier
gnb = GaussianNB()

# Train the classifier
gnb.fit(X_train, y_train)

# Predict on the test set
y_pred = gnb.predict(X_test)

# Evaluate the classifier
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))