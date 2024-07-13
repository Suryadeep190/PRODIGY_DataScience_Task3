import pandas as pd

# Load the dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip'
df = pd.read_csv(url, sep=';')

# Display the first few rows of the dataset
print(df.head())

# Check the columns and their data types
print(df.info())

# Check for missing values
print(df.isnull().sum())

# Convert categorical variables to dummy/indicator variables
df = pd.get_dummies(df, drop_first=True)

# Separate features (X) and target variable (y)
X = df.drop(columns=['y_yes'])
y = df['y_yes'].map({'no': 0, 'yes': 1})  # Convert 'yes' and 'no' to 1 and 0

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Initialize the Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)

# Train the classifier
dt_classifier.fit(X_train, y_train)

# Predict on the test data
y_pred = dt_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

print(classification_report(y_test, y_pred))

# Confusion matrix
print(confusion_matrix(y_test, y_pred))

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(20,10))
plot_tree(dt_classifier, filled=True, feature_names=X.columns, class_names=['No Purchase', 'Purchase'])
plt.show()
