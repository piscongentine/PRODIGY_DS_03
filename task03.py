# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load the dataset
url = "https://raw.githubusercontent.com/piscongentine/PRODIGY_DS_3/main/bank-additional.csv"
data = pd.read_csv(url, sep=";")

# Display basic information about the dataset
print("Dataset Overview:")
print(data.head())
print("\nDataset Information:")
print(data.info())

# Check for missing values
print("\nMissing Values Count:")
print(data.isnull().sum())

# Encode categorical variables
le = LabelEncoder()
for col in data.select_dtypes(include=["object"]).columns:
    data[col] = le.fit_transform(data[col])

# Separate features and target
X = data.drop("y", axis=1)
y = data["y"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Visualize the decision tree in text format
print("\nDecision Tree Structure:")
print(export_text(clf, feature_names=list(X.columns)))

# Plot the decision tree
plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=X.columns, class_names=["no", "yes"], filled=True)
plt.title("Decision Tree")
plt.show()

# Tune the model using GridSearchCV
param_grid = {
    "max_depth": [3, 5, 10, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring="accuracy")
grid_search.fit(X_train, y_train)

# Best parameters and accuracy
print("\nBest Parameters from GridSearchCV:", grid_search.best_params_)
print("Best Cross-Validated Accuracy:", grid_search.best_score_)

# Use the best estimator
best_clf = grid_search.best_estimator_

# Evaluate the tuned model
y_pred_tuned = best_clf.predict(X_test)
print("\nEvaluation of Tuned Model:")
print("Accuracy:", accuracy_score(y_test, y_pred_tuned))
print("\nClassification Report:\n", classification_report(y_test, y_pred_tuned))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_tuned))
