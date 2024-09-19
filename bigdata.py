# Title of Project:

# Predicting Customer Churn Using Big Data Analytics
# Objective:

# The objective of this project is to build a machine learning model to predict whether a customer will churn or not using a dataset with customer information.
# Data Source:

# For this project, we'll use the Telco Customer Churn dataset from Kaggle, which contains customer data, including demographic, account, and service information.
# Import Libraries

# Start by importing the necessary libraries for data analysis, visualization, and modeling.

# python

# Import necessary libraries
import pandas as pd

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Import Data

# Load the data from your source. In Colab, you can upload files manually or use external links.

# python

# Load the data (assuming it is uploaded to Colab)
from google.colab import files
uploaded = files.upload()

# Load dataset
data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Display the first few rows
data.head()

# Describe Data

# Exploring the dataset will help us understand its structure and contents.

# python

# Get the shape of the dataset
print("Dataset Shape:", data.shape)

# Check for missing values
print(data.isnull().sum())

# Summary statistics
data.describe()

# Check data types
data.info()

#  Data Visualization

# Use visualizations to explore relationships between features.

# python

# Distribution of the target variable (churn)
sns.countplot(x="Churn", data=data)
plt.title("Distribution of Churn")
plt.show()

# Visualizing correlations between numerical features
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()

# Data Preprocessing

# Before modeling, we need to handle categorical variables, missing data, and scaling.

# python

# Handle missing values
data = data.dropna()

# Encode categorical variables (Churn is the target variable)
data['Churn'] = data['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Convert other categorical columns using pd.get_dummies
data = pd.get_dummies(data, drop_first=True)

# Scale the numerical features (if necessary)
scaler = StandardScaler()
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']  # Example columns
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Preview the data
data.head()

# Define Target Variable (y) and Feature Variables (X)

# Separate the independent variables (features) from the dependent variable (target).

# python

# Define target variable 'y' and feature variables 'X'
X = data.drop("Churn", axis=1)
y = data["Churn"]

#  Train Test Split

# Split the dataset into training and testing sets to evaluate the model's performance.

# python

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the shape of the datasets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

#  Modeling

# We'll use a Random Forest Classifier for this task. You can choose other models like Logistic Regression, SVM, or XGBoost.

# python

# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

#  Model Evaluation

# Evaluate the model's performance using accuracy, classification report, and confusion matrix.

# python

# Predict on the test data
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Classification report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Prediction

# Use the model to make predictions on new/unseen data.

# python

# Example of making predictions (for first 5 records in the test set)
new_predictions = model.predict(X_test[:5])
print("Predictions for first 5 test samples:", new_predictions)

#  Explanation

# Explain the outcomes and key insights from the model:

#   Accuracy: This model achieves an accuracy of X%, which shows how well it performs on unseen data.
#     Feature Importance: Random Forest provides feature importance scores, which help in understanding which features contributed the most to predicting customer churn.
#     Next Steps: Further tuning of the model, adding more features, or using advanced techniques like XGBoost might improve the results.