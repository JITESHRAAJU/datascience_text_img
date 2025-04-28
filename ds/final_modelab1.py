# Generalized Model Building Code (Classification/Regression + 1 or 2 datasets)

import pandas as pd
import numpy as np

# Load Datasets
try:
    train_df = pd.read_csv("train.csv")     # If train and test given separately
    test_df = pd.read_csv("test.csv")
    separate = True
    print("Train and Test files loaded separately.")
except:
    df = pd.read_csv("dataset.csv")          # If only one file given
    separate = False
    print("Single dataset loaded.")

# Combine if train and test separately
if separate:
    test_df['Target'] = np.nan   # Placeholder for test
    combined = pd.concat([train_df, test_df], ignore_index=True)
else:
    combined = df.copy()

# Basic Analysis
print("\nDataset Head:\n", combined.head())
print("\nDataset Description:\n", combined.describe())
print("\nMissing Values:\n", combined.isnull().sum())

# Fill Missing Values
num_cols = combined.select_dtypes(include=[np.number]).columns
cat_cols = combined.select_dtypes(include=['object']).columns

combined[num_cols] = combined[num_cols].fillna(combined[num_cols].mean())
for col in cat_cols:
    combined[col] = combined[col].fillna(combined[col].mode()[0])

print("\nMissing Values After Filling:\n", combined.isnull().sum())

# Preprocessing - Encoding Categorical
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
le = LabelEncoder()
for col in cat_cols:
    combined[col] = le.fit_transform(combined[col].astype(str))

# Scaling
scaler = MinMaxScaler()
combined[num_cols] = scaler.fit_transform(combined[num_cols])

# Split back if needed
if separate:
    train_processed = combined[combined['Target'].notna()]
    test_processed = combined[combined['Target'].isna()].drop(columns=['Target'])
    X = train_processed.drop(columns=['Target'])
    y = train_processed['Target']
else:
    target_col = input("\nEnter the target/output column name: ")
    X = combined.drop(columns=[target_col])
    y = combined[target_col]

# Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nData Preprocessing Completed!")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

# Model Selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

if y.dtype == 'object' or (y.nunique() <= 10 and y.dtype in ['int64', 'int32']):
    model = RandomForestClassifier(random_state=42)
    task = 'classification'
else:
    model = LinearRegression()
    task = 'regression'

# Train the Model
model.fit(X_train, y_train)
print("\nModel Training Completed!")

# Predict and Evaluate
y_pred = model.predict(X_test)

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report,
                              r2_score, mean_absolute_error, mean_squared_error)

if task == 'classification':
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nPrecision:", precision_score(y_test, y_pred, average='weighted'))
    print("\nRecall:", recall_score(y_test, y_pred, average='weighted'))
    print("\nF1 Score:", f1_score(y_test, y_pred, average='weighted'))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=1))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
else:
    print("\nR2 Score:", r2_score(y_test, y_pred))
    print("\nMean Absolute Error:", mean_absolute_error(y_test, y_pred))
    print("\nMean Squared Error:", mean_squared_error(y_test, y_pred))

# If test dataset was given separately, predict for test
if separate:
    test_predictions = model.predict(test_processed)
    print("\nTest Dataset Predictions:\n", test_predictions)

print("\nAll tasks completed successfully!")
