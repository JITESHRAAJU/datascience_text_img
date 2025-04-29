import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             r2_score, mean_absolute_error, mean_squared_error)

# Load dataset
try:
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    test['Loan_Status'] = np.nan
    df = pd.concat([train, test], ignore_index=True)
    target = 'Loan_Status'
    separate = True
    print("Train and Test files loaded separately.")
except:
    df = pd.read_csv("dataset.csv")
    target = input("Enter the target column: ")
    separate = False
    print("Single dataset loaded.")

# Show basic stats
print("\n--- Basic Statistics ---")
print(df.describe(include='all').T[['mean', '50%', 'std']])  # median = 50%

# Handle missing values
num_cols = df.select_dtypes(include='number').columns.drop(target, errors='ignore')
cat_cols = df.select_dtypes(include='object').columns.drop(target, errors='ignore')
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Encode & Scale
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])
scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Split into features & target
if separate:
    train_df = df[df[target].notna()]
    test_df = df[df[target].isna()].drop(columns=[target])
    X = train_df.drop(columns=[target])
    y = train_df[target]
else:
    X = df.drop(columns=[target])
    y = df[target]

# Split for validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose model
if y.dtype == 'object' or y.nunique() <= 10:
    model = RandomForestClassifier()
    task = 'classification'
else:
    model = LinearRegression()
    task = 'regression'

# Train model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate
if task == 'classification':
    print("\n--- Classification Report ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, zero_division=1))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

else:
    print("\n--- Regression Report ---")
    print("R² Score:", r2_score(y_test, y_pred))
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))

# Predict test set if given
if separate:
    predictions = model.predict(test_df)
    print("\nTest Predictions:\n", predictions)
