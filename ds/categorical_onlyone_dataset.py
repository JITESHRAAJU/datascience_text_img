import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, classification_report, confusion_matrix)

# CONFIG
DATA_PATH = "Iris.csv"   # Only one file
TARGET_COL = "Species"   # Target column
ID_COL = "Id"            # If dataset has an ID column (optional)

# 1. Load data
df = pd.read_csv(DATA_PATH)

# 2. Drop ID column if present
if ID_COL in df.columns:
    df = df.drop(columns=[ID_COL])

# 3. Detect categorical and numerical columns
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
cat_cols = [col for col in cat_cols if col != TARGET_COL]
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# 4. Handle missing values
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# 5. Prepare X and y
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# Encode target
if y.dtype == 'object' or y.dtype.name == 'category':
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y)
else:
    target_encoder = None

# 6. Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 7. Feature scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# 8. Model training
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# 9. Model evaluation
y_pred = model.predict(X_val_scaled)

# Auto detect binary or multi-class
if len(np.unique(y)) == 2:
    average_type = 'binary'
else:
    average_type = 'macro'

print("Accuracy:", accuracy_score(y_val, y_pred))
print("Precision:", precision_score(y_val, y_pred, average=average_type))
print("Recall:", recall_score(y_val, y_pred, average=average_type))
print("F1 Score:", f1_score(y_val, y_pred, average=average_type))
print("\nClassification Report:\n", classification_report(y_val, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))
