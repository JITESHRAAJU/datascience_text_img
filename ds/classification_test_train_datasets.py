import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, classification_report, confusion_matrix)
from sklearn.impute import SimpleImputer

# CONFIG
TRAIN_PATH = "train_u6lujuX_CVtuZ9i.csv"
TEST_PATH = "test_Y3wMUE5_7gLdaTN.csv"
TARGET_COL = "Loan_Status"  # Change here if target column name changes
ID_COL = "Loan_ID"          # Change here if ID column name changes
OUTPUT_CSV = "submission.csv"

# 1. Load data
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# 2. Combine train and test for preprocessing
test_df[TARGET_COL] = np.nan  # Add dummy target
combined = pd.concat([train_df, test_df], ignore_index=True)

# 3. Drop ID column if exists
if ID_COL in combined.columns:
    id_test = test_df[ID_COL]  # Save test IDs for submission
    combined = combined.drop(columns=[ID_COL])
else:
    id_test = None

# 4. Detect categorical and numerical columns
cat_cols = combined.select_dtypes(include=['object']).columns.tolist()
cat_cols = [col for col in cat_cols if col != TARGET_COL]
num_cols = combined.select_dtypes(include=[np.number]).columns.tolist()

# 5. Preprocessing

# a) Categorical columns
for col in cat_cols:
    combined[col] = combined[col].fillna(combined[col].mode()[0])
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col].astype(str))

# b) Numerical columns
for col in num_cols:
    combined[col] = combined[col].fillna(combined[col].median())

# 6. Split back to train and test
train_processed = combined[combined[TARGET_COL].notna()]
test_processed = combined[combined[TARGET_COL].isna()].drop(columns=[TARGET_COL])

# 7. Prepare data
X = train_processed.drop(columns=[TARGET_COL])
y = train_processed[TARGET_COL]

# Encode target if categorical
if y.dtype == 'object' or y.dtype.name == 'category':
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y)
else:
    target_encoder = None

# 8. Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 9. Feature scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# 10. Model training
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# 11. Model evaluation

y_pred = model.predict(X_val_scaled)

print("Accuracy:", accuracy_score(y_val, y_pred))
print("Precision:", precision_score(y_val, y_pred, average='macro'))  # changed
print("Recall:", recall_score(y_val, y_pred, average='macro'))          # changed
print("F1 Score:", f1_score(y_val, y_pred, average='macro'))            # changed
print("\nClassification Report:\n", classification_report(y_val, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))




# 12. Final prediction on test set
test_scaled = scaler.transform(test_processed)
test_pred = model.predict(test_scaled)

# Inverse transform if needed
if target_encoder:
    test_pred = target_encoder.inverse_transform(test_pred)

# 13. Prepare submission
submission = pd.DataFrame({
    ID_COL: id_test,
    TARGET_COL: test_pred
})

submission.to_csv(OUTPUT_CSV, index=False)
print(f"\nSubmission file '{OUTPUT_CSV}' created successfully!")
