import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Paths
train_path = "train.csv"   # replace
test_path = "test.csv"     # replace

# Load data
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Assume target column is known
TARGET_COL = 'SalePrice'  # or whatever target column (you will know from question)

# Handle missing values (simple example)
train_df.fillna(train_df.median(), inplace=True)
test_df.fillna(test_df.median(), inplace=True)

# Feature-target split
X_train = train_df.drop(columns=[TARGET_COL])
y_train = train_df[TARGET_COL]

X_test = test_df.copy()

# Normalize
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Model evaluation (optional, only on train-validation if you split)
# Predict on test
test_predictions = model.predict(X_test_scaled)

# Create submission file
submission = pd.DataFrame({
    'Id': test_df['Id'],   # Assuming Id column exists
    TARGET_COL: test_predictions
})
submission.to_csv('submission.csv', index=False)
