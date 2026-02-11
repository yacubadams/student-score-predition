import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

# Paths
CSV_PATH = "student_scores.csv"
MODEL_PATH = "../.venv/models/linear_regression.pkl"

# Load current data
df = pd.read_csv(CSV_PATH)

# Optionally: new data to append (any number of rows)
# This could also come from a separate CSV or API in the future
new_data = pd.DataFrame({
    "hours_studied": [9.0, 7.0],
    "attendance": [97, 88],
    "previous_score": [85, 72],
    "final_score": [90, 78]
})

# Append new rows to CSV
df = pd.concat([df, new_data], ignore_index=True)

# Save updated CSV (source of truth)
df.to_csv(CSV_PATH, index=False)

# Split features and target
X = df[["hours_studied", "attendance", "previous_score"]]
y = df["final_score"]

# Optional: keep a test split to check performance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print(f"Retrained model: MAE={mae:.2f}, R2={r2:.2f}")

# Save updated model
os.makedirs("../.venv/models", exist_ok=True)
joblib.dump(model, MODEL_PATH)

print(f"Updated model saved to {MODEL_PATH}")
