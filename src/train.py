import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

# Load data
DATA_PATH = "student_scores.csv"
df = pd.read_csv(DATA_PATH)

# Features and target
X = df[["hours_studied", "attendance", "previous_score"]]
y = df["final_score"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"MAE: {mae:.2f}")
print(f"R2 Score: {r2:.2f}")

# Save model
os.makedirs("../.venv/models", exist_ok=True)
joblib.dump(model, "../.venv/models/linear_regression.pkl")
