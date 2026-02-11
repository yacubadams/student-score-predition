# Student Score Prediction

A simple machine learning project to predict students' final scores based on study hours, attendance, and previous scores. This project simulates a **company-style ML workflow**, including training, retraining, and prediction.

---

## 🏗 Project Structure
student-score-prediction/
│
├── data/
│   ├── raw/         # Historical dataset (source of truth)
│   │    └── student_scores.csv
│   └── new/         # New incoming data for retraining
│        └── new_scores_YYYY_MM_DD.csv
│
├── models/
│    └── linear_regression.pkl  # Trained model artifact
│
├── src/
│   ├── train.py       # Initial training script
│   ├── retrain.py     # Appends new data, retrains, updates .pkl
│   └── predict.py     # Loads trained model and predicts scores
│
├── requirements.txt
└── README.md


Usage


1. Initial Training

Train the baseline model using the historical dataset:

python src/train.py


Reads data/raw/student_scores.csv

Trains a Linear Regression model

Saves the trained model to models/linear_regression.pkl

Prints evaluation metrics (MAE, R²)

2. Adding New Data & Retraining

Whenever new student scores arrive:

Add new CSV files to data/new/ (e.g., new_scores_2026_02_11.csv)

Run:

python src/retrain.py


Appends new data to data/raw/student_scores.csv

Retrains the model on the full dataset

Saves updated .pkl model

Archives processed new data (optional)

3. Making Predictions

Use the trained model for predictions:

python src/predict.py


Or import the function:

from predict import predict_score

score = predict_score(hours_studied=6, attendance=85, previous_score=70)
print(f"Predicted final score: {score:.2f}")


Prediction script does not retrain

Always uses the latest .pkl model

Evaluation Metrics

Metrics printed during training/retraining:

MAE (Mean Absolute Error) → Average prediction error

R² Score → How well the model explains the variance

Example Baseline Metrics:

MAE: 1.04

R² Score: 0.99

Professional Workflow

data/raw/ → source of truth

data/new/ → new incoming data

train.py → initial training

retrain.py → batch retraining

predict.py → inference using .pkl

This structure simulates real-world ML pipelines used in companies.

Dependencies

Python ≥ 3.8

pandas

scikit-learn

joblib

Install with:

pip install -r requirements.txt

Next Steps / Extensions

Add Decision Tree / Random Forest for comparison

Wrap predict.py in a FastAPI / Flask API

Add experiment tracking (MLflow or simple logging)

Automate retraining with a scheduled batch job
