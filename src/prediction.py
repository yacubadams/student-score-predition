import joblib
import pandas as pd

MODEL_PATH = "../.venv/models/linear_regression.pkl"

def predict_score(hours_studied, attendance, previous_score):
    model = joblib.load(MODEL_PATH)

    input_data = pd.DataFrame([{
        "hours_studied": hours_studied,
        "attendance": attendance,
        "previous_score": previous_score
    }])

    prediction = model.predict(input_data)
    return prediction[0]


if __name__ == "__main__":
    score = predict_score(
        hours_studied=8,
        attendance=100,
        previous_score=91
    )

    print(f"Predicted final score: {score:.2f}")
