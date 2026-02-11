import pandas as pd

data = pd.read_csv("data/student_scores.csv")

print(data.head())

print(data.info())
print(data.describe())
