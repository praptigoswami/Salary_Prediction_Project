import pickle
import pandas as pd

model_path = "salary_best_model.pkl"
data_path = "one_row_example.csv"  # Change to your input CSV

with open(model_path, "rb") as f:
    model = pickle.load(f)

new_data = pd.read_csv(data_path)
prediction = model.predict(new_data)

print("Predicted Salary:", float(prediction[0]))
