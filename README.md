# Salary Prediction using Machine Learning (Ensemble Model)

## Overview
This project predicts employee salary based on features like:
- Job Title
- Experience (Years)
- Education Level
- Location
- Company Size
- Employment Type
- Work Mode

We trained multiple models and selected the best performing one using:
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- Voting Regressor (Final Model)

The **Voting Regressor** achieved the highest accuracy.

---

## Files in This Repository
| File | Description |
|------|-------------|
| `Salary_Prediction_Ensemble.ipynb` | Jupyter Notebook containing training & visualization |
| `Salary_Data.csv` | Dataset used for training |
| `salary_best_model.pkl` | Trained model saved for reuse |
| `predict_one_row.py` | Script to predict salary for new employees |

---

## How to Predict Salary for New Employee
1. Create a CSV file named `one_row_example.csv` with the same feature columns.
2. Run:
   ```bash
   python predict_one_row.py
