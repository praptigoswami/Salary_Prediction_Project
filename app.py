import streamlit as st
import pandas as pd
import joblib

# Load Model
model = joblib.load("salary_best_model.pkl")

st.title("Employee Salary Prediction App")
st.write("Fill the details below to predict employee salary.")

# User Inputs
Job_Title = st.selectbox("Job Title", ["Data Scientist", "Software Engineer", "HR Manager", "Project Manager", "Business Analyst"])
Experience = st.number_input("Experience (Years)", min_value=0.0, max_value=40.0, value=1.0, step=0.1)
Education = st.selectbox("Education Level", ["High School", "Bachelors", "Masters", "PhD"])
Location = st.selectbox("Location", ["Kolkata", "Mumbai", "Delhi", "Bangalore", "Hyderabad"])
Company_Size = st.selectbox("Company Size", ["Small", "Medium", "Large"])
Employment_Type = st.selectbox("Employment Type", ["Full-time", "Part-time", "Intern"])
Work_Mode = st.selectbox("Work Mode", ["On-site", "Remote", "Hybrid"])

# Create Input DataFrame (with corrected column names)
input_data = pd.DataFrame({
    "Job_Title": [Job_Title],
    "Experience": [Experience],
    "Education": [Education],
    "Location": [Location],
    "Company_Size": [Company_Size],
    "Employment_Type": [Employment_Type],
    "Work_Mode": [Work_Mode]
})

# Align columns to model training columns (Prevents 'missing columns' error)
if hasattr(model, "feature_names_in_"):
    model_cols = list(model.feature_names_in_)
    input_data = input_data.reindex(columns=model_cols)

if st.button("Predict Salary"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Salary: ₹ {round(prediction, 2)}")
