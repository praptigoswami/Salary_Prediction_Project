import streamlit as st
import pandas as pd
import joblib

# Load Model
model = joblib.load("salary_best_model.pkl")

st.title("💰Employee Salary Prediction App")
st.write("Fill the details below to predict employee salary.")

# Input Fields (matching training data)
Job_Title = st.selectbox("Job Title", ["Data Scientist", "Software Engineer", "HR Manager", "Project Manager", "Business Analyst"])
Experience = st.number_input("Experience (Years)", min_value=0.0, max_value=40.0, value=1.0, step=0.1)
Education = st.selectbox("Education Level", ["High School", "Bachelors", "Masters", "PhD"])
Location = st.selectbox("Location", ["Kolkata", "Mumbai", "Delhi", "Bangalore", "Hyderabad"])
Company_Size = st.selectbox("Company Size", ["Small", "Medium", "Large"])
Employment_Type = st.selectbox("Employment Type", ["Full-time", "Part-time", "Intern"])
Work_Mode = st.selectbox("Work Mode", ["On-site", "Remote", "Hybrid"])

# Create DataFrame EXACT same column names as training dataset
input_data = pd.DataFrame({
    "Job Title": [Job_Title],
    "Experience (Years)": [Experience],
    "Education Level": [Education],
    "Location": [Location],
    "Company Size": [Company_Size],
    "Employment Type": [Employment_Type],
    "Work Mode": [Work_Mode]
})

# Reorder columns to match model requirement (prevents errors)
if hasattr(model, "feature_names_in_"):
    input_data = input_data.reindex(columns=model.feature_names_in_)

if st.button("Predict Salary"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Salary: ₹ {round(prediction, 2)}")

