import streamlit as st
import pandas as pd
import pickle

# Load the trained model
model = pickle.load(open("salary_best_model.pkl", "rb"))

st.title("Employee Salary Prediction App")
st.write("Fill the details below to predict employee salary.")

Job_Title = st.selectbox("Job Title", ["Data Scientist", "Software Engineer", "HR Manager", "Project Manager", "Business Analyst"])
Experience = st.number_input("Experience (Years)", min_value=0, max_value=40, value=1)
Education = st.selectbox("Education Level", ["High School", "Bachelors", "Masters", "PhD"])
Location = st.selectbox("Location", ["Kolkata", "Mumbai", "Delhi", "Bangalore", "Hyderabad"])
Company_Size = st.selectbox("Company Size", ["Small", "Medium", "Large"])
Employment_Type = st.selectbox("Employment Type", ["Full-time", "Part-time", "Intern"])
Work_Mode = st.selectbox("Work Mode", ["On-site", "Remote", "Hybrid"])

# Create DataFrame for prediction
input_data = pd.DataFrame({
    "Job Title": [Job_Title],
    "Experience (Years)": [Experience],
    "Education Level": [Education],
    "Location": [Location],
    "Company Size": [Company_Size],
    "Employment Type": [Employment_Type],
    "Work Mode": [Work_Mode]
})

if st.button("Predict Salary"):
    prediction = model.predict(input_data)
    st.success(f"Predicted Salary: ₹ {round(prediction[0], 2)}")




