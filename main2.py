import streamlit as st
import joblib

# Load trained model
model = joblib.load("fake_job_model.pkl")

# Title
st.title("Fake Job Detection System")

st.write("Enter a job description to check if it is Fake or Real")

# User input
job_text = st.text_area("Job Description")

# Prediction
if st.button("Predict"):

    if job_text.strip() == "":
        st.warning("Please enter job description")
    
    else:
        prediction = model.predict([job_text])[0]

        if prediction == 1:
            st.error("This job posting looks FAKE")
        else:
            st.success("This job posting looks REAL")