import streamlit as st
import joblib
import re
from sklearn.base import BaseEstimator, TransformerMixin

# MUST define this before loading
class TextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        cleaned = []
        for text in X:
            text = text.lower()
            text = re.sub(r'[^a-zA-Z]', ' ', text)
            cleaned.append(text)
        return cleaned

# Load trained pipeline
model = joblib.load("fake_job_model.pkl")

st.title("🚨 Fake Job Detection")

job_text = st.text_area("Enter Job Description")

if st.button("Check"):
    if job_text:
        prediction = model.predict([job_text])
        if prediction[0] == 1:
            st.error("🚨 FAKE Job Posting")
        else:
            st.success("✅ REAL Job Posting")