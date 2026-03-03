import streamlit as st
import joblib

# Load model
model = joblib.load("fake_job_model.pkl")

# -------------------------
# Rule-Based Detection
# -------------------------
def rule_based_fake(text):
    red_flags = [
        "tea", "coffee", "clerk", "pretend",
        "no real work", "act as", "joke", "fun job"
    ]
    text = text.lower()
    return any(flag in text for flag in red_flags)

# -------------------------
# UI
# -------------------------
st.title("🚨 Fake Job Detection")

job_text = st.text_area("Enter Job Description")

if st.button("Analyze"):

    if job_text.strip() == "":
        st.warning("Please enter job description")

    else:
        # First apply rule-based check
        if rule_based_fake(job_text):
            st.error("🚨 FAKE Job (Rule-Based Detection)")

        else:
            # If no red flags → use ML model
            prediction = model.predict([job_text])

            if prediction[0] == 1:
                st.error("🚨 FAKE Job (ML Model)")
            else:
                st.success("✅ REAL Job")