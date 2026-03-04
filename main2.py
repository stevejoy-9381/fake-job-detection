import streamlit as st
import joblib
import re

# Load trained ML model
model = joblib.load("fake_job_model.pkl")

# -------------------------
# Rule-Based Detection
# -------------------------
def rule_based_fake(text):
    text = text.lower()

    red_flags = [
        # Payment related
        "registration fee", "processing fee", "security deposit",
        "training fee", "pay before joining", "investment required",
        "earn money fast", "guaranteed income",
        "verification payment", "small verification payment",
        "pay for verification", "verification fee",
        "pay to confirm job", "job confirmation payment",
        "payment before interview", "fee for job",

        # Suspicious contact
        "whatsapp only", "telegram", "contact via dm",
        "no interview", "direct selection",
        "@gmail.com", "@yahoo.com", "@outlook.com",

        # Unrealistic promises
        "work 1 hour daily", "salary per day 5000",
        "no experience high salary", "easy money",
        "instant joining", "limited seats hurry",

        # Vague company info
        "confidential company", "urgent hiring worldwide",
        "no company details",

        # Joke-style / obvious fake
        "tea drinking job", "watch movies and earn",
        "get paid to sleep", "fun job no work",
        "pretend", "act as", "joke job"
    ]

    # 1️⃣ Keyword-based detection
    if any(flag in text for flag in red_flags):
        return True

    # 2️⃣ Detect suspicious salary pattern (very high daily pay)
    salary_pattern = r"(₹|\$)?\s?\d{4,}\s?(per day|daily)"
    if re.search(salary_pattern, text):
        return True

    # 3️⃣ Detect any sentence asking for payment
    payment_pattern = r"(pay|payment|fee|deposit).{0,30}(job|verification|confirm|registration)"
    if re.search(payment_pattern, text):
        return True

    return False


# -------------------------
# Streamlit UI
# -------------------------
st.title("🚨 Fake Job Detection System")

job_text = st.text_area("Enter Job Description")

if st.button("Analyze"):

    if job_text.strip() == "":
        st.warning("Please enter job description")

    else:
        # Rule-based check first
        if rule_based_fake(job_text):
            st.error("🚨 FAKE Job (Rule-Based Detection)")

        else:
            # ML Model Prediction
            prediction = model.predict([job_text])

            if prediction[0] == 1:
                st.error("🚨 FAKE Job (ML Model)")
            else:
                st.success("✅ REAL Job")