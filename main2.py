import streamlit as st
import joblib
import re
import difflib

# -------------------------
# Load trained ML model
# -------------------------
model = joblib.load("fake_job_model.pkl")

# -------------------------
# Utility Functions
# -------------------------

def fuzzy_match(text, keyword, threshold=0.85):
    words = text.split()
    for word in words:
        similarity = difflib.SequenceMatcher(None, word, keyword).ratio()
        if similarity >= threshold:
            return True
    return False


def count_uppercase_words(text):
    return sum(1 for word in text.split() if word.isupper() and len(word) > 2)


def count_exclamations(text):
    return text.count("!")


# -------------------------
# Advanced Rule-Based Detection
# -------------------------

def rule_based_fake(text):
    text_lower = text.lower()
    risk_score = 0
    reasons = []

    red_flags = [
        "registration fee", "processing fee", "security deposit",
        "training fee", "pay before joining", "investment required",
        "earn money fast", "guaranteed income",
        "verification payment", "small verification payment",
        "pay for verification", "verification fee",
        "pay to confirm job", "job confirmation payment",
        "payment before interview", "fee for job",
        "confidential company", "urgent hiring worldwide",
        "no company details", "instant joining",
        "limited seats hurry", "direct selection"
    ]

    # 1️⃣ Exact keyword detection
    for flag in red_flags:
        if flag in text_lower:
            risk_score += 2
            reasons.append(f"Keyword detected: {flag}")

    # 2️⃣ Fuzzy detection for disguised words
    suspicious_words = ["payment", "fee", "deposit", "charge", "amount"]
    for word in suspicious_words:
        if fuzzy_match(text_lower, word):
            risk_score += 2
            reasons.append(f"Suspicious variation of word detected: {word}")

    # 3️⃣ Stronger payment regex
    payment_pattern = r"(pay|fee|deposit|charge|amount|processing|security).{0,40}(job|verification|confirm|registration|onboarding|joining)"
    if re.search(payment_pattern, text_lower):
        risk_score += 3
        reasons.append("Suspicious payment-related sentence detected")

    # 4️⃣ Suspicious salary pattern
    salary_pattern = r"(₹|\$)?\s?\d{4,}\s?(per day|daily)"
    if re.search(salary_pattern, text_lower):
        risk_score += 2
        reasons.append("Unrealistic daily salary detected")

    # 5️⃣ Behavioral signals
    uppercase_count = count_uppercase_words(text)
    exclamation_count = count_exclamations(text)

    if uppercase_count > 5:
        risk_score += 1
        reasons.append("Excessive uppercase words")

    if exclamation_count > 3:
        risk_score += 1
        reasons.append("Too many exclamation marks")

    return risk_score, reasons


# -------------------------
# Streamlit UI
# -------------------------

st.title("🚨 Advanced Fake Job Detection System")

job_text = st.text_area("Enter Job Description")

if st.button("Analyze"):

    if job_text.strip() == "":
        st.warning("Please enter job description")

    else:
        risk_score, reasons = rule_based_fake(job_text)

        st.subheader("📊 Risk Analysis")

        st.write(f"Risk Score: {risk_score}")

        if reasons:
            st.write("Reasons:")
            for r in reasons:
                st.write(f"- {r}")

        # Risk level interpretation
        if risk_score >= 5:
            st.error("🔴 HIGH RISK - Likely FAKE Job (Rule-Based)")
        elif risk_score >= 3:
            st.warning("🟡 MEDIUM RISK - Suspicious Job")
        else:
            # ML model prediction
            prediction = model.predict([job_text])
            proba = None

            try:
                proba = model.predict_proba([job_text])[0][1]
            except:
                pass

            if prediction[0] == 1:
                st.error("🚨 FAKE Job (ML Model)")
                if proba:
                    st.write(f"Confidence: {round(proba * 100, 2)}%")
            else:
                st.success("✅ REAL Job")
                if proba:
                    st.write(f"Confidence: {round((1 - proba) * 100, 2)}%")