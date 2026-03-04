# =====================================================
# COMPLETE HYBRID FAKE JOB DETECTION SYSTEM
# Rule + ML + DistilBERT + Advanced Logging + Streamlit
# =====================================================

import streamlit as st
import re
import torch
import joblib
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import logging

# =====================================================
# LOGGING CONFIGURATION
# =====================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("fake_job_detection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =====================================================
# MODEL LOADING WITH CACHING
# =====================================================

@st.cache_resource
def load_ml_model():
    """Load cached ML model"""
    try:
        model = joblib.load("fake_job_model.pkl")
        logger.info("ML model loaded successfully")
        return model
    except FileNotFoundError:
        logger.error("ML model file not found")
        st.error("ML model file 'fake_job_model.pkl' not found")
        return None

@st.cache_resource
def load_bert_model():
    """Load cached BERT model and tokenizer"""
    try:
        MODEL_NAME = "distilbert-base-uncased"
        tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
        bert_model = DistilBertForSequenceClassification.from_pretrained(
            MODEL_NAME, 
            num_labels=2
        )
        bert_model.eval()
        logger.info("BERT model loaded successfully")
        return tokenizer, bert_model
    except Exception as e:
        logger.error(f"Error loading BERT model: {e}")
        st.error(f"Error loading BERT model: {e}")
        return None, None

# Load models
ml_model = load_ml_model()
tokenizer, bert_model = load_bert_model()

# =====================================================
# COMPREHENSIVE FRAUD PATTERN DATABASE
# =====================================================

FRAUD_PATTERNS = {
    "pre_offer": {
        "patterns": [
            r"before\s+(contract|offer|confirmation)",
            r"prior\s+to\s+(contract|offer|confirmation)",
            r"offer\s+(issued|released)\s+after",
            r"subject\s+to\s+(activation|clearance|verification)",
            r"pending\s+(clearance|activation|approval)",
            r"conditional\s+(offer|employment)",
            r"contingent\s+upon\s+(payment|verification)",
        ],
        "weight": 3,
        "risk_level": "HIGH"
    },
    "activation": {
        "patterns": [
            r"activate\s+(employment|employee|profile|id)",
            r"profile\s+activation",
            r"credential\s+(activation|provisioning)",
            r"reserve\s+(training|slot|schedule)",
            r"setup\s+fee",
            r"activation\s+fee",
            r"initiation\s+fee",
        ],
        "weight": 2,
        "risk_level": "HIGH"
    },
    "third_party": {
        "patterns": [
            r"third[- ]party\s+(onboarding|partner|portal)",
            r"compliance\s+(gateway|portal|partner)",
            r"workforce\s+processing\s+partner",
            r"secure\s+employment\s+portal",
            r"validation\s+partner",
            r"external\s+(processing|verification)",
            r"partner\s+(verification|portal|system)",
        ],
        "weight": 2,
        "risk_level": "MEDIUM"
    },
    "pressure": {
        "patterns": [
            r"prioritize\s+serious\s+candidates",
            r"confirm\s+commitment",
            r"high\s+applicant\s+volume",
            r"limited\s+slots",
            r"act\s+quickly",
            r"urgent",
            r"deadline\s+(today|tomorrow|this\s+week)",
            r"last\s+(minute|call|chance)",
            r"don't\s+miss\s+out",
        ],
        "weight": 1,
        "risk_level": "MEDIUM"
    },
    "payment": {
        "patterns": [
            r"payment\s+required",
            r"upfront\s+fee",
            r"processing\s+fee",
            r"deposit",
            r"bank\s+transfer",
            r"wire\s+transfer",
            r"cryptocurrency",
            r"gift\s+card",
            r"money\s+order",
        ],
        "weight": 3,
        "risk_level": "CRITICAL"
    },
    "suspicious_contact": {
        "patterns": [
            r"whatsapp",
            r"telegram",
            r"signal",
            r"private\s+email",
            r"personal\s+email",
            r"gmail\s+account",
            r"avoid\s+(company|official)\s+channels",
            r"don't\s+contact\s+(company|hr|office)",
        ],
        "weight": 2,
        "risk_level": "HIGH"
    },
    "vague_details": {
        "patterns": [
            r"unclear\s+(job|role|position)",
            r"to\s+be\s+determined",
            r"will\s+be\s+discussed",
            r"salary\s+(not\s+)?(specified|determined|discussed)",
            r"benefits\s+vary",
            r"location\s+(flexible|remote|worldwide)",
        ],
        "weight": 1,
        "risk_level": "LOW"
    },
    "too_good": {
        "patterns": [
            r"easy\s+money",
            r"high\s+salary\s+for\s+less\s+work",
            r"minimal\s+experience",
            r"no\s+experience\s+required",
            r"guaranteed\s+position",
            r"guaranteed\s+income",
        ],
        "weight": 2,
        "risk_level": "MEDIUM"
    }
}

# =====================================================
# DETECTION FUNCTIONS
# =====================================================

def extract_text_features(text):
    """Extract linguistic features from text"""
    features = {
        "word_count": len(text.split()),
        "char_count": len(text),
        "avg_word_length": np.mean([len(w) for w in text.split()]) if text.split() else 0,
        "uppercase_ratio": sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0,
        "digit_ratio": sum(1 for c in text if c.isdigit()) / len(text) if len(text) > 0 else 0,
        "special_char_ratio": sum(1 for c in text if not c.isalnum() and c.isascii()) / len(text) if len(text) > 0 else 0,
    }
    return features

def compute_rule_score(text):
    """Compute rule-based fraud score with detailed analysis"""
    score = 0
    reasons = []
    pattern_matches = {}

    for category, data in FRAUD_PATTERNS.items():
        matched = False
        for pattern in data["patterns"]:
            if re.search(pattern, text, re.IGNORECASE):
                matched = True
                break
        
        if matched:
            score += data["weight"]
            pattern_matches[category] = {
                "weight": data["weight"],
                "risk_level": data["risk_level"]
            }
            reasons.append(f"{category.replace('_', ' ').title()}: {data['risk_level']} risk")

    return score, reasons, pattern_matches

def normalize_rule_score(score, max_score=20):
    """Normalize rule score to 0-1 range"""
    return min(score / max_score, 1.0)

def get_ml_probability(text):
    """Get ML model probability"""
    if ml_model is None:
        logger.warning("ML model not available, returning 0.5")
        return 0.5
    try:
        proba = ml_model.predict_proba([text])[0][1]
        return float(proba)
    except Exception as e:
        logger.error(f"Error in ML prediction: {e}")
        return 0.5

def get_bert_probability(text):
    """Get BERT model probability"""
    if bert_model is None or tokenizer is None:
        logger.warning("BERT model not available, returning 0.5")
        return 0.5
    
    try:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = bert_model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1)
        return float(probs[0][1].item())
    except Exception as e:
        logger.error(f"Error in BERT prediction: {e}")
        return 0.5

def determine_risk_level(score):
    """Determine overall risk level"""
    if score >= 0.85:
        return "CRITICAL"
    elif score >= 0.7:
        return "HIGH"
    elif score >= 0.5:
        return "MEDIUM"
    elif score >= 0.3:
        return "LOW"
    else:
        return "VERY LOW"

def log_suspicious_job(text, result, filepath="suspicious_logs.json"):
    """Log suspicious or fake jobs with full details"""
    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "text": text[:500],  # Truncate for storage
            "analysis": result,
            "text_features": extract_text_features(text)
        }

        # Append to JSON Lines format
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        logger.info(f"Logged suspicious job: {result['label']}")
    except Exception as e:
        logger.error(f"Error logging suspicious job: {e}")

def analyze_job(text):
    """Complete job analysis combining all detection methods"""
    if not text.strip():
        return None

    # Rule-based detection
    rule_score, rule_reasons, pattern_matches = compute_rule_score(text)
    rule_norm = normalize_rule_score(rule_score)

    # ML-based detection
    ml_prob = get_ml_probability(text)

    # BERT-based detection
    bert_prob = get_bert_probability(text)

    # Weighted ensemble
    final_score = (
        0.4 * bert_prob +      # BERT (40%)
        0.3 * ml_prob +        # ML Model (30%)
        0.3 * rule_norm        # Rules (30%)
    )

    # Determine label and risk level
    if final_score >= 0.75:
        label = "FAKE"
        risk_level = "CRITICAL"
    elif final_score >= 0.6:
        label = "SUSPICIOUS"
        risk_level = "HIGH"
    elif final_score >= 0.45:
        label = "QUESTIONABLE"
        risk_level = "MEDIUM"
    else:
        label = "LIKELY REAL"
        risk_level = "LOW"

    # Extract text features
    text_features = extract_text_features(text)

    result = {
        "label": label,
        "risk_level": risk_level,
        "final_score": round(final_score, 4),
        "rule_score": rule_score,
        "rule_reasons": rule_reasons,
        "pattern_matches": pattern_matches,
        "ml_probability": round(ml_prob, 4),
        "bert_probability": round(bert_prob, 4),
        "text_features": text_features,
        "confidence": round(max(bert_prob, ml_prob), 4),
        "timestamp": datetime.now().isoformat()
    }

    # Log if not real
    if label != "LIKELY REAL":
        log_suspicious_job(text, result)
        logger.warning(f"Suspicious job detected: {label} (Score: {final_score})")

    return result

# =====================================================
# ANALYTICS & HISTORY
# =====================================================

def load_analysis_history(filepath="suspicious_logs.json"):
    """Load historical analysis data"""
    if not Path(filepath).exists():
        return pd.DataFrame()
    
    try:
        data = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return pd.DataFrame(data)
    except Exception as e:
        logger.error(f"Error loading history: {e}")
        return pd.DataFrame()

def get_statistics():
    """Get detection statistics"""
    df = load_analysis_history()
    if df.empty:
        return {}
    
    return {
        "total_analyzed": len(df),
        "fake_count": len(df[df['analysis'].apply(lambda x: x.get('label') == 'FAKE')]),
        "suspicious_count": len(df[df['analysis'].apply(lambda x: x.get('label') == 'SUSPICIOUS')]),
        "avg_score": float(df['analysis'].apply(lambda x: x.get('final_score', 0)).mean())
    }

# =====================================================
# STREAMLIT UI
# =====================================================

def main():
    st.set_page_config(
        page_title="Hybrid Fake Job Detector",
        page_icon="🚨",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS
    st.markdown("""
        <style>
        .main-title {
            color: #FF6B6B;
            text-align: center;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .fake-badge { background-color: #FF6B6B; color: white; padding: 5px 10px; border-radius: 5px; }
        .suspicious-badge { background-color: #FFD93D; color: black; padding: 5px 10px; border-radius: 5px; }
        .real-badge { background-color: #6BCB77; color: white; padding: 5px 10px; border-radius: 5px; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="main-title">🚨 Hybrid Fake Job Detection System</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        mode = st.radio("Select Mode", ["Analyze Job", "View History", "Statistics"])

    # Main content
    if mode == "Analyze Job":
        st.header("📝 Job Description Analysis")
        
        job_text = st.text_area(
            "Paste Job Description Here:",
            height=300,
            placeholder="Enter the job description text..."
        )

        col1, col2, col3 = st.columns(3)
        
        with col1:
            analyze_btn = st.button("🔍 Analyze", use_container_width=True)
        with col2:
            clear_btn = st.button("🗑️ Clear", use_container_width=True)
        with col3:
            example_btn = st.button("📋 Example", use_container_width=True)

        if clear_btn:
            st.rerun()

        if example_btn:
            example = """Dear Candidate,

We are pleased to offer you a position! Before your contract is issued, you must activate your employee profile through our secure employment portal. 

This requires a one-time activation fee of $200 to verify your credentials. Please act quickly as we have limited slots available. 

Confirm your commitment immediately via WhatsApp to secure your position."""
            job_text = example
            st.text_area("Example Job Description:", value=example, height=300, disabled=True)

        if analyze_btn:
            if not job_text.strip():
                st.warning("⚠️ Please enter a job description.")
            else:
                with st.spinner("🔄 Analyzing..."):
                    result = analyze_job(job_text)

                if result:
                    # Results header
                    st.subheader("📊 Analysis Results")
                    
                    # Risk indicator
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Final Score", f"{result['final_score']:.1%}")
                    with col2:
                        st.metric("Risk Level", result['risk_level'])
                    with col3:
                        st.metric("Confidence", f"{result['confidence']:.1%}")

                    # Label with color coding
                    st.markdown("---")
                    if result["label"] == "FAKE":
                        st.error(f"🔴 **{result['label']}** - CRITICAL RISK")
                    elif result["label"] == "SUSPICIOUS":
                        st.warning(f"🟠 **{result['label']}** - HIGH RISK")
                    elif result["label"] == "QUESTIONABLE":
                        st.info(f"🟡 **{result['label']}** - MEDIUM RISK")
                    else:
                        st.success(f"✅ **{result['label']}** - LOW RISK")

                    # Detailed scores
                    st.markdown("---")
                    st.subheader("🎯 Component Scores")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("BERT Score", f"{result['bert_probability']:.1%}")
                    with col2:
                        st.metric("ML Score", f"{result['ml_probability']:.1%}")
                    with col3:
                        st.metric("Rule Score", f"{result['rule_score']}/20")

                    # Detected patterns
                    if result["rule_reasons"]:
                        st.markdown("---")
                        st.subheader("⚠️ Detected Red Flags")
                        for reason in result["rule_reasons"]:
                            st.write(f"• {reason}")

                    # Text features
                    st.markdown("---")
                    st.subheader("📈 Text Analysis")
                    features = result["text_features"]
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Word Count", f"{int(features['word_count'])}")
                    with col2:
                        st.metric("Avg Word Length", f"{features['avg_word_length']:.1f}")
                    with col3:
                        st.metric("Uppercase Ratio", f"{features['uppercase_ratio']:.1%}")
                    with col4:
                        st.metric("Special Chars", f"{features['special_char_ratio']:.1%}")

    elif mode == "View History":
        st.header("📋 Analysis History")
        df = load_analysis_history()
        
        if df.empty:
            st.info("No analysis history yet.")
        else:
            st.write(f"Total records: {len(df)}")
            
            # Display history
            display_df = pd.DataFrame({
                'Timestamp': pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S'),
                'Label': df['analysis'].apply(lambda x: x.get('label', 'N/A')),
                'Score': df['analysis'].apply(lambda x: f"{x.get('final_score', 0):.2%}"),
                'Risk Level': df['analysis'].apply(lambda x: x.get('risk_level', 'N/A')),
            })
            
            st.dataframe(display_df, use_container_width=True)
            
            # Export option
            if st.button("📥 Download as CSV"):
                csv = display_df.to_csv(index=False)
                st.download_button(
                    label="Download",
                    data=csv,
                    file_name=f"fake_job_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

    elif mode == "Statistics":
        st.header("📊 Detection Statistics")
        
        stats = get_statistics()
        
        if not stats:
            st.info("No analysis data available yet.")
        else:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Analyzed", stats.get('total_analyzed', 0))
            with col2:
                st.metric("Fake Detected", stats.get('fake_count', 0))
            with col3:
                st.metric("Suspicious", stats.get('suspicious_count', 0))
            with col4:
                st.metric("Avg Score", f"{stats.get('avg_score', 0):.1%}")
            
            # Visualizations
            st.markdown("---")
            
            df = load_analysis_history()
            if not df.empty:
                # Chart data
                chart_data = df['analysis'].apply(lambda x: x.get('label', 'Unknown')).value_counts()
                
                st.subheader("Detection Distribution")
                st.bar_chart(chart_data)

if __name__ == "__main__":
    main()