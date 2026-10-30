
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="FraudShield | Transaction Intelligence",
    page_icon="🛡️",
    layout="wide"
)
# ==========================================
# LOAD TRAINED MODEL
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

@st.cache_resource
def load_model():
    model = joblib.load(
        MODEL_DIR / "logistic_regression.pkl"
    )

    preprocessor = joblib.load(
        MODEL_DIR / "preprocessor.pkl"
    )

    return model, preprocessor

model, preprocessor = load_model()
# ==========================================
# CUSTOM STYLING
# ==========================================

st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
.block-container {
    padding-top: 2rem;
}
.metric-card {
    background-color: #1c2333;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #30394d;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# HEADER
# ==========================================

st.title("🛡️ FraudShield")
st.subheader("Financial Transaction Fraud Detection System")

st.caption(
    "Machine Learning Powered Transaction Risk Assessment"
)

st.divider()

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "Transaction Analysis",
        "Model Performance",
        "About Project"
    ],
    key="sidebar_section"
)

# ==========================================
# TRANSACTION ANALYSIS
# ==========================================

if page == "Transaction Analysis":

    st.header("Transaction Risk Analysis")

    st.write(
        "Enter transaction details to assess potential fraud risk."
    )

    col1, col2 = st.columns(2)

    with col1:

        amount = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=5000.0
        )

        transaction_type = st.selectbox(
            "Transaction Type",
            ["TRANSFER", "PAYMENT", "CASH_OUT", "DEBIT", "CASH_IN"]
        )

        merchant_category = st.selectbox(
            "Merchant Category",
            [
                "grocery",
                "electronics",
                "travel",
                "retail",
                "food",
                "other"
            ]
        )

        device_used = st.selectbox(
            "Device Used",
            ["mobile", "desktop", "tablet"]
        )

        location = st.text_input(
            "Transaction Location",
            "Mumbai"
        )

    with col2:

        payment_channel = st.selectbox(
            "Payment Channel",
            ["UPI", "CARD", "ONLINE", "ATM", "BANK_TRANSFER"]
        )

        spending_deviation_score = st.slider(
            "Spending Deviation Score",
            0.0, 1.0, 0.5
        )

        velocity_score = st.slider(
            "Velocity Score",
            0.0, 1.0, 0.5
        )

        geo_anomaly_score = st.slider(
            "Geographical Anomaly Score",
            0.0, 1.0, 0.5
        )

        sender_txn_count = st.number_input(
            "Previous Sender Transactions",
            min_value=0,
            value=5
        )

        sender_avg_amount = st.number_input(
            "Previous Average Transaction Amount",
            min_value=0.0,
            value=3000.0
        )

        receiver_txn_count = st.number_input(
            "Previous Receiver Transactions",
            min_value=0,
            value=5
        )

    st.divider()

if st.button(
    "🔍 Analyse Transaction",
    use_container_width=True
):
    current_time = datetime.now()

    input_data = pd.DataFrame([{
        "amount": amount,
        "spending_deviation_score": spending_deviation_score,
        "velocity_score": velocity_score,
        "geo_anomaly_score": geo_anomaly_score,
        "hour_of_day": current_time.hour,
        "day_of_week": current_time.weekday(),
        "month": current_time.month,
        "is_weekend": int(current_time.weekday() >= 5),
        "sender_txn_count": sender_txn_count,
        "sender_avg_amount": sender_avg_amount,
        "receiver_txn_count": receiver_txn_count,
        "transaction_type": transaction_type,
        "merchant_category": merchant_category,
        "device_used": device_used,
        "location": location,
        "payment_channel": payment_channel
    }])
    try:

        processed_data = preprocessor.transform(input_data)

        fraud_score = model.predict_proba(
            processed_data
        )[0][1]

        prediction = int(fraud_score >= 0.50)

        st.divider()
        st.subheader("Transaction Analysis Result")

        col1, col2 = st.columns(2)

        with col1:

            if prediction == 1:
                st.error("⚠️ POTENTIALLY FRAUDULENT")
            else:
                st.success("✅ PREDICTED LEGITIMATE")

        with col2:

            st.metric(
                "Model-estimated fraud score",
                f"{fraud_score * 100:.2f}%"
            )

        st.progress(float(fraud_score))

        st.caption(
            "Experimental model score. This is not a calibrated "
            "real-world fraud probability."
        )

    except Exception as e:

        st.error(f"Prediction failed: {e}")

# ==========================================
# MODEL PERFORMANCE
# ==========================================

elif page == "Model Performance":

    st.header("Model Performance Dashboard")

    st.write(
        "Current evaluation results from the existing test-set experiment."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("PR-AUC", "0.0384")
    col2.metric("ROC-AUC", "0.5238")
    col3.metric("Precision", "3.82%")
    col4.metric("Recall", "54.46%")

    st.warning(
        "Current predictive performance is weak. "
        "These models are experimental and not suitable for "
        "real-world fraud prevention."
    )

    st.subheader("Model Comparison")

    comparison = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "LightGBM",
            "Random Forest"
        ],
        "PR-AUC": [
            0.038423,
            0.037156,
            0.037140
        ],
        "ROC-AUC": [
            0.523813,
            0.509653,
            0.509025
        ]
    })

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )

# ==========================================
# ABOUT PROJECT
# ==========================================

elif page == "About Project":

    st.header("About FraudShield")

    st.write("""
    FraudShield is a machine learning based financial
    transaction fraud detection project.

    The project investigates fraudulent transaction patterns
    using a large financial transaction dataset containing
    approximately 5 million records.

    ### Machine Learning Models

    - Logistic Regression
    - Random Forest
    - LightGBM
    - CatBoost (experimental)

    ### Project Methodology

    1. Data collection and auditing
    2. Exploratory Data Analysis
    3. Data preprocessing
    4. Temporal feature engineering
    5. Historical account behaviour analysis
    6. Model training
    7. Model evaluation
    8. Transaction risk prediction

    ### Evaluation Metrics

    - Precision
    - Recall
    - F1-score
    - PR-AUC
    - ROC-AUC
    - Matthews Correlation Coefficient
    """)


import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="FraudShield | Transaction Intelligence",
    page_icon="🛡️",
    layout="wide"
)
# ==========================================
# LOAD TRAINED MODEL
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

@st.cache_resource
def load_model():
    model = joblib.load(
        MODEL_DIR / "logistic_regression.pkl"
    )

    preprocessor = joblib.load(
        MODEL_DIR / "preprocessor.pkl"
    )

    return model, preprocessor

model, preprocessor = load_model()
# ==========================================
# CUSTOM STYLING
# ==========================================

st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
.block-container {
    padding-top: 2rem;
}
.metric-card {
    background-color: #1c2333;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #30394d;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# HEADER
# ==========================================

st.title("🛡️ FraudShield")
st.subheader("Financial Transaction Fraud Detection System")

st.caption(
    "Machine Learning Powered Transaction Risk Assessment"
)

st.divider()

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "Transaction Analysis",
        "Model Performance",
        "About Project"
    ]
)

# ==========================================
# TRANSACTION ANALYSIS
# ==========================================

if page == "Transaction Analysis":

    st.header("Transaction Risk Analysis")

    st.write(
        "Enter transaction details to assess potential fraud risk."
    )

    col1, col2 = st.columns(2)

    with col1:

        amount = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=5000.0
        )

        transaction_type = st.selectbox(
            "Transaction Type",
            ["TRANSFER", "PAYMENT", "CASH_OUT", "DEBIT", "CASH_IN"]
        )

        merchant_category = st.selectbox(
            "Merchant Category",
            [
                "grocery",
                "electronics",
                "travel",
                "retail",
                "food",
                "other"
            ]
        )

        device_used = st.selectbox(
            "Device Used",
            ["mobile", "desktop", "tablet"]
        )

        location = st.text_input(
            "Transaction Location",
            "Mumbai"
        )

    with col2:

        payment_channel = st.selectbox(
            "Payment Channel",
            ["UPI", "CARD", "ONLINE", "ATM", "BANK_TRANSFER"]
        )

        spending_deviation_score = st.slider(
            "Spending Deviation Score",
            0.0, 1.0, 0.5
        )

        velocity_score = st.slider(
            "Velocity Score",
            0.0, 1.0, 0.5
        )

        geo_anomaly_score = st.slider(
            "Geographical Anomaly Score",
            0.0, 1.0, 0.5
        )

        sender_txn_count = st.number_input(
            "Previous Sender Transactions",
            min_value=0,
            value=5
        )

        sender_avg_amount = st.number_input(
            "Previous Average Transaction Amount",
            min_value=0.0,
            value=3000.0
        )

        receiver_txn_count = st.number_input(
            "Previous Receiver Transactions",
            min_value=0,
            value=5
        )

    st.divider()

if st.button(
    "🔍 Analyse Transaction",
    use_container_width=True
):
    current_time = datetime.now()

    input_data = pd.DataFrame([{
        "amount": amount,
        "spending_deviation_score": spending_deviation_score,
        "velocity_score": velocity_score,
        "geo_anomaly_score": geo_anomaly_score,
        "hour_of_day": current_time.hour,
        "day_of_week": current_time.weekday(),
        "month": current_time.month,
        "is_weekend": int(current_time.weekday() >= 5),
        "sender_txn_count": sender_txn_count,
        "sender_avg_amount": sender_avg_amount,
        "receiver_txn_count": receiver_txn_count,
        "transaction_type": transaction_type,
        "merchant_category": merchant_category,
        "device_used": device_used,
        "location": location,
        "payment_channel": payment_channel
    }])
    try:

        processed_data = preprocessor.transform(input_data)

        fraud_score = model.predict_proba(
            processed_data
        )[0][1]

        prediction = int(fraud_score >= 0.50)

        st.divider()
        st.subheader("Transaction Analysis Result")

        col1, col2 = st.columns(2)

        with col1:

            if prediction == 1:
                st.error("⚠️ POTENTIALLY FRAUDULENT")
            else:
                st.success("✅ PREDICTED LEGITIMATE")

        with col2:

            st.metric(
                "Model-estimated fraud score",
                f"{fraud_score * 100:.2f}%"
            )

        st.progress(float(fraud_score))

        st.caption(
            "Experimental model score. This is not a calibrated "
            "real-world fraud probability."
        )

    except Exception as e:

        st.error(f"Prediction failed: {e}")

# ==========================================
# MODEL PERFORMANCE
# ==========================================

elif page == "Model Performance":

    st.header("Model Performance Dashboard")

    st.write(
        "Current evaluation results from the existing test-set experiment."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("PR-AUC", "0.0384")
    col2.metric("ROC-AUC", "0.5238")
    col3.metric("Precision", "3.82%")
    col4.metric("Recall", "54.46%")

    st.warning(
        "Current predictive performance is weak. "
        "These models are experimental and not suitable for "
        "real-world fraud prevention."
    )

    st.subheader("Model Comparison")

    comparison = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "LightGBM",
            "Random Forest"
        ],
        "PR-AUC": [
            0.038423,
            0.037156,
            0.037140
        ],
        "ROC-AUC": [
            0.523813,
            0.509653,
            0.509025
        ]
    })

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )

# ==========================================
# ABOUT PROJECT
# ==========================================

elif page == "About Project":

    st.header("About FraudShield")

    st.write("""
    FraudShield is a machine learning based financial
    transaction fraud detection project.

    The project investigates fraudulent transaction patterns
    using a large financial transaction dataset containing
    approximately 5 million records.

    ### Machine Learning Models

    - Logistic Regression
    - Random Forest
    - LightGBM
    - CatBoost (experimental)

    ### Project Methodology

    1. Data collection and auditing
    2. Exploratory Data Analysis
    3. Data preprocessing
    4. Temporal feature engineering
    5. Historical account behaviour analysis
    6. Model training
    7. Model evaluation
    8. Transaction risk prediction

    ### Evaluation Metrics

    - Precision
    - Recall
    - F1-score
    - PR-AUC
    - ROC-AUC
    - Matthews Correlation Coefficient
    """)

