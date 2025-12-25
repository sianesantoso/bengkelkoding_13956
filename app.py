import streamlit as st
import pandas as pd
import joblib

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return joblib.load("churn_model.pkl")

model = load_model()

st.title("Telco Customer Churn Prediction")

st.write("Masukkan data pelanggan untuk memprediksi kemungkinan churn.")

# =========================
# INPUT USER
# =========================
gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure (bulan)", min_value=0, max_value=100, value=12)

PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

InternetService = st.selectbox(
    "Internet Service", ["DSL", "Fiber optic", "No"]
)

OnlineSecurity = st.selectbox(
    "Online Security", ["Yes", "No", "No internet service"]
)
OnlineBackup = st.selectbox(
    "Online Backup", ["Yes", "No", "No internet service"]
)
DeviceProtection = st.selectbox(
    "Device Protection", ["Yes", "No", "No internet service"]
)
TechSupport = st.selectbox(
    "Tech Support", ["Yes", "No", "No internet service"]
)
StreamingTV = st.selectbox(
    "Streaming TV", ["Yes", "No", "No internet service"]
)
StreamingMovies = st.selectbox(
    "Streaming Movies", ["Yes", "No", "No internet service"]
)

Contract = st.selectbox(
    "Contract", ["Month-to-month", "One year", "Two year"]
)
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

MonthlyCharges = st.number_input(
    "Monthly Charges", min_value=0.0, value=70.0
)
TotalCharges = st.number_input(
    "Total Charges", min_value=0.0, value=1000.0
)

# =========================
# DATAFRAME
# =========================
input_df = pd.DataFrame([{
    "gender": gender,
    "SeniorCitizen": SeniorCitizen,
    "Partner": Partner,
    "Dependents": Dependents,
    "tenure": tenure,
    "PhoneService": PhoneService,
    "MultipleLines": MultipleLines,
    "InternetService": InternetService,
    "OnlineSecurity": OnlineSecurity,
    "OnlineBackup": OnlineBackup,
    "DeviceProtection": DeviceProtection,
    "TechSupport": TechSupport,
    "StreamingTV": StreamingTV,
    "StreamingMovies": StreamingMovies,
    "Contract": Contract,
    "PaperlessBilling": PaperlessBilling,
    "PaymentMethod": PaymentMethod,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges
}])

# =========================
# PREDIKSI
# =========================
if st.button("Predict Churn"):
    proba = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]

    st.subheader("Hasil Prediksi")

    if pred == 1:
        st.error(f"⚠️ Berpotensi CHURN\nProbabilitas: {proba:.2%}")
    else:
        st.success(f"✅ Tidak Churn\nProbabilitas Churn: {proba:.2%}")
