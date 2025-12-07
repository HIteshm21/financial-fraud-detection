# app.py
import streamlit as st
import pandas as pd
import joblib
import os

# --- load model files (relative to this file) ---
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
COLS_PATH = os.path.join(MODEL_DIR, "model_columns.pkl")

st.set_page_config(page_title="Financial Fraud Detection", layout="wide")
st.title("Financial Transaction Fraud Detection System")

# Check files
if not os.path.exists(MODEL_PATH) or not os.path.exists(COLS_PATH):
    st.error("Model files not found. Put model.pkl and model_columns.pkl in the 'model' folder.")
    st.stop()

# Load model and columns
model = joblib.load(MODEL_PATH)
model_columns = joblib.load(COLS_PATH)

st.markdown("**Enter transaction details (or use defaults)**")

# Minimal inputs (you can add more fields later)
amount = st.number_input("Amount", min_value=0.0, value=100.0, step=1.0)
time_val = st.number_input("Time (seconds from first transaction)", min_value=0.0, value=0.0, step=1.0)

# Build input row using saved model columns
input_dict = {col: 0.0 for col in model_columns}
# Set values for known columns if they exist in model
if "Amount" in input_dict:
    input_dict["Amount"] = float(amount)
if "Time" in input_dict:
    input_dict["Time"] = float(time_val)

input_df = pd.DataFrame([input_dict])

# Display the features being sent to model
with st.expander("Features sent to model (show/hide)"):
    st.dataframe(input_df.T, height=300)

if st.button("Predict Fraud Risk"):
    try:
        proba = float(model.predict_proba(input_df)[0, 1])
        pred = int(model.predict(input_df)[0])
    except Exception as e:
        st.error(f"Model prediction failed: {e}")
        st.stop()

    st.metric("Fraud Probability", f"{proba:.4f}")

    if pred == 1:
        st.error("⚠️ Prediction: FRAUD")
    else:
        st.success("✅ Prediction: GENUINE")

# Optional: quick dataset overview (if data file exists)
data_path = os.path.join("data", "creditcard.csv")
if st.sidebar.button("Show dataset overview"):
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        st.sidebar.write("Rows, cols:", df.shape)
        st.sidebar.bar_chart(df["Class"].value_counts())
    else:
        st.sidebar.warning("creditcard.csv not found in /data folder.")
