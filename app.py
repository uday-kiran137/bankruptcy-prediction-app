# app.py
import streamlit as st
import numpy as np
import pickle

# Load the model
with open('rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Bankruptcy Prediction App")

st.write("Predict whether a company will go bankrupt or not based on various risks.")

# Input fields for features
industrial_risk = st.selectbox("Industrial Risk", [0, 0.5, 1])
management_risk = st.selectbox("Management Risk", [0, 0.5, 1])
financial_flexibility = st.selectbox("Financial Flexibility", [0, 0.5, 1])
credibility = st.selectbox("Credibility", [0, 0.5, 1])
competitiveness = st.selectbox("Competitiveness", [0, 0.5, 1])
operating_risk = st.selectbox("Operating Risk", [0, 0.5, 1])

# Predict button
if st.button("Predict"):
    input_data = np.array([[industrial_risk, management_risk, financial_flexibility,
                            credibility, competitiveness, operating_risk]])
    prediction = model.predict(input_data)[0]

    # Map prediction (0 or 1) to readable label
    label_map = {0: "non-bankruptcy", 1: "bankruptcy"}
    predicted_label = label_map[prediction]

    st.success(f"The predicted class is: **{predicted_label}**")

