import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open("model/house_price_model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

st.title("🏠 House Price Prediction App")

area = st.number_input("Enter Area in Sq Ft", min_value=500)
bedrooms = st.number_input("Enter Number of Bedrooms", min_value=1)
age = st.number_input("Enter Age of House (Years)", min_value=0)

if st.button("Predict Price"):
    input_data = np.array([[area, bedrooms, age]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    st.success(f"Predicted House Price: ₹ {prediction[0]:,.2f}")
