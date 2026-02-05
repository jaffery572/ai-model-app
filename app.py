import streamlit as st
import pickle
import numpy as np

# Title
st.title("🏗️ Global Infrastructure AI")
st.write("Enter 2 values for prediction (Population & Area):")

# Load model
with open('ai_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Only 2 inputs - Model expects 2 features
input1 = st.number_input("Population (thousands)", value=100.0, min_value=0.0)
input2 = st.number_input("Area Size (sq km)", value=50.0, min_value=0.0)

# Predict button
if st.button("Predict Now"):
    # Use only 2 features
    inputs = np.array([[input1, input2]])
    prediction = model.predict(inputs)
    
    # Show result
    st.success(f"Prediction: {prediction[0]}")
    
    # Add interpretation
    if prediction[0] == 1:
        st.error("🚨 Infrastructure Development Needed!")
    else:
        st.success("✅ Infrastructure is Sufficient")
    
    st.balloons()