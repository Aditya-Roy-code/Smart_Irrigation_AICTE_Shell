import streamlit as st
import numpy as np
import joblib
import os

# Load the trained model
model = joblib.load("Farm_Irrigation_System.pkl")

# Title
st.title("🚿 Smart Sprinkler Prediction System")
st.markdown("Enter scaled sensor values between **0 to 1** to predict sprinkler status for each parcel.")

# Load model with error handling
model_path = "Farm_Irrigation_System.pkl"
try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"⚠️ Error loading the model: {e}")
    st.stop()

# Get expected input shape from model
try:
    # For illustration only; manually set if unsure
    expected_input_size = model.n_features_in_
except AttributeError:
    expected_input_size = 20  # fallback default

# Sidebar for sensor inputs
st.sidebar.header("🌡 Sensor Inputs")
sensor_values = []

for i in range(expected_input_size):
    val = st.sidebar.slider(f"Sensor {i}", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    sensor_values.append(val)

# Predict button
if st.button("🔍 Predict Sprinklers"):
    try:
        input_array = np.array(sensor_values).reshape(1, -1)
        prediction = model.predict(input_array)[0]

        st.success("✅ Prediction Complete!")
        st.markdown("### 🧠 Sprinkler Status per Parcel:")

        for i, status in enumerate(prediction):
            color = "green" if status == 1 else "red"
            st.markdown(f"<span style='color:{color};font-weight:bold;'>Sprinkler {i} (parcel_{i}): {'ON' if status == 1 else 'OFF'}</span>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
