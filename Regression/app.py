# app.py
import streamlit as st
import joblib
import pandas as pd
import os
import glob
from utils import AdvancedFeatureEngineer, OutlierHandler

# ===========================
# 1️⃣ Auto-detect latest model version
# ===========================
model_folders = glob.glob("models/v1_*")
if not model_folders:
    st.error("❌ No model folders found in 'models/'. Please train and save a model first.")
    st.stop()

# Get the latest folder by modification time
MODEL_DIR = max(model_folders, key=os.path.getmtime)
model_path = os.path.join(MODEL_DIR, "best_model.pkl")
preprocessor_path = os.path.join(MODEL_DIR, "preprocessor.pkl")

# Safety check
if not os.path.exists(model_path):
    st.error(f"❌ Model file not found at {model_path}")
    st.stop()
if not os.path.exists(preprocessor_path):
    st.error(f"❌ Preprocessor file not found at {preprocessor_path}")
    st.stop()

# ✅ Make sure utils.py is imported before unpickling
@st.cache_resource
def load_model():
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    return model, preprocessor

model, preprocessor = load_model()

# ===========================
# 2️⃣ Streamlit UI
# ===========================
st.set_page_config(page_title="🏠 California Housing Price Predictor", layout="wide")
st.title("🏠 California Housing Price Predictor")
st.write(
    """
    Enter the features of a house to predict its price in USD.
    This model is a **Gradient Boosting Regressor** trained on the California Housing dataset.
    """
)

st.sidebar.header("🏠 Input Features")
col1, col2 = st.columns(2)

with col1:
    med_inc = st.slider("Median Income (10k$)", 0.5, 15.0, 5.0, 0.1)
    house_age = st.slider("House Age (years)", 1, 52, 20)
    latitude = st.slider("Latitude", 32.0, 42.0, 34.0, 0.1)
    longitude = st.slider("Longitude", -124.0, -114.0, -118.0, 0.1)

with col2:
    avg_rooms = st.slider("Average Rooms", 1.0, 15.0, 5.0, 0.1)
    avg_bedrooms = st.slider("Average Bedrooms", 0.5, 10.0, 1.0, 0.1)
    population = st.slider("Population", 100, 5000, 1500, 10)
    avg_occupancy = st.slider("Average Occupancy", 1.0, 10.0, 3.0, 0.1)

# ===========================
# 3️⃣ Prepare input DataFrame
# ===========================
feature_names = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude"
]

input_df = pd.DataFrame([[
    med_inc, house_age, avg_rooms, avg_bedrooms,
    population, avg_occupancy, latitude, longitude
]], columns=feature_names)

# ===========================
# 4️⃣ Preprocess & Predict
# ===========================
if st.button("🚀 Predict House Price"):
    try:
        X_input = preprocessor.transform(input_df)
        prediction = model.predict(X_input)[0]
        predicted_price = prediction * 100_000  # Convert from 100k units

        st.success(f"### Predicted House Price: ${predicted_price:,.0f}")
        st.write("#### Input Features") 
        st.write(input_df)
    except Exception as e:
        st.error(f"❌ Error making prediction: {str(e)}")

# ===========================
# 5️⃣ Sidebar Info
# ===========================
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ Model Information")
st.sidebar.write(f"**Loaded from:** `{MODEL_DIR}`")
st.sidebar.write("**Model:** Gradient Boosting Regressor")
st.sidebar.write("**Dataset:** California Housing")
st.sidebar.write("**Validation R²:** 0.8298")
st.sidebar.write("**Test R²:** 0.8333 | RMSE: 0.4673 | MAE: 0.3087")
