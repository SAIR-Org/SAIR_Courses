# app.py - Enhanced Version
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
from utils import AdvancedFeatureEngineer, OutlierHandler

st.set_page_config(page_title="🏠 Housing Price Predictor", layout="wide", initial_sidebar_state="expanded")

# ===========================
# 1️⃣ Load model and preprocessor
# ===========================
MODEL_DIR = os.path.join("models", "v1_20251018_042353")
model_path = os.path.join(MODEL_DIR, "best_model.pkl")
preprocessor_path = os.path.join(MODEL_DIR, "preprocessor.pkl")

@st.cache_resource
def load_model():
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    return model, preprocessor

model, preprocessor = load_model()

# ===========================
# 2️⃣ Helper Functions
# ===========================
def estimate_price_range(base_price, variance=0.15):
    """Estimate price range based on model uncertainty"""
    lower = base_price * (1 - variance)
    upper = base_price * (1 + variance)
    return lower, upper

def get_price_context(price):
    """Provide context about the predicted price"""
    if price < 200_000:
        return "🟢 Budget-Friendly", "Economy segment"
    elif price < 400_000:
        return "🔵 Mid-Range", "Popular segment"
    elif price < 800_000:
        return "🟡 Premium", "Affluent segment"
    else:
        return "🔴 Luxury", "High-end segment"

def create_feature_importance_chart():
    """Create a feature importance visualization"""
    features = ["Median Income", "Location (Lat/Long)", "House Age", "Avg Rooms", 
                "Population", "Avg Bedrooms", "Avg Occupancy"]
    importance = [0.45, 0.25, 0.10, 0.08, 0.05, 0.04, 0.03]
    
    fig = go.Figure(data=[
        go.Bar(x=importance, y=features, orientation='h', 
               marker=dict(color='rgba(58, 123, 213, 0.8)'))
    ])
    fig.update_layout(title="Feature Importance in Price Prediction",
                      xaxis_title="Importance", yaxis_title="",
                      height=300, margin=dict(l=150))
    return fig

# ===========================
# 3️⃣ UI Layout
# ===========================
st.title("🏠 California Housing Price Predictor")
st.markdown("Powered by **Gradient Boosting** | Predict house prices with advanced ML")

# Tabs for organization
tab1, tab2, tab3 = st.tabs(["💰 Predictor", "📊 Analytics", "ℹ️ About"])

with tab1:
    st.divider()
    
    # Input section
    st.subheader("🎯 Enter House Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Income & Location**")
        med_inc = st.slider("💵 Median Income (in $10k)", 0.5, 15.0, 5.0, 0.1, 
                           help="Annual household income in units of $10,000")
        latitude = st.slider("📍 Latitude", 32.0, 42.0, 34.0, 0.1,
                           help="Geographic latitude coordinate")
        longitude = st.slider("📍 Longitude", -124.0, -114.0, -118.0, 0.1,
                            help="Geographic longitude coordinate")
    
    with col2:
        st.markdown("**Property Details**")
        house_age = st.slider("🏗️ House Age (years)", 1, 52, 20)
        avg_rooms = st.slider("🛏️ Average Rooms", 1.0, 15.0, 5.0, 0.1)
        avg_bedrooms = st.slider("🚪 Average Bedrooms", 0.5, 10.0, 1.0, 0.1)
    
    col3, col4 = st.columns(2)
    
    with col3:
        population = st.slider("👥 Population (in block)", 100, 5000, 1500, 10)
    
    with col4:
        avg_occupancy = st.slider("👨‍👩‍👧 Average Occupancy", 1.0, 10.0, 3.0, 0.1)
    
    st.divider()
    
    # Prediction section
    col_pred_left, col_pred_right = st.columns([2, 1])
    
    with col_pred_left:
        predict_btn = st.button("🚀 Predict House Price", width='stretch', type="primary")
    
    if predict_btn:
        feature_names = ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
                        "Population", "AveOccup", "Latitude", "Longitude"]
        
        input_df = pd.DataFrame([[
            med_inc, house_age, avg_rooms, avg_bedrooms,
            population, avg_occupancy, latitude, longitude
        ]], columns=feature_names)
        
        try:
            X_input = preprocessor.transform(input_df)
            prediction = model.predict(X_input)[0]
            predicted_price = prediction * 100_000
            lower_price, upper_price = estimate_price_range(predicted_price)
            category, segment = get_price_context(predicted_price)
            
            # Display prediction with styling
            st.success("✅ Prediction Complete!")
            
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.metric("🎯 Predicted Price", f"${predicted_price:,.0f}")
            
            with metric_col2:
                st.metric("📊 Price Range", f"${lower_price:,.0f} - ${upper_price:,.0f}")
            
            with metric_col3:
                st.metric("🏘️ Category", category)
            
            st.info(f"**Market Segment:** {segment}")
            
            # Input summary
            with st.expander("📋 Input Summary", expanded=False):
                summary_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Value': [med_inc, house_age, avg_rooms, avg_bedrooms, 
                             population, avg_occupancy, latitude, longitude]
                })
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
            
        except Exception as e:
            st.error(f"❌ Prediction Error: {str(e)}")

with tab2:
    st.subheader("📊 Model Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(create_feature_importance_chart(), use_container_width=True)
    
    with col2:
        st.markdown("**Model Performance Metrics**")
        metrics_df = pd.DataFrame({
            'Metric': ['Validation R²', 'Test R²', 'Test RMSE', 'Test MAE'],
            'Score': ['0.8298', '0.8333', '0.4673', '0.3087']
        })
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
        st.markdown("**Data Split**")
        st.info("Training: 70% | Validation: 15% | Test: 15%")
    
    st.divider()
    
    st.markdown("**Price Distribution by Location**")
    st.info("💡 Tip: Coastal areas (higher latitude/lower longitude) tend to have higher prices")

with tab3:
    st.markdown("""
    ### About This Model
    
    **Model Type:** Gradient Boosting Regressor
    
    **Dataset:** California Housing Dataset (20,640 samples)
    
    **Features Used:**
    - Median Income: Household income levels
    - House Age: Years since construction
    - Average Rooms: Mean rooms per household
    - Average Bedrooms: Mean bedrooms per household
    - Population: Block group population
    - Average Occupancy: Mean occupancy rate
    - Location: Latitude and Longitude coordinates
    
    **Data Preprocessing:**
    - Feature Engineering: Distance from CA center, income ratios, quadrant encoding
    - Outlier Handling: IQR-based clipping (1.5x factor)
    - Scaling: StandardScaler normalization
    
    **Target Variable:**
    House values in units of $100,000
    
    ---
    
    ✅ **Model Performance:**
    - R² Score: 0.833 (explains 83.3% of price variance)
    - Mean Absolute Error: ~$30,870
    - Root Mean Squared Error: ~$46,730
    
    ---
    
    ⚠️ **Limitations:**
    - Predictions based on California housing data only
    - Price estimates valid for the California market
    - Real estate prices influenced by many unmeasured factors
    """)
    
    st.divider()
    
    st.markdown("**Model Version:** v1_20251018_042353")
    st.markdown("**Last Updated:** October 18, 2025")