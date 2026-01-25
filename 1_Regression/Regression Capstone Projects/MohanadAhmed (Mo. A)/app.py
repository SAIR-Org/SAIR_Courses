import streamlit as st
import glob
from eda import *
from src.preprocessor import OutlierHandler
#from feature_engineering import OutlierHandler
# ======================================================
# 1️⃣ Auto-detect latest model folder
# ======================================================
model_folders = glob.glob("models/v1_20260109_152722")
if not model_folders:
    st.error("❌ No trained model found.")
    st.stop()

MODEL_DIR = max(model_folders, key=os.path.getmtime)
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.pkl")

if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
    st.error("❌ Model or preprocessor file missing in the latest folder.")
    st.stop()

# ======================================================
# 2️⃣ Load model & preprocessor
# ======================================================
@st.cache_resource
def load_model_and_preprocessor():
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor

model, preprocessor = load_model_and_preprocessor()

# ======================================================
# 3️⃣ UI CONFIG (UNCHANGED)
# ======================================================
st.set_page_config(
    page_title="⚡ CCPP Power Output Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    h1, h2, h3 { color: #f5f5f5; }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        padding: 0.6em 1.2em;
        font-size: 1.5em;
    }
</style>
""", unsafe_allow_html=True)

# ======================================================
# 4️⃣ HEADER
# ======================================================
st.title("⚡ Combined Cycle Power Plant (CCPP) Electricity Predictor")
st.markdown(
    "Predict the **net hourly electrical energy output (PE) in MW** "
    "using a **Gradient Boosting model**."
)

# ======================================================
# 5️⃣ SIDEBAR INPUTS
# ======================================================
st.sidebar.header("🧪 Environmental Inputs")
AT = st.sidebar.slider("🌡 Ambient Temperature (°C)", -5.0, 40.0, 15.0, 0.1)
V  = st.sidebar.slider("🌀 Exhaust Vacuum (cm Hg)", 25.0, 80.0, 40.0, 0.1)
AP = st.sidebar.slider("🌬 Ambient Pressure (mbar)", 980.0, 1050.0, 1013.0, 0.5)
RH = st.sidebar.slider("💧 Relative Humidity (%)", 10.0, 100.0, 70.0, 1.0)

input_df = pd.DataFrame([[AT, V, AP, RH]],
                        columns=["AT", "V", "AP", "RH"])

# ======================================================
# 6️⃣ PREDICTION
# ======================================================
col_pred, col_info = st.columns([1.2, 1])

with col_pred:
    if st.button("🚀 Predict Power Output"):
        try:
            # Apply preprocessing first (adds engineered features)
            X_processed = preprocessor.transform(input_df)

            # Predict
            prediction = model.predict(X_processed)[0]

            # Display prediction in centered green box
            st.markdown(f"""
                <div style="
                    background-color:#d4edda;
                    color:#155724;
                    border-radius:10px;
                    padding:20px;
                    text-align:center;
                    font-size:28px;
                    font-weight:600;
                ">
                    ⚡ Predicted Electrical Output: {prediction:.2f} MW
                </div>
            """, unsafe_allow_html=True)

            # Show input values
            st.markdown("#### 🔎 Input Conditions")
            st.dataframe(input_df, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

# ======================================================
# 7️⃣ MODEL INFO PANEL
# ======================================================
with col_info:
    st.markdown("### ℹ️ Model Information")
    st.write(f"**Loaded from:** `{MODEL_DIR}`")
    st.write("**Model:** Random Forest")
    st.write("**Features:** 4 raw")

    st.markdown("---")
    st.markdown("### 📊 Performance")
    st.write("• **Validation R² :** 0.9642")
    st.write("• **Test R² :** 0.9668")
    st.write("• **Test RMSE :** 3.0378 MW")
