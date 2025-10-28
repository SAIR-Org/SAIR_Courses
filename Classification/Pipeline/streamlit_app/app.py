"""
Main Streamlit Application for Spaceship Titanic ML Pipeline.

This is a multi-page Streamlit app that provides:
- Home dashboard with project overview
- EDA visualizations
- Interactive predictions
- Model performance analysis
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Page configuration
st.set_page_config(
    page_title="Spaceship Titanic ML",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-left: 4px solid #2196f3;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #e8f5e9;
        padding: 1rem;
        border-left: 4px solid #4caf50;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Main content
def main():
    """Main application page."""
    
    # Header
    st.markdown('<div class="main-header">🚀 Spaceship Titanic ML Pipeline</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h3>Welcome to the Spaceship Titanic ML Application!</h3>
    <p>This interactive application demonstrates a complete machine learning pipeline for predicting 
    which passengers were transported to an alternate dimension aboard the Spaceship Titanic.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation guide
    st.markdown('<div class="sub-header">📚 Application Features</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🏠 Home (This Page)
        - Project overview and architecture
        - Quick start guide
        - Performance metrics summary
        
        ### 📊 EDA (Exploratory Data Analysis)
        - Interactive data visualizations
        - Feature distributions
        - Correlation analysis
        - Missing value patterns
        """)
    
    with col2:
        st.markdown("""
        ### 🤖 Predictions
        - Upload passenger data for predictions
        - Real-time prediction results
        - Batch prediction support
        - Confidence scores
        
        ### 📈 Model Performance
        - Model comparison charts
        - Confusion matrices
        - ROC curves
        - Feature importance analysis
        """)
    
    # Project Overview
    st.markdown('<div class="sub-header">🎯 Project Overview</div>', unsafe_allow_html=True)
    
    st.markdown("""
    This project implements a **production-ready machine learning pipeline** that includes:
    
    ✅ **Advanced Feature Engineering** - Extracts meaningful features from raw data  
    ✅ **Multiple Model Training** - Compares 7+ algorithms (Random Forest, XGBoost, LightGBM, etc.)  
    ✅ **Hyperparameter Tuning** - Optimizes model performance with cross-validation  
    ✅ **MLflow Tracking** - Tracks all experiments for reproducibility  
    ✅ **Interactive UI** - This Streamlit app for easy model interaction  
    ✅ **Kaggle Ready** - Generates competition-ready submission files  
    """)
    
    # Architecture diagram
    st.markdown('<div class="sub-header">🏗️ Pipeline Architecture</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ```
    Raw Data → Feature Engineering → Preprocessing → Model Training → Evaluation → Deployment
         ↓              ↓                  ↓               ↓              ↓           ↓
    CSV Files    GroupId, Cabin      Scaling &      Random Forest    Metrics    Streamlit App
                 Spending, Age     Encoding       XGBoost, etc.    ROC-AUC    Kaggle Submit
    ```
    """)
    
    # Quick Start
    st.markdown('<div class="sub-header">🚀 Quick Start Guide</div>', unsafe_allow_html=True)
    
    tabs = st.tabs(["For Users", "For Developers", "For Students"])
    
    with tabs[0]:
        st.markdown("""
        ### Using the Application
        
        1. **📊 Explore the Data**  
           Navigate to the EDA page to understand the dataset
        
        2. **🤖 Make Predictions**  
           Go to the Predictions page and upload passenger data
        
        3. **📈 Analyze Performance**  
           Check the Model Performance page to see how models compare
        
        4. **💾 Download Results**  
           Export predictions as CSV for Kaggle submission
        """)
    
    with tabs[1]:
        st.markdown("""
        ### Running the Pipeline
        
        ```bash
        # Run complete pipeline
        python run_pipeline.py --mode full
        
        # Train models only
        python run_pipeline.py --mode training
        
        # Generate submission
        python run_pipeline.py --mode submission
        
        # Launch Streamlit app
        streamlit run streamlit_app/app.py
        ```
        
        ### MLflow Tracking
        
        ```bash
        # View experiments
        mlflow ui --backend-store-uri spaceship_experiments
        ```
        """)
    
    with tabs[2]:
        st.markdown("""
        ### Learning Resources
        
        📖 **Project Structure**  
        - `project/data/` - Data loading and preprocessing
        - `project/models/` - Model training and evaluation
        - `streamlit_app/` - This interactive application
        
        🎓 **Key Concepts Demonstrated**  
        - Scikit-learn pipelines
        - Feature engineering
        - Cross-validation
        - Ensemble methods
        - Experiment tracking with MLflow
        
        📝 **Adapting for Your Project**  
        1. Replace data loading in `project/data/load_data.py`
        2. Customize features in `project/data/feature_engineering.py`
        3. Modify models in `project/models/base_models.py`
        4. Update this UI to match your needs
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: gray;">
    <p>Built with ❤️ using Scikit-learn, XGBoost, LightGBM, MLflow, and Streamlit</p>
    <p>📧 Questions? Open an issue on GitHub</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()