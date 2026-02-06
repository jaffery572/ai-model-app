import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="InfraScope AI | Infrastructure Intelligence",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
        margin-bottom: 20px;
        border-top: 4px solid #4CAF50;
    }
    
    .prediction-success {
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
        color: white;
        border-radius: 15px;
        padding: 30px;
        margin-bottom: 25px;
    }
    
    .prediction-danger {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
        border-radius: 15px;
        padding: 30px;
        margin-bottom: 25px;
    }
    
    .main-header {
        background: linear-gradient(90deg, #4CAF50, #2196F3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.5rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
        color: white;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ==================== INITIALIZE SESSION STATE ====================
if 'population' not in st.session_state:
    st.session_state.population = 6300.0
if 'area' not in st.session_state:
    st.session_state.area = 4191.0
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False

# ==================== LOAD MODEL ====================
@st.cache_resource
def load_model():
    """Load or create AI model"""
    try:
        with open('infra_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('model_info.json', 'r') as f:
            info = json.load(f)
        
        return model, info, True
    except:
        # Create new model
        np.random.seed(42)
        n_samples = 5000
        
        X = []
        y = []
        
        # Training data
        for _ in range(n_samples):
            pop = np.random.uniform(10, 50000)
            area = np.random.uniform(10, 10000)
            density = pop / area
            
            if density < 1:
                y.append(0)  # Sufficient
            elif density > 5:
                y.append(1)  # Development needed
            else:
                y.append(0 if np.random.random() > 0.4 else 1)
            
            X.append([pop, area])
        
        X = np.array(X)
        y = np.array(y)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Save model
        with open('infra_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        info = {
            "accuracy": float(model.score(X, y)),
            "samples": n_samples,
            "sufficient": int(sum(y == 0)),
            "needed": int(sum(y == 1)),
            "created": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open('model_info.json', 'w') as f:
            json.dump(info, f, indent=2)
        
        return model, info, False

# Load model
model, model_info, _ = load_model()

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>🏗️ InfraScope AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; opacity: 0.9;'>Intelligent Infrastructure Planning</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Input Section
    st.markdown("### 📊 Input Parameters")
    
    population = st.slider(
        "Population (thousands)",
        min_value=0.0,
        max_value=50000.0,
        value=st.session_state.population,
        step=100.0,
        key="population_slider"
    )
    
    area = st.slider(
        "Area (sq km)",
        min_value=1.0,
        max_value=10000.0,
        value=st.session_state.area,
        step=10.0,
        key="area_slider"
    )
    
    # Update session state
    st.session_state.population = population
    st.session_state.area = area
    
    st.markdown("---")
    
    # Model Information
    st.markdown("### 🤖 Model Information")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", f"{model_info['accuracy']*100:.1f}%")
    with col2:
        st.metric("Samples", f"{model_info['samples']:,}")
    
    st.markdown("---")
    
    if st.button("🔄 Reset All", use_container_width=True, type="secondary"):
        st.session_state.population = 5000.0
        st.session_state.area = 1000.0
        st.session_state.prediction_made = False
        st.rerun()

# ==================== MAIN DASHBOARD ====================
# Header
st.markdown("<h1 class='main-header'>Infrastructure Intelligence Dashboard</h1>", unsafe_allow_html=True)

# Top Metrics Row
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

# Calculate metrics
density = population / area if area > 0 else 0
infra_index = min(100, density * 12.5)
capacity_utilization = min(100, (population / (area * 10)) * 100)
development_urgency = min(100, density * 9.5)

with col1:
    st.metric("Population Density", f"{density:.2f}", "people/sq km")

with col2:
    st.metric("Infrastructure Index", f"{infra_index:.0f}/100", 
              "High" if infra_index > 70 else "Medium" if infra_index > 40 else "Low")

with col3:
    st.metric("Capacity Utilization", f"{capacity_utilization:.1f}%",
              "Optimal" if 60 <= capacity_utilization <= 80 else "Over" if capacity_utilization > 80 else "Under")

with col4:
    st.metric("Development Urgency", f"{development_urgency:.0f}/100",
              "High" if development_urgency > 60 else "Medium" if development_urgency > 30 else "Low")

# Main Prediction Button
st.markdown("---")
if st.button("🚀 Run AI Analysis", type="primary", use_container_width=True):
    st.session_state.prediction_made = True

# If prediction has been made, show results
if st.session_state.prediction_made:
    # Get prediction
    X_input = np.array([[population, area]])
    prediction = model.predict(X_input)[0]
    probabilities = model.predict_proba(X_input)[0]
    
    # Display Prediction Result
    st.markdown("---")
    
    if prediction == 0:
        st.markdown("<div class='prediction-success'>", unsafe_allow_html=True)
        st.markdown("# ✅ INFRASTRUCTURE SUFFICIENT")
        st.markdown(f"### Confidence: {probabilities[0]*100:.1f}%")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='prediction-danger'>", unsafe_allow_html=True)
        st.markdown("# 🚨 DEVELOPMENT REQUIRED")
        st.markdown(f"### Confidence: {probabilities[1]*100:.1f}%")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Confidence Metrics
    col_metrics1, col_metrics2 = st.columns(2)
    
    with col_metrics1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("### 📊 Probability Distribution")
        
        # CORRECT PIE CHART - ONLY 2 CATEGORIES
        fig, ax = plt.subplots(figsize=(6, 6))
        labels = ['Sufficient', 'Development Needed']  # ONLY 2 LABELS
        sizes = [probabilities[0]*100, probabilities[1]*100]  # ONLY 2 VALUES
        colors = ['#4CAF50', '#FF5252']
        
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        ax.set_title('AI Prediction Probabilities')
        
        st.pyplot(fig)
        
        # CORRECT PERCENTAGES
        st.markdown(f"**✅ Sufficient:** {probabilities[0]*100:.1f}%")
        st.markdown(f"**🚨 Development Needed:** {probabilities[1]*100:.1f}%")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col_metrics2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("### ⚖️ Confidence Metrics")
        
        confidence = max(probabilities)*100
        st.metric("Model Confidence", f"{confidence:.1f}%")
        
        st.progress(float(max(probabilities)))
        
        st.markdown("**Detailed Breakdown:**")
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.metric("Sufficient", f"{probabilities[0]*100:.1f}%")
        with col_p2:
            st.metric("Needed", f"{probabilities[1]*100:.1f}%")
        
        # CORRECT RISK LEVEL
        if probabilities[1] > 0.7:
            risk = "🔴 HIGH"
        elif probabilities[1] > 0.4:
            risk = "🟡 MEDIUM"
        else:
            risk = "🟢 LOW"
        
        st.markdown(f"**Risk Level:** {risk}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Export Section
    st.markdown("---")
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.markdown("### 📤 Export & Share Results")
    
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    col_exp1, col_exp2 = st.columns(2)
    
    with col_exp1:
        # JSON Export
        export_data = {
            "population": population,
            "area": area,
            "density": density,
            "prediction": "Sufficient" if prediction == 0 else "Development Needed",
            "confidence": float(max(probabilities) * 100),
            "probabilities": {
                "sufficient": float(probabilities[0] * 100),
                "development_needed": float(probabilities[1] * 100)
            }
        }
        
        st.download_button(
            label="📊 Download JSON Data",
            data=json.dumps(export_data, indent=2),
            file_name=f"infrastructure_analysis_{current_time}.json",
            mime="application/json"
        )
    
    with col_exp2:
        # Text Report
        report_text = f"""
        INFRASTRUCTURE INTELLIGENCE REPORT
        
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        INPUT PARAMETERS:
        - Population: {population:,.0f} thousand
        - Area: {area:,.0f} sq km
        - Population Density: {density:.2f}
        
        AI PREDICTION:
        - Result: {"Sufficient" if prediction == 0 else "Development Needed"}
        - Confidence: {max(probabilities)*100:.1f}%
        - Probability (Sufficient): {probabilities[0]*100:.1f}%
        - Probability (Development Needed): {probabilities[1]*100:.1f}%
        
        Generated by InfraScope AI
        """
        
        st.download_button(
            label="📄 Download Text Report",
            data=report_text,
            file_name=f"infrastructure_report_{current_time}.txt",
            mime="text/plain"
        )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Celebration
    st.balloons()

else:
    # Show instructions
    st.markdown("---")
    st.info("Click '🚀 Run AI Analysis' button to get predictions")

# Footer
st.markdown("---")
current_date = datetime.now().strftime('%Y-%m-%d')
st.markdown(f"""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p style='font-size: 1.1rem; font-weight: 600;'>🏗️ InfraScope AI | Version 3.1</p>
    <p style='font-size: 0.9rem;'>Data Updated: {current_date}</p>
</div>
""", unsafe_allow_html=True)

# Cleanup
plt.close('all')
