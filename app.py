import streamlit as st
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import json

# ================= CONFIGURATION =================
st.set_page_config(
    page_title="Global Infrastructure AI",
    page_icon="🏗️",
    layout="wide"
)

# ================= TITLE =================
st.title("🏗️ Global Infrastructure AI")
st.markdown("Predict infrastructure needs based on population density and economic factors")

# ================= MODEL LOADING =================
@st.cache_resource
def load_or_create_model():
    """Load existing model or create a new one"""
    try:
        # Try to load existing model
        with open('ai_model.pkl', 'rb') as f:
            model = pickle.load(f)
        st.sidebar.success("✅ Model loaded successfully")
        return model
    except:
        # Create new model if doesn't exist
        st.sidebar.warning("⚠️ Creating default model...")
        
        # Generate training data
        np.random.seed(42)
        n_samples = 500
        
        # Features: Population, Area, GDP per capita
        X = np.zeros((n_samples, 3))
        
        # Population (thousands) - 50 to 10000
        X[:, 0] = np.random.uniform(50, 10000, n_samples)
        
        # Area (sq km) - 10 to 5000
        X[:, 1] = np.random.uniform(10, 5000, n_samples)
        
        # GDP per capita ($) - 1000 to 50000
        X[:, 2] = np.random.uniform(1000, 50000, n_samples)
        
        # Calculate density and create labels
        density = X[:, 0] / X[:, 1]
        gdp_factor = X[:, 2] / 50000  # Normalize
        
        # Label: 0 = Sufficient, 1 = Development Needed
        y = np.zeros(n_samples)
        
        # High density + low GDP = development needed
        high_density_mask = density > 2
        low_gdp_mask = gdp_factor < 0.3
        y[(high_density_mask & low_gdp_mask) | (density > 5)] = 1
        
        # Some medium cases
        medium_mask = (density > 1) & (density <= 3) & (gdp_factor < 0.5)
        y[medium_mask] = np.random.choice([0, 1], size=medium_mask.sum(), p=[0.4, 0.6])
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            min_samples_split=5,
            min_samples_leaf=2
        )
        model.fit(X[:, :2], y)  # Use only Population and Area for prediction
        
        # Save model
        with open('ai_model.pkl', 'wb') as f:
            pickle.dump(model, f)
            
        # Save training info
        training_info = {
            "samples": n_samples,
            "accuracy": model.score(X[:, :2], y),
            "created": str(np.datetime64('now')),
            "features": ["Population", "Area"]
        }
        with open('model_info.json', 'w') as f:
            json.dump(training_info, f, indent=2)
            
        st.sidebar.success("✅ Default model created and saved")
        return model

# Load model
model = load_or_create_model()

# ================= INPUT SECTION =================
st.markdown("---")
st.header("📊 Input Parameters")

col1, col2 = st.columns(2)

with col1:
    population = st.number_input(
        "**Population (thousands)**", 
        value=500.0,
        min_value=0.0,
        max_value=50000.0,
        step=50.0,
        help="Total population in thousands (e.g., 500 = 500,000 people)"
    )
    
    # Population slider for quick adjustment
    population = st.slider(
        "Adjust population",
        min_value=0.0,
        max_value=5000.0,
        value=float(population),
        step=50.0,
        key="pop_slider"
    )

with col2:
    area = st.number_input(
        "**Area Size (sq km)**", 
        value=200.0,
        min_value=1.0,
        max_value=10000.0,
        step=10.0,
        help="Geographical area in square kilometers"
    )
    
    # Area slider for quick adjustment
    area = st.slider(
        "Adjust area",
        min_value=1.0,
        max_value=2000.0,
        value=float(area),
        step=10.0,
        key="area_slider"
    )

# Additional economic factors (optional)
with st.expander("⚙️ Advanced Economic Factors (Optional)"):
    col3, col4 = st.columns(2)
    
    with col3:
        gdp_per_capita = st.number_input(
            "GDP per Capita (USD)",
            value=5000.0,
            min_value=0.0,
            step=1000.0
        )
    
    with col4:
        urban_population = st.slider(
            "Urban Population %",
            min_value=0,
            max_value=100,
            value=50,
            step=5
        )

# ================= CALCULATIONS =================
if area > 0:
    density = population / area
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Population Density",
            value=f"{density:.2f}",
            help="Thousand people per sq km"
        )
    
    with col2:
        # Density category
        if density < 0.5:
            category = "Very Low"
            color = "🟢"
        elif density < 1:
            category = "Low"
            color = "🟢"
        elif density < 2:
            category = "Medium"
            color = "🟡"
        elif density < 5:
            category = "High"
            color = "🟠"
        else:
            category = "Very High"
            color = "🔴"
        
        st.metric("Density Category", f"{color} {category}")
    
    with col3:
        # Estimate required infrastructure index
        infra_index = min(100, density * 10 + (100 - urban_population) * 0.3)
        st.metric(
            "Infrastructure Index",
            f"{infra_index:.0f}/100",
            delta="High need" if infra_index > 60 else "Moderate"
        )
    
    st.markdown("---")
    
    # ================= PREDICTION =================
    if st.button("🚀 **Predict Infrastructure Needs**", type="primary", use_container_width=True):
        
        # Prepare input
        X_input = np.array([[population, area]])
        
        # Get prediction
        prediction = model.predict(X_input)[0]
        probabilities = model.predict_proba(X_input)[0]
        
        # Display results
        st.header("📈 **Prediction Results**")
        
        if prediction == 1:
            st.error("""
            ## 🚨 **INFRASTRUCTURE DEVELOPMENT NEEDED**
            
            **Urgency Level:** HIGH  
            **Recommended Action:** Immediate planning required
            """)
        else:
            st.success("""
            ## ✅ **INFRASTRUCTURE IS SUFFICIENT**
            
            **Status:** GOOD  
            **Recommended Action:** Regular maintenance recommended
            """)
        
        # Confidence gauge
        confidence = max(probabilities) * 100
        st.progress(float(confidence/100))
        st.caption(f"Model Confidence: {confidence:.1f}%")
        
        # Detailed probabilities
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Probability: Sufficient",
                f"{probabilities[0]*100:.1f}%",
                delta="Low risk" if probabilities[0] > 0.7 else "Monitor"
            )
        
        with col2:
            st.metric(
                "Probability: Needs Development",
                f"{probabilities[1]*100:.1f}%",
                delta="Action needed" if probabilities[1] > 0.7 else "Watch"
            )
        
        # Visualization
        st.subheader("📊 **Visual Analysis**")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Bar chart
        axes[0].bar(['Sufficient', 'Development Needed'], 
                   [probabilities[0]*100, probabilities[1]*100],
                   color=['green', 'red'])
        axes[0].set_ylabel('Probability (%)')
        axes[0].set_title('Prediction Probabilities')
        axes[0].set_ylim(0, 100)
        
        # Add value labels on bars
        for i, v in enumerate([probabilities[0]*100, probabilities[1]*100]):
            axes[0].text(i, v + 1, f'{v:.1f}%', ha='center')
        
        # Scatter plot for context
        sample_densities = np.linspace(0.1, 10, 100)
        sample_predictions = []
        
        for d in sample_densities:
            sample_input = np.array([[d * area, area]])  # Keep area constant, vary population
            sample_pred = model.predict_proba(sample_input)[0][1]
            sample_predictions.append(sample_pred * 100)
        
        axes[1].plot(sample_densities, sample_predictions, 'b-', linewidth=2)
        axes[1].axvline(x=density, color='r', linestyle='--', label='Your Input')
        axes[1].set_xlabel('Population Density')
        axes[1].set_ylabel('Probability of Need (%)')
        axes[1].set_title('Density vs Development Probability')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Recommendations
        st.subheader("💡 **Recommendations**")
        
        if prediction == 1:
            if density > 5:
                st.warning("""
                **CRITICAL PRIORITY AREA**
                
                **Immediate Actions Required:**
                1. 🚧 **Emergency Infrastructure Audit** (Within 1 month)
                2. 💰 **Budget Allocation** for immediate upgrades
                3. 🏗️ **Rapid Development Plan** (6-month timeline)
                4. 📞 **Stakeholder Coordination** meetings
                5. 📊 **Real-time Monitoring System** implementation
                
                **Focus Areas:**
                - Transportation networks
                - Water and sanitation
                - Healthcare facilities
                - Educational institutions
                - Housing development
                """)
            else:
                st.info("""
                **HIGH PRIORITY AREA**
                
                **Recommended Actions:**
                1. 📋 **Comprehensive Planning** (3-month timeline)
                2. 🏦 **Funding Proposal** development
                3. 🗺️ **Zoning and Land Use** review
                4. 👥 **Community Consultation** process
                5. 📈 **Phased Implementation** strategy
                """)
        else:
            st.success("""
            **MAINTENANCE AND MONITORING ZONE**
            
            **Recommended Actions:**
            1. 🔧 **Regular Maintenance Schedule**
            2. 📊 **Performance Monitoring** system
            3. 🌱 **Sustainable Development** initiatives
            4. 💡 **Smart Technology** integration
            5. 📝 **Future Capacity Planning**
            
            **Preventive Measures:**
            - Quarterly infrastructure inspections
            - Annual capacity assessments
            - Technology upgrade planning
            - Environmental impact reviews
            """)
        
        # Export results
        st.markdown("---")
        st.subheader("📥 **Export Results**")
        
        results_text = f"""
        GLOBAL INFRASTRUCTURE AI - PREDICTION REPORT
        ===========================================
        
        Date: {str(np.datetime64('now'))}
        
        INPUT PARAMETERS:
        - Population: {population:,.0f} thousand
        - Area: {area:,.0f} sq km
        - Population Density: {density:.2f} thousand/sq km
        - Density Category: {category}
        
        PREDICTION RESULTS:
        - Prediction: {'Development Needed' if prediction == 1 else 'Infrastructure Sufficient'}
        - Confidence Level: {confidence:.1f}%
        - Probability (Sufficient): {probabilities[0]*100:.1f}%
        - Probability (Development Needed): {probabilities[1]*100:.1f}%
        
        RECOMMENDATIONS:
        - Priority Level: {'HIGH' if prediction == 1 else 'MODERATE'}
        - Timeline: {'Immediate (0-6 months)' if prediction == 1 else 'Ongoing maintenance'}
        
        Generated by Global Infrastructure AI v3.0
        """
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📄 Download Report (TXT)",
                data=results_text,
                file_name=f"infrastructure_report_{population}_{area}.txt",
                mime="text/plain"
            )
        
        with col2:
            # JSON export
            results_json = {
                "population": population,
                "area": area,
                "density": density,
                "prediction": int(prediction),
                "confidence": float(confidence),
                "probabilities": {
                    "sufficient": float(probabilities[0]),
                    "development_needed": float(probabilities[1])
                },
                "timestamp": str(np.datetime64('now'))
            }
            
            st.download_button(
                label="📊 Download Data (JSON)",
                data=json.dumps(results_json, indent=2),
                file_name=f"infrastructure_data_{population}_{area}.json",
                mime="application/json"
            )
        
        st.balloons()

# ================= SIDEBAR =================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3067/3067256.png", width=100)
    
    st.title("🏗️ Infrastructure AI")
    
    st.markdown("""
    ---
    ### **How It Works**
    
    This AI model analyzes:
    1. **Population Density** (People per sq km)
    2. **Economic Factors** (Optional)
    3. **Infrastructure Patterns**
    
    **Prediction Types:**
    - ✅ **Sufficient**: Current infrastructure meets needs
    - 🚨 **Development Needed**: Requires investment
    
    ---
    """)
    
    # Model info
    if os.path.exists('model_info.json'):
        with open('model_info.json', 'r') as f:
            model_info = json.load(f)
        
        st.subheader("📊 Model Information")
        st.write(f"**Samples:** {model_info.get('samples', 'N/A')}")
        st.write(f"**Accuracy:** {model_info.get('accuracy', 0)*100:.1f}%")
        st.write(f"**Features:** {', '.join(model_info.get('features', []))}")
    
    st.markdown("---")
    
    # Quick examples
    st.subheader("🚀 Quick Examples")
    
    examples = [
        {"label": "Small Town", "pop": 100, "area": 200},
        {"label": "Medium City", "pop": 1000, "area": 300},
        {"label": "Metro Area", "pop": 5000, "area": 500},
        {"label": "Rural Zone", "pop": 50, "area": 1000},
    ]
    
    for example in examples:
        if st.button(f"📌 {example['label']}"):
            st.session_state.pop_slider = example['pop']
            st.session_state.area_slider = example['area']
            st.rerun()
    
    st.markdown("---")
    
    # Reset button
    if st.button("🔄 Reset All", use_container_width=True):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()
    
    # About
    st.markdown("""
    ---
    **Version:** 3.0  
    **Last Updated:** 2024  
    **Developed by:** Global Infrastructure AI Team
    """)

# ================= FOOTER =================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🏗️ <b>Global Infrastructure AI</b> | Decision Support System</p>
    <p style='font-size: 0.8em;'>
        For planning and analysis purposes only. Always consult with infrastructure experts.
    </p>
</div>
""", unsafe_allow_html=True)

# ================= STYLE =================
st.markdown("""
<style>
    .stButton > button {
        border-radius: 10px;
        height: 3em;
        font-weight: bold;
    }
    .stProgress > div > div > div {
        background-color: #4CAF50;
    }
    .css-1v0mbdj {
        margin: auto;
        display: block;
    }
</style>
""", unsafe_allow_html=True)
