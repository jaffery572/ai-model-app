import streamlit as st
import pickle
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier

# Title
st.title("🏗️ Global Infrastructure AI")
st.write("Enter 2 values for prediction (Population & Area):")

# Initialize model variable
model = None

# Check if model exists, otherwise create default
if not os.path.exists('ai_model.pkl'):
    st.warning("⚠️ Model file not found. Creating a default model...")
    
    # Create simple training data
    X_train = np.array([[100, 50], [500, 200], [1000, 300], 
                        [50, 100], [200, 500], [300, 1000]])
    y_train = np.array([0, 0, 1, 0, 0, 1])  # 0=Sufficient, 1=Development Needed
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    with open('ai_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    st.success("✅ Default model created!")
else:
    try:
        with open('ai_model.pkl', 'rb') as f:
            model = pickle.load(f)
    except:
        st.error("❌ Error loading model. Creating new one...")
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(np.array([[100, 50], [500, 200]]), np.array([0, 1]))

# Input fields
col1, col2 = st.columns(2)

with col1:
    population = st.number_input(
        "Population (thousands)", 
        value=100.0, 
        min_value=0.0,
        help="Enter population in thousands"
    )

with col2:
    area = st.number_input(
        "Area Size (sq km)", 
        value=50.0, 
        min_value=0.0,
        help="Enter area in square kilometers"
    )

# Calculate population density
if area > 0:
    density = population / area
    st.info(f"📊 Population Density: **{density:.2f}** thousand people per sq km")
else:
    st.warning("Area must be greater than 0")
    density = 0

# Simple rule-based estimation (as backup)
def rule_based_estimate(pop, area):
    if area == 0:
        return 1  # Development needed if area is 0
    
    density = pop / area
    if density > 3:
        return 1  # High density = development needed
    elif density > 1:
        return 0.5  # Medium density
    else:
        return 0  # Low density = sufficient

# Predict button
if st.button("🔮 Predict Now", type="primary") and model is not None:
    try:
        # Prepare input
        inputs = np.array([[population, area]])
        
        # Make prediction
        prediction = model.predict(inputs)[0]
        
        # Get probabilities (FIXED)
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(inputs)[0]
            
            # Ensure we have exactly 2 classes
            if len(probabilities) >= 2:
                prob_sufficient = probabilities[0] * 100
                prob_needed = probabilities[1] * 100
            else:
                # Fallback if model returns different format
                prob_sufficient = 50.0
                prob_needed = 50.0
        else:
            # Fallback if no probabilities available
            prob_sufficient = 50.0 if prediction == 0 else 30.0
            prob_needed = 50.0 if prediction == 1 else 30.0
        
        # Get rule-based estimate
        rule_value = rule_based_estimate(population, area)
        
        # Display results
        st.markdown("---")
        st.subheader("📈 Prediction Results")
        
        # Result column
        col_result, col_prob = st.columns(2)
        
        with col_result:
            if prediction == 1:
                st.error("🚨 **Infrastructure Development Needed!**")
                st.write(f"Rule-based check: {'Confirms need' if rule_value > 0.5 else 'Suggests review'}")
            else:
                st.success("✅ **Infrastructure is Sufficient**")
                st.write(f"Rule-based check: {'Confirms sufficiency' if rule_value < 0.5 else 'Suggests review'}")
        
        with col_prob:
            # Use the higher probability as confidence
            confidence = max(prob_sufficient, prob_needed)
            st.metric(
                label="Model Confidence",
                value=f"{confidence:.1f}%"
            )
        
        # Detailed probabilities (CORRECTED)
        with st.expander("📊 View Detailed Probabilities", expanded=True):
            col_prob1, col_prob2 = st.columns(2)
            
            with col_prob1:
                st.metric(
                    label="Probability of 'Sufficient'",
                    value=f"{prob_sufficient:.1f}%"
                )
            
            with col_prob2:
                st.metric(
                    label="Probability of 'Needed'",
                    value=f"{prob_needed:.1f}%"
                )
            
            # Progress bar for visualization
            st.write("**Probability Distribution:**")
            if prob_sufficient > prob_needed:
                st.progress(float(prob_sufficient/100))
                st.caption(f"Model is {prob_sufficient:.1f}% confident infrastructure is sufficient")
            else:
                st.progress(float(prob_needed/100))
                st.caption(f"Model is {prob_needed:.1f}% confident development is needed")
        
        # Recommendations based on both model and rules
        st.markdown("---")
        st.subheader("💡 Recommendations")
        
        if prediction == 1 or rule_value > 0.7:
            st.write("""
            **🟢 Strong Recommendation: Infrastructure Development Needed**
            
            **Suggested Actions:**
            1. 📋 Conduct comprehensive infrastructure audit
            2. 🚧 Prioritize transportation network upgrades
            3. ⚡ Assess utilities capacity (water, electricity, sewage)
            4. 🏥 Plan healthcare and education facilities
            5. 💰 Budget for 3-5 year development plan
            """)
        elif prediction == 0 and rule_value < 0.3:
            st.write("""
            **🔵 Strong Recommendation: Infrastructure is Sufficient**
            
            **Maintenance Focus:**
            1. 🔧 Regular infrastructure maintenance schedule
            2. 📊 Monitor population growth trends quarterly
            3. 🔄 Optimize existing resource utilization
            4. 📈 Plan for future capacity (5+ years)
            5. 🌱 Sustainable development initiatives
            """)
        else:
            st.write("""
            **🟡 Moderate Recommendation: Further Assessment Needed**
            
            **Next Steps:**
            1. 🔍 Conduct detailed local assessment
            2. 📈 Analyze specific sector needs
            3. 👥 Community needs survey
            4. 📋 Cost-benefit analysis
            5. 🗓️ 6-month review recommended
            """)
        
        st.balloons()
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.info("Using rule-based estimation instead...")
        
        # Fallback to rule-based
        rule_result = rule_based_estimate(population, area)
        if rule_result > 0.5:
            st.error("🚨 Rule-based: Infrastructure Development Likely Needed")
        else:
            st.success("✅ Rule-based: Infrastructure Likely Sufficient")

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ How It Works")
    st.write("""
    **Model Logic:**
    1. Takes Population & Area as inputs
    2. Calculates Population Density
    3. AI Model predicts infrastructure status
    
    **Interpretation:**
    - **0 = Sufficient**: Infrastructure meets current needs
    - **1 = Development Needed**: Requires infrastructure investment
    
    **Rule Backup:**
    - Density < 1: Usually sufficient
    - Density > 3: Usually needs development
    """)
    
    # Model info
    if model and hasattr(model, 'n_estimators'):
        st.metric("Model Trees", model.n_estimators)
    
    if st.button("🔄 Reset All"):
        st.rerun()

# Footer
st.markdown("---")
st.caption("""
**Global Infrastructure AI v2.0** | 
Combines AI prediction with rule-based validation |
For demonstration and planning purposes
""")
