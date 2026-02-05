import streamlit as st
import pickle
import numpy as np
import os

# Title
st.title("🏗️ Global Infrastructure AI")
st.write("Enter 2 values for prediction (Population & Area):")

# Load model with error handling
def load_model():
    try:
        with open('ai_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model, None
    except FileNotFoundError:
        return None, "Model file not found. Please ensure 'ai_model.pkl' exists in the same folder."
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

# Check if model exists
if not os.path.exists('ai_model.pkl'):
    st.warning("⚠️ Model file not found. Creating a default model...")
    
    # Create simple default model
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np
    
    # Simple training data
    X_train = np.array([[100, 50], [500, 200], [1000, 300], [2000, 400]])
    y_train = np.array([0, 0, 1, 1])
    
    default_model = RandomForestClassifier()
    default_model.fit(X_train, y_train)
    
    with open('ai_model.pkl', 'wb') as f:
        pickle.dump(default_model, f)
    
    st.success("Default model created successfully!")

# Load the model
model, error = load_model()

if error:
    st.error(f"❌ {error}")
    st.stop()

# Input fields
col1, col2 = st.columns(2)

with col1:
    input1 = st.number_input(
        "Population (thousands)", 
        value=100.0, 
        min_value=0.0,
        help="Enter population in thousands"
    )

with col2:
    input2 = st.number_input(
        "Area Size (sq km)", 
        value=50.0, 
        min_value=0.0,
        help="Enter area in square kilometers"
    )

# Calculate population density
density = input1 / input2 if input2 > 0 else 0
st.info(f"📊 Population Density: {density:.2f} thousand people per sq km")

# Predict button
if st.button("🔮 Predict Now", type="primary"):
    try:
        # Prepare input
        inputs = np.array([[input1, input2]])
        
        # Make prediction
        prediction = model.predict(inputs)
        prediction_proba = model.predict_proba(inputs)
        
        # Show results
        st.markdown("---")
        st.subheader("📈 Prediction Results")
        
        col_result, col_prob = st.columns(2)
        
        with col_result:
            if prediction[0] == 1:
                st.error("🚨 **Infrastructure Development Needed!**")
                st.write("Based on the inputs, this region may require additional infrastructure investment.")
            else:
                st.success("✅ **Infrastructure is Sufficient**")
                st.write("Current infrastructure appears adequate for the given population and area.")
        
        with col_prob:
            st.metric(
                label="Confidence Level",
                value=f"{max(prediction_proba[0])*100:.1f}%"
            )
        
        # Show detailed probability
        with st.expander("📊 View Detailed Probabilities"):
            st.write(f"**Probability of 'Sufficient':** {prediction_proba[0][0]*100:.2f}%")
            st.write(f"**Probability of 'Development Needed':** {prediction_proba[0][1]*100:.2f}%")
            st.progress(float(prediction_proba[0][1]))
        
        # Recommendations
        st.markdown("---")
        st.subheader("💡 Recommendations")
        
        if prediction[0] == 1:
            st.write("""
            **Suggested Actions:**
            1. Conduct detailed infrastructure assessment
            2. Consider transportation network expansion
            3. Evaluate utilities capacity (water, electricity)
            4. Plan for public service facilities
            5. Budget for phased infrastructure development
            """)
        else:
            st.write("""
            **Maintenance Focus:**
            1. Regular infrastructure maintenance
            2. Monitor population growth trends
            3. Plan for future capacity expansion
            4. Optimize existing resource utilization
            """)
        
        st.balloons()
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.info("Try using different input values.")

# Add sidebar information
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This AI model predicts infrastructure needs based on:
    - **Population**: Number of people (in thousands)
    - **Area**: Geographical size (in sq km)
    
    **Prediction Classes:**
    - 0: Infrastructure is sufficient
    - 1: Infrastructure development needed
    
    *Note: This is a demonstration model.*
    """)
    
    if st.button("🔄 Reset Inputs"):
        st.rerun()

# Footer
st.markdown("---")
st.caption("Global Infrastructure AI v1.0 | For demonstration purposes")
