import streamlit as st
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import json
from sklearn.ensemble import RandomForestClassifier

# ================= AUTO-TRAIN MODEL =================
def create_trained_model():
    """Create and save a trained model"""
    np.random.seed(42)
    
    # Generate training data
    n_samples = 3000
    X = []
    y = []
    
    # 1. Clear "Sufficient" cases (low density)
    for _ in range(n_samples // 3):
        pop = np.random.uniform(10, 500)  # 10k to 500k
        area = np.random.uniform(300, 3000)  # 300 to 3000 sq km
        X.append([pop, area])
        y.append(0)  # Sufficient
    
    # 2. Clear "Development Needed" cases (high density)
    for _ in range(n_samples // 3):
        pop = np.random.uniform(3000, 15000)  # 3M to 15M
        area = np.random.uniform(30, 300)  # 30 to 300 sq km
        X.append([pop, area])
        y.append(1)  # Development needed
    
    # 3. Borderline cases (mix)
    for _ in range(n_samples // 3):
        pop = np.random.uniform(500, 3000)  # 500k to 3M
        area = np.random.uniform(100, 800)  # 100 to 800 sq km
        X.append([pop, area])
        
        # Decision based on density
        density = pop / area
        if density < 1.5:
            y.append(0)  # Sufficient
        elif density > 3:
            y.append(1)  # Development needed
        else:
            # Random for borderline
            y.append(0 if np.random.random() > 0.6 else 1)
    
    X = np.array(X)
    y = np.array(y)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=80,
        max_depth=8,
        random_state=42,
        min_samples_split=4
    )
    model.fit(X, y)
    
    return model, X, y

# Initialize or load model
MODEL_FILE = 'infra_model.pkl'
INFO_FILE = 'model_info.json'

if not os.path.exists(MODEL_FILE):
    st.info("🤖 Training AI model for the first time...")
    model, X_train, y_train = create_trained_model()
    
    # Save model
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
    
    # Save info
    accuracy = model.score(X_train, y_train)
    info = {
        "accuracy": float(accuracy),
        "samples": len(X_train),
        "sufficient": int(sum(y_train == 0)),
        "needed": int(sum(y_train == 1)),
        "created": str(np.datetime64('now'))
    }
    
    with open(INFO_FILE, 'w') as f:
        json.dump(info, f, indent=2)
    
    st.success(f"✅ Model trained! Accuracy: {accuracy:.1%}")
else:
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
    
    if os.path.exists(INFO_FILE):
        with open(INFO_FILE, 'r') as f:
            info = json.load(f)

# ================= STREAMLIT APP =================
st.set_page_config(
    page_title="Infrastructure AI",
    page_icon="🏗️",
    layout="centered"
)

st.title("🏗️ Infrastructure AI")
st.markdown("Predict infrastructure needs using AI")

# Inputs
col1, col2 = st.columns(2)

with col1:
    pop = st.number_input(
        "Population (thousands)",
        min_value=0.0,
        value=500.0,
        step=50.0,
        help="e.g., 500 = 500,000 people"
    )

with col2:
    area = st.number_input(
        "Area (sq km)",
        min_value=1.0,
        value=200.0,
        step=10.0
    )

# Calculate density
if area > 0:
    density = pop / area
    
    # Density indicator
    st.subheader("📊 Population Density")
    
    if density < 0.5:
        st.success(f"**{density:.2f}** - Very Low (Infrastructure likely sufficient)")
    elif density < 1.5:
        st.info(f"**{density:.2f}** - Low (Infrastructure likely sufficient)")
    elif density < 3:
        st.warning(f"**{density:.2f}** - Medium (May need assessment)")
    elif density < 5:
        st.error(f"**{density:.2f}** - High (Likely needs development)")
    else:
        st.error(f"**{density:.2f}** - Very High (Definitely needs development)")

# Predict button
if st.button("🔍 Predict Infrastructure Status", type="primary"):
    
    # Prepare input
    X_input = np.array([[pop, area]])
    
    # Get prediction
    pred = model.predict(X_input)[0]
    probs = model.predict_proba(X_input)[0]
    
    # Display result
    st.markdown("---")
    st.subheader("🎯 AI Prediction")
    
    if pred == 0:
        st.success(f"## ✅ INFRASTRUCTURE SUFFICIENT")
        st.write(f"**Confidence:** {probs[0]*100:.1f}%")
    else:
        st.error(f"## 🚨 DEVELOPMENT NEEDED")
        st.write(f"**Confidence:** {probs[1]*100:.1f}%")
    
    # Probability bars
    st.subheader("📈 Probability Analysis")
    
    fig, ax = plt.subplots(figsize=(8, 3))
    
    bars = ax.barh(['Sufficient', 'Development Needed'], 
                   [probs[0]*100, probs[1]*100],
                   color=['green', 'red'])
    
    ax.set_xlim(0, 100)
    ax.set_xlabel('Probability (%)')
    
    # Add percentage labels
    for bar, prob in zip(bars, [probs[0]*100, probs[1]*100]):
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2,
                f'{prob:.1f}%', va='center')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Recommendations
    st.subheader("💡 Recommendations")
    
    if pred == 0 and probs[0] > 0.7:
        st.info("""
        **Maintenance Focus:**
        - Regular infrastructure inspections
        - Monitor population growth
        - Plan for future capacity
        - Consider sustainability upgrades
        """)
    elif pred == 0:
        st.warning("""
        **Monitor Closely:**
        - Quarterly infrastructure reviews
        - Track density changes
        - Prepare contingency plans
        - Community feedback sessions
        """)
    elif pred == 1 and probs[1] > 0.7:
        st.error("""
        **Immediate Action Required:**
        - Conduct infrastructure audit (30 days)
        - Allocate emergency funding
        - Stakeholder coordination
        - Public communication plan
        """)
    else:
        st.warning("""
        **Development Planning Needed:**
        - Detailed infrastructure planning
        - Secure development funding
        - Environmental impact studies
        - Phased implementation
        """)
    
    # Quick stats
    st.markdown("---")
    cols = st.columns(4)
    
    with cols[0]:
        st.metric("Population", f"{pop:,.0f}k")
    
    with cols[1]:
        st.metric("Area", f"{area:,.0f} sq km")
    
    with cols[2]:
        st.metric("Density", f"{density:.2f}")
    
    with cols[3]:
        if pred == 0:
            st.metric("Status", "✅ Sufficient")
        else:
            st.metric("Status", "🚨 Needed")

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3067/3067256.png", width=80)
    
    st.markdown("### About")
    st.write("AI model trained on 3,000 synthetic scenarios.")
    
    if os.path.exists(INFO_FILE):
        with open(INFO_FILE, 'r') as f:
            info = json.load(f)
        
        st.markdown("### Model Info")
        st.write(f"**Accuracy:** {info.get('accuracy', 0)*100:.1f}%")
        st.write(f"**Training Samples:** {info.get('samples', 0)}")
        st.write(f"**Sufficient Cases:** {info.get('sufficient', 0)}")
        st.write(f"**Needed Cases:** {info.get('needed', 0)}")
    
    st.markdown("---")
    
    st.markdown("### Example Inputs")
    
    examples = [
        ("Village", 50, 300),
        ("Town", 200, 150),
        ("City", 1500, 400),
        ("Metro", 5000, 600)
    ]
    
    for name, p, a in examples:
        if st.button(f"{name}: {p}k, {a} sq km"):
            st.session_state['pop'] = p
            st.session_state['area'] = a
            st.rerun()

# Footer
st.markdown("---")
st.caption("Infrastructure AI v2.0 • AI-powered planning tool")


# Add custom CSS for better UI
st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 0 1rem;
    }
    
    /* Cards styling */
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .stMetric label {
        color: white !important;
        font-weight: bold;
    }
    
    .stMetric div {
        color: white !important;
        font-size: 1.5em;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
    }
    
    /* Headers */
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding-bottom: 10px;
    }
    
    /* Success/Error boxes */
    .stAlert {
        border-radius: 10px;
        border-left: 5px solid;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #4CAF50, #8BC34A);
    }
</style>
""", unsafe_allow_html=True)
