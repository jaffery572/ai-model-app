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
    /* Main styling */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Professional cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
        margin-bottom: 20px;
        border-top: 4px solid #4CAF50;
        transition: transform 0.3s;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.12);
    }
    
    /* Prediction cards */
    .prediction-success {
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
        color: white;
        border-radius: 15px;
        padding: 30px;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
    }
    
    .prediction-warning {
        background: linear-gradient(135deg, #ff7e5f 0%, #feb47b 100%);
        color: white;
        border-radius: 15px;
        padding: 30px;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
    }
    
    .prediction-danger {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
        border-radius: 15px;
        padding: 30px;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
    }
    
    /* Headers */
    .main-header {
        background: linear-gradient(90deg, #4CAF50, #2196F3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    .section-header {
        color: #2c3e50;
        font-weight: 700;
        font-size: 1.8rem;
        margin-bottom: 1.5rem;
        border-bottom: 3px solid #4CAF50;
        padding-bottom: 10px;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(76, 175, 80, 0.4);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 12px 24px;
        font-weight: 600;
        background: #f8f9fa;
    }
    
    .stTabs [aria-selected="true"] {
        background: #4CAF50 !important;
        color: white !important;
    }
    
    /* Progress bars */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #4CAF50, #8BC34A);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1rem !important;
        font-weight: 500 !important;
    }
</style>
""", unsafe_allow_html=True)

# ==================== LOAD MODEL ====================
@st.cache_resource
def load_model():
    """Load or create AI model with advanced features"""
    try:
        with open('infra_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('model_info.json', 'r') as f:
            info = json.load(f)
        
        st.sidebar.success("✅ Model loaded from cache")
        return model, info
    except:
        st.sidebar.info("🔄 Training new model...")
        
        # Create comprehensive training data
        np.random.seed(42)
        n_samples = 5000
        
        # Generate diverse training scenarios
        data = []
        for i in range(n_samples):
            # Varying population sizes
            if i < 1000:
                pop = np.random.uniform(10, 1000)  # Small towns
            elif i < 3000:
                pop = np.random.uniform(1000, 10000)  # Medium cities
            else:
                pop = np.random.uniform(10000, 50000)  # Large metros
            
            # Varying areas
            area = np.random.uniform(50, 10000)
            density = pop / area
            
            # Smart labeling based on multiple factors
            urban_factor = np.random.uniform(0.1, 0.9)
            gdp_factor = np.random.uniform(0.3, 1.0)
            
            # Complex decision logic
            if density < 0.5:
                label = 0  # Definitely sufficient
            elif density < 1 and urban_factor < 0.3:
                label = 0  # Rural with low density
            elif density > 5:
                label = 1  # Definitely needs development
            elif density > 3 and gdp_factor < 0.5:
                label = 1  # High density with low GDP
            elif density > 2 and urban_factor > 0.7:
                label = 1  # Urban high density
            else:
                # Borderline cases
                label = 1 if (density * urban_factor * (1-gdp_factor)) > 0.3 else 0
            
            data.append([pop, area, label])
        
        df = pd.DataFrame(data, columns=['population', 'area', 'label'])
        X = df[['population', 'area']].values
        y = df['label'].values
        
        # Train advanced model
        model = RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        
        model.fit(X, y)
        
        # Calculate cross-validation score
        from sklearn.model_selection import cross_val_score
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        
        # Save model
        with open('infra_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        # Save detailed info
        info = {
            "accuracy": float(np.mean(cv_scores)),
            "samples": n_samples,
            "sufficient": int(sum(y == 0)),
            "needed": int(sum(y == 1)),
            "created": str(datetime.now()),
            "cv_scores": cv_scores.tolist(),
            "feature_importance": model.feature_importances_.tolist(),
            "model_type": "RandomForest",
            "parameters": {
                "n_estimators": 150,
                "max_depth": 12,
                "min_samples_split": 5
            }
        }
        
        with open('model_info.json', 'w') as f:
            json.dump(info, f, indent=2)
        
        st.sidebar.success(f"✅ Model trained ({n_samples} samples)")
        return model, info

# Load the model
model, model_info = load_model()

# ==================== SIDEBAR ====================
with st.sidebar:
    # Header
    st.markdown("<h1 style='text-align: center;'>🏗️ InfraScope AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; opacity: 0.9;'>Intelligent Infrastructure Planning</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Input Section
    st.markdown("### 📊 Input Parameters")
    
    population = st.slider(
        "Population (thousands)",
        min_value=0.0,
        max_value=50000.0,
        value=30600.0,
        step=100.0,
        help="Total population in thousands (e.g., 1000 = 1 million)"
    )
    
    area = st.slider(
        "Area (sq km)",
        min_value=1.0,
        max_value=10000.0,
        value=3901.0,
        step=10.0,
        help="Geographical area in square kilometers"
    )
    
    # Quick calculate button
    if st.button("🧮 Calculate Metrics", use_container_width=True):
        st.rerun()
    
    st.markdown("---")
    
    # Model Information
    st.markdown("### 🤖 Model Information")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", f"{model_info['accuracy']*100:.1f}%")
    with col2:
        st.metric("Samples", f"{model_info['samples']:,}")
    
    st.markdown(f"**Model Type:** {model_info.get('model_type', 'Random Forest')}")
    st.markdown(f"**Created:** {model_info.get('created', 'N/A')[:10]}")
    
    st.markdown("---")
    
    # Quick Actions
    st.markdown("### ⚡ Quick Actions")
    
    if st.button("🔄 Reset Inputs", use_container_width=True, type="secondary"):
        st.session_state.clear()
        st.rerun()
    
    if st.button("📊 View Model Details", use_container_width=True):
        st.info("Model details shown in main panel")
    
    st.markdown("---")
    
    # Footer
    st.markdown("""
    <div style='text-align: center; font-size: 0.8rem; opacity: 0.7;'>
    <p>Version 3.1.0</p>
    <p>© 2024 InfraScope AI</p>
    </div>
    """, unsafe_allow_html=True)

# ==================== MAIN DASHBOARD ====================
# Header
st.markdown("<h1 class='main-header'>Infrastructure Intelligence Dashboard</h1>", unsafe_allow_html=True)

# Top Metrics Row
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    density = population / area if area > 0 else 0
    st.metric(
        "Population Density",
        f"{density:.2f}",
        "people/sq km",
        delta_color="off"
    )

with col2:
    infra_index = min(100, density * 12.5)  # Scale factor
    st.metric(
        "Infrastructure Index",
        f"{infra_index:.0f}/100",
        "High" if infra_index > 70 else "Medium" if infra_index > 40 else "Low"
    )

with col3:
    capacity_utilization = min(100, (population / (area * 10)) * 100)
    st.metric(
        "Capacity Utilization",
        f"{capacity_utilization:.1f}%",
        "Optimal" if 60 <= capacity_utilization <= 80 else "Over" if capacity_utilization > 80 else "Under"
    )

with col4:
    development_urgency = min(100, density * 9.5)
    st.metric(
        "Development Urgency",
        f"{development_urgency:.0f}/100",
        "High" if development_urgency > 60 else "Medium" if development_urgency > 30 else "Low"
    )

# Main Content Tabs
tab1, tab2, tab3 = st.tabs(["🎯 AI Prediction", "📈 Deep Analysis", "📋 Action Plan"])

with tab1:
    # AI Prediction Section
    if st.button("🚀 Run AI Analysis", type="primary", use_container_width=True):
        
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
            st.markdown("The AI model indicates that current infrastructure is adequate for the given population and area.")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='prediction-danger'>", unsafe_allow_html=True)
            st.markdown("# 🚨 DEVELOPMENT REQUIRED")
            st.markdown(f"### Confidence: {probabilities[1]*100:.1f}%")
            st.markdown("The AI model indicates that infrastructure development is urgently needed.")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Confidence Metrics
        col_metrics1, col_metrics2 = st.columns(2)
        
        with col_metrics1:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown("### 📊 Probability Distribution")
            
            # Create pie chart
            fig, ax = plt.subplots(figsize=(6, 6))
            labels = ['Sufficient', 'Development Needed']
            sizes = [probabilities[0]*100, probabilities[1]*100]
            colors = ['#4CAF50', '#FF5252']
            explode = (0.05, 0.05)
            
            ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                  autopct='%1.1f%%', shadow=True, startangle=90)
            ax.axis('equal')
            
            st.pyplot(fig)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col_metrics2:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown("### ⚖️ Confidence Metrics")
            
            st.metric(
                "Model Confidence",
                f"{max(probabilities)*100:.1f}%",
                "High" if max(probabilities) > 0.8 else "Medium"
            )
            
            st.progress(float(max(probabilities)))
            
            st.markdown("**Detailed Breakdown:**")
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.metric("Sufficient", f"{probabilities[0]*100:.1f}%")
            with col_p2:
                st.metric("Needed", f"{probabilities[1]*100:.1f}%")
            
            # Risk Level
            if probabilities[1] > 0.7:
                risk = "🔴 HIGH"
            elif probabilities[1] > 0.4:
                risk = "🟡 MEDIUM"
            else:
                risk = "🟢 LOW"
            
            st.markdown(f"**Risk Level:** {risk}")
            st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    # Deep Analysis Tab
    st.markdown("<h2 class='section-header'>Comprehensive Analysis</h2>", unsafe_allow_html=True)
    
    col_analysis1, col_analysis2 = st.columns(2)
    
    with col_analysis1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("### 📈 Density Trend Analysis")
        
        # Create trend chart
        years = list(range(2024, 2034))
        growth_rates = [0.02, 0.022, 0.025, 0.027, 0.03, 0.032, 0.035, 0.037, 0.04, 0.042]
        projected_pop = [population * (1 + sum(growth_rates[:i])) for i in range(10)]
        projected_density = [p / area for p in projected_pop]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(years, projected_density, 'b-', linewidth=3, marker='o', markersize=8)
        ax.fill_between(years, projected_density, alpha=0.2, color='blue')
        ax.axhline(y=5, color='r', linestyle='--', label='Critical Threshold')
        ax.set_xlabel('Year')
        ax.set_ylabel('Population Density')
        ax.set_title('10-Year Density Projection')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        # Interpretation
        if max(projected_density) > 5:
            st.warning("⚠️ **Projection Alert:** Density will exceed critical threshold")
        elif max(projected_density) > 3:
            st.info("ℹ️ **Projection Note:** Density will reach high levels")
        else:
            st.success("✅ **Projection Stable:** Density remains manageable")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col_analysis2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("### 🔍 Factor Impact Analysis")
        
        # Create horizontal bar chart for factors
        factors = ['Population Density', 'Urbanization', 'Economic Factors', 'Existing Infrastructure']
        impacts = [density * 8, 65, 45, 60]  # Example values
        
        fig, ax = plt.subplots(figsize=(10, 4))
        y_pos = np.arange(len(factors))
        bars = ax.barh(y_pos, impacts, color=['#FF5252', '#FF9800', '#4CAF50', '#2196F3'])
        ax.set_yticks(y_pos)
        ax.set_yticklabels(factors)
        ax.set_xlabel('Impact Score (0-100)')
        ax.set_xlim(0, 100)
        
        # Add value labels
        for bar, impact in zip(bars, impacts):
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2,
                   f'{impact:.0f}', va='center')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("**Key Insights:**")
        if density > 5:
            st.markdown("- 🚨 **Population Density** is the primary concern")
        if density > 3:
            st.markdown("- ⚠️ **Urbanization pressure** is significant")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Model Performance
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.markdown("### 🤖 Model Performance")
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    
    with col_m1:
        st.metric("Accuracy", f"{model_info['accuracy']*100:.1f}%")
    
    with col_m2:
        if 'cv_scores' in model_info:
            cv_std = np.std(model_info['cv_scores']) * 100
            st.metric("CV Stability", f"±{cv_std:.1f}%")
    
    with col_m3:
        st.metric("Training Data", f"{model_info['samples']:,}")
    
    with col_m4:
        balance = model_info['sufficient'] / model_info['samples'] * 100
        st.metric("Class Balance", f"{balance:.1f}%")
    
    # Feature importance
    if 'feature_importance' in model_info:
        st.markdown("**Feature Importance:**")
        imp_df = pd.DataFrame({
            'Feature': ['Population', 'Area'],
            'Importance': model_info['feature_importance']
        })
        st.dataframe(imp_df, use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    # Action Plan Tab
    st.markdown("<h2 class='section-header'>Strategic Action Plan</h2>", unsafe_allow_html=True)
    
    # Get prediction for action plan
    X_input = np.array([[population, area]])
    prediction = model.predict(X_input)[0]
    probabilities = model.predict_proba(X_input)[0]
    
    if prediction == 1 and probabilities[1] > 0.7:
        # Critical situation
        st.markdown("<div class='prediction-danger'>", unsafe_allow_html=True)
        st.markdown("## 🔴 CRITICAL ACTION PLAN")
        st.markdown("**Status: Emergency Development Required**")
        st.markdown("</div>", unsafe_allow_html=True)
        
        col_plan1, col_plan2 = st.columns(2)
        
        with col_plan1:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown("### 🚨 Phase 1: Emergency Response (0-3 Months)")
            st.markdown("""
            1. **Immediate Safety Audit**
               - Infrastructure stability assessment
               - Public safety measures
               - Emergency funding allocation
            
            2. **Crisis Management**
               - Task force formation
               - Public communication plan
               - Temporary infrastructure
            
            3. **Rapid Planning**
               - Fast-track approvals
               - Emergency procurement
               - Stakeholder coordination
            """)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col_plan2:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown("### 🏗️ Phase 2: Rapid Development (3-12 Months)")
            st.markdown("""
            1. **Core Infrastructure**
               - Transportation network expansion
               - Water & sanitation upgrades
               - Power grid enhancement
            
            2. **Essential Services**
               - Healthcare facilities
               - Educational institutions
               - Emergency services
            
            3. **Housing & Shelter**
               - Temporary housing
               - Affordable housing projects
               - Community shelters
            """)
            st.markdown("</div>", unsafe_allow_html=True)
    
    elif prediction == 1:
        # Development needed
        st.markdown("<div class='prediction-warning'>", unsafe_allow_html=True)
        st.markdown("## 🟠 PRIORITY ACTION PLAN")
        st.markdown("**Status: Development Planning Required**")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("### 📋 Development Roadmap")
        
        # Create timeline
        timeline_data = {
            'Phase': ['Assessment', 'Planning', 'Implementation', 'Review'],
            'Duration': ['1-2 months', '2-4 months', '6-18 months', 'Ongoing'],
            'Key Activities': ['Needs analysis, Stakeholder consultation', 
                              'Design, Funding, Approvals', 
                              'Construction, Procurement', 
                              'Monitoring, Optimization']
        }
        
        timeline_df = pd.DataFrame(timeline_data)
        st.dataframe(timeline_df, use_container_width=True, hide_index=True)
        
        st.markdown("### 💰 Budget Allocation")
        
        budget_items = {
            'Planning & Design': '15%',
            'Construction': '50%',
            'Equipment & Materials': '20%',
            'Contingency': '10%',
            'Project Management': '5%'
        }
        
        for item, percent in budget_items.items():
            col_b1, col_b2 = st.columns([3, 1])
            with col_b1:
                st.markdown(f"**{item}**")
            with col_b2:
                st.markdown(f"**{percent}**")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    else:
        # Sufficient infrastructure
        st.markdown("<div class='prediction-success'>", unsafe_allow_html=True)
        st.markdown("## 🟢 MAINTENANCE & OPTIMIZATION PLAN")
        st.markdown("**Status: Infrastructure Adequate**")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("### 🔧 Maintenance Strategy")
        
        col_maint1, col_maint2 = st.columns(2)
        
        with col_maint1:
            st.markdown("#### 🗓️ Quarterly Activities")
            st.markdown("""
            - Infrastructure health checks
            - Performance metrics review
            - Preventive maintenance
            - Technology upgrades assessment
            - Community feedback collection
            """)
        
        with col_maint2:
            st.markdown("#### 📊 Annual Review")
            st.markdown("""
            - Population growth analysis
            - Capacity assessment
            - Future requirements planning
            - Budget allocation review
            - Sustainability initiatives
            """)
        
        st.markdown("### 🚀 Growth Readiness")
        
        readiness_items = [
            "Land bank development for future expansion",
            "Regulatory framework updates",
            "Skill development programs",
            "Innovation adoption roadmap",
            "Public-private partnership opportunities"
        ]
        
        for item in readiness_items:
            st.markdown(f"✅ {item}")
        
        st.markdown("</div>", unsafe_allow_html=True)

# ==================== FOOTER & EXPORT ====================
st.markdown("---")

# Success Message
st.success("""
🎉 **Infrastructure Analysis Complete!** 
Your AI-powered assessment is ready. Use the insights above for strategic planning.
""")

# Export Section
st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
st.markdown("### 📤 Export & Share Results")

col_exp1, col_exp2, col_exp3 = st.columns(3)

with col_exp1:
    # JSON Export
    export_data = {
        "analysis_date": str(datetime.now()),
        "parameters": {
            "population": population,
            "area": area,
            "density": density
        },
        "prediction": {
            "result": "Development Required" if prediction == 1 else "Infrastructure Sufficient",
            "confidence": float(max(probabilities) * 100),
            "probabilities": {
                "sufficient": float(probabilities[0] * 100),
                "development_needed": float(probabilities[1] * 100)
            }
        },
        "metrics": {
            "infrastructure_index": float(infra_index),
            "development_urgency": float(development_urgency),
            "capacity_utilization": float(capacity_utilization)
        },
        "model_info": {
            "accuracy": float(model_info['accuracy'] * 100),
            "samples": model_info['samples']
        }
    }
    
    st.download_button(
        label="📊 Download JSON Data",
        data=json.dumps(export_data, indent=2),
        file_name=f"infrastructure_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

with col_exp2:
    # Text Report
    report_text = f"""
    INFRASTRUCTURE INTELLIGENCE REPORT
    ===================================
    
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    INPUT PARAMETERS
    ----------------
    Population: {population:,.0f} thousand
    Area: {area:,.0f} sq km
    Population Density: {density:.2f} people/sq km
    
    AI PREDICTION
    -------------
    Result: {"DEVELOPMENT REQUIRED" if prediction == 1 else "INFRASTRUCTURE SUFFICIENT"}
    Confidence Level: {max(probabilities)*100:.1f}%
    
    Probability Breakdown:
    - Sufficient Infrastructure: {probabilities[0]*100:.1f}%
    - Development Needed: {probabilities[1]*100:.1f}%
    
    KEY METRICS
    -----------
    Infrastructure Index: {infra_index:.0f}/100
    Development Urgency: {development_urgency:.0f}/100
    Capacity Utilization: {capacity_utilization:.1f}%
    
    MODEL INFORMATION
    -----------------
    Model Accuracy: {model_info['accuracy']*100:.1f}%
    Training Samples: {model_info['samples']:,}
    
    RECOMMENDATIONS
    ---------------
    Priority Level: {"CRITICAL" if prediction == 1 and probabilities[1] > 0.7 else "HIGH" if prediction == 1 else "LOW"}
    Timeline: {"IMMEDIATE (0-3 months)" if prediction == 1 and probabilities[1] > 0.7 else "6-18 months" if prediction == 1 else "ONGOING"}
    
    ===================================
    Generated by InfraScope AI v3.1
    For planning and decision support
    ===================================
    """
    
    st.download_button(
        label="📄 Download Text Report",
        data=report_text,
        file_name=f"infrastructure_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )

with col_exp3:
    # Share options
    st.markdown("### 🔗 Share Analysis")
    
    if st.button("📧 Email Report", use_container_width=True):
        st.info("Email feature coming soon!")
    
    if st.button("💬 Request Consultation", use_container_width=True):
        st.info("Consultation request sent!")

st.markdown("</div>", unsafe_allow_html=True)

# Final Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p style='font-size: 1.1rem; font-weight: 600;'>🏗️ InfraScope AI | Enterprise Infrastructure Intelligence Platform</p>
    <p style='font-size: 0.9rem;'>
        Version 3.1.0 • Data Updated: {datetime.now().strftime('%Y-%m-%d')} • 
        <a href='#' style='color: #4CAF50;'>Privacy Policy</a> • 
        <a href='#' style='color: #4CAF50;'>Terms of Service</a>
    </p>
    <p style='font-size: 0.8rem; opacity: 0.7;'>
        This tool provides AI-powered insights for planning purposes. 
        Always consult with infrastructure experts for critical decisions.
    </p>
</div>
""".format(datetime=datetime), unsafe_allow_html=True)

# Celebration
if st.button("🎉 Celebrate Success!"):
    st.balloons()
    st.success("🎊 Infrastructure analysis completed successfully!")
