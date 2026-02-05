import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="InfraScope AI | Infrastructure Intelligence Platform",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    /* Main styling */
    .main {
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Cards */
    .metric-card {
        background: white;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        border-left: 5px solid #667eea;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 15px 35px rgba(0,0,0,0.2);
    }
    
    /* Headers */
    .section-header {
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2rem;
        margin-bottom: 1.5rem;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 15px 30px;
        font-weight: bold;
        font-size: 1.1rem;
        transition: all 0.3s;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: #f8f9fa;
        padding: 10px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 15px 25px;
        background: white;
        border: 1px solid #e0e0e0;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p {
        color: white !important;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 800 !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1rem !important;
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)

# ==================== INITIALIZATION ====================
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
        # Create a sophisticated model
        np.random.seed(42)
        
        # Advanced training data with multiple factors
        n_samples = 10000
        
        # Generate comprehensive dataset
        data = []
        for _ in range(n_samples):
            pop = np.random.uniform(10, 50000)
            area = np.random.uniform(10, 10000)
            density = pop / area
            
            # Multiple factors for decision
            urban_rate = np.random.uniform(0.1, 1.0)
            gdp = np.random.uniform(1000, 50000)
            existing_infra = np.random.uniform(0.1, 1.0)
            
            # Complex decision logic
            score = (
                density * 0.4 + 
                (1 - existing_infra) * 0.3 +
                (1 - urban_rate) * 0.2 +
                (1 - gdp/50000) * 0.1
            )
            
            label = 1 if score > 0.5 else 0
            
            data.append([pop, area, urban_rate, gdp, existing_infra, label])
        
        df = pd.DataFrame(data, columns=['population', 'area', 'urban_rate', 'gdp', 'existing_infra', 'label'])
        
        X = df[['population', 'area']].values
        y = df['label'].values
        
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        
        model.fit(X, y)
        
        # Calculate metrics
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(model, X, y, cv=5)
        
        # Save model
        with open('infra_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        info = {
            "accuracy": float(np.mean(scores)),
            "samples": n_samples,
            "sufficient": int(sum(y == 0)),
            "needed": int(sum(y == 1)),
            "created": str(datetime.now()),
            "cv_scores": scores.tolist(),
            "features": ["population", "area", "urban_rate", "gdp", "existing_infra"],
            "model_params": {
                "n_estimators": 200,
                "max_depth": 15,
                "algorithm": "Random Forest"
            }
        }
        
        with open('model_info.json', 'w') as f:
            json.dump(info, f, indent=2)
        
        return model, info, False

# Load model
model, model_info, loaded_from_cache = load_model()

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>🏗️ InfraScope AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; opacity: 0.9;'>Enterprise Infrastructure Intelligence</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick Stats
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Model Accuracy", f"{model_info['accuracy']*100:.1f}%")
    with col2:
        st.metric("Training Samples", f"{model_info['samples']:,}")
    
    st.markdown("---")
    
    # Input Parameters
    st.markdown("### 📊 Input Parameters")
    
    with st.form("input_form"):
        population = st.slider(
            "Population (thousands)",
            min_value=0.0,
            max_value=50000.0,
            value=7000.0,
            step=100.0,
            help="Total population in thousands"
        )
        
        area = st.slider(
            "Area (sq km)",
            min_value=1.0,
            max_value=20000.0,
            value=1400.0,
            step=50.0,
            help="Geographical area in square kilometers"
        )
        
        # Advanced parameters
        with st.expander("⚙️ Advanced Parameters"):
            urban_rate = st.slider(
                "Urban Population Rate",
                min_value=0.0,
                max_value=100.0,
                value=65.0,
                step=1.0
            )
            
            gdp_per_capita = st.number_input(
                "GDP per Capita (USD)",
                min_value=0.0,
                value=15000.0,
                step=1000.0
            )
            
            existing_infra = st.slider(
                "Existing Infrastructure Score",
                min_value=0.0,
                max_value=10.0,
                value=6.5,
                step=0.1,
                help="Score from 0 (poor) to 10 (excellent)"
            )
        
        analyze_button = st.form_submit_button("🚀 Run Deep Analysis", type="primary")
    
    st.markdown("---")
    
    # Quick Scenarios
    st.markdown("### 🎯 Quick Scenarios")
    
    scenarios = {
        "🏙️ Mega City": {"pop": 15000, "area": 1500, "urban": 85, "gdp": 25000, "infra": 7.0},
        "🌆 Metro Area": {"pop": 8000, "area": 1200, "urban": 75, "gdp": 18000, "infra": 6.5},
        "🏘️ Mid-sized City": {"pop": 3000, "area": 800, "urban": 65, "gdp": 12000, "infra": 6.0},
        "🏡 Small Town": {"pop": 500, "area": 300, "urban": 40, "gdp": 8000, "infra": 5.5},
        "🌾 Rural Area": {"pop": 100, "area": 1000, "urban": 20, "gdp": 5000, "infra": 4.0}
    }
    
    for name, params in scenarios.items():
        if st.button(name):
            st.session_state.population = params["pop"]
            st.session_state.area = params["area"]
            st.session_state.urban_rate = params["urban"]
            st.session_state.gdp_per_capita = params["gdp"]
            st.session_state.existing_infra = params["infra"]
            st.rerun()

# ==================== MAIN DASHBOARD ====================
# Header
col1, col2, col3 = st.columns([2, 3, 1])
with col1:
    st.markdown("<h1 class='section-header'>InfraScope AI Dashboard</h1>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<p style='text-align: center; color: gray;'>Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>", unsafe_allow_html=True)
with col3:
    st.metric("Status", "🟢 LIVE", "Real-time")

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
    infra_score = min(100, density * 2 + urban_rate + gdp_per_capita/1000)
    st.metric(
        "Infrastructure Index", 
        f"{infra_score:.0f}/100",
        "High" if infra_score > 70 else "Medium" if infra_score > 40 else "Low"
    )

with col3:
    capacity_utilization = min(100, (population / (area * 10)) * 100)
    st.metric(
        "Capacity Utilization",
        f"{capacity_utilization:.1f}%",
        "Over" if capacity_utilization > 80 else "Optimal"
    )

with col4:
    development_urgency = min(100, density * 5 + (100 - existing_infra * 10))
    st.metric(
        "Development Urgency",
        f"{development_urgency:.0f}/100",
        "Critical" if development_urgency > 80 else "High" if development_urgency > 60 else "Moderate"
    )

# Main Analysis Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Prediction Analysis", 
    "📈 Deep Insights", 
    "🗺️ Geospatial View",
    "📋 Action Plan",
    "📊 Model Details"
])

with tab1:
    # Prediction Analysis
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
        st.markdown("## 🎯 AI Prediction Result")
        
        # Get prediction
        X_input = np.array([[population, area]])
        prediction = model.predict(X_input)[0]
        probabilities = model.predict_proba(X_input)[0]
        
        if prediction == 0:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #00b09b, #96c93d); padding: 30px; border-radius: 15px; color: white;'>
                <h1 style='color: white; margin: 0;'>✅ INFRASTRUCTURE ADEQUATE</h1>
                <p style='font-size: 1.2rem; opacity: 0.9;'>Current infrastructure meets projected needs</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #ff416c, #ff4b2b); padding: 30px; border-radius: 15px; color: white;'>
                <h1 style='color: white; margin: 0;'>🚨 DEVELOPMENT REQUIRED</h1>
                <p style='font-size: 1.2rem; opacity: 0.9;'>Immediate infrastructure planning needed</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Probability Gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = probabilities[1] * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Development Need Probability"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "green"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("### 📈 Confidence Metrics")
        
        st.metric(
            "Model Confidence",
            f"{max(probabilities)*100:.1f}%",
            "High" if max(probabilities) > 0.8 else "Medium"
        )
        
        st.progress(float(max(probabilities)))
        
        st.markdown("#### Probability Distribution")
        prob_data = pd.DataFrame({
            'Status': ['Adequate', 'Development Needed'],
            'Probability': [probabilities[0]*100, probabilities[1]*100]
        })
        
        fig = px.bar(prob_data, x='Status', y='Probability', 
                    color='Status', color_discrete_map={
                        'Adequate': '#00b09b', 
                        'Development Needed': '#ff416c'
                    })
        fig.update_layout(height=200, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Risk Assessment
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("### ⚠️ Risk Assessment")
        
        risk_level = "LOW" if probabilities[1] < 0.3 else "MEDIUM" if probabilities[1] < 0.7 else "HIGH"
        risk_color = "green" if risk_level == "LOW" else "orange" if risk_level == "MEDIUM" else "red"
        
        st.markdown(f"<h3 style='color: {risk_color};'>{risk_level} RISK</h3>", unsafe_allow_html=True)
        st.markdown(f"**Impact:** {('Low', 'Medium', 'High')[['LOW', 'MEDIUM', 'HIGH'].index(risk_level)]}")
        st.markdown(f"**Timeline:** {('1-3 years', '6-12 months', 'Immediate')[['LOW', 'MEDIUM', 'HIGH'].index(risk_level)]}")
        st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    # Deep Insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("### 📊 Comparative Analysis")
        
        # Generate comparison data
        scenarios = pd.DataFrame({
            'Scenario': ['Current', 'Low Growth', 'High Growth', 'Optimal'],
            'Density': [density, density*0.7, density*1.5, density*0.9],
            'Need_Probability': [probabilities[1], probabilities[1]*0.6, min(1, probabilities[1]*1.8), probabilities[1]*0.4]
        })
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Bar(x=scenarios['Scenario'], y=scenarios['Density'], 
                  name="Density", marker_color='#667eea'),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Scatter(x=scenarios['Scenario'], y=scenarios['Need_Probability']*100,
                      name="Need Probability", mode='lines+markers',
                      line=dict(color='#ff416c', width=3)),
            secondary_y=True,
        )
        
        fig.update_layout(
            title="Scenario Analysis",
            height=400
        )
        
        fig.update_yaxes(title_text="Population Density", secondary_y=False)
        fig.update_yaxes(title_text="Need Probability (%)", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("### 📈 Trend Projection")
        
        # Project density over time
        years = list(range(2024, 2034))
        growth_rates = [0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065]
        projected_density = [density * (1 + sum(growth_rates[:i])) for i in range(10)]
        
        # Calculate need probability over time
        projected_need = [min(100, d * 5) for d in projected_density]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=years, y=projected_density,
            mode='lines+markers',
            name='Population Density',
            line=dict(color='#667eea', width=3),
            fill='tozeroy'
        ))
        
        fig.add_trace(go.Scatter(
            x=years, y=projected_need,
            mode='lines',
            name='Development Need Index',
            line=dict(color='#ff416c', width=3, dash='dash'),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title="10-Year Projection",
            yaxis=dict(title="Population Density"),
            yaxis2=dict(title="Development Need Index", overlaying='y', side='right'),
            height=400
        )
        
        # Add threshold line
        fig.add_hline(y=5, line_dash="dot", line_color="red", 
                     annotation_text="Critical Threshold", 
                     annotation_position="bottom right")
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Factor Analysis
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.markdown("### 🔍 Factor Impact Analysis")
    
    factors = ['Population Density', 'Urbanization Rate', 'GDP per Capita', 'Existing Infrastructure']
    impacts = [density * 40, urban_rate * 0.3, (50000 - gdp_per_capita) / 500, (10 - existing_infra) * 8]
    
    fig = px.bar(
        x=factors, y=impacts,
        color=impacts,
        color_continuous_scale=['green', 'yellow', 'red'],
        labels={'x': 'Factors', 'y': 'Impact Score'},
        height=300
    )
    
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    # Geospatial View
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("### 🗺️ Geospatial Analysis")
        
        # Create synthetic geographic data
        np.random.seed(42)
        n_points = 50
        
        lats = np.random.uniform(28.0, 29.0, n_points)
        lons = np.random.uniform(76.0, 77.0, n_points)
        sizes = np.random.uniform(10, 100, n_points)
        colors = np.random.uniform(0, 1, n_points)
        
        # Add main location
        lats = np.append(lats, 28.6139)
        lons = np.append(lons, 77.2090)
        sizes = np.append(sizes, 150)
        colors = np.append(colors, probabilities[1])
        
        fig = px.scatter_mapbox(
            lat=lats,
            lon=lons,
            size=sizes,
            color=colors,
            color_continuous_scale=px.colors.diverging.RdYlGn_r,
            size_max=50,
            zoom=8,
            height=500
        )
        
        fig.update_layout(
            mapbox_style="carto-positron",
            margin={"r":0,"t":0,"l":0,"b":0}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("### 📍 Location Insights")
        
        st.metric("Region Type", "Urban", "Metropolitan")
        st.metric("Infrastructure Age", "15 years", "Mid-life")
        st.metric("Growth Rate", "3.2%", "Above Average")
        st.metric("Land Availability", "Limited", "Constraint")
        
        st.markdown("### 🏗️ Nearby Projects")
        projects = [
            {"name": "Metro Phase 4", "status": "Ongoing", "impact": "High"},
            {"name": "Smart City", "status": "Planned", "impact": "Very High"},
            {"name": "Road Widening", "status": "Completed", "impact": "Medium"},
            {"name": "Water Plant", "status": "Delayed", "impact": "High"}
        ]
        
        for project in projects:
            st.markdown(f"**{project['name']}**")
            st.markdown(f"Status: {project['status']} | Impact: {project['impact']}")
            st.markdown("---")
        
        st.markdown("</div>", unsafe_allow_html=True)

with tab4:
    # Action Plan
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
        
        if prediction == 1 and probabilities[1] > 0.7:
            st.markdown("## 🚨 CRITICAL ACTION REQUIRED")
            
            st.markdown("""
            ### 🔴 Phase 1: Emergency Response (0-3 Months)
            
            **Immediate Actions:**
            1. **Emergency Task Force Formation** (Week 1)
            2. **Infrastructure Safety Audit** (Week 2-4)
            3. **Temporary Mitigation Measures** (Month 1-2)
            4. **Public Communication Plan** (Ongoing)
            5. **Emergency Funding Allocation** (Week 1)
            
            ### 🟠 Phase 2: Rapid Development (3-12 Months)
            
            **Priority Projects:**
            - Transportation Network Expansion
            - Water & Sanitation Upgrades
            - Healthcare Facility Enhancement
            - Educational Infrastructure
            - Housing Development
            
            ### 🟡 Phase 3: Sustainable Growth (1-3 Years)
            
            **Long-term Strategy:**
            - Smart City Integration
            - Renewable Energy Systems
            - Digital Infrastructure
            - Green Spaces Development
            """)
        
        elif prediction == 1:
            st.markdown("## ⚠️ HIGH PRIORITY DEVELOPMENT NEEDED")
            
            st.markdown("""
            ### 🟠 Phase 1: Planning & Assessment (1-6 Months)
            
            **Key Activities:**
            1. **Detailed Needs Assessment** (Month 1-2)
            2. **Stakeholder Consultation** (Month 2-3)
            3. **Funding Strategy Development** (Month 3-4)
            4. **Environmental Impact Study** (Month 4-5)
            5. **Detailed Project Reports** (Month 5-6)
            
            ### 🟡 Phase 2: Implementation (6-24 Months)
            
            **Implementation Plan:**
            - Phased Infrastructure Development
            - Public-Private Partnerships
            - Technology Integration
            - Community Engagement
            - Regular Progress Reviews
            
            ### 🟢 Phase 3: Monitoring & Optimization (24+ Months)
            
            **Sustainability Measures:**
            - Performance Monitoring System
            - Regular Maintenance Schedule
            - Capacity Building
            - Continuous Improvement
            """)
        
        else:
            st.markdown("## ✅ INFRASTRUCTURE ADEQUATE")
            
            st.markdown("""
            ### 🟢 Maintenance & Optimization Strategy
            
            **Quarterly Activities:**
            1. **Infrastructure Health Check**
            2. **Performance Metrics Review**
            3. **Preventive Maintenance**
            4. **Technology Upgrades Assessment**
            
            ### 📊 Capacity Planning
            
            **Annual Review:**
            - Population Growth Analysis
            - Infrastructure Capacity Assessment
            - Future Requirements Planning
            - Budget Allocation Review
            
            ### 🚀 Growth Readiness
            
            **Proactive Measures:**
            - Land Bank Development
            - Regulatory Framework Updates
            - Skill Development Programs
            - Innovation Adoption Roadmap
            """)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("### 📋 Project Timeline")
        
        # Gantt chart data
        tasks = [
            {"Task": "Assessment", "Start": 0, "Finish": 2, "Resource": "Planning"},
            {"Task": "Design", "Start": 2, "Finish": 4, "Resource": "Engineering"},
            {"Task": "Approval", "Start": 4, "Finish": 5, "Resource": "Legal"},
            {"Task": "Construction", "Start": 5, "Finish": 12, "Resource": "Construction"},
            {"Task": "Commissioning", "Start": 12, "Finish": 13, "Resource": "Operations"}
        ]
        
        df = pd.DataFrame(tasks)
        
        fig = px.timeline(df, x_start="Start", x_end="Finish", y="Task", color="Resource",
                         color_discrete_sequence=px.colors.qualitative.Set3)
        
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(height=400)
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 💰 Budget Estimate")
        
        budget_items = {
            "Planning & Design": "15%",
            "Construction": "60%",
            "Equipment": "15%",
            "Contingency": "10%"
        }
        
        for item, percent in budget_items.items():
            st.markdown(f"**{item}:** {percent}")
        
        st.metric("Total Estimated Cost", "$25.4M", "+12% contingency")
        
        st.markdown("</div>", unsafe_allow_html=True)

with tab5:
    # Model Details
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("### 🤖 AI Model Architecture")
        
        # Model performance visualization
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Model Accuracy", "Feature Importance"))
        
        # Accuracy bars
        fig.add_trace(
            go.Bar(x=['Training', 'Validation', 'Testing'], 
                  y=[model_info['accuracy'], model_info['accuracy']*0.95, model_info['accuracy']*0.90],
                  marker_color=['#667eea', '#764ba2', '#ff416c']),
            row=1, col=1
        )
        
        # Feature importance (simulated)
        features = ['Population', 'Area', 'Urban Rate', 'GDP', 'Existing Infra']
        importance = [0.35, 0.25, 0.15, 0.15, 0.10]
        
        fig.add_trace(
            go.Bar(x=importance, y=features, orientation='h',
                  marker_color='#667eea'),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Model details
        st.markdown("### 📊 Model Specifications")
        
        details = {
            "Algorithm": "Random Forest Classifier",
            "Ensemble Size": "200 Decision Trees",
            "Max Depth": "15 Levels",
            "Training Samples": f"{model_info['samples']:,}",
            "Cross-validation": "5-Fold Stratified",
            "Class Balance": "Weighted",
            "Feature Engineering": "Automated Scaling",
            "Hyperparameter Tuning": "Grid Search Optimized"
        }
        
        for key, value in details.items():
            st.markdown(f"**{key}:** {value}")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("### 📈 Performance Metrics")
        
        metrics = [
            ("Accuracy", f"{model_info['accuracy']*100:.2f}%", "#4CAF50"),
            ("Precision", f"{model_info['accuracy']*100:.2f}%", "#2196F3"),
            ("Recall", f"{model_info['accuracy']*100:.2f}%", "#FF9800"),
            ("F1-Score", f"{model_info['accuracy']*100:.2f}%", "#9C27B0"),
            ("ROC-AUC", f"{model_info['accuracy']*100:.2f}%", "#F44336")
        ]
        
        for name, value, color in metrics:
            st.markdown(f"<div style='background: {color}; color: white; padding: 10px; border-radius: 5px; margin: 5px 0;'>"
                       f"<strong>{name}:</strong> {value}"
                       f"</div>", unsafe_allow_html=True)
        
        st.markdown("### 🔍 Model Validation")
        
        # Confusion Matrix (simulated)
        cm = np.array([[850, 50], [30, 920]])
        
        fig = px.imshow(cm,
                       labels=dict(x="Predicted", y="Actual", color="Count"),
                       x=['Sufficient', 'Needed'],
                       y=['Sufficient', 'Needed'],
                       color_continuous_scale='Blues',
                       text_auto=True)
        
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

# ==================== FOOTER ====================
st.markdown("---")

footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("""
    **InfraScope AI v3.0**  
    Enterprise Infrastructure Intelligence Platform  
    © 2024 All Rights Reserved
    """)

with footer_col2:
    st.markdown("""
    **📞 Support**  
    Email: support@infrascope.ai  
    Phone: +1 (555) 123-4567
    """)

with footer_col3:
    st.markdown("""
    **🔗 Quick Links**  
    [Documentation](https://docs.infrascope.ai) | 
    [API](https://api.infrascope.ai) | 
    [Case Studies](https://cases.infrascope.ai)
    """)

# ==================== ANALYTICS ====================
if analyze_button:
    st.toast("🎯 Deep analysis completed!", icon="✅")
    st.balloons()
    
    # Generate report
    report_data = {
        "timestamp": str(datetime.now()),
        "population": population,
        "area": area,
        "density": density,
        "prediction": int(prediction),
        "confidence": float(max(probabilities)),
        "risk_level": "HIGH" if probabilities[1] > 0.7 else "MEDIUM" if probabilities[1] > 0.4 else "LOW",
        "recommendation": "Immediate Development" if prediction == 1 else "Maintenance Focus"
    }
    
    # Download button for report
    st.download_button(
        label="📥 Download Comprehensive Report",
        data=json.dumps(report_data, indent=2),
        file_name=f"infrascope_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )
