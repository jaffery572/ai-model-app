# app.py — Streamlit Cloud compatible (joblib + ONNX fallback) — UPDATED (model path fix)

import os
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go

# Optional deps (safe on Cloud)
try:
    import joblib
    JOBLIB_OK = True
except Exception:
    JOBLIB_OK = False

try:
    import onnxruntime as ort
    ONNX_OK = True
except Exception:
    ONNX_OK = False


# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="🌍 Global Infrastructure AI",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Styling
# ----------------------------
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        background: linear-gradient(90deg, #1a2980, #26d0ce);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: 800;
    }
    .sub-header {
        font-size: 1.25rem;
        color: #dfe6e9;
        border-left: 5px solid #3498db;
        padding-left: 12px;
        margin: 1rem 0;
    }
    .pill {
        display:inline-block;
        padding:4px 10px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 12px;
        margin-left: 8px;
    }
    .risk-high { background:#ff4d4d; color:white; }
    .risk-med  { background:#ffb020; color:#111; }
    .risk-low  { background:#00cc66; color:white; }
    .card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 14px 16px;
    }
    .muted { color: #b2bec3; font-size: 0.9rem; }
</style>
""",
    unsafe_allow_html=True
)


# ----------------------------
# Paths / constants
# ----------------------------
MODEL_DIR = "models"
JOBLIB_CANDIDATES = [
    os.path.join(MODEL_DIR, "infra_model.joblib"),
    os.path.join(MODEL_DIR, "infra_model_cloud.joblib"),
    os.path.join(MODEL_DIR, "infra_model.joblib.pkl"),
    os.path.join(MODEL_DIR, "infra_model.joblib.gz"),
    os.path.join(MODEL_DIR, "infra_model.joblib"),
    os.path.join(MODEL_DIR, "infra_model.joblib"),
    os.path.join(MODEL_DIR, "infra_model.joblib"),
    os.path.join(MODEL_DIR, "infra_model.joblib"),
    os.path.join(MODEL_DIR, "infra_model.joblib"),
    os.path.join(MODEL_DIR, "infra_model.joblib"),
    os.path.join(MODEL_DIR, "infra_model.joblib"),
    os.path.join(MODEL_DIR, "infra_model.joblib"),
    os.path.join(MODEL_DIR, "infra_model.joblib"),
    os.path.join(MODEL_DIR, "infra_model.joblib"),
    os.path.join(MODEL_DIR, "infra_model.joblib"),
    os.path.join(MODEL_DIR, "infra_model.joblib"),
    os.path.join(MODEL_DIR, "infra_model.joblib"),
    os.path.join(MODEL_DIR, "infra_model.joblib"),
    os.path.join(MODEL_DIR, "infra_model.joblib"),
    os.path.join(MODEL_DIR, "infra_model.joblib"),
    os.path.join(MODEL_DIR, "infra_model.joblib"),
    os.path.join(MODEL_DIR, "infra_model.joblib"),
    os.path.join(MODEL_DIR, "infra_model.joblib"),
    os.path.join(MODEL_DIR, "infra_model.joblib"),
    os.path.join(MODEL_DIR, "infra_model.joblib"),
    os.path.join(MODEL_DIR, "infra_model.joblib"),
    os.path.join(MODEL_DIR, "infra_model.joblib"),
    os.path.join(MODEL_DIR, "infra_model.joblib"),
    os.path.join(MODEL_DIR, "infra_model.joblib"),
    os.path.join(MODEL_DIR, "infra_model.joblib"),
    os.path.join(MODEL_DIR, "infra_model.joblib"),
    os.path.join(MODEL_DIR, "infra_model.joblib"),
    os.path.join(MODEL_DIR, "infra_model.joblib"),
    os.path.join(MODEL_DIR, "infra_model.joblib"),
    os.path.join(MODEL_DIR, "infra_model.joblib"),
    os.path.join(MODEL_DIR, "infra_model.joblib"),
    os.path.join(MODEL_DIR, "infra_model.joblib"),
    os.path.join(MODEL_DIR, "infra_model.joblib"),
    os.path.join(MODEL_DIR, "infra_model.joblib"),
    os.path.join(MODEL_DIR, "infra_model.joblib"),
    os.path.join(MODEL_DIR, "infra_model.joblib"),
    os.path.join(MODEL_DIR, "infra_model.joblib"),
    os.path.join(MODEL_DIR, "infra_model.joblib"),
    os.path.join(MODEL_DIR, "infra_model.joblib"),
    os.path.join(MODEL_DIR, "infra_model.joblib"),
    os.path.join(MODEL_DIR, "infra_model.joblib"),
    os.path.join(MODEL_DIR, "infra_model.joblib"),
]
# Keep only unique while preserving order
seen = set()
JOBLIB_CANDIDATES = [p for p in JOBLIB_CANDIDATES if not (p in seen or seen.add(p))]

ONNX_CANDIDATES = [
    os.path.join(MODEL_DIR, "infra_model.onnx"),
    os.path.join(MODEL_DIR, "infra_model_cloud.onnx"),
    os.path.join(MODEL_DIR, "infra_model_cloud.onnx"),
    os.path.join(MODEL_DIR, "infra_model_cloud.onnx"),
]

FEATURES = ["Population_Millions", "GDP_per_capita_USD", "HDI_Index", "Urbanization_Rate"]


# ----------------------------
# Helpers
# ----------------------------
def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def risk_badge(prob):
    if prob >= 70:
        return "HIGH", "risk-high"
    if prob >= 40:
        return "MEDIUM", "risk-med"
    return "LOW", "risk-low"


def heuristic_predict(row):
    # Deterministic heuristic (works without model)
    gdp = safe_float(row.get("GDP_per_capita_USD", np.nan))
    hdi = safe_float(row.get("HDI_Index", np.nan))
    urb = safe_float(row.get("Urbanization_Rate", np.nan))
    pop = safe_float(row.get("Population_Millions", np.nan))

    if np.isnan(gdp) or np.isnan(hdi) or np.isnan(urb) or np.isnan(pop):
        return 50.0

    need_score = (
        (1 - min(gdp / 50000.0, 1.0)) * 40.0 +
        (1.0 - hdi) * 30.0 +
        ((urb / 100.0) if gdp < 10000 else 0.0) * 20.0 +
        (1.0 if (pop > 100 and gdp < 5000) else 0.0) * 10.0
    )
    return float(np.clip(need_score, 0, 100))


@st.cache_data(show_spinner=False)
def load_default_country_data():
    # Fallback small dataset (safe)
    data = {
        "Country": ["USA","CHN","IND","DEU","GBR","JPN","BRA","RUS","FRA","ITA","CAN","AUS","KOR","MEX","IDN","TUR","SAU","CHE","NLD","ESP","PAK","BGD","NGA","EGY","VNM","THA","ZAF","ARG","COL","MYS"],
        "Population_Millions": [331,1412,1408,83,68,125,215,144,67,59,38,26,51,129,278,85,36,8.6,17,47,240,170,216,109,98,70,60,45,52,33],
        "GDP_per_capita_USD": [63500,12500,2300,45700,42200,40100,8900,11200,40400,32000,43200,52000,35000,9900,4300,9500,23500,81900,52400,29400,1500,2600,2300,3900,2800,7800,6300,10600,6400,11400],
        "HDI_Index": [0.926,0.761,0.645,0.947,0.932,0.925,0.765,0.824,0.901,0.892,0.929,0.944,0.916,0.779,0.718,0.820,0.857,0.955,0.944,0.904,0.557,0.632,0.539,0.707,0.704,0.777,0.709,0.845,0.767,0.803],
        "Urbanization_Rate": [83,64,35,77,84,92,87,75,81,71,81,86,81,81,57,76,84,74,93,81,37,39,52,43,38,51,68,92,81,78],
        "Continent": ["North America","Asia","Asia","Europe","Europe","Asia","South America","Europe","Europe","Europe","North America","Oceania","Asia","North America","Asia","Asia","Asia","Europe","Europe","Europe","Asia","Asia","Africa","Africa","Asia","Asia","Africa","South America","South America","Asia"]
    }
    df = pd.DataFrame(data)
    df["Infrastructure_Need_Heuristic"] = df.apply(lambda r: heuristic_predict(r), axis=1)
    return df


def validate_schema(df: pd.DataFrame):
    missing = [c for c in FEATURES if c not in df.columns]
    return missing


# ----------------------------
# Model loading (Cloud-safe)
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_joblib_model():
    """
    Returns: (kind, model_obj, model_path, features)
      kind: "joblib" | None
    """
    if not JOBLIB_OK:
        return (None, None, None, None)

    for p in JOBLIB_CANDIDATES:
        if os.path.exists(p):
            try:
                obj = joblib.load(p)
                # allow either raw model OR dict {"model":..., "features":[...]}
                if isinstance(obj, dict) and "model" in obj:
                    feats = obj.get("features", FEATURES)
                    return ("joblib", obj["model"], p, feats)
                return ("joblib", obj, p, FEATURES)
            except Exception as e:
                # joblib mismatch happens due to numpy pickle differences
                # we'll fallback to ONNX
                return (None, None, None, None)
    return (None, None, None, None)


@st.cache_resource(show_spinner=False)
def load_onnx_session():
    """
    Returns: (kind, session, model_path)
      kind: "onnx" | None

    FIX: Now returns model_path too (so sidebar shows it).
    """
    if not ONNX_OK:
        return (None, None, None)

    for p in ONNX_CANDIDATES:
        if os.path.exists(p):
            try:
                sess = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
                return ("onnx", sess, p)
            except Exception:
                return (None, None, None)
    return (None, None, None)


def model_predict(model_kind, model_obj, features, row_dict):
    """
    Returns probability/score in 0..100
    """
    x = np.array([[row_dict.get(f, np.nan) for f in features]], dtype=np.float32)

    if model_kind == "onnx":
        # ONNX inference
        inputs = model_obj.get_inputs()
        input_name = inputs[0].name
        out = model_obj.run(None, {input_name: x})
        # handle typical outputs (prob or score)
        y = out[0]
        y = np.array(y).reshape(-1)

        # If model outputs probability in [0,1], scale; else assume [0,100]
        val = float(y[0])
        if 0.0 <= val <= 1.0:
            return val * 100.0
        return float(np.clip(val, 0, 100))

    if model_kind == "joblib":
        # sklearn model; prefer predict_proba
        if hasattr(model_obj, "predict_proba"):
            proba = model_obj.predict_proba(x)[0]
            # take positive class if exists
            if len(proba) > 1:
                return float(proba[1] * 100.0)
            return float(proba[0] * 100.0)
        if hasattr(model_obj, "predict"):
            pred = model_obj.predict(x)[0]
            # if binary pred, map to rough score
            return float(pred) * 100.0

    # fallback
    return heuristic_predict(row_dict)


# ----------------------------
# UI
# ----------------------------
def main():
    # Sidebar settings
    st.sidebar.markdown("## Settings")
    show_debug = st.sidebar.toggle("Show debug info", value=False)

    # Header
    st.markdown('<div class="main-header">🌍 Global Infrastructure AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="muted" style="text-align:center;">World-style indicators + ML model. Streamlit Cloud compatible (joblib + ONNX fallback).</div>',
        unsafe_allow_html=True
    )

    # Load model (prefer ONNX on Cloud)
    model_kind, model, model_path, model_features = None, None, None, None

    # 1) Try ONNX first (Cloud safest)
    ok, sess, onnx_path = load_onnx_session()
    if ok == "onnx":
        model_kind, model, model_path, model_features = "onnx", sess, onnx_path, FEATURES
    else:
        # 2) fallback to joblib
        jk, jm, jp, jf = load_joblib_model()
        if jk == "joblib":
            model_kind, model, model_path, model_features = jk, jm, jp, jf

    if show_debug:
        st.sidebar.markdown("---")
        st.sidebar.write("Model kind:", model_kind)
        st.sidebar.write("Model path:", model_path)
        st.sidebar.write("Features:")
        st.sidebar.write(model_features if model_features else FEATURES)

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🧠 Predictor", "🧾 Model Status"])

    # Data (fallback + optional upload)
    df_default = load_default_country_data()

    with tab1:
        st.markdown('<div class="sub-header">Dashboard</div>', unsafe_allow_html=True)

        with st.container():
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f'<div class="card"><div class="muted">Countries</div><h2>{len(df_default)}</h2></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="card"><div class="muted">Avg GDP/capita</div><h2>${df_default["GDP_per_capita_USD"].mean():,.0f}</h2></div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="card"><div class="muted">Avg HDI</div><h2>{df_default["HDI_Index"].mean():.3f}</h2></div>', unsafe_allow_html=True)
            c4.markdown(f'<div class="card"><div class="muted">Model</div><h2>{"ONNX" if model_kind=="onnx" else ("JOBLIB" if model_kind=="joblib" else "HEURISTIC")}</h2></div>', unsafe_allow_html=True)

        st.markdown("### Charts")

        col1, col2 = st.columns(2)

        with col1:
            # Top GDP bar
            d_top = df_default.sort_values("GDP_per_capita_USD", ascending=False).head(10).copy()
            fig1 = px.bar(
                d_top,
                x="Country",
                y="GDP_per_capita_USD",
                title="Top 10 Countries by GDP per Capita",
            )
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            # Scatter GDP vs HDI
            fig2 = px.scatter(
                df_default,
                x="GDP_per_capita_USD",
                y="HDI_Index",
                size="Population_Millions",
                color="Continent",
                hover_name="Country",
                title="GDP vs HDI (bubble = population)",
                log_x=True
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("### Country Table (fallback dataset)")
        st.dataframe(df_default, use_container_width=True)

    with tab2:
        st.markdown('<div class="sub-header">Infrastructure Predictor</div>', unsafe_allow_html=True)

        left, right = st.columns([2, 1])

        with left:
            mode = st.radio("Input method", ["Select country (fallback)", "Custom input", "Upload CSV"], horizontal=True)

            country_row = None
            input_row = {}

            if mode == "Select country (fallback)":
                country = st.selectbox("Country", df_default["Country"].tolist())
                country_row = df_default[df_default["Country"] == country].iloc[0].to_dict()
                input_row = {
                    "Population_Millions": float(country_row["Population_Millions"]),
                    "GDP_per_capita_USD": float(country_row["GDP_per_capita_USD"]),
                    "HDI_Index": float(country_row["HDI_Index"]),
                    "Urbanization_Rate": float(country_row["Urbanization_Rate"]),
                }

            elif mode == "Custom input":
                input_row = {
                    "Population_Millions": st.number_input("Population (Millions)", min_value=0.1, max_value=2000.0, value=100.0, step=1.0),
                    "GDP_per_capita_USD": st.number_input("GDP per Capita (USD)", min_value=100.0, max_value=200000.0, value=5000.0, step=100.0),
                    "HDI_Index": st.slider("HDI Index", min_value=0.2, max_value=1.0, value=0.7, step=0.01),
                    "Urbanization_Rate": st.slider("Urbanization Rate (%)", min_value=0.0, max_value=100.0, value=50.0, step=1.0),
                }

            else:
                up = st.file_uploader("Upload CSV with columns: " + ", ".join(FEATURES), type=["csv"])
                if up is not None:
                    try:
                        df_up = pd.read_csv(up)
                        missing = validate_schema(df_up)
                        if missing:
                            st.error(f"Missing columns: {missing}")
                        else:
                            st.success("CSV ok. Select a row to score.")
                            idx = st.number_input("Row index", min_value=0, max_value=max(len(df_up)-1, 0), value=0, step=1)
                            r = df_up.iloc[int(idx)].to_dict()
                            input_row = {f: safe_float(r.get(f)) for f in FEATURES}
                            st.dataframe(df_up.head(20), use_container_width=True)
                    except Exception as e:
                        st.error(f"CSV read failed: {e}")

        with right:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### Run prediction")
            run = st.button("🚀 PREDICT", type="primary", use_container_width=True)

            if run:
                # Validate numbers
                if any(np.isnan(safe_float(input_row.get(f))) for f in FEATURES):
                    st.error("Please provide valid numeric values for all features.")
                else:
                    score = model_predict(model_kind, model, model_features or FEATURES, input_row)
                    risk, cls = risk_badge(score)
                    conf = float(np.clip(95.0 - abs(50.0 - score) / 2.0, 50.0, 99.0))

                    st.session_state["last_pred"] = {
                        "score": score,
                        "risk": risk,
                        "confidence": conf,
                        "input": input_row,
                        "model_kind": model_kind or "heuristic",
                        "ts": datetime.utcnow().isoformat()
                    }

                    st.markdown(
                        f"""
                        <div style="text-align:center; padding: 8px 0;">
                            <h1 style="margin:0;">{score:.1f}%</h1>
                            <div class="pill {cls}">{risk} RISK</div>
                            <div class="muted" style="margin-top:8px;">Confidence: {conf:.1f}%</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            st.markdown("</div>", unsafe_allow_html=True)

        if "last_pred" in st.session_state:
            pred = st.session_state["last_pred"]
            st.markdown("---")
            st.markdown("### Detailed analysis")

            a, b = st.columns(2)
            with a:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=pred["score"],
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Infrastructure Need"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'steps': [
                            {'range': [0, 30], 'color': "#00cc66"},
                            {'range': [30, 70], 'color': "#ffb020"},
                            {'range': [70, 100], 'color': "#ff4d4d"},
                        ],
                    }
                ))
                fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig, use_container_width=True)

            with b:
                score = pred["score"]
                if score >= 70:
                    st.error("🚨 High priority investment needed")
                    st.write("- Urgent infrastructure development required")
                    st.write("- Focus: transport, utilities, electricity access")
                    st.write("- Suggested budget: $10B+ (indicative)")
                elif score >= 40:
                    st.warning("⚠️ Moderate investment recommended")
                    st.write("- Strategic planning and upgrades")
                    st.write("- Focus: sustainable development, resilience")
                    st.write("- Suggested budget: $1–10B (indicative)")
                else:
                    st.success("✅ Maintenance & optimization")
                    st.write("- Maintain existing infra")
                    st.write("- Focus: smart city initiatives")
                    st.write("- Suggested budget: <$1B (indicative)")

            st.markdown("### Inputs used")
            st.json(pred["input"])

    with tab3:
        st.markdown('<div class="sub-header">Model Status</div>', unsafe_allow_html=True)

        if model_kind == "onnx":
            st.success("Model loaded ✅ (onnx)")
        elif model_kind == "joblib":
            st.success("Model loaded ✅ (joblib)")
        else:
            st.warning("Model not loaded. Using heuristic scoring.")

        st.markdown("**Tip:** If joblib load fails due to numpy mismatch, ONNX fallback is the safest approach on Cloud.")
        st.markdown("---")
        st.write(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Streamlit Cloud safe")

    st.markdown("---")
    st.caption("🌍 Global Infrastructure AI | Streamlit Cloud compatible | No paid APIs")


if __name__ == "__main__":
    main()
