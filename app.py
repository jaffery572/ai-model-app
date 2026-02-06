# app.py — Streamlit Cloud robust (joblib + ONNX fallback)
import os
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Optional ONNX runtime (fallback)
try:
    import onnxruntime as ort
    ONNX_OK = True
except Exception:
    ONNX_OK = False


# -----------------------------
# Page config (ALWAYS top-level)
# -----------------------------
st.set_page_config(
    page_title="🌍 Global Infrastructure AI",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

JOBLIB_CANDIDATES = [
    os.path.join(MODELS_DIR, "infra_model.joblib"),
    os.path.join(MODELS_DIR, "infra_model_cloud.joblib"),
]
ONNX_CANDIDATES = [
    os.path.join(MODELS_DIR, "infra_model.onnx"),
    os.path.join(MODELS_DIR, "infra_model_cloud.onnx"),
]


# -----------------------------
# UI CSS (safe)
# -----------------------------
st.markdown(
    """
<style>
.main-header {
    font-size: 2.2rem;
    font-weight: 800;
    margin: 0.2rem 0 0.6rem 0;
}
.sub-header {
    font-size: 1.2rem;
    margin: 0.8rem 0 0.4rem 0;
    opacity: 0.9;
}
.badge {
    display:inline-block;
    padding: 0.25rem 0.6rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.85rem;
}
.high { background:#ff4d4d; color:white; }
.med  { background:#ffb020; color:#111; }
.low  { background:#18c964; color:white; }
.card {
    padding: 1rem;
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.08);
    background: rgba(255,255,255,0.03);
}
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------
# Defaults / fallback dataset
# -----------------------------
FEATURES = ["Population_Millions", "GDP_per_capita_USD", "HDI_Index", "Urbanization_Rate"]

@st.cache_data(show_spinner=False)
def load_default_country_data() -> pd.DataFrame:
    data = {
        "Country": [
            "USA","CHN","IND","DEU","GBR","JPN","BRA","RUS","FRA","ITA","CAN","AUS","KOR","MEX","IDN",
            "TUR","SAU","CHE","NLD","ESP","PAK","BGD","NGA","EGY","VNM","THA","ZAF","ARG","COL","MYS"
        ],
        "Population_Millions": [
            331,1412,1408,83,68,125,215,144,67,59,38,26,51,129,278,85,36,8.6,17,47,
            240,170,216,109,98,70,60,45,52,33
        ],
        "GDP_per_capita_USD": [
            63500,12500,2300,45700,42200,40100,8900,11200,40400,32000,43200,52000,35000,9900,4300,
            9500,23500,81900,52400,29400,1500,2600,2300,3900,2800,7800,6300,10600,6400,11400
        ],
        "HDI_Index": [
            0.926,0.761,0.645,0.947,0.932,0.925,0.765,0.824,0.901,0.892,0.929,0.944,0.916,0.779,
            0.718,0.820,0.857,0.955,0.944,0.904,0.557,0.632,0.539,0.707,0.704,0.777,0.709,0.845,
            0.767,0.803
        ],
        "Urbanization_Rate": [
            83,64,35,77,84,92,87,75,81,71,81,86,81,81,57,76,84,74,93,81,
            37,39,52,43,38,51,68,92,81,78
        ],
        "Continent": [
            "North America","Asia","Asia","Europe","Europe","Asia","South America","Europe","Europe","Europe",
            "North America","Oceania","Asia","North America","Asia","Asia","Asia","Europe","Europe","Europe",
            "Asia","Asia","Africa","Africa","Asia","Asia","Africa","South America","South America","Asia"
        ],
    }
    df = pd.DataFrame(data)

    # Simple label (same as your original rule idea)
    def calc_need(r):
        score = 0
        if r["GDP_per_capita_USD"] < 5000: score += 2
        if r["HDI_Index"] < 0.7: score += 2
        if r["Urbanization_Rate"] < 50 and r["Population_Millions"] > 50: score += 1
        return 1 if score >= 3 else 0

    df["Infrastructure_Need"] = df.apply(calc_need, axis=1)
    return df


def risk_badge(prob: float):
    if prob >= 0.70:
        return "HIGH", "high"
    if prob >= 0.40:
        return "MEDIUM", "med"
    return "LOW", "low"


def heuristic_prob(x: dict) -> float:
    # Safe fallback if model not available
    gdp = float(x["GDP_per_capita_USD"])
    hdi = float(x["HDI_Index"])
    urb = float(x["Urbanization_Rate"])
    pop = float(x["Population_Millions"])

    need_score = (
        (1 - min(gdp / 50000, 1)) * 0.40 +
        (1 - hdi) * 0.30 +
        ((urb / 100) if gdp < 10000 else 0) * 0.20 +
        (1 if (pop > 100 and gdp < 5000) else 0) * 0.10
    )
    return float(np.clip(need_score, 0.0, 1.0))


# -----------------------------
# Model loading (robust)
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_joblib_model():
    for p in JOBLIB_CANDIDATES:
        if os.path.exists(p):
            obj = joblib.load(p)
            # allow either direct model or dict {model, features}
            if isinstance(obj, dict) and "model" in obj:
                return ("joblib", obj["model"], obj.get("features", FEATURES), p)
            return ("joblib", obj, FEATURES, p)
    return (None, None, FEATURES, None)


@st.cache_resource(show_spinner=False)
def load_onnx_session():
    if not ONNX_OK:
        return (None, None)
    for p in ONNX_CANDIDATES:
        if os.path.exists(p):
            # CPU only on Streamlit Cloud
            sess = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
            return ("onnx", sess)
    return (None, None)


def predict_with_model(model_kind, model_or_sess, features, row_dict):
    X = np.array([[float(row_dict[f]) for f in features]], dtype=np.float32)

    if model_kind == "joblib":
        # sklearn model expects float64 usually; safe to cast
        y_prob = None
        if hasattr(model_or_sess, "predict_proba"):
            y_prob = model_or_sess.predict_proba(X.astype(np.float64))[0, 1]
        else:
            # fallback to decision output
            y_pred = model_or_sess.predict(X.astype(np.float64))[0]
            y_prob = float(y_pred)
        return float(np.clip(y_prob, 0.0, 1.0))

    if model_kind == "onnx":
        # ONNX: find first input name
        input_name = model_or_sess.get_inputs()[0].name
        out = model_or_sess.run(None, {input_name: X})[0]
        # common outputs: [[p0, p1]] or [[p1]]
        out = np.asarray(out)
        if out.ndim == 2 and out.shape[1] >= 2:
            return float(np.clip(out[0, 1], 0.0, 1.0))
        return float(np.clip(out.ravel()[0], 0.0, 1.0))

    return heuristic_prob(row_dict)


# -----------------------------
# Main
# -----------------------------
def main():
    st.markdown('<div class="main-header">🌍 Global Infrastructure AI</div>', unsafe_allow_html=True)
    st.caption("World-style indicators + ML model. Streamlit Cloud compatible (joblib + ONNX fallback).")

    # Always render something (prevents “blank dark page” feel)
    with st.sidebar:
        st.subheader("Settings")
        st.write("Model load is safe. If it fails, app still runs with heuristic predictions.")
        show_debug = st.toggle("Show debug info", value=False)

    # Load data
    df = load_default_country_data()

    # Load model (with safe error handling)
    model_kind = None
    model = None
    features = FEATURES
    model_path = None

    # Joblib first
    try:
        mk, m, feats, mp = load_joblib_model()
        model_kind, model, features, model_path = mk, m, feats, mp
    except Exception as e:
        if show_debug:
            st.exception(e)
        model_kind, model, features, model_path = None, None, FEATURES, None

    # ONNX fallback if joblib failed
    if model_kind is None:
        try:
            ok, sess = load_onnx_session()
            if ok == "onnx":
                model_kind, model = "onnx", sess
        except Exception as e:
            if show_debug:
                st.exception(e)

    if show_debug:
        st.sidebar.write("Model kind:", model_kind)
        st.sidebar.write("Model path:", model_path)
        st.sidebar.write("Features:", features)

    tabs = st.tabs(["📊 Dashboard", "🤖 Predictor", "🧠 Model Status"])

    # ---------------- Dashboard
    with tabs[0]:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Countries", len(df))
        c2.metric("Needs (label)", int(df["Infrastructure_Need"].sum()))
        c3.metric("Avg GDP/cap", f"${df['GDP_per_capita_USD'].mean():,.0f}")
        c4.metric("Model", "Loaded ✅" if model_kind else "Heuristic ⚠️")

        colA, colB = st.columns(2)
        with colA:
            fig1 = px.bar(
                df.sort_values("GDP_per_capita_USD", ascending=False).head(10),
                x="Country",
                y="GDP_per_capita_USD",
                title="Top 10 Countries by GDP/cap",
                color="Continent",  # discrete-safe
            )
            st.plotly_chart(fig1, use_container_width=True)

        with colB:
            # log_x can break if zeros; ensure min > 0
            safe_df = df.copy()
            safe_df["GDP_per_capita_USD"] = safe_df["GDP_per_capita_USD"].clip(lower=1)
            fig2 = px.scatter(
                safe_df,
                x="GDP_per_capita_USD",
                y="HDI_Index",
                size="Population_Millions",
                color="Continent",
                hover_name="Country",
                title="GDP/cap vs HDI (log X)",
                log_x=True,
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="sub-header">Country Data</div>', unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True)

    # ---------------- Predictor
    with tabs[1]:
        st.markdown('<div class="sub-header">AI Infrastructure Predictor</div>', unsafe_allow_html=True)

        left, right = st.columns([2, 1])

        with left:
            option = st.radio("Input Method", ["Select Country", "Custom Input"], horizontal=True)

            if option == "Select Country":
                selected = st.selectbox("Choose Country", df["Country"].tolist())
                row = df[df["Country"] == selected].iloc[0].to_dict()
                country_data = {
                    "Population_Millions": float(row["Population_Millions"]),
                    "GDP_per_capita_USD": float(row["GDP_per_capita_USD"]),
                    "HDI_Index": float(row["HDI_Index"]),
                    "Urbanization_Rate": float(row["Urbanization_Rate"]),
                }
            else:
                country_data = {
                    "Population_Millions": st.number_input("Population (Millions)", 1.0, 2000.0, 100.0, step=1.0),
                    "GDP_per_capita_USD": st.number_input("GDP per Capita (USD)", 500.0, 200000.0, 5000.0, step=100.0),
                    "HDI_Index": st.slider("HDI Index", 0.30, 1.00, 0.70),
                    "Urbanization_Rate": st.slider("Urbanization Rate (%)", 5.0, 100.0, 50.0),
                }

        with right:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.write("### Run Prediction")
            run = st.button("🚀 PREDICT", use_container_width=True, type="primary")

            if run:
                try:
                    prob = predict_with_model(model_kind, model, features, country_data)
                    risk, cls = risk_badge(prob)

                    st.session_state["pred_prob"] = prob
                    st.session_state["pred_risk"] = risk
                    st.session_state["pred_cls"] = cls
                except Exception as e:
                    st.error("Prediction failed. Check logs.")
                    if show_debug:
                        st.exception(e)
            st.markdown("</div>", unsafe_allow_html=True)

        if "pred_prob" in st.session_state:
            prob = float(st.session_state["pred_prob"])
            risk = st.session_state["pred_risk"]
            cls = st.session_state["pred_cls"]

            st.markdown("---")
            st.markdown("### Result")

            a, b = st.columns([1, 1])
            with a:
                st.markdown(
                    f"""
<div class="card">
  <div style="font-size:2rem;font-weight:800;">{prob*100:.1f}%</div>
  <div style="opacity:0.9;">Infrastructure Need</div>
  <div style="margin-top:0.6rem;"><span class="badge {cls}">{risk} RISK</span></div>
</div>
""",
                    unsafe_allow_html=True,
                )

            with b:
                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=prob * 100.0,
                        domain={"x": [0, 1], "y": [0, 1]},
                        title={"text": "Need Score"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "steps": [
                                {"range": [0, 40], "color": "rgba(24,201,100,0.25)"},
                                {"range": [40, 70], "color": "rgba(255,176,32,0.25)"},
                                {"range": [70, 100], "color": "rgba(255,77,77,0.25)"},
                            ],
                        },
                    )
                )
                fig.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Recommendations")
            if prob >= 0.70:
                st.error("🚨 High priority investment recommended (transport, utilities, reliability).")
            elif prob >= 0.40:
                st.warning("⚠️ Moderate investment recommended (sustainable growth + upgrades).")
            else:
                st.success("✅ Focus on maintenance + optimization (smart infra initiatives).")

    # ---------------- Model status
    with tabs[2]:
        st.markdown('<div class="sub-header">Model Status</div>', unsafe_allow_html=True)

        if model_kind:
            st.success(f"Model loaded ✅ ({model_kind})")
            if model_path:
                st.code(model_path)
        else:
            st.warning("No model loaded. App is running in heuristic mode.")
            st.info(
                "Upload one of these into /models:\n"
                "- infra_model.joblib OR infra_model_cloud.joblib\n"
                "- (optional) infra_model.onnx OR infra_model_cloud.onnx"
            )

        st.caption("Tip: If joblib load fails due to numpy mismatch, ONNX fallback is the safest approach on Cloud.")

    st.markdown("---")
    st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Streamlit Cloud safe")

if __name__ == "__main__":
    main()
