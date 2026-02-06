# app.py - Streamlit Cloud Robust Version (ONNX + Joblib + Heuristic fallback)
import os
import io
import json
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Optional imports (safe)
try:
    import joblib
except Exception:
    joblib = None

try:
    import onnxruntime as ort
except Exception:
    ort = None


# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="🌍 Global Infrastructure AI",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------
# Styling
# ----------------------------
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 900;
        text-align: center;
        margin-bottom: 0.2rem;
        background: linear-gradient(90deg, #1a2980, #26d0ce);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle {
        text-align:center;
        color:#8aa;
        margin-bottom: 1.2rem;
    }
    .badge {
        display:inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .badge-high { background:#ff4d4d; color:white; }
    .badge-med  { background:#ffb020; color:black; }
    .badge-low  { background:#22c55e; color:white; }
    .card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 16px;
    }
    .small { color:#9ab; font-size:0.9rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------
# Constants
# ----------------------------
FEATURES = ["Population_Millions", "GDP_per_capita_USD", "HDI_Index", "Urbanization_Rate"]

MODEL_JOBLIB_PATH = os.path.join("models", "infra_model.joblib")
MODEL_ONNX_PATH = os.path.join("models", "infra_model.onnx")  # recommended for Cloud
MODEL_KIND = "none"

REQUIRED_COLS = FEATURES[:]  # required CSV schema


# ----------------------------
# Helpers
# ----------------------------
def risk_badge(needed_pct: float):
    if needed_pct >= 70:
        return "HIGH", "badge-high"
    elif needed_pct >= 40:
        return "MEDIUM", "badge-med"
    return "LOW", "badge-low"


def heuristic_predict(row: dict):
    """
    Fallback heuristic model (always works).
    Returns probability (0..100) + confidence.
    """
    gdp = float(row["GDP_per_capita_USD"])
    hdi = float(row["HDI_Index"])
    urb = float(row["Urbanization_Rate"])
    pop = float(row["Population_Millions"])

    need_score = (
        (1 - min(gdp / 50000.0, 1.0)) * 40.0
        + (1 - hdi) * 35.0
        + (urb / 100.0 if gdp < 10000 else 0) * 15.0
        + (10.0 if (pop > 100 and gdp < 5000) else 0.0)
    )
    needed = float(np.clip(need_score, 0, 100))
    conf = float(np.clip(92.0 - abs(50 - needed) / 2.0, 60.0, 95.0))
    return needed, conf


def make_template_csv_bytes():
    template_df = pd.DataFrame(
        [
            {"Country": "Pakistan", "Population_Millions": 240, "GDP_per_capita_USD": 1500, "HDI_Index": 0.557, "Urbanization_Rate": 37},
            {"Country": "France", "Population_Millions": 67, "GDP_per_capita_USD": 40400, "HDI_Index": 0.901, "Urbanization_Rate": 81},
        ]
    )
    return template_df.to_csv(index=False).encode("utf-8")


def validate_and_clean_uploaded_csv(uploaded_file):
    try:
        df_up = pd.read_csv(uploaded_file)
    except Exception as e:
        raise ValueError(f"CSV read failed: {e}")

    missing = [c for c in REQUIRED_COLS if c not in df_up.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Required: {REQUIRED_COLS} (optional: Country)")

    keep = REQUIRED_COLS + (["Country"] if "Country" in df_up.columns else [])
    df_up = df_up[keep].copy()

    for c in REQUIRED_COLS:
        df_up[c] = pd.to_numeric(df_up[c], errors="coerce")

    before = len(df_up)
    df_up = df_up.dropna(subset=REQUIRED_COLS)
    after = len(df_up)
    if after == 0:
        raise ValueError("All rows invalid after cleaning. Check numeric values in required columns.")

    return df_up, (before - after)


@st.cache_data
def load_default_country_data():
    """
    Small safe fallback dataset (no external APIs).
    """
    data = {
        "Country": [
            "USA","CHN","IND","DEU","GBR","JPN","BRA","RUS","FRA","ITA",
            "CAN","AUS","KOR","MEX","IDN","TUR","SAU","CHE","NLD","ESP",
            "PAK","BGD","NGA","EGY","VNM","THA","ZAF","ARG","COL","MYS"
        ],
        "Population_Millions": [331,1412,1408,83,68,125,215,144,67,59,38,26,51,129,278,85,36,8.6,17,47,240,170,216,109,98,70,60,45,52,33],
        "GDP_per_capita_USD": [63500,12500,2300,45700,42200,40100,8900,11200,40400,32000,43200,52000,35000,9900,4300,9500,23500,81900,52400,29400,1500,2600,2300,3900,2800,7800,6300,10600,6400,11400],
        "HDI_Index": [0.926,0.761,0.645,0.947,0.932,0.925,0.765,0.824,0.901,0.892,0.929,0.944,0.916,0.779,0.718,0.820,0.857,0.955,0.944,0.904,0.557,0.632,0.539,0.707,0.704,0.777,0.709,0.845,0.767,0.803],
        "Urbanization_Rate": [83,64,35,77,84,92,87,75,81,71,81,86,81,81,57,76,84,74,93,81,37,39,52,43,38,51,68,92,81,78],
        "Continent": [
            "North America","Asia","Asia","Europe","Europe","Asia","South America","Europe","Europe","Europe",
            "North America","Oceania","Asia","North America","Asia","Asia","Asia","Europe","Europe","Europe",
            "Asia","Asia","Africa","Africa","Asia","Asia","Africa","South America","South America","Asia"
        ],
    }
    df = pd.DataFrame(data)
    return df


# ----------------------------
# Model loading (Cloud-safe)
# ----------------------------
@st.cache_resource
def load_model():
    """
    Returns:
      model_kind: "onnx" | "joblib" | "none"
      model_obj: onnx session OR sklearn model bundle
      meta: dict
    """
    meta = {"path": None, "error": None, "features": FEATURES}

    # 1) Prefer ONNX on Streamlit Cloud (most stable across numpy versions)
    if ort is not None and os.path.exists(MODEL_ONNX_PATH):
        try:
            sess = ort.InferenceSession(MODEL_ONNX_PATH, providers=["CPUExecutionProvider"])
            meta["path"] = MODEL_ONNX_PATH
            return "onnx", sess, meta
        except Exception as e:
            meta["error"] = f"ONNX load failed: {e}"

    # 2) Joblib fallback (may fail if numpy/sklearn mismatch)
    if joblib is not None and os.path.exists(MODEL_JOBLIB_PATH):
        try:
            obj = joblib.load(MODEL_JOBLIB_PATH)
            meta["path"] = MODEL_JOBLIB_PATH
            return "joblib", obj, meta
        except Exception as e:
            meta["error"] = f"Joblib load failed: {e}"

    # 3) None -> heuristic
    return "none", None, meta


def model_predict(model_kind, model_obj, row_dict):
    """
    Unified predictor.
    Returns needed_pct (0..100), confidence_pct (0..100), model_used str
    """
    # Always ensure correct order
    x = np.array([[float(row_dict[f]) for f in FEATURES]], dtype=np.float32)

    if model_kind == "onnx" and ort is not None and model_obj is not None:
        try:
            inputs = model_obj.get_inputs()
            input_name = inputs[0].name
            outputs = model_obj.run(None, {input_name: x})

            # handle common output shapes
            out = outputs[0]
            out = np.array(out).squeeze()

            # if output is prob for class=1
            if out.ndim == 0:
                prob1 = float(out)
            elif out.ndim == 1 and len(out) >= 2:
                # sometimes returns [p0, p1]
                prob1 = float(out[-1])
            else:
                prob1 = float(out.ravel()[-1])

            needed = float(np.clip(prob1 * 100.0, 0, 100))
            conf = float(np.clip(90.0 - abs(50 - needed) / 2.2, 60.0, 95.0))
            return needed, conf, "onnx"

        except Exception:
            # fallback to heuristic if ONNX inference fails
            needed, conf = heuristic_predict(row_dict)
            return needed, conf, "heuristic"

    if model_kind == "joblib" and model_obj is not None:
        try:
            # accept either {"model":..., "features":...} or direct estimator
            est = model_obj["model"] if isinstance(model_obj, dict) and "model" in model_obj else model_obj

            if hasattr(est, "predict_proba"):
                prob1 = float(est.predict_proba(x)[:, 1][0])
            else:
                # fallback: predict -> 0/1
                pred = float(est.predict(x)[0])
                prob1 = pred

            needed = float(np.clip(prob1 * 100.0, 0, 100))
            conf = float(np.clip(90.0 - abs(50 - needed) / 2.2, 60.0, 95.0))
            return needed, conf, "joblib"
        except Exception:
            needed, conf = heuristic_predict(row_dict)
            return needed, conf, "heuristic"

    needed, conf = heuristic_predict(row_dict)
    return needed, conf, "heuristic"


def score_dict(row_dict, model_kind, model_obj):
    needed, conf, used = model_predict(model_kind, model_obj, row_dict)
    level, badge_class = risk_badge(needed)
    return {
        "needed": needed,
        "sufficient": 100 - needed,
        "confidence": conf,
        "risk_level": level,
        "risk_class": badge_class,
        "model_used": used,
    }


# ----------------------------
# Main UI
# ----------------------------
def main():
    st.markdown('<div class="main-header">🌍 Global Infrastructure AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">World-style indicators + ML model. Streamlit Cloud compatible (ONNX + joblib fallback).</div>',
        unsafe_allow_html=True,
    )

    # Sidebar controls
    st.sidebar.header("Settings")
    show_debug = st.sidebar.toggle("Show debug info", value=False)

    model_kind, model_obj, meta = load_model()

    st.sidebar.write("Model load is safe. If it fails, app still runs with heuristic predictions.")
    st.sidebar.write(f"Model kind: **{model_kind}**")
    st.sidebar.write(f"Model path: `{meta.get('path')}`" if meta.get("path") else "Model path: `None`")
    if show_debug and meta.get("error"):
        st.sidebar.error(meta["error"])
    if show_debug:
        st.sidebar.write("Features:")
        st.sidebar.code(json.dumps(FEATURES, indent=2))

    df = load_default_country_data()

    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🧠 Predictor", "📦 Model Status"])

    # ----------------------------
    # Dashboard
    # ----------------------------
    with tab1:
        left, right = st.columns([1.2, 0.8])

        with left:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Overview")
            st.write("Fallback dataset is built-in (safe). Upload CSV in Predictor for your own data.")
            st.write(f"Countries in demo dataset: **{len(df)}**")
            st.write(f"Updated: **{datetime.now().strftime('%Y-%m-%d %H:%M')}**")
            st.markdown('</div>', unsafe_allow_html=True)

        with right:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Model")
            if model_kind == "onnx":
                st.success("Model loaded ✅ (ONNX)")
            elif model_kind == "joblib":
                st.success("Model loaded ✅ (joblib)")
            else:
                st.warning("No model found → using heuristic fallback")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("### Charts")

        c1, c2 = st.columns(2)

        with c1:
            # Discrete color (stable)
            dff = df.sort_values("GDP_per_capita_USD", ascending=False).head(12).copy()
            fig = px.bar(
                dff,
                x="Country",
                y="GDP_per_capita_USD",
                title="Top GDP per Capita (demo dataset)",
                color="Continent",
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig2 = px.scatter(
                df,
                x="GDP_per_capita_USD",
                y="HDI_Index",
                size="Population_Millions",
                color="Continent",
                hover_name="Country",
                title="GDP vs HDI (demo dataset)",
                log_x=True,  # safe for >0 values
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.dataframe(df, use_container_width=True)

    # ----------------------------
    # Predictor
    # ----------------------------
    with tab2:
        st.subheader("Infrastructure Predictor")
        st.caption("Input method: select a demo country, custom values, or upload CSV (batch scoring).")

        col1, col2 = st.columns([2, 1])

        with col1:
            option = st.radio("Input method", ["Select country (demo)", "Custom input", "Upload CSV"], horizontal=True)

            # Template download always available
            st.download_button(
                label="⬇️ Download template CSV",
                data=make_template_csv_bytes(),
                file_name="infra_template.csv",
                mime="text/csv",
                use_container_width=False,
            )

            country_data = None
            upload_df = None

            if option == "Select country (demo)":
                selected_country = st.selectbox("Choose country:", df["Country"].tolist(), index=0)
                row = df[df["Country"] == selected_country].iloc[0]
                country_data = {f: row[f] for f in FEATURES}

            elif option == "Custom input":
                country_data = {
                    "Population_Millions": st.number_input("Population (Millions)", 1.0, 2000.0, 100.0),
                    "GDP_per_capita_USD": st.number_input("GDP per Capita (USD)", 100.0, 200000.0, 5000.0),
                    "HDI_Index": st.slider("HDI Index", 0.2, 1.0, 0.70),
                    "Urbanization_Rate": st.slider("Urbanization Rate (%)", 1.0, 100.0, 50.0),
                }

            else:
                st.caption(f"Upload CSV must contain: {', '.join(REQUIRED_COLS)} (optional: Country)")
                uploaded = st.file_uploader("Upload CSV", type=["csv"])
                if uploaded is not None:
                    try:
                        upload_df, dropped = validate_and_clean_uploaded_csv(uploaded)
                        st.success(f"Loaded ✅ Rows: {len(upload_df)} (dropped invalid: {dropped})")
                        st.dataframe(upload_df.head(20), use_container_width=True)

                        idx = st.number_input("Preview row index", 0, len(upload_df) - 1, 0, 1)
                        r = upload_df.iloc[int(idx)]
                        country_data = {f: float(r[f]) for f in FEATURES}

                    except Exception as e:
                        st.error(str(e))
                        upload_df = None

        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Run prediction")

            if st.button("🚀 PREDICT", type="primary", use_container_width=True):
                if country_data is None:
                    st.error("No input found. Select country, fill custom input, or upload CSV.")
                else:
                    pred = score_dict(country_data, model_kind, model_obj)

                    st.session_state["last_pred"] = pred
                    st.session_state["last_input"] = country_data
                    st.session_state["last_upload_df"] = upload_df

            # show last prediction
            if "last_pred" in st.session_state:
                pred = st.session_state["last_pred"]
                badge = f'<span class="badge {pred["risk_class"]}">{pred["risk_level"]} RISK</span>'

                st.markdown(f"### {pred['needed']:.1f}% need")
                st.markdown(badge, unsafe_allow_html=True)
                st.caption(f"Confidence: {pred['confidence']:.1f}% | Model used: {pred['model_used']}")

            st.markdown('</div>', unsafe_allow_html=True)

        # Detailed section
        if "last_pred" in st.session_state and "last_input" in st.session_state:
            pred = st.session_state["last_pred"]
            st.markdown("---")
            st.subheader("Detailed analysis")

            c1, c2 = st.columns(2)
            with c1:
                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=float(pred["needed"]),
                        title={"text": "Infrastructure Need"},
                        gauge={"axis": {"range": [0, 100]}},
                    )
                )
                fig.update_layout(height=280)
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                if pred["needed"] >= 70:
                    st.warning("🚨 High priority investment recommended")
                    st.write("- Focus: transport, utilities, electricity access")
                    st.write("- Improve human development outcomes")
                elif pred["needed"] >= 40:
                    st.info("⚠️ Moderate investment recommended")
                    st.write("- Sustainable upgrades + targeted infra")
                    st.write("- Urban planning improvements")
                else:
                    st.success("✅ Maintenance & optimization")
                    st.write("- Maintain infrastructure + smart upgrades")
                    st.write("- Efficiency & resilience focus")

            # Batch scoring + download if CSV uploaded
            last_upload_df = st.session_state.get("last_upload_df")
            if last_upload_df is not None:
                st.markdown("### Batch scoring (CSV)")
                scored = last_upload_df.copy()

                needed_list = []
                risk_list = []
                conf_list = []
                used_list = []

                for _, rr in scored.iterrows():
                    row_dict = {f: float(rr[f]) for f in FEATURES}
                    p = score_dict(row_dict, model_kind, model_obj)
                    needed_list.append(p["needed"])
                    risk_list.append(p["risk_level"])
                    conf_list.append(p["confidence"])
                    used_list.append(p["model_used"])

                scored["InfraNeed_%"] = np.round(needed_list, 2)
                scored["Risk_Level"] = risk_list
                scored["Confidence_%"] = np.round(conf_list, 2)
                scored["Model_Used"] = used_list

                out_bytes = scored.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "⬇️ Download scored CSV",
                    data=out_bytes,
                    file_name="infra_scored.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

    # ----------------------------
    # Model Status
    # ----------------------------
    with tab3:
        st.subheader("Model status")

        if model_kind in ["onnx", "joblib"]:
            st.success(f"Model loaded ✅ ({model_kind})")
        else:
            st.warning("No model found. App is running with heuristic fallback.")

        st.markdown("#### Expected files")
        st.code(
            "\n".join(
                [
                    f"- {MODEL_ONNX_PATH}  (recommended for Streamlit Cloud)",
                    f"- {MODEL_JOBLIB_PATH} (optional fallback; can fail if numpy mismatch)",
                ]
            )
        )

        if show_debug:
            st.markdown("#### Debug")
            st.write(meta)

        st.caption("Tip: ONNX is the safest format for Streamlit Cloud (avoids numpy pickling issues).")

    st.markdown("---")
    st.caption("🌍 Global Infrastructure AI | Streamlit Cloud safe | No paid APIs")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("App crashed unexpectedly. Enable debug in sidebar and check logs.")
        st.exception(e)
