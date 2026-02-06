# app.py - Global Infrastructure AI (Streamlit Cloud robust)
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# Optional imports (safe fallback)
try:
    import joblib
except Exception:
    joblib = None

try:
    import onnxruntime as ort
except Exception:
    ort = None


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="🌍 Global Infrastructure AI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Constants
# -----------------------------
APP_UPDATED = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
SEED = 42

# Expected features for inference (must match training)
FEATURES = [
    "Population_Millions",
    "GDP_per_capita_USD",
    "HDI_Index",
    "Urbanization_Rate",
]

# Repo paths
MODEL_DIR = "models"
ONNX_PATH = os.path.join(MODEL_DIR, "infra_model.onnx")
JOBLIB_PATH = os.path.join(MODEL_DIR, "infra_model.joblib")

# CSV upload schema
UPLOAD_REQUIRED_COLS = FEATURES[:]  # same as model input

# Set global random seed (reproducibility)
np.random.seed(SEED)


# -----------------------------
# CSS (clean, stable)
# -----------------------------
st.markdown(
    """
<style>
.main-title {
  font-size: 2.2rem;
  font-weight: 800;
  letter-spacing: 0.5px;
}
.subtitle {
  color: #9aa0a6;
  margin-top: -8px;
}
.badge {
  display: inline-block;
  padding: 6px 12px;
  border-radius: 999px;
  font-weight: 700;
  font-size: 0.9rem;
}
.badge-low { background: #1f7a3a; color: white; }
.badge-med { background: #b56b00; color: white; }
.badge-high { background: #a31111; color: white; }
.card {
  padding: 14px 16px;
  border-radius: 14px;
  border: 1px solid rgba(255,255,255,0.08);
  background: rgba(255,255,255,0.03);
}
.small-muted { color: #9aa0a6; font-size: 0.9rem; }
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------
# Utility: safe debug
# -----------------------------
def show_debug(msg: str):
    if st.session_state.get("debug", False):
        st.sidebar.code(msg)


# -----------------------------
# Default fallback dataset (small, safe)
# -----------------------------
@st.cache_data(show_spinner=False)
def load_default_country_data() -> pd.DataFrame:
    data = {
        "Country": [
            "USA", "CHN", "IND", "DEU", "GBR", "JPN", "BRA", "RUS", "FRA", "ITA",
            "CAN", "AUS", "KOR", "MEX", "IDN", "TUR", "SAU", "CHE", "NLD", "ESP",
            "PAK", "BGD", "NGA", "EGY", "VNM", "THA", "ZAF", "ARG", "COL", "MYS",
        ],
        "Population_Millions": [
            331, 1412, 1408, 83, 68, 125, 215, 144, 67, 59,
            38, 26, 51, 129, 278, 85, 36, 9, 17, 47,
            240, 170, 216, 109, 98, 70, 60, 45, 52, 33
        ],
        "GDP_per_capita_USD": [
            63500, 12500, 2300, 45700, 42200, 40100, 8900, 11200, 40400, 32000,
            43200, 52000, 35000, 9900, 4300, 9500, 23500, 81900, 52400, 29400,
            1500, 2600, 2300, 3900, 2800, 7800, 6300, 10600, 6400, 11400
        ],
        "HDI_Index": [
            0.926, 0.761, 0.645, 0.947, 0.932, 0.925, 0.765, 0.824, 0.901, 0.892,
            0.929, 0.944, 0.916, 0.779, 0.718, 0.820, 0.857, 0.955, 0.944, 0.904,
            0.557, 0.632, 0.539, 0.707, 0.704, 0.777, 0.709, 0.845, 0.767, 0.803
        ],
        "Urbanization_Rate": [
            83, 64, 35, 77, 84, 92, 87, 75, 81, 71,
            81, 86, 81, 81, 57, 76, 84, 74, 93, 81,
            37, 39, 52, 43, 38, 51, 68, 92, 81, 78
        ],
        "Continent": [
            "North America", "Asia", "Asia", "Europe", "Europe", "Asia", "South America",
            "Europe", "Europe", "Europe", "North America", "Oceania", "Asia", "North America",
            "Asia", "Asia", "Asia", "Europe", "Europe", "Europe", "Asia", "Asia",
            "Africa", "Africa", "Asia", "Asia", "Africa", "South America", "South America", "Asia"
        ],
    }
    df = pd.DataFrame(data)
    return df


# -----------------------------
# Model loaders (Cloud-safe)
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_onnx_session(path: str):
    """Prefer ONNX on Streamlit Cloud to avoid joblib/numpy mismatch issues."""
    if ort is None:
        return None
    if not os.path.exists(path):
        return None
    try:
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        return sess
    except Exception as e:
        return {"_error": str(e)}


@st.cache_resource(show_spinner=False)
def load_joblib_bundle(path: str):
    """Joblib bundle optional fallback. Can fail if numpy/sklearn versions mismatch."""
    if joblib is None:
        return None
    if not os.path.exists(path):
        return None
    try:
        obj = joblib.load(path)
        # supports either plain model or dict bundle
        return obj
    except Exception as e:
        return {"_error": str(e)}


def resolve_model() -> Tuple[str, Optional[object], List[str], Optional[str]]:
    """
    Returns:
      (model_kind, model_obj, features, model_path)
    model_kind: "onnx" | "joblib" | "heuristic"
    """
    # 1) ONNX first
    onnx_sess = load_onnx_session(ONNX_PATH)
    if isinstance(onnx_sess, dict) and "_error" in onnx_sess:
        show_debug(f"ONNX load error: {onnx_sess['_error']}")
    elif onnx_sess is not None:
        return ("onnx", onnx_sess, FEATURES, ONNX_PATH)

    # 2) joblib fallback
    jb = load_joblib_bundle(JOBLIB_PATH)
    if isinstance(jb, dict) and "_error" in jb:
        show_debug(f"Joblib load error: {jb['_error']}")
    elif jb is not None:
        # If bundle saved as dict
        if isinstance(jb, dict) and "model" in jb and "features" in jb:
            return ("joblib", jb, jb["features"], JOBLIB_PATH)
        return ("joblib", jb, FEATURES, JOBLIB_PATH)

    # 3) heuristic fallback
    return ("heuristic", None, FEATURES, None)


# -----------------------------
# Input validation
# -----------------------------
def validate_input_dict(d: Dict) -> Tuple[bool, List[str]]:
    errs = []
    for col in FEATURES:
        if col not in d:
            errs.append(f"Missing: {col}")
            continue

        val = d[col]
        try:
            v = float(val)
        except Exception:
            errs.append(f"{col} must be numeric")
            continue

        # sanity ranges (soft)
        if col == "Population_Millions" and not (0.1 <= v <= 2000):
            errs.append("Population_Millions out of range (0.1–2000)")
        if col == "GDP_per_capita_USD" and not (100 <= v <= 200000):
            errs.append("GDP_per_capita_USD out of range (100–200000)")
        if col == "HDI_Index" and not (0.2 <= v <= 1.0):
            errs.append("HDI_Index out of range (0.2–1.0)")
        if col == "Urbanization_Rate" and not (0 <= v <= 100):
            errs.append("Urbanization_Rate out of range (0–100)")

    return (len(errs) == 0, errs)


def dict_to_features_row(d: Dict, feature_list: List[str]) -> np.ndarray:
    return np.array([[float(d[f]) for f in feature_list]], dtype=np.float32)


# -----------------------------
# Prediction (ONNX / joblib / heuristic)
# -----------------------------
def heuristic_predict(d: Dict) -> Dict:
    # Simple stable scoring
    gdp = float(d["GDP_per_capita_USD"])
    hdi = float(d["HDI_Index"])
    urb = float(d["Urbanization_Rate"])
    pop = float(d["Population_Millions"])

    score = (
        (1 - min(gdp / 50000, 1)) * 45
        + (1 - hdi) * 35
        + (1 - min(urb / 100, 1)) * 10
        + (1 if (pop > 100 and gdp < 5000) else 0) * 10
    )
    need = float(np.clip(score, 0, 100))
    conf = float(np.clip(60 + (abs(need - 50) / 2), 60, 85))  # heuristic confidence

    return format_prediction(need, conf, model_used="heuristic")


def format_prediction(need_percent: float, confidence: float, model_used: str) -> Dict:
    need_percent = float(np.clip(need_percent, 0, 100))
    sufficient = 100 - need_percent

    if need_percent >= 70:
        risk = "HIGH"
        badge = "badge-high"
    elif need_percent >= 40:
        risk = "MEDIUM"
        badge = "badge-med"
    else:
        risk = "LOW"
        badge = "badge-low"

    return {
        "needed": need_percent,
        "sufficient": sufficient,
        "confidence": float(np.clip(confidence, 0, 100)),
        "risk_level": risk,
        "risk_badge": badge,
        "model_used": model_used,
    }


def predict(d: Dict, model_kind: str, model_obj: object, feature_list: List[str]) -> Dict:
    if model_kind == "onnx" and model_obj is not None:
        try:
            x = dict_to_features_row(d, feature_list)
            input_name = model_obj.get_inputs()[0].name
            out = model_obj.run(None, {input_name: x})[0]
            # out shape could be (1,1) probability or (1,2)
            out = np.array(out).reshape(1, -1)
            if out.shape[1] == 1:
                prob = float(out[0, 0])
            else:
                # assume class1 prob in column 1
                prob = float(out[0, 1])
            need = prob * 100.0
            conf = 70 + (abs(need - 50) / 3)  # simple stable confidence
            return format_prediction(need, conf, model_used="onnx")
        except Exception as e:
            show_debug(f"ONNX predict error: {e}")
            return heuristic_predict(d)

    if model_kind == "joblib" and model_obj is not None:
        try:
            model = model_obj["model"] if isinstance(model_obj, dict) and "model" in model_obj else model_obj
            x = pd.DataFrame([{f: float(d[f]) for f in feature_list}])
            if hasattr(model, "predict_proba"):
                prob = float(model.predict_proba(x)[0, 1])
                need = prob * 100.0
            else:
                # fallback to decision/predict
                pred_cls = int(model.predict(x)[0])
                need = 80.0 if pred_cls == 1 else 20.0
            conf = 75 + (abs(need - 50) / 4)
            return format_prediction(need, conf, model_used="joblib")
        except Exception as e:
            show_debug(f"Joblib predict error: {e}")
            return heuristic_predict(d)

    return heuristic_predict(d)


# -----------------------------
# CSV Upload utilities
# -----------------------------
def validate_uploaded_csv(df: pd.DataFrame) -> Tuple[bool, str]:
    missing = [c for c in UPLOAD_REQUIRED_COLS if c not in df.columns]
    if missing:
        return False, f"Missing required columns: {missing}"

    # numeric checks
    for c in UPLOAD_REQUIRED_COLS:
        if not pd.api.types.is_numeric_dtype(df[c]):
            # try convert
            try:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            except Exception:
                return False, f"Column {c} must be numeric."

        if df[c].isna().all():
            return False, f"Column {c} has no numeric values."

    df2 = df[UPLOAD_REQUIRED_COLS].dropna()
    if len(df2) == 0:
        return False, "After removing invalid rows (NaNs), no rows remain."

    return True, "OK"


# -----------------------------
# UI blocks
# -----------------------------
def metric_cards(df: pd.DataFrame):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="card"><div class="small-muted">🌍 Countries</div>'
                    f'<div style="font-size:1.7rem;font-weight:800;">{len(df):,}</div></div>',
                    unsafe_allow_html=True)
    with c2:
        # We don’t rely on label column now; just show sample coverage
        st.markdown('<div class="card"><div class="small-muted">📦 Dataset</div>'
                    f'<div style="font-size:1.7rem;font-weight:800;">Default + Upload</div></div>',
                    unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="card"><div class="small-muted">💰 Avg GDP/cap</div>'
                    f'<div style="font-size:1.7rem;font-weight:800;">${df["GDP_per_capita_USD"].mean():,.0f}</div></div>',
                    unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="card"><div class="small-muted">🧠 Model</div>'
                    f'<div style="font-size:1.7rem;font-weight:800;">{st.session_state.get("model_kind","-")}</div></div>',
                    unsafe_allow_html=True)


def dashboard_tab(df: pd.DataFrame):
    st.markdown("## Dashboard")

    metric_cards(df)

    c1, c2 = st.columns(2)
    with c1:
        top = df.sort_values("GDP_per_capita_USD", ascending=False).head(12)
        fig1 = px.bar(
            top,
            x="Country",
            y="GDP_per_capita_USD",
            title="Top Countries by GDP per Capita",
            text_auto=True,
        )
        fig1.update_layout(height=420)
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        fig2 = px.scatter(
            df,
            x="GDP_per_capita_USD",
            y="HDI_Index",
            size="Population_Millions",
            color="Continent",
            hover_name="Country",
            title="GDP vs HDI (bubble = population)",
            log_x=True,
        )
        fig2.update_layout(height=420)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Country data (default)")
    st.dataframe(df, use_container_width=True, height=360)


def predictor_tab(df: pd.DataFrame, model_kind: str, model_obj: object, feature_list: List[str]):
    st.markdown("## Infrastructure Predictor")

    input_mode = st.radio(
        "Input method",
        ["Select country (fallback)", "Custom input", "Upload CSV"],
        horizontal=True,
    )

    input_dict = None

    if input_mode == "Select country (fallback)":
        country = st.selectbox("Choose Country (3-letter)", df["Country"].tolist(), index=0)
        row = df[df["Country"] == country].iloc[0]
        input_dict = {f: float(row[f]) for f in FEATURES}

    elif input_mode == "Custom input":
        c1, c2 = st.columns(2)
        with c1:
            pop = st.number_input("Population (Millions)", min_value=0.1, max_value=2000.0, value=100.0, step=1.0)
            gdp = st.number_input("GDP per Capita (USD)", min_value=100.0, max_value=200000.0, value=5000.0, step=100.0)
        with c2:
            hdi = st.slider("HDI Index", min_value=0.2, max_value=1.0, value=0.7, step=0.01)
            urb = st.slider("Urbanization Rate (%)", min_value=0.0, max_value=100.0, value=50.0, step=1.0)

        input_dict = {
            "Population_Millions": pop,
            "GDP_per_capita_USD": gdp,
            "HDI_Index": hdi,
            "Urbanization_Rate": urb,
        }

    else:
        st.info("Upload CSV with columns: " + ", ".join(UPLOAD_REQUIRED_COLS))
        f = st.file_uploader("Upload CSV", type=["csv"])
        if f is not None:
            try:
                up = pd.read_csv(f)
                ok, msg = validate_uploaded_csv(up)
                if not ok:
                    st.error(msg)
                else:
                    st.success("CSV validated ✅")
                    st.dataframe(up.head(20), use_container_width=True)
                    st.session_state["uploaded_df"] = up
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")

    cA, cB = st.columns([2, 1])

    with cB:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Run prediction")
        clicked = st.button("🚀 PREDICT", use_container_width=True, type="primary")
        st.markdown("</div>", unsafe_allow_html=True)

    # Single prediction
    if clicked and input_mode != "Upload CSV":
        if input_dict is None:
            st.error("No input found.")
            return
        ok, errs = validate_input_dict(input_dict)
        if not ok:
            st.error("Input errors:\n- " + "\n- ".join(errs))
            return

        pred = predict(input_dict, model_kind, model_obj, feature_list)

        st.session_state["last_pred"] = pred
        st.session_state["last_input"] = input_dict

    # Batch prediction from CSV
    if clicked and input_mode == "Upload CSV":
        up = st.session_state.get("uploaded_df")
        if up is None:
            st.error("Upload a CSV first.")
            return

        ok, msg = validate_uploaded_csv(up)
        if not ok:
            st.error(msg)
            return

        work = up[UPLOAD_REQUIRED_COLS].copy()
        work = work.apply(pd.to_numeric, errors="coerce").dropna()

        # Predict row by row (safe on cloud for modest CSV)
        results = []
        for _, r in work.iterrows():
            d = {c: float(r[c]) for c in UPLOAD_REQUIRED_COLS}
            results.append(predict(d, model_kind, model_obj, feature_list))

        out = work.copy()
        out["Need_%"] = [x["needed"] for x in results]
        out["Risk"] = [x["risk_level"] for x in results]
        out["Model"] = [x["model_used"] for x in results]
        st.session_state["batch_out"] = out

    # Show single result
    if "last_pred" in st.session_state and input_mode != "Upload CSV":
        pred = st.session_state["last_pred"]

        st.markdown("---")
        st.markdown("## Result")

        left, right = st.columns([1, 1])
        with right:
            st.markdown(
                f'<span class="badge {pred["risk_badge"]}">{pred["risk_level"]} RISK</span>',
                unsafe_allow_html=True,
            )
            st.metric("Infrastructure Need", f"{pred['needed']:.1f}%")
            st.caption(f"Confidence: {pred['confidence']:.1f}% | Model used: {pred.get('model_used','-')}")

        with left:
            st.markdown("### Need meter (stable)")
            st.progress(int(round(pred["needed"])))
            bar_df = pd.DataFrame(
                {"Metric": ["Need", "Sufficient"], "Value": [pred["needed"], pred["sufficient"]]}
            )
            fig = px.bar(bar_df, x="Metric", y="Value", title="Need vs Sufficient")
            fig.update_layout(height=320)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Recommendations")
        if pred["needed"] >= 70:
            st.warning("🚨 High priority investment recommended")
            st.write("- Focus: transport + utilities + electricity access")
            st.write("- Improve HDI outcomes + basic service delivery")
            st.write("- Consider resilience projects (flood, heat, grid reliability)")
        elif pred["needed"] >= 40:
            st.info("⚠️ Moderate investment recommended")
            st.write("- Sustainable upgrades + targeted infrastructure")
            st.write("- Urban planning + mobility + water/waste systems")
        else:
            st.success("✅ Maintenance & optimization")
            st.write("- Maintain infrastructure + smart upgrades")
            st.write("- Efficiency + resilience focus")

    # Show batch results
    if "batch_out" in st.session_state and input_mode == "Upload CSV":
        st.markdown("---")
        st.markdown("## Batch results")
        out = st.session_state["batch_out"]
        st.dataframe(out, use_container_width=True)

        fig = px.histogram(out, x="Need_%", nbins=20, title="Distribution of Need_% (uploaded CSV)")
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)

        csv_bytes = out.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download results CSV", data=csv_bytes, file_name="infra_predictions.csv", mime="text/csv")


def model_status_tab(model_kind: str, model_obj: object, feature_list: List[str], model_path: Optional[str]):
    st.markdown("## Model Status")

    if model_kind in ("onnx", "joblib"):
        st.success(f"Model loaded ✅ ({model_kind})")
    else:
        st.warning("Model not found — using heuristic fallback")

    st.write(f"**Model kind:** {model_kind}")
    st.write(f"**Model path:** {model_path if model_path else 'None'}")
    st.write("**Features:**")
    st.code(json.dumps(feature_list, indent=2))

    # Feature importance (only if sklearn model present)
    if model_kind == "joblib" and model_obj is not None:
        try:
            model = model_obj["model"] if isinstance(model_obj, dict) and "model" in model_obj else model_obj
            if hasattr(model, "feature_importances_"):
                imps = np.array(model.feature_importances_, dtype=float)
                imp_df = pd.DataFrame({"feature": feature_list, "importance": imps}).sort_values("importance", ascending=False)
                fig = px.bar(imp_df, x="feature", y="importance", title="Feature importance (sklearn)")
                fig.update_layout(height=360)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Feature importance not available for this model.")
        except Exception as e:
            st.error(f"Feature importance error: {e}")

    st.caption("Tip: Streamlit Cloud pe joblib kabhi kabhi numpy/sklearn mismatch se fail hota hai — ONNX safest hai.")


# -----------------------------
# Main
# -----------------------------
def main():
    # Sidebar settings
    st.sidebar.header("Settings")
    st.session_state["debug"] = st.sidebar.toggle("Show debug info", value=False)

    st.sidebar.markdown("---")
    st.sidebar.caption("Model load is safe. If it fails, app still runs with heuristic predictions.")

    # Resolve model
    model_kind, model_obj, feature_list, model_path = resolve_model()
    st.session_state["model_kind"] = model_kind

    # Header
    st.markdown('<div class="main-title">🌍 Global Infrastructure AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">World-style indicators + ML model. Streamlit Cloud compatible (ONNX preferred + joblib fallback).</div>', unsafe_allow_html=True)

    # Load default data
    df = load_default_country_data()

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🧭 Predictor", "🧠 Model Status"])

    with tab1:
        dashboard_tab(df)

    with tab2:
        predictor_tab(df, model_kind, model_obj, feature_list)

    with tab3:
        model_status_tab(model_kind, model_obj, feature_list, model_path)

    st.markdown("---")
    st.caption(f"Updated: {APP_UPDATED} | Streamlit Cloud safe | No paid APIs")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("Fatal error occurred. Open Manage app → Logs for full details.")
        st.exception(e)
