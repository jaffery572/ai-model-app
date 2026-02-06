# app.py - Streamlit Cloud safe (FREE ONLY) version
# Features: ONNX inference + fallback heuristic, Chat UI, Map + Overpass infra signals, CSV upload validation
# Author: You

import os
import json
import time
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Optional ONNX runtime (recommended for Streamlit Cloud)
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except Exception:
    ort = None
    ONNX_AVAILABLE = False

# Optional folium map + click handler
try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except Exception:
    folium = None
    st_folium = None
    FOLIUM_AVAILABLE = False


# -----------------------------
# CONFIG
# -----------------------------
APP_TITLE = "🌍 Global Infrastructure AI (FREE)"
MODEL_DIR = "models"
ONNX_MODEL_PATH = os.path.join(MODEL_DIR, "infra_model.onnx")  # <- rename your onnx to this
JOBLIB_MODEL_PATH = os.path.join(MODEL_DIR, "infra_model.joblib")  # optional (not recommended on Cloud)

FEATURES = ["Population_Millions", "GDP_per_capita_USD", "HDI_Index", "Urbanization_Rate"]
DEFAULT_TIMEOUT = 12  # seconds for network calls
RANDOM_SEED = 42

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Global CSS
st.markdown(
    """
<style>
    .title {
        font-size: 2.4rem;
        font-weight: 900;
        letter-spacing: 0.5px;
        margin-bottom: 0.1rem;
    }
    .subtitle {
        color: #9aa4b2;
        margin-top: 0;
        margin-bottom: 1.2rem;
    }
    .card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .badge {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        font-weight: 800;
        font-size: 0.85rem;
    }
    .high { background: #b91c1c; color: white; }
    .med { background: #f59e0b; color: #111827; }
    .low { background: #16a34a; color: white; }
    .muted {
        color: #9aa4b2;
        font-size: 0.9rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------
# HELPERS
# -----------------------------
def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def risk_label(need_pct: float) -> Tuple[str, str]:
    if need_pct >= 70:
        return "HIGH RISK", "high"
    elif need_pct >= 40:
        return "MEDIUM RISK", "med"
    return "LOW RISK", "low"


def safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def validate_row(row: Dict[str, Any]) -> Tuple[bool, str]:
    for k in FEATURES:
        if k not in row:
            return False, f"Missing column: {k}"
        if row[k] is None or (isinstance(row[k], str) and not row[k].strip()):
            return False, f"Empty value for: {k}"
        try:
            float(row[k])
        except Exception:
            return False, f"Non-numeric value in: {k}"
    # logical bounds (soft)
    if not (0.3 <= float(row["HDI_Index"]) <= 1.0):
        return False, "HDI_Index must be between 0.3 and 1.0"
    if not (0 <= float(row["Urbanization_Rate"]) <= 100):
        return False, "Urbanization_Rate must be between 0 and 100"
    if float(row["Population_Millions"]) <= 0:
        return False, "Population_Millions must be > 0"
    if float(row["GDP_per_capita_USD"]) <= 0:
        return False, "GDP_per_capita_USD must be > 0"
    return True, "OK"


def heuristic_predict(features: Dict[str, float]) -> Dict[str, Any]:
    """
    Stable free fallback if model not available.
    Outputs need% + confidence estimate.
    """
    pop = safe_float(features["Population_Millions"], 50.0)
    gdp = safe_float(features["GDP_per_capita_USD"], 5000.0)
    hdi = safe_float(features["HDI_Index"], 0.7)
    urb = safe_float(features["Urbanization_Rate"], 50.0)

    # A simple interpretable scoring
    gdp_term = (1 - clamp(gdp / 60000.0, 0, 1)) * 45
    hdi_term = (1 - clamp(hdi, 0, 1)) * 35
    urb_term = (clamp(1 - (urb / 100.0), 0, 1) * 10)
    pop_term = (10 if (pop > 100 and gdp < 8000) else 0)

    need = clamp(gdp_term + hdi_term + urb_term + pop_term, 0, 100)

    # confidence rough
    confidence = clamp(60 + (abs(50 - need) / 2), 55, 90)

    label, css = risk_label(need)
    return {
        "need_pct": need,
        "sufficient_pct": 100 - need,
        "confidence": confidence,
        "risk_label": label,
        "risk_css": css,
        "model_used": "heuristic",
    }


@st.cache_resource(show_spinner=False)
def load_onnx_model() -> Optional["ort.InferenceSession"]:
    if not ONNX_AVAILABLE:
        return None
    if not os.path.exists(ONNX_MODEL_PATH):
        return None
    try:
        # CPU only provider for Streamlit Cloud
        sess = ort.InferenceSession(ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])
        return sess
    except Exception:
        return None


def onnx_predict(sess: "ort.InferenceSession", features: Dict[str, float]) -> Dict[str, Any]:
    """
    Assumes model outputs either probability or score.
    We'll map to 0..100 need%.
    """
    x = np.array([[safe_float(features[f]) for f in FEATURES]], dtype=np.float32)
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: x})

    # Try to interpret output shape robustly
    y = outputs[0]
    val = None
    try:
        arr = np.array(y).reshape(-1)
        # common: [p0, p1] or [p1]
        if arr.size >= 2:
            val = float(arr[-1])
        else:
            val = float(arr[0])
    except Exception:
        val = 0.5

    # If looks like logit or score, squash
    if val < 0 or val > 1:
        val = 1 / (1 + np.exp(-val))

    need = clamp(val * 100.0, 0, 100)
    confidence = clamp(65 + (abs(50 - need) / 2), 55, 92)

    label, css = risk_label(need)
    return {
        "need_pct": need,
        "sufficient_pct": 100 - need,
        "confidence": confidence,
        "risk_label": label,
        "risk_css": css,
        "model_used": "onnx",
    }


def predict(features: Dict[str, float]) -> Dict[str, Any]:
    sess = load_onnx_model()
    if sess is not None:
        try:
            return onnx_predict(sess, features)
        except Exception:
            return heuristic_predict(features)
    return heuristic_predict(features)


@st.cache_data(show_spinner=False)
def default_country_df() -> pd.DataFrame:
    """
    Small fallback dataset (safe). Not meant to be 'real world full dataset'—just a UI helper.
    """
    data = {
        "Country": ["USA", "CHN", "IND", "DEU", "GBR", "JPN", "BRA", "RUS", "FRA", "PAK", "BGD", "NGA", "EGY", "VNM", "ZAF"],
        "Population_Millions": [331, 1412, 1408, 83, 68, 125, 215, 144, 67, 240, 170, 216, 109, 98, 60],
        "GDP_per_capita_USD": [63500, 12500, 2300, 45700, 42200, 40100, 8900, 11200, 40400, 1500, 2600, 2300, 3900, 2800, 6300],
        "HDI_Index": [0.926, 0.761, 0.645, 0.947, 0.932, 0.925, 0.765, 0.824, 0.901, 0.557, 0.632, 0.539, 0.707, 0.704, 0.709],
        "Urbanization_Rate": [83, 64, 35, 77, 84, 92, 87, 75, 81, 37, 39, 52, 43, 38, 68],
        "Continent": ["NA", "Asia", "Asia", "EU", "EU", "Asia", "SA", "EU", "EU", "Asia", "Asia", "Africa", "Africa", "Asia", "Africa"],
    }
    return pd.DataFrame(data)


def meter_bar(need_pct: float) -> None:
    """
    Stable meter that always shows (no plotly gauge issues).
    """
    st.markdown("#### Need meter (stable)")
    st.progress(int(clamp(need_pct, 0, 100)))
    st.caption("0% = low need, 100% = very high need")


def recommendations(need_pct: float, infra_signals: Optional[Dict[str, Any]] = None) -> Tuple[str, List[str]]:
    """
    Returns title + bullet points. Uses infra_signals if available.
    """
    bullets = []
    if need_pct >= 70:
        title = "Maintenance & optimization is NOT enough — heavy investment required"
        bullets += [
            "Prioritize electricity access, water/sanitation, and primary transport corridors",
            "Create a 2–5 year national infrastructure roadmap + financing plan",
            "Target high-impact projects first (grid, roads, basic health coverage)",
        ]
    elif need_pct >= 40:
        title = "Targeted upgrades + resilience improvements recommended"
        bullets += [
            "Upgrade key networks (roads, power reliability, basic public services)",
            "Prioritize high-growth urban zones and logistics nodes",
            "Implement monitoring + maintenance cycles to avoid future collapse",
        ]
    else:
        title = "Maintain infrastructure + smart upgrades"
        bullets += [
            "Focus on optimization (smart operations, preventive maintenance)",
            "Invest in resilience (flood/heat readiness) and data-driven planning",
            "Scale best practices across cities/regions",
        ]

    if infra_signals:
        bullets.append(f"OSM signals (nearby): roads={infra_signals.get('roads',0)}, hospitals={infra_signals.get('hospitals',0)}, schools={infra_signals.get('schools',0)}, power={infra_signals.get('power',0)}")

    return title, bullets


# -----------------------------
# OVERPASS (FREE OSM SIGNALS)
# -----------------------------
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

@st.cache_data(show_spinner=False, ttl=3600)
def overpass_signals(lat: float, lon: float, radius_m: int = 5000) -> Dict[str, int]:
    """
    Counts some key infrastructure features within radius (meters).
    Free, but rate-limited. Cached for 1 hour.
    """
    lat = float(lat); lon = float(lon)
    r = int(radius_m)

    query = f"""
    [out:json][timeout:25];
    (
      way(around:{r},{lat},{lon})["highway"];
      node(around:{r},{lat},{lon})["amenity"="hospital"];
      way(around:{r},{lat},{lon})["amenity"="hospital"];
      node(around:{r},{lat},{lon})["amenity"="school"];
      way(around:{r},{lat},{lon})["amenity"="school"];
      node(around:{r},{lat},{lon})["power"];
      way(around:{r},{lat},{lon})["power"];
    );
    out tags;
    """

    try:
        resp = requests.post(OVERPASS_URL, data=query.encode("utf-8"), timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        elements = data.get("elements", [])

        roads = 0
        hospitals = 0
        schools = 0
        power = 0

        for el in elements:
            tags = el.get("tags", {})
            if "highway" in tags:
                roads += 1
            if tags.get("amenity") == "hospital":
                hospitals += 1
            if tags.get("amenity") == "school":
                schools += 1
            if "power" in tags:
                power += 1

        return {"roads": roads, "hospitals": hospitals, "schools": schools, "power": power}
    except Exception:
        # Graceful failure
        return {"roads": 0, "hospitals": 0, "schools": 0, "power": 0}


# -----------------------------
# CHAT (FREE, TEMPLATE-BASED)
# -----------------------------
def chat_answer(user_msg: str,
                last_features: Optional[Dict[str, float]],
                last_pred: Optional[Dict[str, Any]],
                last_osm: Optional[Dict[str, int]]) -> str:
    """
    Free "GPT-style" assistant:
    - Intent detect
    - Uses your model output + OSM signals + templates
    """
    msg = (user_msg or "").strip().lower()

    if not msg:
        return "Type something like: 'predict', 'explain', 'what does roads mean?', 'how to upload csv?'"

    # Help intents
    if "csv" in msg and ("upload" in msg or "format" in msg or "columns" in msg):
        return (
            "CSV upload ka kaam: aap apna dataset import kar ke bulk prediction kar sakte ho.\n\n"
            "Required columns:\n"
            "- Population_Millions\n- GDP_per_capita_USD\n- HDI_Index\n- Urbanization_Rate\n\n"
            "Example row:\n"
            "100, 5000, 0.70, 50\n"
        )

    if "map" in msg or "osm" in msg or "overpass" in msg:
        return (
            "Map tab me aap kisi location pe click karte ho to main OSM (OpenStreetMap) se free infra signals count karta hoon "
            "(roads/hospitals/schools/power). Ye paid API nahi hai, but rate-limited hota hai — isliye caching on hai."
        )

    if "explain" in msg or "why" in msg or "reason" in msg:
        if not last_pred or not last_features:
            return "Abhi koi prediction run nahi hua. Predictor tab se predict run karo, phir main explain kar dunga."
        title, bullets = recommendations(last_pred["need_pct"], last_osm)
        expl = [
            f"Explaination (model={last_pred['model_used']}):",
            f"- Need score: {last_pred['need_pct']:.1f}% ({last_pred['risk_label']})",
            f"- Key drivers (simple view): GDP low + HDI low => need high; Urbanization + Population pressure can increase need."
        ]
        if last_osm:
            expl.append(f"- OSM signals nearby: {json.dumps(last_osm)}")
        expl.append("\nRecommendations:")
        expl += [f"- {b}" for b in bullets]
        return "\n".join(expl)

    if "predict" in msg or "run" in msg or "score" in msg:
        if not last_pred:
            return "Predictor tab me jaa ke input do, phir 'PREDICT' dabao. Main yahan result summarize kar dunga."
        return (
            f"Result: Need={last_pred['need_pct']:.1f}% | {last_pred['risk_label']} | "
            f"Confidence={last_pred['confidence']:.1f}% | Model={last_pred['model_used']}\n"
            f"Agar aap 'explain' likho to main reasons + recommendations detail me de dunga."
        )

    # Default
    return (
        "Main aapki madad kar sakta hoon:\n"
        "1) 'predict' — last result summarize\n"
        "2) 'explain' — reasons + recommendations\n"
        "3) 'csv format' — CSV upload guide\n"
        "4) 'map' — OSM signals explanation\n"
    )


# -----------------------------
# UI
# -----------------------------
def header():
    st.markdown(f"<div class='title'>{APP_TITLE}</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtitle'>World-style indicators + ML model. FREE ONLY: ONNX + OSM/Overpass signals + template chat.</div>",
        unsafe_allow_html=True,
    )


def sidebar(debug: bool) -> Dict[str, Any]:
    st.sidebar.header("Settings")
    st.sidebar.caption("Model load safe. If it fails, heuristic fallback runs.")

    show_debug = st.sidebar.toggle("Show debug info", value=debug)
    radius_km = st.sidebar.slider("Map signal radius (km)", 1, 20, 5)
    return {"show_debug": show_debug, "radius_m": int(radius_km * 1000)}


def model_status_panel(show_debug: bool):
    st.subheader("Model Status")
    onnx_exists = os.path.exists(ONNX_MODEL_PATH)
    joblib_exists = os.path.exists(JOBLIB_MODEL_PATH)

    if onnx_exists and ONNX_AVAILABLE and load_onnx_model() is not None:
        st.success("Model loaded ✅ (onnx)")
        st.caption("Tip: ONNX is the safest approach on Streamlit Cloud.")
    elif onnx_exists and not ONNX_AVAILABLE:
        st.warning("ONNX model exists but onnxruntime not installed. Add onnxruntime to requirements.")
    elif onnx_exists:
        st.warning("ONNX model exists but failed to load. App will use heuristic fallback.")
    else:
        st.error(f"ONNX model missing: upload `{ONNX_MODEL_PATH}`")

    if joblib_exists:
        st.info("joblib model found (optional). On Cloud, joblib can fail due to numpy mismatch—use ONNX.")
    else:
        st.caption("joblib model not found (optional)")

    st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Streamlit Cloud safe")

    if show_debug:
        st.markdown("**Debug**")
        st.write({"ONNX_AVAILABLE": ONNX_AVAILABLE, "FOLIUM_AVAILABLE": FOLIUM_AVAILABLE})
        st.write({"ONNX_MODEL_PATH": ONNX_MODEL_PATH, "exists": onnx_exists})
        st.write({"FEATURES": FEATURES})


def dashboard_tab(df: pd.DataFrame):
    st.subheader("Dashboard")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Countries (fallback)", len(df))
    with c2:
        st.metric("Avg GDP/capita", f"${df['GDP_per_capita_USD'].mean():,.0f}")
    with c3:
        st.metric("Avg HDI", f"{df['HDI_Index'].mean():.3f}")
    with c4:
        st.metric("Model", "ONNX + fallback")

    col1, col2 = st.columns(2)

    with col1:
        # Discrete color for continent (stable)
        fig = px.scatter(
            df,
            x="GDP_per_capita_USD",
            y="HDI_Index",
            size="Population_Millions",
            color="Continent",
            hover_name="Country",
            log_x=True,
            title="GDP vs HDI (fallback sample)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        top = df.sort_values("GDP_per_capita_USD", ascending=False).head(10).copy()
        fig2 = px.bar(
            top,
            x="Country",
            y="GDP_per_capita_USD",
            color="Continent",
            title="Top 10 GDP/capita (fallback sample)",
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Country table (fallback sample)")
    st.dataframe(df, use_container_width=True)


def predictor_tab(df: pd.DataFrame, radius_m: int, show_debug: bool):
    st.subheader("Infrastructure Predictor")

    # Input method
    method = st.radio("Input method", ["Select country (fallback)", "Custom input", "Upload CSV"], horizontal=True)

    features = None
    selected_country = None
    csv_df = None

    if method == "Select country (fallback)":
        selected_country = st.selectbox("Country", df["Country"].tolist())
        row = df[df["Country"] == selected_country].iloc[0].to_dict()
        features = {k: float(row[k]) for k in FEATURES}

    elif method == "Custom input":
        colA, colB = st.columns(2)
        with colA:
            pop = st.number_input("Population (Millions)", min_value=0.1, max_value=2000.0, value=100.0, step=1.0)
            gdp = st.number_input("GDP per Capita (USD)", min_value=100.0, max_value=200000.0, value=5000.0, step=100.0)
        with colB:
            hdi = st.slider("HDI Index", 0.3, 1.0, 0.7)
            urb = st.slider("Urbanization Rate (%)", 0.0, 100.0, 50.0)
        features = {"Population_Millions": pop, "GDP_per_capita_USD": gdp, "HDI_Index": hdi, "Urbanization_Rate": urb}

    else:  # Upload CSV
        st.caption("CSV upload ka kaam: aap bulk predictions run kar sakte ho.")
        st.code("Required columns: Population_Millions, GDP_per_capita_USD, HDI_Index, Urbanization_Rate", language="text")
        up = st.file_uploader("Upload CSV", type=["csv"])
        if up is not None:
            try:
                csv_df = pd.read_csv(up)
                st.write("Preview:", csv_df.head())

                missing = [c for c in FEATURES if c not in csv_df.columns]
                if missing:
                    st.error(f"Missing columns: {missing}")
                else:
                    st.success("CSV schema OK ✅")
            except Exception as e:
                st.error(f"CSV read failed: {e}")

    # Map + Overpass signals (optional)
    st.markdown("---")
    st.markdown("### Optional: Map signals (FREE OSM)")
    st.caption("Map pe click karo, main nearby roads/hospitals/schools/power count kar dunga (Overpass API).")

    latlon = None
    osm = None

    if FOLIUM_AVAILABLE:
        colM1, colM2 = st.columns([1.2, 1])

        with colM1:
            m = folium.Map(location=[20, 0], zoom_start=2, tiles="OpenStreetMap")
            out = st_folium(m, height=420, width=None)
            if out and out.get("last_clicked"):
                latlon = out["last_clicked"]
                st.info(f"Selected location: lat={latlon['lat']:.5f}, lon={latlon['lng']:.5f}")

        with colM2:
            if latlon:
                with st.spinner("Fetching OSM signals..."):
                    osm = overpass_signals(latlon["lat"], latlon["lng"], radius_m=radius_m)
                st.write("OSM infra signals:", osm)
            else:
                st.caption("Click on map to get signals.")
    else:
        st.warning("Map libraries not installed. Add `folium` + `streamlit-folium` in requirements.txt.")

    # Predict
    st.markdown("---")
    c1, c2 = st.columns([1.2, 1])

    with c2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Run prediction")
        run = st.button("🚀 PREDICT", use_container_width=True, type="primary")
        st.markdown("</div>", unsafe_allow_html=True)

    with c1:
        if run:
            # CSV bulk prediction
            if csv_df is not None and all(c in csv_df.columns for c in FEATURES):
                results = []
                for i, r in csv_df.iterrows():
                    row_dict = {k: r.get(k) for k in FEATURES}
                    ok, msg = validate_row(row_dict)
                    if not ok:
                        results.append({"row": i, "error": msg})
                        continue
                    feat = {k: float(row_dict[k]) for k in FEATURES}
                    pred = predict(feat)
                    results.append({"row": i, **feat, **pred})

                res_df = pd.DataFrame(results)
                st.markdown("### Bulk results")
                st.dataframe(res_df, use_container_width=True)
                st.download_button("Download results CSV", res_df.to_csv(index=False).encode("utf-8"), "predictions.csv", "text/csv")
                st.session_state.last_pred = None
                st.session_state.last_features = None
                st.session_state.last_osm = osm
            else:
                # Single prediction
                if not features:
                    st.error("No input provided.")
                else:
                    ok, msg = validate_row(features)
                    if not ok:
                        st.error(msg)
                    else:
                        pred = predict(features)

                        # store for chat
                        st.session_state.last_pred = pred
                        st.session_state.last_features = features
                        st.session_state.last_osm = osm

                        badge = f"<span class='badge {pred['risk_css']}'>{pred['risk_label']}</span>"
                        st.markdown(f"## Result {badge}", unsafe_allow_html=True)

                        # Stable meter
                        meter_bar(pred["need_pct"])

                        st.markdown(
                            f"### Infrastructure Need\n"
                            f"**{pred['need_pct']:.1f}%**  \n"
                            f"<span class='muted'>Confidence: {pred['confidence']:.1f}% | Model used: {pred['model_used']}</span>",
                            unsafe_allow_html=True,
                        )

                        # Need vs sufficient chart (simple)
                        fig = px.bar(
                            pd.DataFrame(
                                {"Metric": ["Need", "Sufficient"], "Value": [pred["need_pct"], pred["sufficient_pct"]]}
                            ),
                            x="Metric",
                            y="Value",
                            title="Need vs Sufficient",
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Recommendations
                        st.markdown("### Recommendations")
                        title, bullets = recommendations(pred["need_pct"], osm)
                        st.success(title)
                        for b in bullets:
                            st.write(f"- {b}")

                        if show_debug:
                            st.write("Features:", features)
                            st.write("OSM signals:", osm)


def chat_tab():
    st.subheader("Chat (FREE assistant)")
    st.caption("Ye GPT nahi hai — but free mode me templates + model output + OSM signals se smart chat responses deta hai.")

    if "chat" not in st.session_state:
        st.session_state.chat = []
    if "last_pred" not in st.session_state:
        st.session_state.last_pred = None
    if "last_features" not in st.session_state:
        st.session_state.last_features = None
    if "last_osm" not in st.session_state:
        st.session_state.last_osm = None

    # show history
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    user_msg = st.chat_input("Type: predict / explain / csv format / map")
    if user_msg:
        st.session_state.chat.append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.write(user_msg)

        reply = chat_answer(user_msg, st.session_state.last_features, st.session_state.last_pred, st.session_state.last_osm)
        st.session_state.chat.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.write(reply)

    st.markdown("---")
    colA, colB = st.columns(2)
    with colA:
        if st.button("🧹 Clear chat"):
            st.session_state.chat = []
            st.rerun()
    with colB:
        st.caption("Tip: Predictor me run karke phir yahan 'explain' likho.")


def main():
    header()

    debug_default = False
    cfg = sidebar(debug_default)

    df = default_country_df()

    tabs = st.tabs(["📊 Dashboard", "🧠 Predictor", "🗺️ Model Status", "💬 Chat"])
    with tabs[0]:
        dashboard_tab(df)
    with tabs[1]:
        predictor_tab(df, radius_m=cfg["radius_m"], show_debug=cfg["show_debug"])
    with tabs[2]:
        model_status_panel(cfg["show_debug"])
    with tabs[3]:
        chat_tab()

    st.markdown("---")
    st.caption("🟢 Global Infrastructure AI | FREE ONLY | No paid APIs | OSM (Overpass) + ONNX model (if available)")


if __name__ == "__main__":
    main()
