import os
import time
import json
import math
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Map
import folium
from streamlit_folium import st_folium

# ONNX (Cloud-safe)
try:
    import onnxruntime as ort
except Exception:
    ort = None

# Optional joblib fallback (may fail if numpy mismatch; ONNX recommended)
try:
    import joblib
except Exception:
    joblib = None


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Global Infrastructure AI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Constants
# -----------------------------
FEATURES = ["Population_Millions", "GDP_per_capita_USD", "HDI_Index", "Urbanization_Rate"]

MODEL_ONNX_PATH = os.path.join("models", "infra_model.onnx")
MODEL_JOBLIB_PATH = os.path.join("models", "infra_model.joblib")

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# FREE news/disaster sources
GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
RELIEFWEB_URL = "https://api.reliefweb.int/v1/reports"

DEFAULT_RADIUS_M = 5000  # Overpass radius around click point


# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
<style>
.block-container { padding-top: 1.2rem; }
.small-muted { color: #9aa4b2; font-size: 0.9rem; }
.badge {
  display:inline-block; padding:6px 10px; border-radius:999px;
  font-weight:700; font-size: 0.85rem; letter-spacing: .3px;
}
.badge-low { background:#1f8f4a; color:white; }
.badge-med { background:#c68b00; color:black; }
.badge-high { background:#b42318; color:white; }
.card {
  border: 1px solid rgba(255,255,255,0.08);
  background: rgba(255,255,255,0.03);
  border-radius: 16px;
  padding: 14px 16px;
}
hr { border: none; border-top: 1px solid rgba(255,255,255,0.10); margin: 1rem 0; }
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------
# Utilities
# -----------------------------
def safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return default


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def risk_bucket(score_0_100: float) -> Tuple[str, str]:
    if score_0_100 >= 70:
        return "HIGH RISK", "badge-high"
    if score_0_100 >= 40:
        return "MEDIUM RISK", "badge-med"
    return "LOW RISK", "badge-low"


def ensure_schema(df: pd.DataFrame) -> Tuple[bool, str]:
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        return False, f"Missing columns: {missing}. Required: {FEATURES}"
    return True, "OK"


def to_feature_row(d: Dict[str, Any]) -> np.ndarray:
    row = [safe_float(d.get(k)) for k in FEATURES]
    return np.array(row, dtype=np.float32).reshape(1, -1)


def build_default_country_df() -> pd.DataFrame:
    data = {
        "Country": [
            "USA", "CHN", "IND", "DEU", "GBR", "JPN", "BRA", "RUS", "FRA", "ITA",
            "CAN", "AUS", "KOR", "MEX", "IDN", "TUR", "SAU", "CHE", "NLD", "ESP",
            "PAK", "BGD", "NGA", "EGY", "VNM", "THA", "ZAF", "ARG", "COL", "MYS",
        ],
        "Population_Millions": [
            331, 1412, 1408, 83, 68, 125, 215, 144, 67, 59,
            38, 26, 51, 129, 278, 85, 36, 8.6, 17, 47,
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
    }
    return pd.DataFrame(data)


# -----------------------------
# Model loading
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_onnx_session(path: str):
    if ort is None:
        return None
    if not os.path.exists(path):
        return None
    try:
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        return sess
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def load_joblib_model(path: str):
    if joblib is None:
        return None
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


def model_predict_proba(features_row: np.ndarray) -> Tuple[float, str]:
    """
    Returns (need_probability_0_100, model_kind).
    If model fails, falls back to heuristic.
    """
    # Try ONNX first (recommended for Cloud)
    sess = load_onnx_session(MODEL_ONNX_PATH)
    if sess is not None:
        try:
            input_name = sess.get_inputs()[0].name
            outputs = sess.run(None, {input_name: features_row.astype(np.float32)})
            out = outputs[0]

            # Support common output shapes:
            # (1,2) probs, or (1,) score/logit. We'll handle both.
            if isinstance(out, list):
                out = np.array(out)

            out = np.array(out)
            if out.ndim == 2 and out.shape[1] >= 2:
                prob_need = float(out[0, 1])
            else:
                # If it's a score, squash
                score = float(out.reshape(-1)[0])
                prob_need = 1 / (1 + math.exp(-score))

            return clamp(prob_need * 100.0, 0.0, 100.0), "onnx"
        except Exception:
            pass

    # Try joblib
    mdl = load_joblib_model(MODEL_JOBLIB_PATH)
    if mdl is not None:
        try:
            if hasattr(mdl, "predict_proba"):
                p = float(mdl.predict_proba(features_row)[0, 1])
                return clamp(p * 100.0, 0.0, 100.0), "joblib"
            pred = float(mdl.predict(features_row)[0])
            return clamp(pred * 100.0, 0.0, 100.0), "joblib"
        except Exception:
            pass

    # Heuristic fallback (always works)
    d = {
        "Population_Millions": float(features_row[0, 0]),
        "GDP_per_capita_USD": float(features_row[0, 1]),
        "HDI_Index": float(features_row[0, 2]),
        "Urbanization_Rate": float(features_row[0, 3]),
    }
    need = heuristic_need_score(d)
    return need, "heuristic"


def heuristic_need_score(country_data: Dict[str, float]) -> float:
    gdp = country_data["GDP_per_capita_USD"]
    hdi = country_data["HDI_Index"]
    urb = country_data["Urbanization_Rate"]
    pop = country_data["Population_Millions"]

    score = (
        (1 - min(gdp / 50000, 1)) * 42
        + (1 - hdi) * 33
        + (urb / 100 if gdp < 10000 else 0) * 18
        + (1 if pop > 100 and gdp < 5000 else 0) * 7
    )
    return clamp(score, 0, 100)


# -----------------------------
# Overpass: map signals
# -----------------------------
def overpass_query(lat: float, lon: float, radius_m: int) -> str:
    # Keep it small and safe. Count a few core infrastructure POIs.
    return f"""
    [out:json][timeout:25];
    (
      way(around:{radius_m},{lat},{lon})["highway"];
      node(around:{radius_m},{lat},{lon})["amenity"="hospital"];
      node(around:{radius_m},{lat},{lon})["amenity"="school"];
      node(around:{radius_m},{lat},{lon})["amenity"="clinic"];
      node(around:{radius_m},{lat},{lon})["power"];
      node(around:{radius_m},{lat},{lon})["man_made"="water_tower"];
      node(around:{radius_m},{lat},{lon})["emergency"];
    );
    out body;
    """


@st.cache_data(ttl=60 * 10, show_spinner=False)
def fetch_overpass_signals(lat: float, lon: float, radius_m: int) -> Dict[str, Any]:
    q = overpass_query(lat, lon, radius_m)
    r = requests.post(OVERPASS_URL, data=q.encode("utf-8"), headers={"Content-Type": "text/plain"})
    r.raise_for_status()
    data = r.json()

    elements = data.get("elements", [])
    signals = {
        "roads": 0,
        "hospitals": 0,
        "schools": 0,
        "clinics": 0,
        "power": 0,
        "water": 0,
        "emergency": 0,
    }

    for el in elements:
        tags = el.get("tags", {}) or {}
        if "highway" in tags:
            signals["roads"] += 1
        if tags.get("amenity") == "hospital":
            signals["hospitals"] += 1
        if tags.get("amenity") == "school":
            signals["schools"] += 1
        if tags.get("amenity") == "clinic":
            signals["clinics"] += 1
        if "power" in tags:
            signals["power"] += 1
        if tags.get("man_made") == "water_tower":
            signals["water"] += 1
        if "emergency" in tags:
            signals["emergency"] += 1

    signals["radius_m"] = radius_m
    return signals


def map_context_adjustment(signals: Dict[str, Any]) -> float:
    """
    Returns adjustment in range [-10, +15] points.
    Idea:
      - More infra density => slightly lower need
      - Very low infra density => raise need
    """
    roads = signals.get("roads", 0)
    key_pois = (
        signals.get("hospitals", 0)
        + signals.get("schools", 0)
        + signals.get("clinics", 0)
        + signals.get("power", 0)
        + signals.get("water", 0)
    )

    # Normalize (very rough)
    infra_density = (min(roads, 200) / 200.0) * 0.6 + (min(key_pois, 40) / 40.0) * 0.4
    # If density is low, add up to +15. If high, reduce up to -10.
    adj = (0.35 - infra_density) * 45.0
    return clamp(adj, -10.0, 15.0)


# -----------------------------
# News / disaster feeds (FREE)
# -----------------------------
@st.cache_data(ttl=60 * 15, show_spinner=False)
def fetch_gdelt(query: str, mode: str = "ArtList", max_records: int = 20) -> pd.DataFrame:
    params = {
        "query": query,
        "mode": mode,
        "format": "json",
        "maxrecords": max_records,
        "formatdatetime": "true",
        "sort": "HybridRel",
    }
    r = requests.get(GDELT_DOC_URL, params=params, timeout=20)
    r.raise_for_status()
    j = r.json()
    arts = (j.get("articles") or [])
    if not arts:
        return pd.DataFrame(columns=["title", "url", "sourceCountry", "seendate", "tone"])
    rows = []
    for a in arts:
        rows.append({
            "title": a.get("title"),
            "url": a.get("url"),
            "sourceCountry": a.get("sourceCountry"),
            "seendate": a.get("seendate"),
            "tone": a.get("tone"),
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=60 * 30, show_spinner=False)
def fetch_reliefweb(query: str, limit: int = 15) -> pd.DataFrame:
    payload = {
        "appname": "global-infra-ai",
        "query": {
            "value": query
        },
        "limit": limit,
        "profile": "full",
        "sort": ["date:desc"],
    }
    r = requests.post(RELIEFWEB_URL, json=payload, timeout=25)
    r.raise_for_status()
    j = r.json()
    data = j.get("data") or []
    rows = []
    for item in data:
        fields = item.get("fields") or {}
        rows.append({
            "title": fields.get("title"),
            "url": (fields.get("url") or ""),
            "date": fields.get("date", {}).get("created"),
            "status": fields.get("status"),
            "type": ", ".join([t.get("name") for t in (fields.get("type") or []) if isinstance(t, dict)]),
        })
    return pd.DataFrame(rows)


def disaster_overlay_score(gdelt_df: pd.DataFrame, relief_df: pd.DataFrame) -> float:
    """
    Returns 0..15 overlay points based on volume + negative tone.
    """
    score = 0.0
    if not gdelt_df.empty:
        score += min(len(gdelt_df), 20) * 0.3
        # tone: negative often < 0. We'll boost if very negative
        tones = pd.to_numeric(gdelt_df.get("tone", pd.Series([])), errors="coerce").dropna()
        if len(tones) > 0:
            avg_tone = float(tones.mean())
            if avg_tone < -2:
                score += 3.5
            elif avg_tone < -1:
                score += 2.0

    if not relief_df.empty:
        score += min(len(relief_df), 15) * 0.4

    return clamp(score, 0.0, 15.0)


def recommendations(need: float, overlay: float) -> List[str]:
    # need: 0..100, overlay: 0..15
    total = clamp(need + overlay, 0, 100)
    if total >= 70:
        return [
            "Prioritize critical infrastructure assessment (roads, power, water, healthcare).",
            "Create an emergency maintenance plan with response timelines and contractors.",
            "Strengthen flood drainage, slope stability, and backup power readiness.",
            "Run rapid vulnerability audits for bridges, hospitals, and substations.",
        ]
    if total >= 40:
        return [
            "Plan targeted upgrades in transport, utilities, and public services.",
            "Focus on resilience improvements (redundancy, maintenance cycles, inspections).",
            "Track local hazard alerts and prioritize at-risk assets.",
        ]
    return [
        "Maintain assets with preventive maintenance and routine inspections.",
        "Invest in smart monitoring (sensors, reporting, asset inventory).",
        "Run resilience checks for extreme weather preparedness.",
    ]


# -----------------------------
# UI
# -----------------------------
def render_header():
    st.title("🌍 Global Infrastructure AI")
    st.caption("World-style indicators + ML model. Streamlit Cloud compatible (ONNX-first).")
    st.markdown("<div class='small-muted'>Tip: ONNX is the safest format on Streamlit Cloud (avoids numpy/joblib mismatches).</div>", unsafe_allow_html=True)


def render_sidebar() -> Dict[str, Any]:
    with st.sidebar:
        st.subheader("Settings")
        show_debug = st.toggle("Show debug info", value=False)

        st.markdown("---")
        st.write("Model files expected:")
        st.code("models/infra_model.onnx\nmodels/infra_model.joblib (optional)", language="text")

        radius_m = st.slider("Map signals radius (meters)", 1000, 20000, DEFAULT_RADIUS_M, step=500)

        st.markdown("---")
        st.write("Live sources (free):")
        st.write("- OpenStreetMap (Overpass)")
        st.write("- GDELT (news)")
        st.write("- ReliefWeb (disaster reports)")

        return {"show_debug": show_debug, "radius_m": radius_m}


def render_dashboard(df: pd.DataFrame):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("<div class='card'><h4>Countries</h4><h2>%d</h2></div>" % len(df), unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='card'><h4>Avg GDP/capita</h4><h2>$%s</h2></div>" % f"{df['GDP_per_capita_USD'].mean():,.0f}", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='card'><h4>Avg HDI</h4><h2>%0.3f</h2></div>" % df["HDI_Index"].mean(), unsafe_allow_html=True)
    with c4:
        st.markdown("<div class='card'><h4>Updated</h4><h2>%s</h2></div>" % datetime.utcnow().strftime("%Y-%m-%d"), unsafe_allow_html=True)

    st.markdown("---")

    colA, colB = st.columns(2)
    with colA:
        top = df.sort_values("GDP_per_capita_USD", ascending=False).head(12)
        fig = px.bar(top, x="Country", y="GDP_per_capita_USD", title="Top GDP per capita (sample)")
        st.plotly_chart(fig, use_container_width=True)

    with colB:
        fig2 = px.scatter(
            df,
            x="GDP_per_capita_USD",
            y="HDI_Index",
            size="Population_Millions",
            hover_name="Country",
            title="GDP vs HDI (sample)",
            log_x=True,
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Sample Country Dataset")
    st.dataframe(df, use_container_width=True, height=420)


def render_predictor(default_df: pd.DataFrame, cfg: Dict[str, Any]):
    st.subheader("Infrastructure Predictor")

    left, right = st.columns([1.2, 1])

    # INPUT
    with left:
        method = st.radio("Input method", ["Select country (fallback)", "Custom input", "Upload CSV"], horizontal=True)

        row = None
        country_label = None

        if method == "Select country (fallback)":
            country_label = st.selectbox("Country code", default_df["Country"].tolist(), index=0)
            r = default_df[default_df["Country"] == country_label].iloc[0]
            row = {
                "Population_Millions": float(r["Population_Millions"]),
                "GDP_per_capita_USD": float(r["GDP_per_capita_USD"]),
                "HDI_Index": float(r["HDI_Index"]),
                "Urbanization_Rate": float(r["Urbanization_Rate"]),
            }

        elif method == "Custom input":
            row = {
                "Population_Millions": st.number_input("Population (Millions)", 0.1, 2000.0, 50.0, step=1.0),
                "GDP_per_capita_USD": st.number_input("GDP per Capita (USD)", 100.0, 200000.0, 5000.0, step=100.0),
                "HDI_Index": st.slider("HDI Index", 0.2, 1.0, 0.7, step=0.01),
                "Urbanization_Rate": st.slider("Urbanization Rate (%)", 0.0, 100.0, 50.0, step=1.0),
            }

        else:
            st.markdown("Upload CSV with columns:")
            st.code(", ".join(FEATURES), language="text")
            up = st.file_uploader("CSV file", type=["csv"])
            if up is not None:
                df_up = pd.read_csv(up)
                ok, msg = ensure_schema(df_up)
                if not ok:
                    st.error(msg)
                else:
                    st.success(f"Loaded {len(df_up):,} rows. Click Predict to run bulk predictions.")
                    st.session_state["uploaded_df"] = df_up

    # PREDICT
    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Run prediction")

        if st.button("🚀 PREDICT", use_container_width=True, type="primary"):
            if method == "Upload CSV":
                df_up = st.session_state.get("uploaded_df")
                if df_up is None:
                    st.error("Upload a valid CSV first.")
                else:
                    preds = []
                    model_kind_used = None
                    for _, r in df_up.iterrows():
                        row_dict = {k: float(r[k]) for k in FEATURES}
                        x = to_feature_row(row_dict)
                        need, mk = model_predict_proba(x)
                        model_kind_used = mk
                        preds.append(need)

                    out = df_up.copy()
                    out["Need_%"] = preds
                    out["Risk"] = out["Need_%"].apply(lambda v: risk_bucket(float(v))[0])
                    st.session_state["bulk_pred_df"] = out
                    st.session_state["bulk_model_kind"] = model_kind_used
                    st.success(f"Bulk prediction complete. Model used: {model_kind_used}")
            else:
                if row is None:
                    st.error("Provide input first.")
                else:
                    x = to_feature_row(row)
                    need, mk = model_predict_proba(x)
                    st.session_state["single_pred"] = {"need": need, "model_kind": mk, "input": row, "country": country_label}

        st.markdown("</div>", unsafe_allow_html=True)

    # RESULTS
    if "single_pred" in st.session_state:
        pred = st.session_state["single_pred"]
        need = float(pred["need"])
        mk = pred["model_kind"]
        badge_text, badge_cls = risk_bucket(need)

        st.markdown("---")
        st.subheader("Result")

        # Stable meter (Streamlit native) to avoid gauge rendering issues
        st.markdown("#### Need meter (stable)")
        st.progress(int(clamp(need, 0, 100)))
        st.markdown(
            f"<span class='badge {badge_cls}'>{badge_text}</span> &nbsp; "
            f"<b>Infrastructure Need</b>: <span style='font-size:1.8rem'>{need:0.1f}%</span>"
            f"<div class='small-muted'>Model used: {mk}</div>",
            unsafe_allow_html=True,
        )

        # Need vs Sufficient
        fig = px.bar(
            pd.DataFrame({"Metric": ["Need", "Sufficient"], "Value": [need, 100 - need]}),
            x="Metric",
            y="Value",
            title="Need vs Sufficient",
        )
        st.plotly_chart(fig, use_container_width=True)

    if "bulk_pred_df" in st.session_state:
        st.markdown("---")
        st.subheader("Bulk results")
        out = st.session_state["bulk_pred_df"]
        st.dataframe(out, use_container_width=True, height=420)

        csv_bytes = out.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download predictions CSV",
            data=csv_bytes,
            file_name="predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )


def render_map_signals(cfg: Dict[str, Any]):
    st.subheader("Optional: Map signals (FREE OSM)")
    st.caption("Click on the map to fetch nearby infrastructure counts (Overpass API). These signals can adjust the displayed risk (overlay).")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        m = folium.Map(location=[20, 0], zoom_start=2, tiles="OpenStreetMap")
        folium.LatLngPopup().add_to(m)
        map_data = st_folium(m, height=420, use_container_width=True)

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Click on map to get signals")
        st.markdown("<div class='small-muted'>Overpass has rate limits. We cache results for 10 minutes.</div>", unsafe_allow_html=True)

        lat, lon = None, None
        if map_data and map_data.get("last_clicked"):
            lat = map_data["last_clicked"]["lat"]
            lon = map_data["last_clicked"]["lng"]

        if lat is None:
            st.info("Click on the map first.")
        else:
            st.write(f"Selected point: `{lat:.5f}, {lon:.5f}`")
            radius_m = int(cfg["radius_m"])
            with st.spinner("Fetching map signals (Overpass)..."):
                try:
                    sig = fetch_overpass_signals(lat, lon, radius_m)
                    st.session_state["map_signals"] = sig
                    st.success("Signals loaded.")
                except Exception as e:
                    st.error(f"Overpass error: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

    if "map_signals" in st.session_state:
        sig = st.session_state["map_signals"]
        st.markdown("---")
        st.subheader("Signals")
        st.json(sig)

        adj = map_context_adjustment(sig)
        st.markdown("### Context adjustment")
        st.write(f"Adjustment suggested: **{adj:+.1f}** points (adds to displayed risk overlay).")
        st.session_state["map_adjustment"] = adj


def render_events_monitor(cfg: Dict[str, Any]):
    st.subheader("Live Events Monitor (FREE)")
    st.caption("Legal/free feeds only: GDELT + ReliefWeb. No social media scraping.")

    colA, colB = st.columns([1.2, 1])

    with colA:
        query = st.text_input(
            "Keywords (examples: flood OR cyclone OR bridge collapse OR infrastructure damage)",
            value="flood OR infrastructure damage OR bridge collapse OR cyclone",
        )
        max_records = st.slider("Max news records", 5, 50, 20, step=5)

        if st.button("Fetch latest", type="primary"):
            with st.spinner("Fetching feeds..."):
                try:
                    gd = fetch_gdelt(query=query, max_records=max_records)
                except Exception as e:
                    gd = pd.DataFrame()
                    st.error(f"GDELT fetch failed: {e}")

                try:
                    rw = fetch_reliefweb(query=query, limit=min(20, max_records))
                except Exception as e:
                    rw = pd.DataFrame()
                    st.error(f"ReliefWeb fetch failed: {e}")

                st.session_state["gdelt_df"] = gd
                st.session_state["relief_df"] = rw

    with colB:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Overlay risk (events)")
        gd = st.session_state.get("gdelt_df", pd.DataFrame())
        rw = st.session_state.get("relief_df", pd.DataFrame())
        overlay = disaster_overlay_score(gd, rw) if (gd is not None and rw is not None) else 0.0
        st.metric("Overlay points", f"{overlay:0.1f} / 15")
        st.markdown("<div class='small-muted'>This is an additive overlay, not model retraining.</div>", unsafe_allow_html=True)
        st.session_state["news_overlay"] = overlay
        st.markdown("</div>", unsafe_allow_html=True)

    gd = st.session_state.get("gdelt_df", pd.DataFrame())
    rw = st.session_state.get("relief_df", pd.DataFrame())

    if isinstance(gd, pd.DataFrame) and not gd.empty:
        st.markdown("### GDELT (news)")
        st.dataframe(gd, use_container_width=True, height=260)

    if isinstance(rw, pd.DataFrame) and not rw.empty:
        st.markdown("### ReliefWeb (disaster reports)")
        st.dataframe(rw, use_container_width=True, height=260)


def render_combined_risk_panel():
    st.markdown("---")
    st.subheader("Combined Risk View")

    base_need = None
    base_kind = None
    if "single_pred" in st.session_state:
        base_need = float(st.session_state["single_pred"]["need"])
        base_kind = st.session_state["single_pred"]["model_kind"]

    map_adj = float(st.session_state.get("map_adjustment", 0.0))
    news_overlay = float(st.session_state.get("news_overlay", 0.0))
    total_overlay = map_adj + news_overlay

    if base_need is None:
        st.info("Run a single prediction first (Predictor tab). Then come back here for combined view.")
        return

    combined = clamp(base_need + total_overlay, 0, 100)
    badge_text, badge_cls = risk_bucket(combined)

    st.markdown(
        f"<span class='badge {badge_cls}'>{badge_text}</span> &nbsp; "
        f"<b>Combined Need</b>: <span style='font-size:1.8rem'>{combined:0.1f}%</span>"
        f"<div class='small-muted'>Base model: {base_kind} | Map adj: {map_adj:+.1f} | Events overlay: {news_overlay:+.1f}</div>",
        unsafe_allow_html=True,
    )

    st.markdown("### Precautions & prevention (client-ready)")
    for item in recommendations(base_need, total_overlay):
        st.write(f"- {item}")


def render_model_status(cfg: Dict[str, Any]):
    st.subheader("Model Status")

    onnx_ok = os.path.exists(MODEL_ONNX_PATH)
    joblib_ok = os.path.exists(MODEL_JOBLIB_PATH)

    sess = load_onnx_session(MODEL_ONNX_PATH) if onnx_ok else None
    mdl = load_joblib_model(MODEL_JOBLIB_PATH) if joblib_ok else None

    if sess is not None:
        st.success("Model loaded ✅ (onnx)")
    elif mdl is not None:
        st.warning("Model loaded (joblib) — may break on Cloud if numpy mismatches. Prefer ONNX.")
    else:
        st.error("No model loaded. Upload ONNX model to: models/infra_model.onnx")

    if cfg.get("show_debug"):
        st.markdown("#### Debug")
        st.write("ONNX path:", MODEL_ONNX_PATH, "exists:", onnx_ok)
        st.write("JOBLIB path:", MODEL_JOBLIB_PATH, "exists:", joblib_ok)
        st.write("onnxruntime available:", ort is not None)
        st.write("joblib available:", joblib is not None)
        st.write("Features:", FEATURES)


def main():
    cfg = render_sidebar()
    render_header()

    default_df = build_default_country_df()

    tabs = st.tabs(["📊 Dashboard", "🤖 Predictor", "🗺️ Map Signals", "🛰️ Live Events", "✅ Model Status"])

    with tabs[0]:
        render_dashboard(default_df)

    with tabs[1]:
        render_predictor(default_df, cfg)

    with tabs[2]:
        render_map_signals(cfg)

    with tabs[3]:
        render_events_monitor(cfg)

    with tabs[4]:
        render_model_status(cfg)

    render_combined_risk_panel()

    st.markdown("---")
    st.caption(f"Updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} | Free sources: OSM Overpass + GDELT + ReliefWeb")


if __name__ == "__main__":
    main()
