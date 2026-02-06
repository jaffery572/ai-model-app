import os
import time
import json
import math
from datetime import datetime
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px

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

# Overpass endpoints (fallback list)
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.nchc.org.tw/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]

# FREE news/disaster sources
GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
RELIEFWEB_URL = "https://api.reliefweb.int/v1/reports"

DEFAULT_RADIUS_M = 5000  # Overpass radius around click point

# Throttling
EVENTS_COOLDOWN_SECONDS = 45  # UI cooldown for news fetch


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
code { white-space: pre-wrap !important; }
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


def sample_csv_bytes() -> bytes:
    df = pd.DataFrame(
        [
            {"Population_Millions": 50, "GDP_per_capita_USD": 5000, "HDI_Index": 0.70, "Urbanization_Rate": 50},
            {"Population_Millions": 120, "GDP_per_capita_USD": 2200, "HDI_Index": 0.62, "Urbanization_Rate": 40},
        ]
    )
    return df.to_csv(index=False).encode("utf-8")


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


def model_predict_proba(features_row: np.ndarray) -> Tuple[float, str]:
    """
    Returns (need_probability_0_100, model_kind).
    If model fails, falls back to heuristic.
    """
    # Try ONNX first
    sess = load_onnx_session(MODEL_ONNX_PATH)
    if sess is not None:
        try:
            input_name = sess.get_inputs()[0].name
            outputs = sess.run(None, {input_name: features_row.astype(np.float32)})
            out = np.array(outputs[0])

            # Common cases:
            # (1,2): probabilities -> take class1
            # (1,): score/logit -> sigmoid
            if out.ndim == 2 and out.shape[1] >= 2:
                prob_need = float(out[0, 1])
            else:
                score = float(out.reshape(-1)[0])
                prob_need = 1 / (1 + math.exp(-score))

            return clamp(prob_need * 100.0, 0.0, 100.0), "onnx"
        except Exception:
            pass

    # joblib fallback
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

    # Heuristic always works
    d = {
        "Population_Millions": float(features_row[0, 0]),
        "GDP_per_capita_USD": float(features_row[0, 1]),
        "HDI_Index": float(features_row[0, 2]),
        "Urbanization_Rate": float(features_row[0, 3]),
    }
    return heuristic_need_score(d), "heuristic"


# -----------------------------
# Overpass: robust map signals
# -----------------------------
def overpass_query(lat: float, lon: float, radius_m: int) -> str:
    # Keep query bounded to reduce timeouts
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


def _parse_overpass_signals(data: Dict[str, Any], radius_m: int) -> Dict[str, Any]:
    elements = data.get("elements", []) or []
    signals = {
        "roads": 0,
        "hospitals": 0,
        "schools": 0,
        "clinics": 0,
        "power": 0,
        "water": 0,
        "emergency": 0,
        "radius_m": radius_m,
        "elements": len(elements),
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
    return signals


@st.cache_data(ttl=60 * 10, show_spinner=False)
def fetch_overpass_signals_robust(lat: float, lon: float, radius_m: int) -> Dict[str, Any]:
    """
    Robust fetch:
      - endpoint fallback
      - retries with backoff
      - auto-shrink radius on timeouts
    """
    headers = {"Content-Type": "text/plain; charset=utf-8"}
    q = overpass_query(lat, lon, radius_m).strip()

    if not q:
        raise ValueError("Overpass query is empty.")

    # Try with shrinking radius if needed
    radii_try = [radius_m]
    if radius_m > 3500:
        radii_try.append(3500)
    if radius_m > 2000:
        radii_try.append(2000)
    if radius_m > 1200:
        radii_try.append(1200)

    last_err = None

    for r_m in radii_try:
        q2 = overpass_query(lat, lon, r_m).strip()
        for endpoint in OVERPASS_ENDPOINTS:
            # up to 3 retries per endpoint
            for attempt in range(3):
                try:
                    resp = requests.post(
                        endpoint,
                        data=q2.encode("utf-8"),
                        headers=headers,
                        timeout=(10, 35),  # connect/read
                    )
                    # Overpass sometimes returns HTML or empty when overloaded
                    if resp.status_code >= 500:
                        raise RuntimeError(f"Server error {resp.status_code}")
                    if resp.status_code == 429:
                        raise RuntimeError("Rate limited (429)")
                    resp.raise_for_status()

                    text = (resp.text or "").strip()
                    if not text or text.startswith("<!DOCTYPE html") or "Your input contains only whitespace" in text:
                        raise RuntimeError("Non-JSON/empty response from endpoint")

                    data = resp.json()
                    signals = _parse_overpass_signals(data, r_m)
                    signals["endpoint_used"] = endpoint
                    signals["attempt"] = attempt + 1
                    return signals

                except Exception as e:
                    last_err = f"{endpoint} (radius {r_m}) attempt {attempt + 1}: {e}"
                    # backoff
                    time.sleep(1.0 * (2 ** attempt))

    raise RuntimeError(f"Overpass failed after fallbacks. Last error: {last_err}")


def map_context_adjustment(signals: Dict[str, Any]) -> float:
    """
    Returns adjustment in range [-10, +15] points.
    More infrastructure density -> slightly lower need; very low density -> higher need.
    """
    roads = int(signals.get("roads", 0))
    key_pois = (
        int(signals.get("hospitals", 0))
        + int(signals.get("schools", 0))
        + int(signals.get("clinics", 0))
        + int(signals.get("power", 0))
        + int(signals.get("water", 0))
    )

    infra_density = (min(roads, 200) / 200.0) * 0.6 + (min(key_pois, 40) / 40.0) * 0.4
    adj = (0.35 - infra_density) * 45.0
    return clamp(adj, -10.0, 15.0)


# -----------------------------
# News / disaster feeds (FREE) - robust
# -----------------------------
def _safe_json(resp: requests.Response) -> Optional[dict]:
    try:
        txt = (resp.text or "").strip()
        if not txt:
            return None
        if txt.startswith("<!DOCTYPE html") or txt.startswith("<html"):
            return None
        return resp.json()
    except Exception:
        return None


@st.cache_data(ttl=60 * 15, show_spinner=False)
def fetch_gdelt(query: str, max_records: int = 20) -> pd.DataFrame:
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": int(max_records),
        "formatdatetime": "true",
        "sort": "HybridRel",
    }

    # One-shot request (cache + UI cooldown handles most rate limiting)
    r = requests.get(GDELT_DOC_URL, params=params, timeout=25)

    # Handle rate limiting gracefully
    if r.status_code == 429:
        raise RuntimeError("GDELT rate limited (429). Try again in ~1 minute.")
    if r.status_code >= 500:
        raise RuntimeError(f"GDELT server error ({r.status_code}). Try again later.")

    r.raise_for_status()
    j = _safe_json(r)
    if not j:
        raise RuntimeError("GDELT returned an invalid/empty response.")

    arts = j.get("articles") or []
    if not arts:
        return pd.DataFrame(columns=["title", "url", "sourceCountry", "seendate", "tone"])

    rows = []
    for a in arts:
        rows.append(
            {
                "title": a.get("title"),
                "url": a.get("url"),
                "sourceCountry": a.get("sourceCountry"),
                "seendate": a.get("seendate"),
                "tone": a.get("tone"),
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(ttl=60 * 30, show_spinner=False)
def fetch_reliefweb(query: str, limit: int = 15) -> pd.DataFrame:
    """
    ReliefWeb is strict about payload. This format is typically accepted:
      - appname
      - query.value
      - limit
      - profile: "list" (lighter and less error-prone)
      - sort: ["date:desc"]
    """
    payload = {
        "appname": "global-infra-ai",
        "query": {"value": str(query)},
        "limit": int(limit),
        "profile": "list",
        "sort": ["date:desc"],
    }

    r = requests.post(RELIEFWEB_URL, json=payload, timeout=30)
    if r.status_code >= 500:
        raise RuntimeError(f"ReliefWeb server error ({r.status_code}).")
    r.raise_for_status()

    j = _safe_json(r)
    if not j:
        raise RuntimeError("ReliefWeb returned an invalid/empty response.")

    data = j.get("data") or []
    rows = []
    for item in data:
        fields = item.get("fields") or {}
        # in list profile, some fields may be absent; keep safe defaults
        rows.append(
            {
                "title": fields.get("title"),
                "url": fields.get("url") or "",
                "date": (fields.get("date") or {}).get("created"),
                "source": ", ".join([s.get("name") for s in (fields.get("source") or []) if isinstance(s, dict)])[:120],
            }
        )
    return pd.DataFrame(rows)


def disaster_overlay_score(gdelt_df: pd.DataFrame, relief_df: pd.DataFrame) -> float:
    """
    Returns 0..15 overlay points based on volume + negative tone.
    """
    score = 0.0
    if isinstance(gdelt_df, pd.DataFrame) and not gdelt_df.empty:
        score += min(len(gdelt_df), 20) * 0.3
        tones = pd.to_numeric(gdelt_df.get("tone", pd.Series(dtype=float)), errors="coerce").dropna()
        if len(tones) > 0:
            avg_tone = float(tones.mean())
            if avg_tone < -2:
                score += 3.5
            elif avg_tone < -1:
                score += 2.0

    if isinstance(relief_df, pd.DataFrame) and not relief_df.empty:
        score += min(len(relief_df), 15) * 0.4

    return clamp(score, 0.0, 15.0)


def recommendations(need: float, overlay: float) -> List[str]:
    total = clamp(need + overlay, 0, 100)
    if total >= 70:
        return [
            "Prioritize critical infrastructure assessment (roads, power, water, healthcare).",
            "Create an emergency maintenance plan with response timelines and accountable owners.",
            "Strengthen flood drainage, slope stability, and backup power readiness.",
            "Run rapid vulnerability audits for bridges, hospitals, and substations.",
            "Pre-position resources for fast repairs: fuel, pumps, generators, spare parts.",
        ]
    if total >= 40:
        return [
            "Plan targeted upgrades in transport, utilities, and public services.",
            "Focus on resilience improvements (redundancy, maintenance cycles, inspections).",
            "Improve asset inventory and preventive maintenance to reduce failures.",
            "Track hazard alerts and prioritize at-risk assets for inspection.",
        ]
    return [
        "Maintain assets with preventive maintenance and routine inspections.",
        "Improve monitoring (basic reporting, asset inventory, maintenance logging).",
        "Review extreme weather preparedness (drainage, backups, response plans).",
    ]


# -----------------------------
# Analyst (free, rule-based chat)
# -----------------------------
def build_analyst_context() -> Dict[str, Any]:
    ctx = {
        "base_need": None,
        "model_kind": None,
        "map_adjustment": float(st.session_state.get("map_adjustment", 0.0)),
        "news_overlay": float(st.session_state.get("news_overlay", 0.0)),
        "combined_need": None,
        "signals": st.session_state.get("map_signals"),
        "gdelt_rows": int(len(st.session_state.get("gdelt_df", pd.DataFrame()))) if isinstance(st.session_state.get("gdelt_df"), pd.DataFrame) else 0,
        "relief_rows": int(len(st.session_state.get("relief_df", pd.DataFrame()))) if isinstance(st.session_state.get("relief_df"), pd.DataFrame) else 0,
    }
    if "single_pred" in st.session_state:
        ctx["base_need"] = float(st.session_state["single_pred"]["need"])
        ctx["model_kind"] = st.session_state["single_pred"]["model_kind"]
        ctx["combined_need"] = clamp(ctx["base_need"] + ctx["map_adjustment"] + ctx["news_overlay"], 0, 100)
    return ctx


def analyst_reply(user_text: str, ctx: Dict[str, Any]) -> str:
    t = (user_text or "").strip().lower()
    if not t:
        return "Ask a question like: “Summarize risk”, “What actions should we take?”, or “Explain map signals”."

    # No prediction yet
    if ctx.get("base_need") is None:
        return (
            "No single prediction is available yet.\n\n"
            "Go to the Predictor tab, run a single prediction, then come back here. "
            "After that, I can summarize base risk and overlays from map signals and live events."
        )

    base = float(ctx["base_need"])
    combined = float(ctx["combined_need"])
    mk = ctx.get("model_kind", "unknown")
    badge, _ = risk_bucket(combined)

    if any(k in t for k in ["summary", "summarize", "overview", "risk"]):
        parts = [
            f"**Status:** {badge}",
            f"**Base need:** {base:0.1f}% (model: {mk})",
            f"**Map adjustment:** {ctx['map_adjustment']:+0.1f}",
            f"**Events overlay:** {ctx['news_overlay']:+0.1f} (GDELT: {ctx['gdelt_rows']} items, ReliefWeb: {ctx['relief_rows']} items)",
            f"**Combined need:** {combined:0.1f}%",
        ]
        return "\n".join(parts)

    if any(k in t for k in ["map", "signals", "overpass", "roads", "hospital", "school"]):
        sig = ctx.get("signals")
        if not isinstance(sig, dict):
            return "No map signals loaded. Open the Map Signals tab, click on the map, and fetch signals."
        return (
            "Map signals summarize nearby infrastructure density (within the selected radius).\n\n"
            f"- Roads: {sig.get('roads', 0)}\n"
            f"- Hospitals: {sig.get('hospitals', 0)}\n"
            f"- Schools: {sig.get('schools', 0)}\n"
            f"- Clinics: {sig.get('clinics', 0)}\n"
            f"- Power-related: {sig.get('power', 0)}\n"
            f"- Water towers: {sig.get('water', 0)}\n\n"
            "These signals adjust the displayed risk (overlay). Lower density typically increases need."
        )

    if any(k in t for k in ["actions", "recommend", "plan", "precaution", "prevention", "mitigation"]):
        overlay = float(ctx["map_adjustment"] + ctx["news_overlay"])
        recs = recommendations(base, overlay)
        return "**Recommended actions:**\n" + "\n".join([f"- {x}" for x in recs])

    if any(k in t for k in ["events", "news", "gdelt", "reliefweb"]):
        return (
            "Live events overlay is an additive signal (it does not retrain the model).\n\n"
            f"- GDELT items: {ctx['gdelt_rows']}\n"
            f"- ReliefWeb items: {ctx['relief_rows']}\n"
            f"- Events overlay: {ctx['news_overlay']:+0.1f} / 15\n\n"
            "If feeds are rate-limited or temporarily unavailable, the app continues with overlay = 0."
        )

    return (
        "I can help with:\n"
        "- **Risk summary** (type: summary)\n"
        "- **Actions / precautions** (type: actions)\n"
        "- **Map signals explanation** (type: map)\n"
        "- **Events overlay explanation** (type: events)\n"
    )


# -----------------------------
# UI
# -----------------------------
def render_header():
    st.title("🌍 Global Infrastructure AI")
    st.caption("World-style indicators + ML model. Streamlit Cloud compatible (ONNX-first).")
    st.markdown(
        "<div class='small-muted'>ONNX is recommended on Streamlit Cloud to avoid numpy/joblib mismatches.</div>",
        unsafe_allow_html=True,
    )


def render_sidebar() -> Dict[str, Any]:
    with st.sidebar:
        st.subheader("Settings")

        show_debug = st.toggle("Show debug info", value=False)

        st.markdown("---")
        st.write("Model files expected:")
        st.code("models/infra_model.onnx\nmodels/infra_model.joblib (optional)", language="text")

        radius_m = st.slider("Map signals radius (meters)", 1000, 20000, DEFAULT_RADIUS_M, step=500)

        st.markdown("---")
        st.write("Sample CSV (for Upload CSV):")
        st.download_button(
            "Download sample.csv",
            data=sample_csv_bytes(),
            file_name="sample.csv",
            mime="text/csv",
            use_container_width=True,
        )

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
        st.markdown(
            "<div class='card'><h4>Avg GDP/capita</h4><h2>$%s</h2></div>" % f"{df['GDP_per_capita_USD'].mean():,.0f}",
            unsafe_allow_html=True,
        )
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


def render_predictor(default_df: pd.DataFrame):
    st.subheader("Infrastructure Predictor")

    left, right = st.columns([1.2, 1])

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
                    for _, rr in df_up.iterrows():
                        row_dict = {k: float(rr[k]) for k in FEATURES}
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

        st.markdown("#### Need meter (stable)")
        st.progress(int(clamp(need, 0, 100)))
        st.markdown(
            f"<span class='badge {badge_cls}'>{badge_text}</span> &nbsp; "
            f"<b>Infrastructure Need</b>: <span style='font-size:1.8rem'>{need:0.1f}%</span>"
            f"<div class='small-muted'>Model used: {mk}</div>",
            unsafe_allow_html=True,
        )

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
    st.caption("Click on the map to fetch nearby infrastructure counts (Overpass API). These signals adjust displayed risk (overlay).")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        m = folium.Map(location=[20, 0], zoom_start=2, tiles="OpenStreetMap")
        folium.LatLngPopup().add_to(m)
        map_data = st_folium(m, height=420, use_container_width=True)

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Click on map to get signals")
        st.markdown("<div class='small-muted'>Overpass has rate limits. This app retries and uses multiple endpoints.</div>", unsafe_allow_html=True)

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
                    sig = fetch_overpass_signals_robust(lat, lon, radius_m)
                    st.session_state["map_signals"] = sig
                    st.success("Signals loaded.")
                    st.markdown("Endpoint used (not clickable):")
                    st.code(str(sig.get("endpoint_used", "")), language="text")
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


def _cooldown_remaining() -> int:
    last = st.session_state.get("events_last_fetch_ts", 0.0)
    now = time.time()
    rem = int(max(0, EVENTS_COOLDOWN_SECONDS - (now - float(last))))
    return rem


def render_events_monitor():
    st.subheader("Live Events Monitor (FREE)")
    st.caption("Free feeds only: GDELT + ReliefWeb. If rate-limited, the app continues without events overlay.")

    colA, colB = st.columns([1.2, 1])

    with colA:
        query = st.text_input(
            "Keywords (examples: flood OR cyclone OR bridge collapse OR infrastructure damage)",
            value="flood OR infrastructure damage OR bridge collapse OR cyclone",
        )
        max_records = st.slider("Max news records", 5, 50, 20, step=5)

        rem = _cooldown_remaining()
        if rem > 0:
            st.info(f"Cooldown active: wait {rem}s to fetch again (prevents rate limiting).")

        if st.button("Fetch latest", type="primary", disabled=(rem > 0)):
            st.session_state["events_last_fetch_ts"] = time.time()
            with st.spinner("Fetching feeds..."):
                gd, rw = pd.DataFrame(), pd.DataFrame()

                # GDELT
                try:
                    gd = fetch_gdelt(query=query, max_records=min(30, int(max_records)))
                except Exception as e:
                    st.warning(f"GDELT unavailable: {e}")

                # ReliefWeb
                try:
                    rw = fetch_reliefweb(query=query, limit=min(20, int(max_records)))
                except Exception as e:
                    st.warning(f"ReliefWeb unavailable: {e}")

                st.session_state["gdelt_df"] = gd
                st.session_state["relief_df"] = rw

    with colB:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Overlay risk (events)")
        gd = st.session_state.get("gdelt_df", pd.DataFrame())
        rw = st.session_state.get("relief_df", pd.DataFrame())
        overlay = disaster_overlay_score(gd, rw)
        st.metric("Overlay points", f"{overlay:0.1f} / 15")
        st.markdown("<div class='small-muted'>Additive overlay (does not retrain the model).</div>", unsafe_allow_html=True)
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

    if "single_pred" not in st.session_state:
        st.info("Run a single prediction first (Predictor tab). Then come back here for combined view.")
        return

    base_need = float(st.session_state["single_pred"]["need"])
    base_kind = st.session_state["single_pred"]["model_kind"]

    map_adj = float(st.session_state.get("map_adjustment", 0.0))
    news_overlay = float(st.session_state.get("news_overlay", 0.0))
    total_overlay = map_adj + news_overlay

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
        st.warning("Model loaded (joblib). Prefer ONNX on Streamlit Cloud for stability.")
    else:
        st.error("No model loaded. Upload ONNX model to: models/infra_model.onnx")

    if cfg.get("show_debug"):
        st.markdown("#### Debug")
        st.write("ONNX path:", MODEL_ONNX_PATH, "exists:", onnx_ok)
        st.write("JOBLIB path:", MODEL_JOBLIB_PATH, "exists:", joblib_ok)
        st.write("onnxruntime available:", ort is not None)
        st.write("joblib available:", joblib is not None)
        st.write("Features:", FEATURES)


def render_analyst_chat():
    st.subheader("Analyst (free)")
    st.caption("This assistant uses your current prediction + overlays to generate guidance without paid APIs.")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Render history
    for m in st.session_state["chat_history"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_text = st.chat_input("Ask: summary / actions / map / events")
    if user_text is not None:
        st.session_state["chat_history"].append({"role": "user", "content": user_text})
        ctx = build_analyst_context()
        reply = analyst_reply(user_text, ctx)
        st.session_state["chat_history"].append({"role": "assistant", "content": reply})
        st.rerun()


def main():
    cfg = render_sidebar()
    render_header()

    default_df = build_default_country_df()

    tabs = st.tabs(
        ["📊 Dashboard", "🤖 Predictor", "🗺️ Map Signals", "🛰️ Live Events", "💬 Analyst", "✅ Model Status"]
    )

    with tabs[0]:
        render_dashboard(default_df)

    with tabs[1]:
        render_predictor(default_df)

    with tabs[2]:
        render_map_signals(cfg)

    with tabs[3]:
        render_events_monitor()

    with tabs[4]:
        render_analyst_chat()

    with tabs[5]:
        render_model_status(cfg)

    render_combined_risk_panel()

    st.markdown("---")
    st.caption(f"Updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} | Free sources: OSM Overpass + GDELT + ReliefWeb")


if __name__ == "__main__":
    main()
