import os
import time
import math
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd
import requests

# Optional ONNX (Cloud-safe)
try:
    import onnxruntime as ort
except Exception:
    ort = None

# Optional joblib fallback (may fail if numpy mismatch; ONNX recommended)
try:
    import joblib
except Exception:
    joblib = None


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

# Feeds (free)
GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
RELIEFWEB_URL = "https://api.reliefweb.int/v1/reports"

# New FREE feeds
USGS_EQ_URL = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson"
EONET_EVENTS_URL = "https://eonet.gsfc.nasa.gov/api/v3/events"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


# -----------------------------
# Generic helpers
# -----------------------------
def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return default


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


# -----------------------------
# Retry / backoff network layer
# -----------------------------
def _sleep_backoff(attempt: int, base: float = 0.8, cap: float = 8.0) -> None:
    t = min(cap, base * (2 ** attempt))
    time.sleep(t)


def request_json(
    method: str,
    url: str,
    *,
    params: Optional[dict] = None,
    json_body: Optional[dict] = None,
    data: Optional[bytes] = None,
    headers: Optional[dict] = None,
    timeout: int = 20,
    max_retries: int = 3,
) -> Tuple[Optional[dict], Optional[str], Optional[int]]:
    """
    Returns: (json_dict_or_none, error_message_or_none, status_code_or_none)
    Safe: never throws.
    """
    headers = headers or {}
    headers.setdefault("User-Agent", "GlobalInfrastructureAI/1.0 (streamlit)")

    for attempt in range(max_retries):
        try:
            r = requests.request(
                method,
                url,
                params=params,
                json=json_body,
                data=data,
                headers=headers,
                timeout=timeout,
            )
            code = r.status_code

            # Handle rate limiting / temporary failures with retry
            if code in (429, 500, 502, 503, 504):
                if attempt < max_retries - 1:
                    _sleep_backoff(attempt)
                    continue
                return None, f"{code} error from provider (rate-limited or temporary).", code

            if code >= 400:
                snippet = (r.text or "")[:400]
                return None, f"{code} error: {snippet}", code

            try:
                return r.json(), None, code
            except Exception:
                return None, "Response was not valid JSON.", code

        except Exception as e:
            if attempt < max_retries - 1:
                _sleep_backoff(attempt)
                continue
            return None, f"Network error: {e}", None

    return None, "Unknown error.", None


# -----------------------------
# Model loading / prediction
# -----------------------------
_onnx_sess_cache: Dict[str, Any] = {}
_joblib_cache: Dict[str, Any] = {}


def load_onnx_session(path: str):
    if ort is None:
        return None
    if not os.path.exists(path):
        return None
    if path in _onnx_sess_cache:
        return _onnx_sess_cache[path]
    try:
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        _onnx_sess_cache[path] = sess
        return sess
    except Exception:
        return None


def load_joblib_model(path: str):
    if joblib is None:
        return None
    if not os.path.exists(path):
        return None
    if path in _joblib_cache:
        return _joblib_cache[path]
    try:
        mdl = joblib.load(path)
        _joblib_cache[path] = mdl
        return mdl
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


def model_predict_need(features_row: np.ndarray) -> Tuple[float, str]:
    """
    Returns (need_0_100, model_kind). Never throws.
    """
    sess = load_onnx_session(MODEL_ONNX_PATH)
    if sess is not None:
        try:
            input_name = sess.get_inputs()[0].name
            outputs = sess.run(None, {input_name: features_row.astype(np.float32)})
            out = np.array(outputs[0])

            if out.ndim == 2 and out.shape[1] >= 2:
                prob_need = float(out[0, 1])
            else:
                score = float(out.reshape(-1)[0])
                prob_need = 1 / (1 + math.exp(-score))

            return clamp(prob_need * 100.0, 0.0, 100.0), "onnx"
        except Exception:
            pass

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

    d = {
        "Population_Millions": float(features_row[0, 0]),
        "GDP_per_capita_USD": float(features_row[0, 1]),
        "HDI_Index": float(features_row[0, 2]),
        "Urbanization_Rate": float(features_row[0, 3]),
    }
    return heuristic_need_score(d), "heuristic"


# -----------------------------
# Overpass signals (multi-endpoint)
# -----------------------------
def overpass_query(lat: float, lon: float, radius_m: int) -> str:
    return f"""
[out:json][timeout:25];
(
  way(around:{radius_m},{lat},{lon})["highway"]["highway"~"motorway|trunk|primary|secondary|tertiary|residential"];
  node(around:{radius_m},{lat},{lon})["amenity"="hospital"];
  node(around:{radius_m},{lat},{lon})["amenity"="school"];
  node(around:{radius_m},{lat},{lon})["amenity"="clinic"];
  node(around:{radius_m},{lat},{lon})["power"];
  node(around:{radius_m},{lat},{lon})["man_made"="water_tower"];
  node(around:{radius_m},{lat},{lon})["emergency"];
);
out body;
"""


def parse_overpass_signals(data: dict) -> Dict[str, Any]:
    elements = data.get("elements", []) or []
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

    signals["roads"] = int(min(signals["roads"], 10000))
    return signals


def fetch_overpass_signals(lat: float, lon: float, radius_m: int) -> Tuple[Optional[Dict[str, Any]], str, Optional[str]]:
    q = overpass_query(lat, lon, radius_m).encode("utf-8")

    for endpoint in OVERPASS_ENDPOINTS:
        j, err, code = request_json(
            "POST",
            endpoint,
            data=q,
            headers={"Content-Type": "text/plain"},
            timeout=25,
            max_retries=2,
        )
        if j is not None:
            sig = parse_overpass_signals(j)
            sig["radius_m"] = int(radius_m)
            return sig, "Signals loaded.", endpoint

    return None, "Overpass unavailable right now. Try again in a minute.", None


def map_context_adjustment(signals: Dict[str, Any]) -> float:
    roads = float(signals.get("roads", 0))
    key_pois = float(
        signals.get("hospitals", 0)
        + signals.get("schools", 0)
        + signals.get("clinics", 0)
        + signals.get("power", 0)
        + signals.get("water", 0)
    )

    infra_density = (min(roads, 2000) / 2000.0) * 0.6 + (min(key_pois, 80) / 80.0) * 0.4
    adj = (0.35 - infra_density) * 45.0
    return clamp(adj, -10.0, 15.0)


# -----------------------------
# News / disaster feeds (safe)
# -----------------------------
def fetch_gdelt(query: str, max_records: int = 20) -> Tuple[pd.DataFrame, Optional[str]]:
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": int(max_records),
        "formatdatetime": "true",
        "sort": "HybridRel",
    }

    j, err, code = request_json("GET", GDELT_DOC_URL, params=params, timeout=20, max_retries=3)
    if j is None:
        if code == 429:
            return pd.DataFrame(), "GDELT rate-limited (429). Try again in ~1 minute."
        return pd.DataFrame(), f"GDELT unavailable: {err or 'unknown error'}"

    arts = (j.get("articles") or [])
    if not arts:
        return pd.DataFrame(columns=["title", "url", "sourceCountry", "seendate", "tone"]), None

    rows = []
    for a in arts:
        rows.append({
            "title": a.get("title"),
            "url": a.get("url"),
            "sourceCountry": a.get("sourceCountry"),
            "seendate": a.get("seendate"),
            "tone": a.get("tone"),
        })
    return pd.DataFrame(rows), None


def fetch_reliefweb(query: str, limit: int = 15) -> Tuple[pd.DataFrame, Optional[str]]:
    payload = {
        "appname": "global-infrastructure-ai",
        "query": {"value": query},
        "limit": int(limit),
        "profile": "lite",
        "sort": ["date:desc"],
    }

    j, err, code = request_json(
        "POST",
        RELIEFWEB_URL,
        json_body=payload,
        headers={"Content-Type": "application/json"},
        timeout=25,
        max_retries=2,
    )
    if j is None:
        if code == 400:
            return pd.DataFrame(), "ReliefWeb rejected the query (400). Try simpler keywords."
        return pd.DataFrame(), f"ReliefWeb unavailable: {err or 'unknown error'}"

    data = j.get("data") or []
    rows = []
    for item in data:
        fields = item.get("fields") or {}
        rows.append({
            "title": fields.get("title"),
            "url": fields.get("url") or "",
            "date": (fields.get("date") or {}).get("created"),
            "status": fields.get("status"),
            "type": ", ".join([t.get("name") for t in (fields.get("type") or []) if isinstance(t, dict)]),
        })

    if not rows:
        return pd.DataFrame(columns=["title", "url", "date", "status", "type"]), None

    return pd.DataFrame(rows), None


def disaster_overlay_score(gdelt_df: pd.DataFrame, relief_df: pd.DataFrame) -> float:
    score = 0.0

    if isinstance(gdelt_df, pd.DataFrame) and not gdelt_df.empty:
        score += min(len(gdelt_df), 20) * 0.3
        tones = pd.to_numeric(gdelt_df.get("tone", pd.Series([], dtype=float)), errors="coerce").dropna()
        if len(tones) > 0:
            avg_tone = float(tones.mean())
            if avg_tone < -2:
                score += 3.5
            elif avg_tone < -1:
                score += 2.0

    if isinstance(relief_df, pd.DataFrame) and not relief_df.empty:
        score += min(len(relief_df), 15) * 0.4

    return clamp(score, 0.0, 15.0)


# -----------------------------
# NEW: USGS earthquakes (free)
# -----------------------------
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2) + math.cos(p1) * math.cos(p2) * (math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def fetch_usgs_earthquakes_near(lat: float, lon: float, radius_km: float = 300.0) -> Tuple[pd.DataFrame, Optional[str]]:
    j, err, code = request_json("GET", USGS_EQ_URL, timeout=20, max_retries=3)
    if j is None:
        return pd.DataFrame(), f"USGS unavailable: {err or code}"

    feats = j.get("features") or []
    rows = []
    for f in feats:
        props = f.get("properties") or {}
        geom = f.get("geometry") or {}
        coords = geom.get("coordinates") or []
        if len(coords) < 2:
            continue
        eq_lon, eq_lat = float(coords[0]), float(coords[1])
        dist = haversine_km(lat, lon, eq_lat, eq_lon)
        if dist <= radius_km:
            mag = props.get("mag")
            rows.append({
                "time": props.get("time"),
                "place": props.get("place"),
                "mag": mag,
                "url": props.get("url"),
                "distance_km": dist,
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["distance_km"] = df["distance_km"].round(1)
        df = df.sort_values(["mag", "distance_km"], ascending=[False, True]).head(30)
    return df, None


def usgs_overlay_points(eq_df: pd.DataFrame) -> float:
    """
    0..8 overlay points based on nearby quakes (last day feed).
    """
    if eq_df is None or not isinstance(eq_df, pd.DataFrame) or eq_df.empty:
        return 0.0

    mags = pd.to_numeric(eq_df.get("mag", pd.Series([], dtype=float)), errors="coerce").dropna()
    if mags.empty:
        return min(2.0, len(eq_df) * 0.2)

    max_mag = float(mags.max())
    count = float(len(eq_df))

    pts = 0.0
    pts += min(3.0, count * 0.25)
    if max_mag >= 6.0:
        pts += 5.0
    elif max_mag >= 5.0:
        pts += 3.0
    elif max_mag >= 4.0:
        pts += 1.5

    return clamp(pts, 0.0, 8.0)


# -----------------------------
# NEW: Open-Meteo (free)
# -----------------------------
def fetch_open_meteo(lat: float, lon: float) -> Tuple[Optional[dict], Optional[str]]:
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "precipitation_sum,wind_speed_10m_max,temperature_2m_max,temperature_2m_min",
        "timezone": "UTC",
        "forecast_days": 3,
    }

    j, err, code = request_json(
        "GET",
        OPEN_METEO_URL,
        params=params,
        timeout=20,
        max_retries=5,
    )

    if j is None:
        if code == 429:
            return None, "Open-Meteo rate-limited (429). Try again in ~1–2 minutes."
        return None, f"Open-Meteo unavailable: {err or code}"
    return j, None


def open_meteo_overlay_points(meteo_json: Optional[dict]) -> Tuple[float, Dict[str, Any]]:
    """
    0..6 overlay points from heavy rain / strong wind heuristic.
    Also returns a small summary dict.
    """
    if not meteo_json:
        return 0.0, {}

    daily = meteo_json.get("daily") or {}
    dates = daily.get("time") or []
    rain = daily.get("precipitation_sum") or []
    wind = daily.get("wind_speed_10m_max") or []
    tmax = daily.get("temperature_2m_max") or []
    tmin = daily.get("temperature_2m_min") or []

    if not dates:
        return 0.0, {}

    def _safe_idx(arr, i):
        try:
            return float(arr[i])
        except Exception:
            return float("nan")

    rain0 = _safe_idx(rain, 0)
    wind0 = _safe_idx(wind, 0)

    pts = 0.0
    if not math.isnan(rain0):
        if rain0 >= 50:
            pts += 4.0
        elif rain0 >= 25:
            pts += 2.5
        elif rain0 >= 10:
            pts += 1.0

    if not math.isnan(wind0):
        if wind0 >= 70:
            pts += 3.0
        elif wind0 >= 50:
            pts += 2.0
        elif wind0 >= 35:
            pts += 1.0

    pts = clamp(pts, 0.0, 6.0)

    summary = {
        "date_utc": dates[0] if dates else None,
        "precipitation_sum_mm": rain0,
        "wind_speed_10m_max": wind0,
        "tmax": _safe_idx(tmax, 0),
        "tmin": _safe_idx(tmin, 0),
    }
    return pts, summary


# -----------------------------
# NEW: NASA EONET (free)
# -----------------------------
def fetch_eonet_events_near(lat: float, lon: float, radius_km: float = 500.0, limit: int = 30) -> Tuple[pd.DataFrame, Optional[str]]:
    params = {"status": "open", "limit": int(limit)}
    j, err, code = request_json("GET", EONET_EVENTS_URL, params=params, timeout=20, max_retries=3)
    if j is None:
        return pd.DataFrame(), f"EONET unavailable: {err or code}"

    events = j.get("events") or []
    rows = []

    for ev in events:
        title = ev.get("title")
        link = ev.get("link")
        categories = ev.get("categories") or []
        cat_names = ", ".join([c.get("title") for c in categories if isinstance(c, dict)])

        geoms = ev.get("geometry") or []
        best_dist = None
        best_lat = None
        best_lon = None
        best_date = None

        for g in geoms:
            gtype = g.get("type")
            coords = g.get("coordinates")
            gdate = g.get("date")
            if gtype == "Point" and isinstance(coords, list) and len(coords) >= 2:
                glon, glat = float(coords[0]), float(coords[1])
                dist = haversine_km(lat, lon, glat, glon)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_lat, best_lon = glat, glon
                    best_date = gdate

        if best_dist is not None and best_dist <= radius_km:
            rows.append({
                "title": title,
                "categories": cat_names,
                "date": best_date,
                "url": link,
                "distance_km": round(best_dist, 1),
                "lat": best_lat,
                "lon": best_lon,
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("distance_km", ascending=True).head(30)
    return df, None


def eonet_overlay_points(eo_df: pd.DataFrame) -> float:
    """
    0..6 overlay points based on number + category rough weighting.
    """
    if eo_df is None or not isinstance(eo_df, pd.DataFrame) or eo_df.empty:
        return 0.0

    pts = min(3.0, len(eo_df) * 0.4)

    cats = " ".join([str(x).lower() for x in eo_df.get("categories", [])])
    if "wildfire" in cats:
        pts += 1.5
    if "severe storms" in cats or "storm" in cats:
        pts += 1.0
    if "volcano" in cats:
        pts += 1.5
    if "flood" in cats:
        pts += 1.0

    return clamp(pts, 0.0, 6.0)


# -----------------------------
# NEW: Detailed Recommendations + Evidence + Report
# -----------------------------
def _confidence_and_limitations(base_need: float, overlays: Dict[str, float], model_kind: str, signals_used: Dict[str, bool]) -> Tuple[str, List[str]]:
    """
    Returns: (confidence_label, limitations_list)
    Confidence is heuristic (not statistical), based on model availability + signal coverage.
    """
    coverage = 0
    if signals_used.get("map"):
        coverage += 1
    if signals_used.get("news"):
        coverage += 1
    if signals_used.get("usgs"):
        coverage += 1
    if signals_used.get("weather"):
        coverage += 1
    if signals_used.get("eonet"):
        coverage += 1

    # Model quality assumption
    if model_kind == "onnx":
        base_conf = 0.70
    elif model_kind == "joblib":
        base_conf = 0.60
    else:
        base_conf = 0.45  # heuristic-only

    conf = base_conf + min(0.20, coverage * 0.05)

    # Penalty when overlays dominate
    overlay_total = float(sum([abs(v) for v in overlays.values()]))
    if overlay_total > 20:
        conf -= 0.08
    if overlay_total > 35:
        conf -= 0.10

    conf = clamp(conf, 0.25, 0.90)

    if conf >= 0.78:
        label = "High"
    elif conf >= 0.62:
        label = "Medium"
    else:
        label = "Low"

    limitations = [
        "This score is an indicator-level estimate; it is not an engineering assessment of specific assets.",
        "OpenStreetMap/Overpass data may be incomplete or temporarily rate-limited; counts are approximate.",
        "News/relief feeds reflect media/reporting volume and may be biased by coverage and language.",
        "Hazard feeds (USGS/EONET/Open-Meteo) are near-real-time snapshots; conditions can change quickly.",
        "The overlays are heuristic adjustments, not causal proof of damage or infrastructure failure.",
        "Use this report to prioritize follow-up inspections, audits, and local authority coordination.",
    ]

    return label, limitations


def recommendations_detailed(
    *,
    base_need: float,
    overlays: Dict[str, float],
    inputs: Dict[str, Any],
    map_signals: Optional[Dict[str, Any]] = None,
    evidence: Optional[Dict[str, Any]] = None,
    model_kind: str = "onnx",
) -> Dict[str, Any]:
    """
    Returns a structured plan with:
      - summary (immediate actions, context reasons)
      - items (each: why, data_gathered, steps, deliverables)
      - evidence (dataframes for link rendering + snapshots)
      - confidence + limitations
    """
    evidence = evidence or {}

    combined = clamp(base_need + sum(overlays.values()), 0, 100)
    risk, _ = risk_bucket(combined)

    # Signals used
    signals_used = {
        "map": bool(map_signals),
        "news": float(overlays.get("events", 0.0)) != 0.0,
        "usgs": float(overlays.get("usgs", 0.0)) != 0.0,
        "weather": float(overlays.get("weather", 0.0)) != 0.0,
        "eonet": float(overlays.get("eonet", 0.0)) != 0.0,
    }
    confidence_label, limitations = _confidence_and_limitations(base_need, overlays, model_kind, signals_used)

    # Context reasons (readable drivers)
    reasons = []
    try:
        gdp = float(inputs.get("GDP_per_capita_USD", 0))
        hdi = float(inputs.get("HDI_Index", 0))
        urb = float(inputs.get("Urbanization_Rate", 0))
        pop = float(inputs.get("Population_Millions", 0))

        if gdp < 5000:
            reasons.append("Low GDP per capita increases capital constraints and raises infrastructure backlog risk.")
        elif gdp < 12000:
            reasons.append("Moderate GDP per capita suggests selective investment capacity but still potential gaps.")
        else:
            reasons.append("Higher GDP per capita usually supports maintenance and faster upgrades (lower baseline risk).")

        if hdi < 0.70:
            reasons.append("Lower HDI correlates with service access gaps (health, utilities, transport).")
        elif hdi < 0.80:
            reasons.append("Mid-range HDI suggests mixed service quality and uneven regional coverage.")
        else:
            reasons.append("High HDI typically aligns with better service coverage and resilience capacity.")

        if urb > 70:
            reasons.append("High urbanization increases load on roads, water, and power networks (maintenance pressure).")
        elif urb < 40 and pop > 50:
            reasons.append("Low urbanization + large population often indicates dispersed service delivery challenges.")
        else:
            reasons.append("Urbanization level suggests standard planning demands on public services.")

        if pop > 150 and gdp < 6000:
            reasons.append("Large population with low GDP per capita increases demand pressure with limited funding.")
    except Exception:
        reasons.append("Indicator drivers could not be fully computed due to missing inputs.")

    # Map signals notes
    map_notes = []
    if map_signals:
        roads = map_signals.get("roads")
        hospitals = map_signals.get("hospitals")
        schools = map_signals.get("schools")
        power = map_signals.get("power")
        water = map_signals.get("water")
        map_notes.append(f"Nearby roads count: {roads}")
        map_notes.append(f"Hospitals: {hospitals}, Schools: {schools}, Power nodes: {power}, Water towers: {water}")

    # Immediate actions (based on risk)
    if combined >= 70:
        immediate = [
            "Initiate a 72-hour critical infrastructure condition check (roads, bridges, power, water, hospitals).",
            "Stand up an incident coordination cell (operations + utilities + public safety) with daily reporting.",
            "Confirm backup power readiness for hospitals and water pumping stations (fuel + load tests).",
            "Prioritize drainage, culverts, embankments, and known flood bottlenecks for rapid clearance.",
        ]
    elif combined >= 40:
        immediate = [
            "Create a prioritized inspection list for high-traffic corridors and utility bottlenecks.",
            "Review preventive maintenance schedule and tighten inspection frequency for at-risk assets.",
            "Prepare a quick-response contractor roster and material stock list.",
        ]
    else:
        immediate = [
            "Maintain preventive maintenance and routine inspections with standardized checklists.",
            "Build/refresh an asset inventory with condition scoring and repair backlog tracking.",
            "Add low-cost monitoring (weekly reporting + incident log + periodic audits).",
        ]

    # Detailed plan items
    items = []

    items.append({
        "title": "1) Asset inventory & condition scoring (foundation)",
        "why": [
            "Without a complete asset list, upgrades become reactive and funding allocation is inefficient.",
            "A condition score enables prioritization and accountability.",
        ],
        "data_gathered": [
            "Input indicators (GDP/HDI/Urbanization/Population).",
            "Map-based infrastructure density signals (if available).",
        ] + (map_notes if map_notes else ["Map signals not available; using indicators only."]),
        "steps": [
            "List critical assets by category: roads, bridges, water, power, hospitals, schools.",
            "Assign condition scoring (1–5) using rapid field survey templates.",
            "Create a backlog list with: issue type, severity, estimated cost, and deadline.",
            "Set a monthly review cycle and track progress in a simple dashboard/spreadsheet.",
        ],
        "deliverables": [
            "Asset register (CSV/Sheet) with condition score and owner.",
            "Criticality matrix (high/medium/low) and prioritized backlog list.",
        ],
    })

    items.append({
        "title": "2) Resilience upgrades (utilities + transport)",
        "why": [
            "Utilities and transport failures cause cascading disruption (health access, supply chain, safety).",
            "Redundancy and maintenance are often the cheapest risk reducers.",
        ],
        "data_gathered": [
            "Hazard overlays (USGS/Open-Meteo/EONET) and news signals (if refreshed).",
            f"Overlay totals: map={overlays.get('map', 0):+.1f}, events={overlays.get('events', 0):+.1f}, "
            f"usgs={overlays.get('usgs', 0):+.1f}, weather={overlays.get('weather', 0):+.1f}, eonet={overlays.get('eonet', 0):+.1f}",
        ],
        "steps": [
            "Identify single points of failure: substations, trunk mains, bridges, main corridors.",
            "Add redundancy where possible: alternate feeders, bypass valves, backup generators.",
            "Implement preventive maintenance triggers (threshold-based) for high-risk assets.",
            "Run tabletop exercises for disruption scenarios (flood, quake, storm).",
        ],
        "deliverables": [
            "Resilience improvement plan with cost tiers (low/medium/high).",
            "Maintenance SOP + trigger thresholds + escalation contacts.",
        ],
    })

    items.append({
        "title": "3) Emergency readiness & response workflow",
        "why": [
            "Fast coordination reduces downtime and secondary damage.",
            "Clear responsibilities prevent delays during incidents.",
        ],
        "data_gathered": [
            "Live event feeds (GDELT/ReliefWeb) and hazards (USGS/EONET/Open-Meteo) when available.",
            "Known high-risk drivers from indicators and infrastructure density signals.",
        ],
        "steps": [
            "Define incident roles (lead, comms, utilities liaison, field ops, procurement).",
            "Create a rapid assessment checklist (first 6 hours, 24 hours, 72 hours).",
            "Pre-position critical spares: pumps, generators, repair kits, temporary bridges/barriers.",
            "Set public communication templates and update cycles.",
        ],
        "deliverables": [
            "Emergency playbook (PDF/MD) and checklists for field teams.",
            "Resource & contractor roster with 24/7 contact details.",
        ],
    })

    # Evidence package (for report links)
    plan = {
        "meta": {
            "risk_level": risk,
            "base_need": float(base_need),
            "combined_need": float(combined),
            "model_kind": model_kind,
            "overlays": {k: float(v) for k, v in overlays.items()},
        },
        "summary": {
            "confidence": confidence_label,
            "immediate_actions": immediate,
            "context_reasons": reasons,
        },
        "items": items,
        "evidence": {
            "map_signals": map_signals or {},
            "gdelt_df": evidence.get("gdelt_df"),
            "relief_df": evidence.get("relief_df"),
            "usgs_df": evidence.get("usgs_df"),
            "eonet_df": evidence.get("eonet_df"),
            "meteo_summary": evidence.get("meteo_summary") or {},
        },
        "limitations": limitations,
    }
    return plan


def render_plan_markdown(plan: Dict[str, Any]) -> str:
    """
    Client-ready markdown report with:
      - summary, confidence, limitations
      - readable evidence section
      - evidence links (GDELT/ReliefWeb/USGS/EONET)
    """
    meta = plan.get("meta", {})
    summary = plan.get("summary", {})
    items = plan.get("items", [])
    evidence = plan.get("evidence", {})
    limitations = plan.get("limitations", [])

    lines: List[str] = []
    lines.append("# Infrastructure Risk Report")
    lines.append("")
    lines.append(f"**Risk level:** {meta.get('risk_level', 'N/A')}")
    lines.append(f"**Base need (model):** {float(meta.get('base_need', 0.0)):.1f}%")
    lines.append(f"**Combined need:** {float(meta.get('combined_need', 0.0)):.1f}%")
    lines.append(f"**Model kind:** {meta.get('model_kind', 'N/A')}")
    lines.append("")
    lines.append("## Confidence")
    lines.append(f"- **Confidence:** {summary.get('confidence', 'N/A')}")
    lines.append("- Confidence is a heuristic based on model availability and signal coverage (not a statistical guarantee).")
    lines.append("")

    lines.append("## Immediate actions")
    for a in summary.get("immediate_actions", []):
        lines.append(f"- {a}")
    lines.append("")

    lines.append("## Why this score (key drivers)")
    for r in summary.get("context_reasons", []):
        lines.append(f"- {r}")
    lines.append("")

    lines.append("## Overlay breakdown")
    overlays = meta.get("overlays", {}) or {}
    lines.append(f"- Map adjustment: {float(overlays.get('map', 0.0)):+.1f}")
    lines.append(f"- News/Reports: {float(overlays.get('events', 0.0)):+.1f}")
    lines.append(f"- USGS earthquakes: {float(overlays.get('usgs', 0.0)):+.1f}")
    lines.append(f"- Weather (Open-Meteo): {float(overlays.get('weather', 0.0)):+.1f}")
    lines.append(f"- NASA EONET hazards: {float(overlays.get('eonet', 0.0)):+.1f}")
    lines.append("")

    lines.append("## Evidence (readable snapshot)")
    map_sig = evidence.get("map_signals") or {}
    if map_sig:
        lines.append("### Map signals (OpenStreetMap/Overpass)")
        for k, v in map_sig.items():
            lines.append(f"- {k}: {v}")
        lines.append("")
    else:
        lines.append("- Map signals not available.")
        lines.append("")

    meteo = evidence.get("meteo_summary") or {}
    if meteo:
        lines.append("### Weather snapshot (Open-Meteo, UTC)")
        for k, v in meteo.items():
            lines.append(f"- {k}: {v}")
        lines.append("")

    # ---- Evidence links (clickable)
    lines.append("## Evidence links (sources)")
    links_added = False

    def _add_links_from_df(df: Any, label: str, title_col: str = "title", url_col: str = "url", max_items: int = 12):
        nonlocal links_added
        try:
            if df is None or not isinstance(df, pd.DataFrame) or df.empty:
                lines.append(f"- {label}: _No items available._")
                return

            lines.append(f"### {label}")
            n = 0
            for _, r in df.iterrows():
                if n >= max_items:
                    break
                t = str(r.get(title_col, "") or "").strip()
                u = str(r.get(url_col, "") or "").strip()
                if not u:
                    continue
                if not t:
                    t = u
                if len(t) > 120:
                    t = t[:117] + "..."
                lines.append(f"- [{t}]({u})")
                n += 1

            if n == 0:
                lines.append("- _No valid links found in feed._")
            links_added = True
            lines.append("")
        except Exception:
            lines.append(f"- {label}: _Could not render links._")

    gdelt_df = evidence.get("gdelt_df")
    relief_df = evidence.get("relief_df")
    usgs_df = evidence.get("usgs_df")
    eonet_df = evidence.get("eonet_df")

    _add_links_from_df(gdelt_df, "GDELT (news)", title_col="title", url_col="url", max_items=12)
    _add_links_from_df(relief_df, "ReliefWeb (disaster reports)", title_col="title", url_col="url", max_items=12)
    _add_links_from_df(usgs_df, "USGS (earthquakes)", title_col="place", url_col="url", max_items=12)
    _add_links_from_df(eonet_df, "NASA EONET (hazards)", title_col="title", url_col="url", max_items=12)

    if not links_added:
        lines.append("_No evidence links available. Try refreshing feeds in the app._")
    lines.append("")

    lines.append("## Detailed plan")
    for it in items:
        lines.append(f"### {it.get('title', 'Recommendation')}")
        lines.append("")
        lines.append("**Why it matters**")
        for w in it.get("why", []):
            lines.append(f"- {w}")
        lines.append("")
        lines.append("**Data gathered**")
        for d in it.get("data_gathered", []):
            lines.append(f"- {d}")
        lines.append("")
        lines.append("**Steps (recommended sequence)**")
        for i, step in enumerate(it.get("steps", []), start=1):
            lines.append(f"{i}. {step}")
        lines.append("")
        lines.append("**Deliverables (client-ready)**")
        for d in it.get("deliverables", []):
            lines.append(f"- {d}")
        lines.append("")

    lines.append("## Limitations")
    for l in limitations:
        lines.append(f"- {l}")
    lines.append("")

    lines.append("---")
    lines.append("Generated using local model inference and free public sources (OSM/Overpass, GDELT, ReliefWeb, USGS, Open-Meteo, NASA EONET).")
    return "\n".join(lines)
