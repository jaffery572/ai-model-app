# utils.py
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

            # Retry for temporary failures / rate limiting
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
# USGS earthquakes (free)
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
# Open-Meteo (free)
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
# NASA EONET (free)
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


# =============================================================================
# REPLACEMENT SECTION:
# Detailed precautions + new report generator
# =============================================================================

def _tier(base_need: float, total_overlay: float) -> str:
    total = clamp(base_need + total_overlay, 0, 100)
    if total >= 70:
        return "high"
    if total >= 40:
        return "medium"
    return "low"


def recommendations_detailed(
    *,
    base_need: float,
    overlays: Dict[str, float],
    inputs: Dict[str, Any],
    map_signals: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Returns a structured, client-ready plan:
      {
        "summary": { "immediate_actions": [...], "context_reasons":[...], "tier": "...", "combined": float },
        "items": [ {title, why[], data_gathered[], steps[], deliverables[]} ... ],
        "inputs": {...},
        "overlays": {...},
        "generated_at_utc": "..."
      }
    """
    total_overlay = float(sum([float(v) for v in (overlays or {}).values()]))
    combined = float(clamp(base_need + total_overlay, 0, 100))
    tier = _tier(base_need, total_overlay)

    pop = safe_float(inputs.get("Population_Millions"), np.nan)
    gdp = safe_float(inputs.get("GDP_per_capita_USD"), np.nan)
    hdi = safe_float(inputs.get("HDI_Index"), np.nan)
    urb = safe_float(inputs.get("Urbanization_Rate"), np.nan)

    reasons: List[str] = []
    if not math.isnan(gdp) and gdp < 5000:
        reasons.append("Low GDP per capita increases maintenance backlog risk and funding constraints.")
    if not math.isnan(hdi) and hdi < 0.65:
        reasons.append("Lower HDI often correlates with weaker service reliability and limited redundancy.")
    if not math.isnan(urb) and urb > 70:
        reasons.append("High urbanization increases demand load and cascade risk during disruptions.")
    if not math.isnan(pop) and pop > 100:
        reasons.append("Large population increases consequence severity and response complexity.")

    if map_signals:
        roads = map_signals.get("roads", 0)
        hos = map_signals.get("hospitals", 0)
        powc = map_signals.get("power", 0)
        wat = map_signals.get("water", 0)
        if roads == 0:
            reasons.append("Very low mapped road density near the selected point suggests access constraints.")
        if (hos or 0) == 0:
            reasons.append("No hospitals detected near the selected point (OSM), indicating limited nearby healthcare access.")
        if (powc or 0) == 0:
            reasons.append("No power infrastructure tags detected near the selected point (OSM), possible resilience concern.")
        if (wat or 0) == 0:
            reasons.append("No water-tower tags detected near the selected point (OSM), water storage visibility is limited.")

    # Overlay-driven context
    if overlays.get("events", 0) >= 6:
        reasons.append("Elevated news/disaster activity increases near-term disruption likelihood.")
    if overlays.get("usgs", 0) >= 3:
        reasons.append("Recent nearby seismic activity increases infrastructure stress risk.")
    if overlays.get("weather", 0) >= 2:
        reasons.append("Weather indicators suggest near-term wind/rain stress.")
    if overlays.get("eonet", 0) >= 2:
        reasons.append("Nearby active hazard events detected in NASA EONET.")

    immediate_actions: List[str] = []
    if tier == "high":
        immediate_actions = [
            "Run a 24–72 hour critical asset status check (power, water, bridges, hospitals).",
            "Establish an incident response contact list and confirm contractors/repair capacity.",
            "Verify backup power and fuel logistics for critical sites.",
            "Start daily situation reporting and define escalation triggers.",
        ]
    elif tier == "medium":
        immediate_actions = [
            "Prioritize top 10 highest-risk assets and schedule inspections.",
            "Confirm spare parts availability and maintenance response time targets.",
            "Review emergency procedures for storms/flooding and test communications.",
        ]
    else:
        immediate_actions = [
            "Maintain routine preventive maintenance and inspection cycles.",
            "Create/refresh asset inventory and basic condition scoring.",
            "Review seasonal preparedness checklist (storm, heat, heavy rain).",
        ]

    # Plan items (always provide 4 core items)
    items: List[Dict[str, Any]] = []

    # 1) Asset inventory + condition scoring
    items.append({
        "title": "Asset inventory and condition scoring (foundation)",
        "why": [
            "You cannot prioritize upgrades without knowing what assets exist and their condition.",
            "Condition scoring reduces wasted spending and improves budget justification.",
        ],
        "data_gathered": [
            "Inputs: GDP/HDI/urbanization/population indicators (macro risk drivers).",
            "Map signals: nearby roads/hospitals/power/water tags (local context, if available).",
            "Hazard overlays: USGS/Open-Meteo/EONET (near-term risk signals).",
        ],
        "steps": [
            "List critical assets by sector: roads/bridges, power substations, water plants, hospitals, telecom nodes.",
            "Assign a simple condition score (1–5) using inspection notes, age, failures, and visible defects.",
            "Add criticality score (1–5): impact on population and cascading failures if down.",
            "Compute priority = condition_weight + criticality_weight + hazard_weight, then rank top assets.",
            "Store in a CSV and update monthly; keep evidence (photos, inspection forms).",
        ],
        "deliverables": [
            "Asset register (CSV) with location, owner, condition score, criticality score, priority rank.",
            "Top-risk list with justification notes.",
            "Maintenance backlog list with cost/effort estimates (rough order).",
        ],
    })

    # 2) Preventive maintenance + inspections
    items.append({
        "title": "Preventive maintenance and inspection program (reduce failures)",
        "why": [
            "Most outages come from deferred maintenance and undetected defects.",
            "Structured inspections reduce emergency repair cost and downtime.",
        ],
        "data_gathered": [
            "Local access/coverage from OSM signals (roads/services density).",
            "Near-term triggers from weather (wind/rain) and hazard overlays.",
        ],
        "steps": [
            "Define inspection frequencies: critical monthly/quarterly; non-critical semiannual/annual.",
            "Create checklists per asset type (bridge deck, drainage, pumps, transformers, generators).",
            "Set thresholds for action (e.g., corrosion level, vibration, leakage, repeated outages).",
            "Log all work orders and failures; compute MTBF and repeat-failure hotspots.",
            "Introduce quick fixes first: drainage clearing, vegetation control, sealing leaks, spare parts staging.",
        ],
        "deliverables": [
            "Inspection calendar + checklists (PDF/Docs).",
            "Work-order log template + KPI sheet (downtime, repeat failures, response time).",
            "Monthly maintenance report summary for stakeholders.",
        ],
    })

    # 3) Resilience upgrades + redundancy
    items.append({
        "title": "Resilience upgrades and redundancy (limit cascading failures)",
        "why": [
            "Redundancy prevents single-point failures from collapsing multiple services.",
            "Upgrades targeted at critical nodes yield the highest ROI under constraints.",
        ],
        "data_gathered": [
            "Macro constraints implied by GDP/HDI (budget/funding reality).",
            "Hazard overlays for likely stressors (seismic, storms, flooding, wildfire).",
        ],
        "steps": [
            "Identify single points of failure (one substation, one water main, one bridge access route).",
            "Implement low-cost redundancy: alternate routing, backup generators, spare pumps/valves.",
            "Harden critical sites: elevation of equipment, waterproofing, surge protection, fire breaks.",
            "Define minimum service levels (e.g., hospitals require 48–72h backup power).",
            "Plan phased upgrades: Phase 1 quick wins; Phase 2 medium capex; Phase 3 long-term modernization.",
        ],
        "deliverables": [
            "Single-point-of-failure map/list and mitigation plan.",
            "Resilience upgrade roadmap (phased) with estimated effort and expected impact.",
            "Standard operating procedures for backup activation (power/water/IT).",
        ],
    })

    # 4) Incident response + triggers
    items.append({
        "title": "Incident response plan with triggers (fast coordinated action)",
        "why": [
            "During floods/storms/seismic events, speed and coordination reduce damage and downtime.",
            "Trigger-based actions prevent delayed decision making.",
        ],
        "data_gathered": [
            "News/disaster activity (GDELT/ReliefWeb) as situational awareness input.",
            "Weather and hazard overlays for early warning.",
        ],
        "steps": [
            "Define triggers: rainfall/wind thresholds, quake magnitude nearby, active hazard proximity, repeated outage counts.",
            "Assign roles: incident lead, utilities lead, transport lead, communications lead, logistics lead.",
            "Prepare contact lists: contractors, spare parts vendors, fuel suppliers, emergency services.",
            "Create a 1-page playbook: what to check first, how to isolate faults, how to communicate status.",
            "Run a tabletop drill monthly; update the plan based on lessons learned.",
        ],
        "deliverables": [
            "Incident Response Plan (IRP) + trigger matrix.",
            "Contact list + escalation tree.",
            "Daily/weekly situation report template.",
        ],
    })

    # Adjust emphasis by tier (add more urgency text)
    if tier == "high":
        items.insert(0, {
            "title": "Rapid critical infrastructure assessment (0–72 hours)",
            "why": [
                "High combined risk suggests elevated probability of disruption or existing fragility.",
                "Early detection prevents secondary failures (power→water→healthcare).",
            ],
            "data_gathered": [
                "Combined risk level derived from model output + overlays (map/news/hazards).",
                "Local infrastructure density and hazard signals.",
            ],
            "steps": [
                "Inspect: bridges/culverts/drainage, substations, water pumps, hospitals/clinics backup systems.",
                "Check access routes for blockage risk; identify alternate routes.",
                "Verify generator readiness, fuel stock, spare parts, and on-call repair teams.",
                "Create a red/amber/green status board and update twice daily during the risk window.",
            ],
            "deliverables": [
                "Critical asset status board (RAG) with owners and next actions.",
                "Immediate repair list (top 5) and resource assignment.",
            ],
        })

    plan = {
        "summary": {
            "tier": tier,
            "combined_need": round(combined, 1),
            "base_need": round(float(base_need), 1),
            "total_overlay": round(float(total_overlay), 1),
            "immediate_actions": immediate_actions,
            "context_reasons": reasons[:10],
        },
        "items": items,
        "inputs": {k: inputs.get(k) for k in FEATURES},
        "overlays": overlays,
        "generated_at_utc": datetime_utc_str(),
    }
    return plan


def datetime_utc_str() -> str:
    # avoid importing datetime at top to keep file minimal
    import datetime as _dt
    return _dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")


def render_plan_markdown(plan: Dict[str, Any]) -> str:
    """
    Convert plan dict into a client-ready Markdown report.
    """
    if not plan:
        return "# Infrastructure Risk Brief\n\nNo plan data available.\n"

    summary = plan.get("summary", {}) or {}
    items = plan.get("items", []) or []
    inputs = plan.get("inputs", {}) or {}
    overlays = plan.get("overlays", {}) or {}

    tier = str(summary.get("tier", "unknown")).upper()
    combined = summary.get("combined_need", "n/a")
    base_need = summary.get("base_need", "n/a")
    total_overlay = summary.get("total_overlay", "n/a")

    lines: List[str] = []
    lines.append("# Infrastructure Risk Brief")
    lines.append("")
    lines.append(f"**Risk tier:** {tier}")
    lines.append(f"**Combined need:** {combined}%")
    lines.append(f"**Base need (model):** {base_need}%")
    lines.append(f"**Total overlay points:** {total_overlay}")
    lines.append(f"**Generated:** {plan.get('generated_at_utc', '')}")
    lines.append("")

    lines.append("## Overlay breakdown")
    for k in ["map", "events", "usgs", "weather", "eonet"]:
        if k in overlays:
            lines.append(f"- {k}: {overlays.get(k):+0.1f}")
    lines.append("")

    lines.append("## Input indicators")
    for k in FEATURES:
        lines.append(f"- {k}: {inputs.get(k)}")
    lines.append("")

    ia = summary.get("immediate_actions", []) or []
    if ia:
        lines.append("## Immediate actions")
        for x in ia:
            lines.append(f"- {x}")
        lines.append("")

    cr = summary.get("context_reasons", []) or []
    if cr:
        lines.append("## Key drivers (why this score)")
        for x in cr:
            lines.append(f"- {x}")
        lines.append("")

    lines.append("## Detailed precautions & prevention plan")
    for item in items:
        lines.append(f"### {item.get('title','Recommendation')}")
        lines.append("")
        lines.append("**Why it matters**")
        for w in item.get("why", []) or []:
            lines.append(f"- {w}")
        lines.append("")
        lines.append("**Data gathered**")
        for d in item.get("data_gathered", []) or []:
            lines.append(f"- {d}")
        lines.append("")
        lines.append("**Steps (recommended sequence)**")
        for i, s in enumerate(item.get("steps", []) or [], start=1):
            lines.append(f"{i}. {s}")
        lines.append("")
        lines.append("**Deliverables**")
        for d in item.get("deliverables", []) or []:
            lines.append(f"- {d}")
        lines.append("")

    lines.append("---")
    lines.append("This document is generated from the application's indicators and free public data feeds.")
    return "\n".join(lines)
