import os
import time
import math
import json
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

# FREE hazard feeds
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
# Geo distance helper
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
# Offline GeoJSON signals (UPLOAD)
# -----------------------------
def parse_geojson_bytes(file_bytes: bytes) -> Tuple[Optional[dict], str]:
    try:
        txt = file_bytes.decode("utf-8", errors="ignore")
        j = json.loads(txt)
        if not isinstance(j, dict) or j.get("type") not in ("FeatureCollection", "Feature"):
            return None, "Invalid GeoJSON: expected FeatureCollection."
        return j, "OK"
    except Exception as e:
        return None, f"Invalid GeoJSON: {e}"


def _iter_features(geojson: dict) -> List[dict]:
    if geojson.get("type") == "Feature":
        return [geojson]
    feats = geojson.get("features") or []
    return feats if isinstance(feats, list) else []


def _coords_from_geometry(geom: dict) -> List[Tuple[float, float]]:
    """
    Returns list of (lat, lon) points extracted from Point/LineString/MultiLineString/MultiPoint.
    """
    if not geom or not isinstance(geom, dict):
        return []
    gtype = geom.get("type")
    coords = geom.get("coordinates")

    pts: List[Tuple[float, float]] = []

    def add_point(c):
        try:
            lon, lat = float(c[0]), float(c[1])
            pts.append((lat, lon))
        except Exception:
            pass

    if gtype == "Point" and isinstance(coords, list) and len(coords) >= 2:
        add_point(coords)

    elif gtype == "MultiPoint" and isinstance(coords, list):
        for c in coords:
            if isinstance(c, list) and len(c) >= 2:
                add_point(c)

    elif gtype == "LineString" and isinstance(coords, list):
        for c in coords:
            if isinstance(c, list) and len(c) >= 2:
                add_point(c)

    elif gtype == "MultiLineString" and isinstance(coords, list):
        for line in coords:
            if isinstance(line, list):
                for c in line:
                    if isinstance(c, list) and len(c) >= 2:
                        add_point(c)

    return pts


def offline_signals_within_radius(
    roads_geojson: Optional[dict],
    pois_geojson: Optional[dict],
    lat: float,
    lon: float,
    radius_m: int,
) -> Dict[str, Any]:
    """
    Counts features within radius using vertex/point proximity (lightweight approximation).
    """
    radius_km = float(radius_m) / 1000.0

    out = {
        "radius_m": int(radius_m),
        "roads_features_near": 0,
        "poi_points_near": 0,
        "hospitals_near": 0,
        "schools_near": 0,
        "clinics_near": 0,
        "power_near": 0,
        "water_near": 0,
        "emergency_near": 0,
    }

    # Roads: count a road feature if ANY vertex falls within radius
    if roads_geojson:
        for f in _iter_features(roads_geojson):
            geom = (f.get("geometry") or {})
            pts = _coords_from_geometry(geom)
            hit = False
            for (plat, plon) in pts:
                if haversine_km(lat, lon, plat, plon) <= radius_km:
                    hit = True
                    break
            if hit:
                out["roads_features_near"] += 1

    # POIs: count points within radius; classify by properties if available
    if pois_geojson:
        for f in _iter_features(pois_geojson):
            geom = (f.get("geometry") or {})
            pts = _coords_from_geometry(geom)
            if not pts:
                continue
            # Use first point for POI distance
            plat, plon = pts[0]
            if haversine_km(lat, lon, plat, plon) <= radius_km:
                out["poi_points_near"] += 1
                props = f.get("properties") or {}
                # Try common OSM tags: amenity, power, man_made, emergency
                amenity = str(props.get("amenity", "")).lower()
                man_made = str(props.get("man_made", "")).lower()
                power = str(props.get("power", "")).lower()
                emergency = str(props.get("emergency", "")).lower()

                if amenity == "hospital":
                    out["hospitals_near"] += 1
                elif amenity == "school":
                    out["schools_near"] += 1
                elif amenity == "clinic":
                    out["clinics_near"] += 1

                if power:
                    out["power_near"] += 1
                if man_made == "water_tower":
                    out["water_near"] += 1
                if emergency:
                    out["emergency_near"] += 1

    # Clamp a bit for sanity
    out["roads_features_near"] = int(min(out["roads_features_near"], 50000))
    out["poi_points_near"] = int(min(out["poi_points_near"], 50000))
    return out


def offline_overlay_points(offline_sig: Dict[str, Any]) -> float:
    """
    0..10 overlay points from offline map density (lower density => higher overlay).
    This is an "environment coverage" overlay, not a hazard overlay.
    """
    if not offline_sig:
        return 0.0

    roads = float(offline_sig.get("roads_features_near", 0))
    pois = float(offline_sig.get("poi_points_near", 0))

    # Normalize around typical values
    roads_norm = min(roads, 2000.0) / 2000.0
    pois_norm = min(pois, 300.0) / 300.0
    density = 0.7 * roads_norm + 0.3 * pois_norm

    # If density is low -> add overlay (up to 10). If high -> near 0.
    pts = (0.45 - density) * 22.0
    return clamp(pts, 0.0, 10.0)


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
            rows.append({
                "time": props.get("time"),
                "place": props.get("place"),
                "mag": props.get("mag"),
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


# -----------------------------
# Evidence + confidence + limitations
# -----------------------------
def compute_confidence(
    model_kind: str,
    base_need: float,
    overlays: Dict[str, float],
    has_location: bool,
    has_offline_map: bool,
) -> Tuple[float, List[str]]:
    """
    Returns (confidence_0_1, limitations[])
    Confidence is a heuristic meant for readability, not a statistical guarantee.
    """
    limitations: List[str] = []

    # Base confidence by model type
    if model_kind == "onnx":
        conf = 0.78
    elif model_kind == "joblib":
        conf = 0.72
        limitations.append("Joblib models can be sensitive to environment/version differences on cloud.")
    else:
        conf = 0.60
        limitations.append("Fallback scoring was used (heuristic), not a trained model output.")

    # Signal availability
    sig_strength = 0.0
    for k, v in (overlays or {}).items():
        if abs(float(v)) > 0.01:
            sig_strength += 1.0

    conf += min(0.10, sig_strength * 0.02)

    if not has_location:
        limitations.append("No map location selected, so hazard overlays (USGS/Open-Meteo/EONET) are zero.")

    if not has_offline_map:
        limitations.append("Offline map overlay is not included unless GeoJSON files are uploaded.")

    # Score extremes can be less stable (heuristic)
    if base_need < 10 or base_need > 90:
        limitations.append("Very low/high scores can be less stable without ground-truth local asset condition data.")
        conf -= 0.03

    # Clamp
    conf = clamp(conf, 0.45, 0.90)

    # General limitations
    limitations.append("This is a decision-support signal; it does not replace engineering inspection or official warnings.")
    limitations.append("Live feeds may be rate-limited; missing items reduce coverage but the app still runs.")

    # De-duplicate
    seen = set()
    clean = []
    for x in limitations:
        if x not in seen:
            seen.add(x)
            clean.append(x)

    return conf, clean


def build_evidence_links(
    gdelt_df: Optional[pd.DataFrame],
    relief_df: Optional[pd.DataFrame],
    usgs_df: Optional[pd.DataFrame],
    eonet_df: Optional[pd.DataFrame],
    max_items: int = 6,
) -> Dict[str, List[Dict[str, str]]]:
    def take_links(df: Optional[pd.DataFrame], title_col: str, url_col: str, date_col: str) -> List[Dict[str, str]]:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return []
        out = []
        for _, r in df.head(max_items).iterrows():
            t = str(r.get(title_col, "")).strip()
            u = str(r.get(url_col, "")).strip()
            d = str(r.get(date_col, "")).strip()
            if u:
                out.append({"title": t[:180], "url": u, "date": d[:60]})
        return out

    return {
        "GDELT": take_links(gdelt_df, "title", "url", "seendate"),
        "ReliefWeb": take_links(relief_df, "title", "url", "date"),
        "USGS": take_links(usgs_df, "place", "url", "time"),
        "EONET": take_links(eonet_df, "title", "url", "date"),
    }


# -----------------------------
# Detailed recommendations + report markdown
# -----------------------------
def recommendations_detailed(
    *,
    base_need: float,
    overlays: Dict[str, float],
    inputs: Dict[str, Any],
    map_signals: Optional[Dict[str, Any]] = None,
    offline_signals: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    total_overlay = sum(float(v) for v in (overlays or {}).values())
    combined = clamp(base_need + total_overlay, 0, 100)

    risk, _ = risk_bucket(combined)

    reasons = []
    gdp = safe_float(inputs.get("GDP_per_capita_USD"))
    hdi = safe_float(inputs.get("HDI_Index"))
    urb = safe_float(inputs.get("Urbanization_Rate"))
    pop = safe_float(inputs.get("Population_Millions"))

    if not np.isnan(gdp) and gdp < 5000:
        reasons.append("Low GDP per capita increases constraints on asset maintenance and expansion.")
    if not np.isnan(hdi) and hdi < 0.70:
        reasons.append("Lower HDI often correlates with gaps in basic services and resilience capacity.")
    if not np.isnan(urb) and urb < 50 and not np.isnan(pop) and pop > 50:
        reasons.append("Large population with low urbanization can increase strain on distributed infrastructure delivery.")

    # Overlay drivers (readable)
    ov_reasons = []
    for k, v in (overlays or {}).items():
        v = float(v)
        if abs(v) < 0.01:
            continue
        label = {
            "map": "OpenStreetMap live signals",
            "events": "News & disaster reports",
            "usgs": "Earthquake activity (USGS)",
            "weather": "Weather stress (Open-Meteo)",
            "eonet": "Active hazards (NASA EONET)",
            "offline_map": "Offline map environment signals",
        }.get(k, k)
        if v > 0:
            ov_reasons.append(f"{label} increased risk by +{v:0.1f} points.")
        else:
            ov_reasons.append(f"{label} reduced risk by {v:0.1f} points.")

    # Plan items (client-ready)
    items: List[Dict[str, Any]] = []

    def add_item(title: str, why: List[str], data_gathered: List[str], steps: List[str], deliverables: List[str]):
        items.append({
            "title": title,
            "why": why,
            "data_gathered": data_gathered,
            "steps": steps,
            "deliverables": deliverables,
        })

    # Common data gathered
    dg_common = [
        "Country-level indicators (Population, GDP per capita, HDI, Urbanization).",
        "Risk overlays from free feeds (news/reports, hazards, weather) when available.",
    ]
    if map_signals:
        dg_common.append("Live local infrastructure counts from OpenStreetMap Overpass around selected point.")
    if offline_signals:
        dg_common.append("Uploaded offline map (GeoJSON) feature density around selected point.")

    if combined >= 70:
        add_item(
            "Critical infrastructure rapid assessment (0–7 days)",
            [
                "High combined risk suggests priority exposure of essential services (health, power, transport, water).",
                "Early detection prevents cascading failures (e.g., power outage → water pumping failure).",
            ],
            dg_common + [
                "Overlay drivers: " + ("; ".join(ov_reasons) if ov_reasons else "No active overlays applied."),
            ],
            [
                "Define the critical asset list (bridges, substations, water plants, hospitals, evacuation routes).",
                "Run a quick condition scoring: access, structural signs, outages, bottlenecks, spare parts.",
                "Assign owners and response SLAs for each asset category (power/water/roads/health).",
                "Set a 7-day action board with daily status checks and escalation rules.",
            ],
            [
                "Critical asset inventory (CSV/Sheet) with condition score and owner.",
                "7-day response plan with SLAs and escalation matrix.",
                "Map of critical routes + chokepoints (for logistics planning).",
            ],
        )
        add_item(
            "Emergency maintenance & continuity planning (0–30 days)",
            [
                "When hazard/news overlays are elevated, continuity planning reduces downtime and losses.",
                "Contractor readiness + spare stock often determines recovery speed.",
            ],
            dg_common,
            [
                "Pre-negotiate contractor call-outs and define scope templates (roads, drainage, power).",
                "Create spare parts & fuel lists (generators, pumps, transformers, bridge components).",
                "Validate backup power availability for healthcare/water pumping sites.",
                "Run a tabletop exercise for flood/wind/quake scenarios with communications plan.",
            ],
            [
                "Continuity plan (PDF/MD) with contact tree and contractor scope templates.",
                "Spare parts & resource checklist with quantities and storage points.",
                "Scenario playbook (flood/wind/earthquake) with triggers and actions.",
            ],
        )
    elif combined >= 40:
        add_item(
            "Targeted resilience upgrades (30–180 days)",
            [
                "Medium risk benefits most from targeted upgrades rather than full rebuilds.",
                "Reducing single points of failure improves service reliability.",
            ],
            dg_common + [
                "Focus areas: drainage, road bottlenecks, distribution reliability, healthcare access.",
            ],
            [
                "Rank assets by criticality + exposure (population served, hazard proximity, redundancy).",
                "Prioritize 10–20 quick wins: drainage cleaning, culvert expansion, slope protection, backup power.",
                "Schedule periodic inspections and preventive maintenance with reporting templates.",
                "Introduce low-cost monitoring: checklists + photo logs + monthly review.",
            ],
            [
                "Prioritized backlog (Top 20) with cost bands and expected impact.",
                "Maintenance calendar and inspection templates.",
                "Resilience quick-win package list ready for procurement.",
            ],
        )
    else:
        add_item(
            "Preventive maintenance & monitoring (ongoing)",
            [
                "Low combined risk still requires maintenance to avoid silent degradation.",
                "Small improvements now prevent major rehab later.",
            ],
            dg_common,
            [
                "Maintain a basic asset registry (location, age, last inspection, condition score).",
                "Implement routine inspections for bridges, drainage, substations, key roads.",
                "Create incident reporting for outages and road closures (simple form + log).",
                "Review seasonal weather readiness and update response SOPs.",
            ],
            [
                "Asset registry (CSV/Sheet) with inspection history.",
                "Maintenance SOPs and monthly reporting dashboard.",
                "Seasonal preparedness checklist and response SOP document.",
            ],
        )

    # Summary block
    summary = {
        "risk_level": risk,
        "combined_need": float(combined),
        "base_need": float(base_need),
        "overlay_total": float(total_overlay),
        "context_reasons": reasons + ov_reasons,
        "immediate_actions": [
            "Select a map location and refresh overlays (USGS/Open-Meteo/EONET/news) for best coverage.",
            "Download the client-ready report and use it as an action tracker.",
        ],
    }

    return {"summary": summary, "items": items}


def render_plan_markdown(plan: Dict[str, Any]) -> str:
    s = plan.get("summary", {}) or {}
    items = plan.get("items", []) or []

    lines: List[str] = []
    lines.append("# Infrastructure Risk Report")
    lines.append("")
    lines.append(f"**Risk level:** {s.get('risk_level', 'N/A')}")
    lines.append(f"**Base need:** {s.get('base_need', 0):0.1f}%")
    lines.append(f"**Overlay total:** {s.get('overlay_total', 0):+0.1f}")
    lines.append(f"**Combined need:** {s.get('combined_need', 0):0.1f}%")
    lines.append("")

    lines.append("## Evidence (readable)")
    for r in (s.get("context_reasons") or [])[:12]:
        lines.append(f"- {r}")
    lines.append("")

    lines.append("## Immediate actions")
    for a in (s.get("immediate_actions") or []):
        lines.append(f"- {a}")
    lines.append("")

    lines.append("## Detailed plan")
    for it in items:
        lines.append(f"### {it.get('title','Recommendation')}")
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

    lines.append("---")
    lines.append("Generated by the app using free sources and local model inference.")
    return "\n".join(lines)


def build_full_report_markdown(
    *,
    country_label: Optional[str],
    inputs: Dict[str, Any],
    model_kind: str,
    base_need: float,
    overlays: Dict[str, float],
    map_signals: Optional[Dict[str, Any]],
    offline_signals: Optional[Dict[str, Any]],
    gdelt_df: Optional[pd.DataFrame],
    relief_df: Optional[pd.DataFrame],
    usgs_df: Optional[pd.DataFrame],
    eonet_df: Optional[pd.DataFrame],
    meteo_summary: Optional[Dict[str, Any]],
    confidence: float,
    limitations: List[str],
) -> str:
    combined = clamp(base_need + sum(float(v) for v in (overlays or {}).values()), 0, 100)
    risk, _ = risk_bucket(combined)

    links = build_evidence_links(gdelt_df, relief_df, usgs_df, eonet_df)

    lines: List[str] = []
    lines.append("# Infrastructure Risk Brief")
    lines.append("")
    lines.append(f"**Risk level:** {risk}")
    lines.append(f"**Combined need:** {combined:0.1f}%")
    lines.append(f"**Base need:** {base_need:0.1f}% _(model: {model_kind})_")
    lines.append(f"**Confidence (heuristic):** {confidence*100:0.0f}%")
    lines.append("")

    if country_label:
        lines.append(f"**Context:** Country code `{country_label}`")
        lines.append("")

    lines.append("## Inputs")
    for k in FEATURES:
        lines.append(f"- {k}: {inputs.get(k)}")
    lines.append("")

    lines.append("## Overlay breakdown")
    for k, v in (overlays or {}).items():
        lines.append(f"- {k}: {float(v):+0.1f}")
    lines.append("")

    if meteo_summary:
        lines.append("## Weather snapshot (Open-Meteo)")
        for k, v in meteo_summary.items():
            lines.append(f"- {k}: {v}")
        lines.append("")

    if map_signals:
        lines.append("## Live map signals (Overpass)")
        for k, v in map_signals.items():
            lines.append(f"- {k}: {v}")
        lines.append("")

    if offline_signals:
        lines.append("## Offline map signals (Uploaded GeoJSON)")
        for k, v in offline_signals.items():
            lines.append(f"- {k}: {v}")
        lines.append("")

    lines.append("## Evidence links")
    for src, arr in links.items():
        if not arr:
            continue
        lines.append(f"### {src}")
        for it in arr:
            t = it.get("title", "").strip()
            u = it.get("url", "").strip()
            d = it.get("date", "").strip()
            if u:
                lines.append(f"- [{t}]({u}) ({d})")
            else:
                lines.append(f"- {t} ({d})")
        lines.append("")

    lines.append("## Limitations")
    for x in (limitations or [])[:12]:
        lines.append(f"- {x}")
    lines.append("")

    lines.append("---")
    lines.append("Generated by the app using free sources and local model inference.")
    return "\n".join(lines)
