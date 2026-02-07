# app.py
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional

import pandas as pd
import streamlit as st
import plotly.express as px

import folium
from streamlit_folium import st_folium

from utils import (
    FEATURES,
    MODEL_ONNX_PATH,
    MODEL_JOBLIB_PATH,
    ensure_schema,
    to_feature_row,
    risk_bucket,
    clamp,
    model_predict_need,
    fetch_overpass_signals,
    map_context_adjustment,
    fetch_gdelt,
    fetch_reliefweb,
    disaster_overlay_score,
    fetch_usgs_earthquakes_near,
    usgs_overlay_points,
    fetch_open_meteo,
    open_meteo_overlay_points,
    fetch_eonet_events_near,
    eonet_overlay_points,
    # Offline map
    parse_geojson_bytes,
    offline_signals_within_radius,
    offline_overlay_points,
    # Plan + report
    recommendations_detailed,
    render_plan_markdown,
    compute_confidence,
    build_evidence_links,
    build_full_report_markdown,
)

st.set_page_config(page_title="Global Infrastructure AI", page_icon="🌍", layout="wide", initial_sidebar_state="expanded")

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

DEFAULT_RADIUS_M = 5000


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
            {"Population_Millions": 200, "GDP_per_capita_USD": 2500, "HDI_Index": 0.62, "Urbanization_Rate": 38},
        ]
    )
    return df.to_csv(index=False).encode("utf-8")


def render_header():
    st.title("🌍 Global Infrastructure AI")
    st.caption("Local model inference + free live signals (OSM, USGS, Open-Meteo, EONET, GDELT, ReliefWeb) + Offline GeoJSON environment overlay.")
    st.markdown("<div class='small-muted'>If a feed is rate-limited, the app shows a message and continues.</div>", unsafe_allow_html=True)


def render_sidebar() -> Dict[str, Any]:
    with st.sidebar:
        st.subheader("Settings")
        show_debug = st.toggle("Show debug info", value=False)

        st.markdown("---")
        st.write("Model files expected:")
        st.code("models/infra_model.onnx\nmodels/infra_model.joblib (optional)", language="text")

        radius_m = st.slider("Analysis radius (meters)", 1000, 20000, DEFAULT_RADIUS_M, step=500)
        if radius_m > 10000:
            st.warning("Large radius can timeout. If signals fail, use 3000–7000m.")

        st.markdown("---")
        st.write("Sample CSV (Upload CSV):")
        st.download_button("Download sample.csv", data=sample_csv_bytes(), file_name="sample.csv", mime="text/csv", use_container_width=True)

        st.markdown("---")
        st.write("Free sources used:")
        st.write("- OpenStreetMap (Overpass)")
        st.write("- USGS Earthquakes")
        st.write("- Open-Meteo Weather")
        st.write("- NASA EONET Hazards")
        st.write("- GDELT News")
        st.write("- ReliefWeb Reports")
        st.write("- Offline GeoJSON uploads (roads/POIs)")

        return {"show_debug": show_debug, "radius_m": int(radius_m)}


def render_dashboard(df: pd.DataFrame):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"<div class='card'><h4>Countries</h4><h2>{len(df)}</h2></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='card'><h4>Avg GDP/capita</h4><h2>${df['GDP_per_capita_USD'].mean():,.0f}</h2></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='card'><h4>Avg HDI</h4><h2>{df['HDI_Index'].mean():0.3f}</h2></div>", unsafe_allow_html=True)
    with c4:
        st.markdown(f"<div class='card'><h4>Updated</h4><h2>{datetime.utcnow().strftime('%Y-%m-%d')}</h2></div>", unsafe_allow_html=True)

    st.markdown("---")
    colA, colB = st.columns(2)
    with colA:
        top = df.sort_values("GDP_per_capita_USD", ascending=False).head(12)
        st.plotly_chart(px.bar(top, x="Country", y="GDP_per_capita_USD", title="Top GDP per capita (sample)"), use_container_width=True)
    with colB:
        st.plotly_chart(
            px.scatter(df, x="GDP_per_capita_USD", y="HDI_Index", size="Population_Millions", hover_name="Country", title="GDP vs HDI (sample)", log_x=True),
            use_container_width=True,
        )

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
            row = {k: float(r[k]) for k in FEATURES}

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
                try:
                    df_up = pd.read_csv(up)
                    ok, msg = ensure_schema(df_up)
                    if not ok:
                        st.error(msg)
                    else:
                        st.success(f"Loaded {len(df_up):,} rows. Click Predict to run bulk predictions.")
                        st.session_state["uploaded_df"] = df_up
                except Exception as e:
                    st.error(f"Could not read CSV: {e}")

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
                        x = to_feature_row({k: float(r[k]) for k in FEATURES})
                        need, mk = model_predict_need(x)
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
                    need, mk = model_predict_need(x)
                    st.session_state["single_pred"] = {"need": need, "model_kind": mk, "input": row, "country": country_label}

        st.markdown("</div>", unsafe_allow_html=True)

    if "single_pred" in st.session_state:
        pred = st.session_state["single_pred"]
        need = float(pred["need"])
        badge_text, badge_cls = risk_bucket(need)

        st.markdown("---")
        st.subheader("Result")
        st.markdown("#### Need meter (stable)")
        st.progress(int(clamp(need, 0, 100)))

        st.markdown(
            f"<span class='badge {badge_cls}'>{badge_text}</span> &nbsp; "
            f"<b>Infrastructure Need</b>: <span style='font-size:1.8rem'>{need:0.1f}%</span>"
            f"<div class='small-muted'>Model used: {pred['model_kind']}</div>",
            unsafe_allow_html=True,
        )

        st.plotly_chart(
            px.bar(pd.DataFrame({"Metric": ["Need", "Sufficient"], "Value": [need, 100 - need]}), x="Metric", y="Value", title="Need vs Sufficient"),
            use_container_width=True,
        )

    if "bulk_pred_df" in st.session_state:
        st.markdown("---")
        st.subheader("Bulk results")
        out = st.session_state["bulk_pred_df"]
        st.dataframe(out, use_container_width=True, height=420)
        st.download_button("Download predictions CSV", data=out.to_csv(index=False).encode("utf-8"), file_name="predictions.csv", mime="text/csv", use_container_width=True)


def render_map_and_location(cfg: Dict[str, Any]):
    st.subheader("Map signals (OpenStreetMap)")
    st.caption("Click the map to set analysis location. Location is used by hazard overlays and offline map overlay.")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        m = folium.Map(location=[20, 0], zoom_start=2, tiles="OpenStreetMap")
        folium.LatLngPopup().add_to(m)
        map_data = st_folium(m, height=420, use_container_width=True)

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Click on map to set analysis location")

        lat, lon = None, None
        if map_data and map_data.get("last_clicked"):
            lat = map_data["last_clicked"]["lat"]
            lon = map_data["last_clicked"]["lng"]

        if lat is None:
            st.info("Click on the map first.")
        else:
            st.session_state["analysis_lat"] = float(lat)
            st.session_state["analysis_lon"] = float(lon)

            st.write(f"Selected point: `{lat:.5f}, {lon:.5f}`")

            radius_m = int(cfg["radius_m"])
            with st.spinner("Fetching live map signals (Overpass)..."):
                sig, msg, endpoint = fetch_overpass_signals(lat, lon, radius_m)
                if sig is None:
                    st.warning(msg)
                else:
                    st.session_state["map_signals"] = sig
                    st.success(msg)
                    if endpoint:
                        st.caption("Endpoint used:")
                        st.code(endpoint, language="text")

        st.markdown("</div>", unsafe_allow_html=True)

    if "map_signals" in st.session_state:
        sig = st.session_state["map_signals"]
        st.markdown("---")
        st.subheader("Live signals")
        st.json(sig)
        adj = map_context_adjustment(sig)
        st.markdown("### Context adjustment")
        st.write(f"Adjustment suggested: **{adj:+.1f}** points.")
        st.session_state["map_adjustment"] = float(adj)


def render_offline_map(cfg: Dict[str, Any]):
    st.subheader("Offline Map Upload (GeoJSON)")
    st.caption("Upload your offline environment as GeoJSON. Use: roads.geojson (LineString) + pois.geojson (Point).")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        roads_file = st.file_uploader("Upload roads.geojson", type=["geojson", "json"], key="roads_geojson")
        pois_file = st.file_uploader("Upload pois.geojson", type=["geojson", "json"], key="pois_geojson")

        if roads_file is not None:
            j, msg = parse_geojson_bytes(roads_file.read())
            if j is None:
                st.error(msg)
            else:
                st.success("roads.geojson loaded.")
                st.session_state["offline_roads_geojson"] = j

        if pois_file is not None:
            j, msg = parse_geojson_bytes(pois_file.read())
            if j is None:
                st.error(msg)
            else:
                st.success("pois.geojson loaded.")
                st.session_state["offline_pois_geojson"] = j

        st.markdown("---")
        st.caption("Tip: Select a location on the Map tab first, then calculate offline signals here.")

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Compute offline signals at selected location")

        lat = st.session_state.get("analysis_lat")
        lon = st.session_state.get("analysis_lon")
        if lat is None or lon is None:
            st.info("Select a location on the Map tab first.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        st.write(f"Selected point: `{lat:.5f}, {lon:.5f}`")
        radius_m = int(cfg["radius_m"])

        has_roads = "offline_roads_geojson" in st.session_state
        has_pois = "offline_pois_geojson" in st.session_state

        if not (has_roads or has_pois):
            st.warning("Upload at least one GeoJSON file first.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        if st.button("Compute offline overlay", use_container_width=True, type="primary"):
            roads_j = st.session_state.get("offline_roads_geojson")
            pois_j = st.session_state.get("offline_pois_geojson")
            sig = offline_signals_within_radius(roads_j, pois_j, float(lat), float(lon), radius_m)
            st.session_state["offline_signals"] = sig
            pts = offline_overlay_points(sig)
            st.session_state["offline_pts"] = float(pts)
            st.success(f"Offline signals computed. Overlay points: {pts:0.1f}/10")

        st.markdown("</div>", unsafe_allow_html=True)

    if "offline_signals" in st.session_state:
        st.markdown("---")
        st.subheader("Offline signals")
        st.json(st.session_state["offline_signals"])
        st.metric("Offline overlay points", f"{float(st.session_state.get('offline_pts', 0.0)):0.1f} / 10")


def render_live_events():
    st.subheader("Live Events Monitor (free)")
    st.caption("News & disaster feeds may be rate-limited; the app continues to run.")

    colA, colB = st.columns([1.2, 1])

    with colA:
        query = st.text_input(
            "Keywords (example: flood OR infrastructure damage OR bridge collapse OR cyclone)",
            value="flood OR infrastructure damage OR bridge collapse OR cyclone",
        )
        max_records = st.slider("Max records", 5, 50, 20, step=5)

        if st.button("Fetch latest", type="primary"):
            with st.spinner("Fetching GDELT + ReliefWeb..."):
                gd, gd_err = fetch_gdelt(query=query, max_records=max_records)
                rw, rw_err = fetch_reliefweb(query=query, limit=min(20, max_records))
                st.session_state["gdelt_df"] = gd
                st.session_state["relief_df"] = rw
                st.session_state["gdelt_err"] = gd_err
                st.session_state["relief_err"] = rw_err

    with colB:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Overlay risk (news)")
        gd = st.session_state.get("gdelt_df", pd.DataFrame())
        rw = st.session_state.get("relief_df", pd.DataFrame())
        overlay = disaster_overlay_score(gd, rw)
        st.metric("Overlay points", f"{overlay:0.1f} / 15")
        st.session_state["news_overlay"] = float(overlay)
        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.get("gdelt_err"):
        st.warning(st.session_state["gdelt_err"])
    if st.session_state.get("relief_err"):
        st.warning(st.session_state["relief_err"])

    gd = st.session_state.get("gdelt_df", pd.DataFrame())
    rw = st.session_state.get("relief_df", pd.DataFrame())
    if isinstance(gd, pd.DataFrame) and not gd.empty:
        st.markdown("### GDELT (news)")
        st.dataframe(gd, use_container_width=True, height=260)
    if isinstance(rw, pd.DataFrame) and not rw.empty:
        st.markdown("### ReliefWeb (disaster reports)")
        st.dataframe(rw, use_container_width=True, height=260)


def render_hazard_overlays():
    st.subheader("Hazard overlays (free)")
    st.caption("Uses map-selected location. If not selected, overlays remain zero.")

    lat = st.session_state.get("analysis_lat")
    lon = st.session_state.get("analysis_lon")

    if lat is None or lon is None:
        st.info("Select a location on the Map tab first (Map tab).")
        return

    col1, col2, col3 = st.columns(3)

    # USGS
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Earthquakes (USGS)")
        if st.button("Refresh USGS", use_container_width=True):
            with st.spinner("Fetching USGS earthquakes..."):
                df, err = fetch_usgs_earthquakes_near(lat, lon, radius_km=300.0)
                st.session_state["usgs_df"] = df
                st.session_state["usgs_err"] = err
        df = st.session_state.get("usgs_df", pd.DataFrame())
        pts = usgs_overlay_points(df)
        st.metric("Overlay points", f"{pts:0.1f} / 8")
        st.session_state["usgs_pts"] = float(pts)
        if st.session_state.get("usgs_err"):
            st.warning(st.session_state["usgs_err"])
        st.markdown("</div>", unsafe_allow_html=True)

    # Open-Meteo
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Weather (Open-Meteo)")
        if st.button("Refresh Weather", use_container_width=True):
            with st.spinner("Fetching weather..."):
                j, err = fetch_open_meteo(lat, lon)
                st.session_state["meteo_json"] = j
                st.session_state["meteo_err"] = err
                pts2, summary = open_meteo_overlay_points(j)
                st.session_state["meteo_pts"] = float(pts2)
                st.session_state["meteo_summary"] = summary
        pts2 = float(st.session_state.get("meteo_pts", 0.0))
        st.metric("Overlay points", f"{pts2:0.1f} / 6")
        if st.session_state.get("meteo_err"):
            st.warning(st.session_state["meteo_err"])
        summary = st.session_state.get("meteo_summary")
        if isinstance(summary, dict) and summary:
            st.caption("Today (UTC)")
            st.json(summary)
        st.markdown("</div>", unsafe_allow_html=True)

    # EONET
    with col3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Hazards (NASA EONET)")
        if st.button("Refresh EONET", use_container_width=True):
            with st.spinner("Fetching EONET events..."):
                df3, err = fetch_eonet_events_near(lat, lon, radius_km=500.0, limit=50)
                st.session_state["eonet_df"] = df3
                st.session_state["eonet_err"] = err
        df3 = st.session_state.get("eonet_df", pd.DataFrame())
        pts3 = eonet_overlay_points(df3)
        st.metric("Overlay points", f"{pts3:0.1f} / 6")
        st.session_state["eonet_pts"] = float(pts3)
        if st.session_state.get("eonet_err"):
            st.warning(st.session_state["eonet_err"])
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    dfu = st.session_state.get("usgs_df", pd.DataFrame())
    dfe = st.session_state.get("eonet_df", pd.DataFrame())

    if isinstance(dfu, pd.DataFrame) and not dfu.empty:
        st.markdown("### Nearby earthquakes (USGS)")
        st.dataframe(dfu, use_container_width=True, height=260)

    if isinstance(dfe, pd.DataFrame) and not dfe.empty:
        st.markdown("### Nearby hazards (NASA EONET)")
        st.dataframe(dfe, use_container_width=True, height=260)


def render_model_status(cfg: Dict[str, Any]):
    st.subheader("Model Status")
    onnx_ok = os.path.exists(MODEL_ONNX_PATH)
    joblib_ok = os.path.exists(MODEL_JOBLIB_PATH)

    if onnx_ok:
        st.success("ONNX model found ✅")
    else:
        st.error("ONNX model missing. Upload to: models/infra_model.onnx")

    if joblib_ok:
        st.info("Joblib model present (optional).")
    else:
        st.caption("Joblib model not found (optional).")

    if cfg.get("show_debug"):
        st.markdown("#### Debug")
        st.write("ONNX path:", MODEL_ONNX_PATH, "exists:", onnx_ok)
        st.write("JOBLIB path:", MODEL_JOBLIB_PATH, "exists:", joblib_ok)
        st.write("Features:", FEATURES)


def render_combined(cfg: Dict[str, Any]):
    st.markdown("---")
    st.subheader("Combined Risk View")

    if "single_pred" not in st.session_state:
        st.info("Run a single prediction first (Predictor tab).")
        return

    pred = st.session_state["single_pred"]
    base_need = float(pred["need"])
    base_kind = pred["model_kind"]
    inputs = pred.get("input", {}) or {}
    country_label = pred.get("country")

    map_adj = float(st.session_state.get("map_adjustment", 0.0))
    news_overlay = float(st.session_state.get("news_overlay", 0.0))
    usgs_pts = float(st.session_state.get("usgs_pts", 0.0))
    meteo_pts = float(st.session_state.get("meteo_pts", 0.0))
    eonet_pts = float(st.session_state.get("eonet_pts", 0.0))
    offline_pts = float(st.session_state.get("offline_pts", 0.0))

    combined = clamp(base_need + map_adj + news_overlay + usgs_pts + meteo_pts + eonet_pts + offline_pts, 0, 100)
    badge_text, badge_cls = risk_bucket(combined)

    st.markdown(
        f"<span class='badge {badge_cls}'>{badge_text}</span> &nbsp; "
        f"<b>Combined Need</b>: <span style='font-size:1.8rem'>{combined:0.1f}%</span>"
        f"<div class='small-muted'>Base: {base_kind} | Live map: {map_adj:+.1f} | News: {news_overlay:+.1f} | "
        f"USGS: {usgs_pts:+.1f} | Weather: {meteo_pts:+.1f} | EONET: {eonet_pts:+.1f} | Offline: {offline_pts:+.1f}</div>",
        unsafe_allow_html=True,
    )

    overlays = {
        "map": map_adj,
        "events": news_overlay,
        "usgs": usgs_pts,
        "weather": meteo_pts,
        "eonet": eonet_pts,
        "offline_map": offline_pts,
    }

    plan = recommendations_detailed(
        base_need=base_need,
        overlays=overlays,
        inputs=inputs,
        map_signals=st.session_state.get("map_signals"),
        offline_signals=st.session_state.get("offline_signals"),
    )

    # Evidence / Confidence / Limitations
    has_location = (st.session_state.get("analysis_lat") is not None and st.session_state.get("analysis_lon") is not None)
    has_offline_map = ("offline_signals" in st.session_state)
    conf, limitations = compute_confidence(
        model_kind=base_kind,
        base_need=base_need,
        overlays=overlays,
        has_location=has_location,
        has_offline_map=has_offline_map,
    )

    st.markdown("---")
    st.subheader("Precautions & prevention")

    summary = plan.get("summary", {})
    st.markdown("#### Immediate actions")
    for a in summary.get("immediate_actions", []):
        st.write(f"- {a}")

    # Readable evidence section
    with st.expander("Evidence (readable) + Confidence + Limitations", expanded=True):
        st.markdown(f"**Confidence (heuristic):** `{conf*100:0.0f}%`")
        st.markdown("**What drove this score:**")
        reasons = summary.get("context_reasons", []) or []
        if reasons:
            for r in reasons[:14]:
                st.write(f"- {r}")
        else:
            st.write("- No specific drivers captured (run map + overlays for richer evidence).")

        st.markdown("**Limitations:**")
        for x in (limitations or [])[:12]:
            st.write(f"- {x}")

        st.markdown("**Evidence links (feeds):**")
        links = build_evidence_links(
            st.session_state.get("gdelt_df", pd.DataFrame()),
            st.session_state.get("relief_df", pd.DataFrame()),
            st.session_state.get("usgs_df", pd.DataFrame()),
            st.session_state.get("eonet_df", pd.DataFrame()),
        )
        any_links = False
        for src, arr in links.items():
            if not arr:
                continue
            any_links = True
            st.markdown(f"**{src}:**")
            for it in arr[:6]:
                title = it.get("title", "Link")
                url = it.get("url", "")
                date = it.get("date", "")
                if url:
                    st.markdown(f"- [{title}]({url})  <span class='small-muted'>{date}</span>", unsafe_allow_html=True)
        if not any_links:
            st.write("- No feed links available yet (fetch News/Reports and refresh hazards).")

    st.markdown("#### Detailed plan")
    for item in plan.get("items", []):
        with st.expander(item.get("title", "Recommendation"), expanded=False):
            st.markdown("**Why it matters**")
            for w in item.get("why", []):
                st.write(f"- {w}")

            st.markdown("**Data gathered**")
            for d in item.get("data_gathered", []):
                st.write(f"- {d}")

            st.markdown("**Steps (recommended sequence)**")
            for i, step in enumerate(item.get("steps", []), start=1):
                st.write(f"{i}. {step}")

            st.markdown("**Deliverables (client-ready)**")
            for d in item.get("deliverables", []):
                st.write(f"- {d}")

    st.markdown("---")
    st.subheader("Download client-ready report")

    # Full report (includes confidence/limitations + evidence links)
    report_md = build_full_report_markdown(
        country_label=country_label,
        inputs=inputs,
        model_kind=base_kind,
        base_need=base_need,
        overlays=overlays,
        map_signals=st.session_state.get("map_signals"),
        offline_signals=st.session_state.get("offline_signals"),
        gdelt_df=st.session_state.get("gdelt_df", pd.DataFrame()),
        relief_df=st.session_state.get("relief_df", pd.DataFrame()),
        usgs_df=st.session_state.get("usgs_df", pd.DataFrame()),
        eonet_df=st.session_state.get("eonet_df", pd.DataFrame()),
        meteo_summary=st.session_state.get("meteo_summary", {}),
        confidence=conf,
        limitations=limitations,
    )

    st.download_button(
        "Download report.md",
        data=report_md.encode("utf-8"),
        file_name="infrastructure_risk_report.md",
        mime="text/markdown",
        use_container_width=True,
    )


def main():
    cfg = render_sidebar()
    render_header()

    default_df = build_default_country_df()

    tabs = st.tabs(["📊 Dashboard", "🤖 Predictor", "🗺️ Map", "🗃️ Offline Map", "🛰️ News/Reports", "🌋 Hazards", "✅ Model Status"])

    with tabs[0]:
        render_dashboard(default_df)

    with tabs[1]:
        render_predictor(default_df)

    with tabs[2]:
        render_map_and_location(cfg)

    with tabs[3]:
        render_offline_map(cfg)

    with tabs[4]:
        render_live_events()

    with tabs[5]:
        render_hazard_overlays()

    with tabs[6]:
        render_model_status(cfg)

    render_combined(cfg)

    st.markdown("---")
    st.caption(f"Updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} | Free feeds + offline GeoJSON overlay")


if __name__ == "__main__":
    main()
