import os
import time
from datetime import datetime

import pandas as pd
import streamlit as st
import plotly.express as px

import folium
from streamlit_folium import st_folium

from utils import (
    FEATURES,
    MODEL_ONNX_PATH,
    MODEL_JOBLIB_PATH,
    DEFAULT_RADIUS_M,
    build_default_country_df,
    sample_csv_bytes,
    ensure_schema,
    to_feature_row,
    clamp,
    risk_bucket,
    model_predict_proba,
    fetch_overpass_signals_robust,
    map_context_adjustment,
    fetch_gdelt,
    fetch_reliefweb,
    disaster_overlay_score,
    recommendations,
    build_analyst_context,
    analyst_reply,
)


EVENTS_COOLDOWN_SECONDS = 45


st.set_page_config(
    page_title="Global Infrastructure AI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

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


def render_header():
    st.title("🌍 Global Infrastructure AI")
    st.caption("Streamlit Cloud compatible (ONNX-first). Free sources: OSM Overpass + GDELT + ReliefWeb.")


def render_sidebar():
    with st.sidebar:
        st.subheader("Settings")
        show_debug = st.toggle("Show debug info", value=False)

        st.markdown("---")
        st.write("Model files expected:")
        st.code("models/infra_model.onnx\nmodels/infra_model.joblib (optional)", language="text")

        radius_m = st.slider("Map signals radius (meters)", 1000, 20000, DEFAULT_RADIUS_M, step=500)

        st.markdown("---")
        st.write("Sample CSV for Upload CSV:")
        st.download_button(
            "Download sample.csv",
            data=sample_csv_bytes(),
            file_name="sample.csv",
            mime="text/csv",
            use_container_width=True,
        )

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
        st.download_button(
            "Download predictions CSV",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )


def render_map_signals(cfg):
    st.subheader("Map signals (FREE OSM)")
    st.caption("Click on the map to fetch nearby infrastructure counts. Robust fallback endpoints + caching enabled.")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        m = folium.Map(location=[20, 0], zoom_start=2, tiles="OpenStreetMap")
        folium.LatLngPopup().add_to(m)
        map_data = st_folium(m, height=420, use_container_width=True)

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Click on map to get signals")

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
    st.caption("Free feeds only. App never crashes if feeds are rate-limited.")

    colA, colB = st.columns([1.2, 1])

    with colA:
        query = st.text_input(
            "Keywords",
            value="flood OR infrastructure damage OR bridge collapse OR cyclone",
        )
        max_records = st.slider("Max records", 5, 50, 20, step=5)

        rem = _cooldown_remaining()
        if rem > 0:
            st.info(f"Cooldown active: wait {rem}s")

        if st.button("Fetch latest", type="primary", disabled=(rem > 0)):
            st.session_state["events_last_fetch_ts"] = time.time()

            with st.spinner("Fetching feeds..."):
                gd, rw = pd.DataFrame(), pd.DataFrame()
                try:
                    gd = fetch_gdelt(query=query, max_records=min(30, int(max_records)))
                except Exception as e:
                    st.warning(f"GDELT unavailable: {e}")

                try:
                    rw = fetch_reliefweb(query=query, limit=min(30, int(max_records)))
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


def render_analyst_chat():
    st.subheader("Analyst (free)")
    st.caption("Chat-like assistant using your prediction + overlays (no paid APIs).")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for m in st.session_state["chat_history"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_text = st.chat_input("Ask: summary / actions / map / events")
    if user_text is not None:
        st.session_state["chat_history"].append({"role": "user", "content": user_text})
        ctx = build_analyst_context(st.session_state)
        reply = analyst_reply(user_text, ctx)
        st.session_state["chat_history"].append({"role": "assistant", "content": reply})
        st.rerun()


def render_combined_risk_panel():
    st.markdown("---")
    st.subheader("Combined Risk View")

    if "single_pred" not in st.session_state:
        st.info("Run a single prediction first (Predictor tab).")
        return

    base_need = float(st.session_state["single_pred"]["need"])
    base_kind = st.session_state["single_pred"]["model_kind"]

    map_adj = float(st.session_state.get("map_adjustment", 0.0))
    news_overlay = float(st.session_state.get("news_overlay", 0.0))
    combined = clamp(base_need + map_adj + news_overlay, 0, 100)

    badge_text, badge_cls = risk_bucket(combined)

    st.markdown(
        f"<span class='badge {badge_cls}'>{badge_text}</span> &nbsp; "
        f"<b>Combined Need</b>: <span style='font-size:1.8rem'>{combined:0.1f}%</span>"
        f"<div class='small-muted'>Base model: {base_kind} | Map adj: {map_adj:+.1f} | Events overlay: {news_overlay:+.1f}</div>",
        unsafe_allow_html=True,
    )

    st.markdown("### Precautions & prevention")
    for item in recommendations(base_need, map_adj + news_overlay):
        st.write(f"- {item}")


def render_model_status(cfg):
    st.subheader("Model Status")
    onnx_ok = os.path.exists(MODEL_ONNX_PATH)
    joblib_ok = os.path.exists(MODEL_JOBLIB_PATH)

    if onnx_ok:
        st.success("ONNX model found ✅")
    elif joblib_ok:
        st.warning("JOBLIB model found (ONNX recommended for Streamlit Cloud).")
    else:
        st.error("No model found. Add: models/infra_model.onnx")

    if cfg["show_debug"]:
        st.markdown("### Debug")
        st.write("ONNX path:", MODEL_ONNX_PATH, "exists:", onnx_ok)
        st.write("JOBLIB path:", MODEL_JOBLIB_PATH, "exists:", joblib_ok)


def main():
    cfg = render_sidebar()
    render_header()

    default_df = build_default_country_df()

    tabs = st.tabs(["📊 Dashboard", "🤖 Predictor", "🗺️ Map Signals", "🛰️ Live Events", "💬 Analyst", "✅ Model Status"])

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
    st.caption(f"Updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")


if __name__ == "__main__":
    main()
