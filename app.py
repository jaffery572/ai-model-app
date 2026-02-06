import os
import joblib
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="🌍 Global Infrastructure AI", page_icon="🌐", layout="wide")

MODEL_PATH = "models/infra_model.joblib"
FEATURES = ["Population_Millions", "GDP_per_capita_USD", "HDI_Index", "Urbanization_Rate"]

WB_INDICATORS = {
    "GDP_per_capita_USD": "NY.GDP.PCAP.CD",
    "Population_Total": "SP.POP.TOTL",
    "Urbanization_Rate": "SP.URB.TOTL.IN.ZS",
    "Electricity_Access": "EG.ELC.ACCS.ZS",
}

def _wb_latest(country: str, indicator: str):
    url = f"https://api.worldbank.org/v2/country/{country}/indicator/{indicator}?format=json"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list) or len(data) < 2 or not data[1]:
        return None, None
    for row in data[1]:
        if row.get("value") is not None:
            return float(row["value"]), row.get("date")
    return None, None

@st.cache_data(ttl=3600)
def fetch_country_snapshot(country_code: str) -> dict:
    country_code = country_code.strip().lower()
    out = {"country_code": country_code.upper()}
    gdp, gdp_year = _wb_latest(country_code, WB_INDICATORS["GDP_per_capita_USD"])
    pop, pop_year = _wb_latest(country_code, WB_INDICATORS["Population_Total"])
    urb, urb_year = _wb_latest(country_code, WB_INDICATORS["Urbanization_Rate"])
    elec, elec_year = _wb_latest(country_code, WB_INDICATORS["Electricity_Access"])

    out.update({
        "GDP_per_capita_USD": gdp,
        "Population_Total": pop,
        "Urbanization_Rate": urb,
        "Electricity_Access": elec,
        "gdp_year": gdp_year,
        "pop_year": pop_year,
        "urb_year": urb_year,
        "elec_year": elec_year,
    })
    return out

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, FEATURES
    try:
        payload = joblib.load(MODEL_PATH)
        return payload.get("model"), payload.get("features", FEATURES)
    except Exception as e:
        st.error(f"Model load failed: {e}")
        return None, FEATURES

def risk_badge(p):
    if p >= 70:
        return "HIGH", "🔴", "Urgent infrastructure investment needed."
    if p >= 40:
        return "MEDIUM", "🟠", "Strategic improvements recommended."
    return "LOW", "🟢", "Maintain & optimize existing infrastructure."

def main():
    st.title("🌍 Global Infrastructure AI")
    st.caption("World Bank indicators + trained ML model. Streamlit Cloud compatible.")

    model, model_features = load_model()
    if model is None:
        st.error("Model missing. Upload: models/infra_model.joblib")
        st.stop()

    tab1, tab2 = st.tabs(["🤖 Predictor", "📊 Explore CSV"])

    with tab1:
        colA, colB = st.columns([1.3, 1])

        with colA:
            method = st.radio("Input method:", ["World Bank (country code)", "Manual"], horizontal=True)

            country_data = None
            hdi = 0.70

            if method == "World Bank (country code)":
                code = st.text_input("Country code (ISO2/ISO3) e.g., PK, IN, US", value="PK")
                if st.button("Fetch Real Data"):
                    try:
                        snap = fetch_country_snapshot(code)
                        country_data = snap
                        st.success("Fetched (cached 1 hour).")
                        st.json(snap)
                    except Exception as e:
                        st.error(f"Fetch failed: {e}")

                hdi = st.slider("HDI Index (manual)", 0.30, 0.99, 0.70, 0.01)

            else:
                pop_m = st.number_input("Population (Millions)", 0.1, 2500.0, 100.0)
                gdp = st.number_input("GDP per Capita (USD)", 100.0, 250000.0, 5000.0)
                hdi = st.slider("HDI Index", 0.30, 0.99, 0.70, 0.01)
                urb = st.slider("Urbanization Rate (%)", 0.0, 100.0, 50.0)

                country_data = {
                    "Population_Total": pop_m * 1e6,
                    "GDP_per_capita_USD": gdp,
                    "Urbanization_Rate": urb,
                }

            if st.button("🚀 Predict", type="primary"):
                if not country_data:
                    st.warning("Please provide inputs first.")
                    st.stop()

                pop_total = country_data.get("Population_Total") or 0
                X = pd.DataFrame([{
                    "Population_Millions": pop_total / 1e6,
                    "GDP_per_capita_USD": float(country_data.get("GDP_per_capita_USD") or 0),
                    "HDI_Index": float(hdi),
                    "Urbanization_Rate": float(country_data.get("Urbanization_Rate") or 0),
                }])[model_features]

                proba = float(model.predict_proba(X.values)[0][1] * 100.0)
                st.session_state["pred"] = {"proba": proba, "X": X}

        with colB:
            st.subheader("Result")
            if "pred" in st.session_state:
                proba = st.session_state["pred"]["proba"]
                X = st.session_state["pred"]["X"]

                level, emoji, msg = risk_badge(proba)
                st.metric("Need Probability", f"{proba:.1f}%")
                st.write(f"**Risk:** {emoji} **{level}**")
                st.write(msg)

                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=proba,
                    title={"text": "Infrastructure Need"},
                    gauge={"axis": {"range": [0, 100]}}
                ))
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)

                df_plot = X.T.reset_index()
                df_plot.columns = ["feature", "value"]
                st.plotly_chart(px.bar(df_plot, x="feature", y="value", title="Inputs used"), use_container_width=True)
            else:
                st.info("Run a prediction to see output.")

    with tab2:
        up = st.file_uploader("Upload CSV (optional)", type=["csv"])
        if up:
            try:
                df = pd.read_csv(up)
                st.dataframe(df.head(100), use_container_width=True)
                if "GDP_per_capita_USD" in df.columns:
                    st.plotly_chart(px.histogram(df, x="GDP_per_capita_USD", nbins=50), use_container_width=True)
            except Exception as e:
                st.error(f"CSV error: {e}")

if __name__ == "__main__":
    main()
