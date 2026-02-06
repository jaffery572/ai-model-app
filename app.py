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

# same features used in training script
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
        return None, None
    payload = joblib.load(MODEL_PATH)
    return payload.get("model"), payload.get("features", FEATURES)

def risk_badge(p):
    if p >= 70:
        return "HIGH", "🔴", "Urgent infrastructure investment needed."
    if p >= 40:
        return "MEDIUM", "🟠", "Strategic improvements recommended."
    return "LOW", "🟢", "Maintain & optimize existing infrastructure."

def main():
    st.markdown("## 🌍 Global Infrastructure AI")
    st.caption("World Bank indicators + trained ML model (local-trained). Streamlit Cloud compatible.")

    model, model_features = load_model()
    if model is None:
        st.error("Model file missing. Upload: models/infra_model.joblib")
        st.stop()

    tab1, tab2 = st.tabs(["📊 Dashboard", "🤖 Predictor"])

    with tab1:
        st.subheader("Quick Overview")
        st.info("Tip: Use Predictor tab to fetch World Bank data + run inference.")

        # Just a small explainability placeholder (Histogram of features distribution if CSV uploaded)
        st.markdown("**Optional:** Upload a CSV (same columns) to explore distributions.")
        up = st.file_uploader("Upload CSV", type=["csv"])
        if up:
            try:
                df = pd.read_csv(up)
                missing = [c for c in FEATURES if c not in df.columns]
                if missing:
                    st.error(f"CSV missing columns: {missing}")
                else:
                    st.dataframe(df.head(50), use_container_width=True)
                    fig = px.histogram(df, x="GDP_per_capita_USD", nbins=50, title="GDP per Capita Distribution")
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"CSV read failed: {e}")

    with tab2:
        st.subheader("AI Predictor (Real Data Fetch + Manual Inputs)")

        colA, colB = st.columns([1.3, 1])

        with colA:
            method = st.radio("Choose input method:", ["World Bank (country code)", "Manual input"], horizontal=True)

            country_data = None

            if method == "World Bank (country code)":
                code = st.text_input("Country code (ISO2/ISO3) e.g., PK, IN, US", value="PK")
                if st.button("Fetch Real Data"):
                    try:
                        snap = fetch_country_snapshot(code)
                        if snap["GDP_per_capita_USD"] is None or snap["Population_Total"] is None:
                            st.warning("Some indicators missing for this country code. Try ISO3 or another country.")
                        country_data = snap
                        st.success("Fetched successfully (cached 1 hour).")

                        st.write({
                            "GDP_per_capita_USD": snap["GDP_per_capita_USD"],
                            "Population_Total": snap["Population_Total"],
                            "Urbanization_Rate": snap["Urbanization_Rate"],
                            "Electricity_Access": snap["Electricity_Access"],
                            "Years": {
                                "gdp": snap["gdp_year"],
                                "pop": snap["pop_year"],
                                "urb": snap["urb_year"],
                                "elec": snap["elec_year"],
                            }
                        })
                    except Exception as e:
                        st.error(f"World Bank fetch failed: {e}")

                # extra required feature not from WB: HDI (manual slider)
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
                    "Electricity_Access": None
                }

            if st.button("🚀 Predict Infrastructure Need", type="primary"):
                try:
                    if not country_data:
                        st.warning("Please provide inputs first.")
                        st.stop()

                    pop_total = country_data.get("Population_Total") or 0
                    pop_m = pop_total / 1e6

                    X = pd.DataFrame([{
                        "Population_Millions": pop_m,
                        "GDP_per_capita_USD": float(country_data.get("GDP_per_capita_USD") or 0),
                        "HDI_Index": float(hdi),
                        "Urbanization_Rate": float(country_data.get("Urbanization_Rate") or 0),
                    }])

                    X = X[model_features]
                    proba = float(model.predict_proba(X.values)[0][1] * 100.0)

                    st.session_state["last_pred"] = {"proba": proba, "X": X}

                except Exception as e:
                    st.error(f"Prediction failed: {e}")

        with colB:
            st.markdown("### Result")
            if "last_pred" in st.session_state:
                proba = st.session_state["last_pred"]["proba"]
                X = st.session_state["last_pred"]["X"]

                level, emoji, msg = risk_badge(proba)
                st.metric("Need Probability", f"{proba:.1f}%")
                st.write(f"**Risk:** {emoji} **{level}**")
                st.write(msg)

                # Gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=proba,
                    title={"text": "Infrastructure Need"},
                    gauge={"axis": {"range": [0, 100]}}
                ))
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)

                # Explainability (simple feature bars)
                df_plot = X.T.reset_index()
                df_plot.columns = ["feature", "value"]
                fig2 = px.bar(df_plot, x="feature", y="value", title="Inputs used (feature values)")
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Run a prediction to see results here.")

    st.markdown("---")
    st.caption("Deployed via GitHub → Streamlit Cloud. Model trained locally and uploaded as joblib.")

if __name__ == "__main__":
    main()
