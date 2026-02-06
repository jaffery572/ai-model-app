# app.py (Streamlit Cloud compatible, production-lean version)
import os
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

warnings.filterwarnings("ignore")

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="🌍 Global Infrastructure AI",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------
# CSS
# ----------------------------
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        background: linear-gradient(90deg, #1a2980, #26d0ce);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: 800;
    }
    .sub-header {
        font-size: 1.25rem;
        color: #2c3e50;
        border-left: 5px solid #3498db;
        padding-left: 12px;
        margin: 1rem 0;
        font-weight: 700;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 12px;
        color: white;
        box-shadow: 0 6px 18px rgba(0,0,0,0.12);
        margin: 0.35rem 0;
    }
    .prediction-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.25rem;
        border-radius: 12px;
        color: white;
        box-shadow: 0 6px 18px rgba(0,0,0,0.12);
    }
    .risk-high { background: #ff4d4d; color: white; padding: 6px 10px; border-radius: 8px; font-weight: 800; display: inline-block; }
    .risk-medium { background: #ff9900; color: #1a1a1a; padding: 6px 10px; border-radius: 8px; font-weight: 800; display: inline-block; }
    .risk-low { background: #00cc66; color: white; padding: 6px 10px; border-radius: 8px; font-weight: 800; display: inline-block; }
    .tiny-note { color: #6b7280; font-size: 0.9rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------
# Constants / paths
# ----------------------------
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "infrastructure_gb.joblib")
RANDOM_SEED_DEFAULT = 42

FEATURE_COLS = ["Population_Millions", "GDP_per_capita_USD", "HDI_Index", "Urbanization_Rate"]
TARGET_COL = "Infrastructure_Need"

# ----------------------------
# Session state init
# ----------------------------
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "model_meta" not in st.session_state:
    st.session_state.model_meta = None


# ----------------------------
# Data (base) + validation
# ----------------------------
@st.cache_data(show_spinner=False)
def load_global_data_base() -> pd.DataFrame:
    data = {
        "Country": [
            "USA", "China", "India", "Germany", "UK", "Japan", "Brazil",
            "Russia", "France", "Italy", "Canada", "Australia", "South Korea",
            "Mexico", "Indonesia", "Turkey", "Saudi Arabia", "Switzerland",
            "Netherlands", "Spain", "Pakistan", "Bangladesh", "Nigeria",
            "Egypt", "Vietnam", "Thailand", "South Africa", "Argentina",
            "Colombia", "Malaysia"
        ],
        "Population_Millions": [
            331, 1412, 1408, 83, 68, 125, 215, 144, 67, 59,
            38, 26, 51, 129, 278, 85, 36, 8.6, 17, 47,
            240, 170, 216, 109, 98, 70, 60, 45, 52, 33
        ],
        "GDP_per_capita_USD": [
            63500, 12500, 2300, 45700, 42200, 40100, 8900,
            11200, 40400, 32000, 43200, 52000, 35000, 9900,
            4300, 9500, 23500, 81900, 52400, 29400,
            1500, 2600, 2300, 3900, 2800, 7800, 6300,
            10600, 6400, 11400
        ],
        "HDI_Index": [
            0.926, 0.761, 0.645, 0.947, 0.932, 0.925, 0.765,
            0.824, 0.901, 0.892, 0.929, 0.944, 0.916, 0.779,
            0.718, 0.820, 0.857, 0.955, 0.944, 0.904,
            0.557, 0.632, 0.539, 0.707, 0.704, 0.777, 0.709,
            0.845, 0.767, 0.803
        ],
        "Urbanization_Rate": [
            83, 64, 35, 77, 84, 92, 87, 75, 81, 71,
            81, 86, 81, 81, 57, 76, 84, 74, 93, 81,
            37, 39, 52, 43, 38, 51, 68, 92, 81, 78
        ],
        "Continent": [
            "North America", "Asia", "Asia", "Europe", "Europe", "Asia",
            "South America", "Europe", "Europe", "Europe", "North America",
            "Oceania", "Asia", "North America", "Asia", "Asia", "Asia",
            "Europe", "Europe", "Europe", "Asia", "Asia", "Africa",
            "Africa", "Asia", "Asia", "Africa", "South America",
            "South America", "Asia"
        ],
    }

    df = pd.DataFrame(data)

    def calculate_need(row) -> int:
        score = 0
        if row["GDP_per_capita_USD"] < 5000:
            score += 2
        if row["HDI_Index"] < 0.7:
            score += 2
        if row["Urbanization_Rate"] < 50 and row["Population_Millions"] > 50:
            score += 1
        return 1 if score >= 3 else 0

    df[TARGET_COL] = df.apply(calculate_need, axis=1)
    return df


def validate_input_df(df: pd.DataFrame) -> tuple[bool, str]:
    needed = set(["Country"] + FEATURE_COLS)
    missing = needed - set(df.columns)
    if missing:
        return False, f"Missing columns: {sorted(list(missing))}"
    # numeric checks
    for c in FEATURE_COLS:
        if not pd.api.types.is_numeric_dtype(df[c]):
            return False, f"Column '{c}' must be numeric."
    if (df["GDP_per_capita_USD"] <= 0).any():
        return False, "GDP_per_capita_USD must be > 0 for log-scale charts."
    if (df["HDI_Index"] < 0).any() or (df["HDI_Index"] > 1).any():
        return False, "HDI_Index must be between 0 and 1."
    if (df["Urbanization_Rate"] < 0).any() or (df["Urbanization_Rate"] > 100).any():
        return False, "Urbanization_Rate must be between 0 and 100."
    if (df["Population_Millions"] <= 0).any():
        return False, "Population_Millions must be > 0."
    return True, ""


# ----------------------------
# Robust scoring for prediction (human-friendly)
# ----------------------------
def heuristic_need_probability(country_data: dict) -> dict:
    gdp = float(country_data["GDP_per_capita_USD"])
    hdi = float(country_data["HDI_Index"])
    urb = float(country_data["Urbanization_Rate"])
    pop = float(country_data["Population_Millions"])

    need_score = (
        (1 - min(gdp / 50000.0, 1.0)) * 40.0
        + (1 - hdi) * 30.0
        + ((urb / 100.0) * 20.0 if gdp < 10000 else 0.0)
        + (10.0 if (pop > 100 and gdp < 5000) else 0.0)
    )

    need_probability = float(np.clip(need_score, 0, 100))

    if need_probability >= 70:
        risk_level, risk_color = "HIGH", "risk-high"
    elif need_probability >= 40:
        risk_level, risk_color = "MEDIUM", "risk-medium"
    else:
        risk_level, risk_color = "LOW", "risk-low"

    return {
        "needed": need_probability,
        "sufficient": 100.0 - need_probability,
        "confidence": float(95.0 - abs(50.0 - need_probability) / 2.0),
        "risk_level": risk_level,
        "risk_color": risk_color,
    }


# ----------------------------
# ML pipeline (Cloud-safe)
# ----------------------------
def make_synthetic_training_data(base_df: pd.DataFrame, n_samples: int, seed: int) -> pd.DataFrame:
    """
    Create synthetic rows around base data with noise, then label using the same rule.
    Keeps memory low and training stable on Streamlit Cloud.
    """
    rng = np.random.default_rng(seed)

    # sample base rows
    idx = rng.integers(0, len(base_df), size=n_samples)
    core = base_df.iloc[idx][FEATURE_COLS].copy()

    # add noise (bounded)
    core["Population_Millions"] = np.clip(core["Population_Millions"] * rng.normal(1.0, 0.12, n_samples), 0.5, 2000)
    core["GDP_per_capita_USD"] = np.clip(core["GDP_per_capita_USD"] * rng.normal(1.0, 0.18, n_samples), 300, 200000)
    core["HDI_Index"] = np.clip(core["HDI_Index"] + rng.normal(0.0, 0.03, n_samples), 0.3, 0.99)
    core["Urbanization_Rate"] = np.clip(core["Urbanization_Rate"] + rng.normal(0.0, 6.0, n_samples), 5, 99)

    df = core

    # label with rule (same as base)
    score = (
        (df["GDP_per_capita_USD"] < 5000).astype(int) * 2
        + (df["HDI_Index"] < 0.7).astype(int) * 2
        + ((df["Urbanization_Rate"] < 50) & (df["Population_Millions"] > 50)).astype(int) * 1
    )
    df[TARGET_COL] = (score >= 3).astype(int)
    return df


def train_gb_model(train_df: pd.DataFrame, seed: int) -> tuple[GradientBoostingClassifier, dict]:
    X = train_df[FEATURE_COLS].values
    y = train_df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y if len(np.unique(y)) > 1 else None
    )

    model = GradientBoostingClassifier(random_state=seed)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    return model, metrics


def save_model(model: GradientBoostingClassifier, meta: dict) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({"model": model, "meta": meta}, MODEL_PATH)


@st.cache_resource(show_spinner=False)
def load_model_if_exists():
    if os.path.exists(MODEL_PATH):
        payload = joblib.load(MODEL_PATH)
        return payload["model"], payload.get("meta", {})
    return None, None


def ml_predict_proba(model: GradientBoostingClassifier, country_data: dict) -> float:
    X = np.array([[country_data[c] for c in FEATURE_COLS]], dtype=float)
    proba = model.predict_proba(X)[0, 1]  # probability of class 1
    return float(np.clip(proba * 100.0, 0, 100))


# ----------------------------
# UI
# ----------------------------
def main():
    st.markdown('<div class="main-header">🌍 GLOBAL INFRASTRUCTURE AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align:center; color:#666; margin-top:-5px;">AI-powered analysis of infrastructure needs (Cloud-safe, real ML)</p>',
        unsafe_allow_html=True,
    )

    # Load base data
    base_df = load_global_data_base()

    # Optional CSV upload
    with st.sidebar:
        st.markdown("### 📥 Data Source")
        uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"])
        st.markdown('<p class="tiny-note">CSV must include: Country + Population_Millions, GDP_per_capita_USD, HDI_Index, Urbanization_Rate</p>', unsafe_allow_html=True)

    df = base_df
    if uploaded is not None:
        try:
            user_df = pd.read_csv(uploaded)
            ok, msg = validate_input_df(user_df)
            if not ok:
                st.sidebar.error(msg)
            else:
                # If target missing, compute it so dashboard still works
                if TARGET_COL not in user_df.columns:
                    tmp = user_df.copy()
                    score = (
                        (tmp["GDP_per_capita_USD"] < 5000).astype(int) * 2
                        + (tmp["HDI_Index"] < 0.7).astype(int) * 2
                        + ((tmp["Urbanization_Rate"] < 50) & (tmp["Population_Millions"] > 50)).astype(int) * 1
                    )
                    tmp[TARGET_COL] = (score >= 3).astype(int)
                    user_df = tmp
                df = user_df
                st.sidebar.success(f"Loaded {len(df)} rows from CSV ✅")
        except Exception as e:
            st.sidebar.error(f"CSV load failed: {e}")

    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🤖 AI Predictor", "⚡ Model Training"])

    # ---------------- Dashboard
    with tab1:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <h4>🌍 Countries</h4>
                    <h2>{len(df)}</h2>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <h4>🏗️ Needs</h4>
                    <h2>{int(df[TARGET_COL].sum())}</h2>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <h4>💰 Avg GDP</h4>
                    <h2>${df["GDP_per_capita_USD"].mean():,.0f}</h2>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col4:
            # show saved model metrics if any
            model, meta = load_model_if_exists()
            acc = meta.get("metrics", {}).get("accuracy", None) if meta else None
            acc_str = f"{acc*100:.1f}%" if isinstance(acc, (int, float)) else "—"
            st.markdown(
                f"""
                <div class="metric-card">
                    <h4>📈 Model Accuracy</h4>
                    <h2>{acc_str}</h2>
                </div>
                """,
                unsafe_allow_html=True,
            )

        colA, colB = st.columns(2)

        with colA:
            # FIX: discrete color for 0/1
            top10 = df.sort_values("GDP_per_capita_USD", ascending=False).head(10).copy()
            top10["Need_Label"] = top10[TARGET_COL].map({0: "No", 1: "Yes"}).astype(str)

            fig1 = px.bar(
                top10,
                x="Country",
                y="GDP_per_capita_USD",
                title="Top 10 Countries by GDP per Capita",
                color="Need_Label",
                color_discrete_map={"Yes": "#ff4d4d", "No": "#00cc66"},
            )
            st.plotly_chart(fig1, use_container_width=True)

        with colB:
            safe_df = df[df["GDP_per_capita_USD"] > 0].copy()
            fig2 = px.scatter(
                safe_df,
                x="GDP_per_capita_USD",
                y="HDI_Index",
                size="Population_Millions",
                color="Continent" if "Continent" in safe_df.columns else None,
                hover_name="Country" if "Country" in safe_df.columns else None,
                title="GDP per Capita vs HDI Index",
                log_x=True,
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="sub-header">Country Data</div>', unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True)

    # ---------------- Predictor
    with tab2:
        st.markdown('<div class="sub-header">AI Infrastructure Predictor</div>', unsafe_allow_html=True)

        model, meta = load_model_if_exists()
        using_model = model is not None

        col1, col2 = st.columns([2, 1])

        with col1:
            option = st.radio("Input Method:", ["Select Country", "Custom Input"], horizontal=True)

            country_data = None
            selected_country = None

            if option == "Select Country":
                selected_country = st.selectbox("Choose Country:", df["Country"].tolist() if "Country" in df.columns else [])
                if selected_country:
                    row = df[df["Country"] == selected_country].iloc[0]
                    country_data = {c: float(row[c]) for c in FEATURE_COLS}
            else:
                country_data = {
                    "Population_Millions": float(st.number_input("Population (Millions)", 0.5, 2500.0, 100.0)),
                    "GDP_per_capita_USD": float(st.number_input("GDP per Capita (USD)", 300.0, 250000.0, 5000.0)),
                    "HDI_Index": float(st.slider("HDI Index", 0.3, 0.99, 0.70)),
                    "Urbanization_Rate": float(st.slider("Urbanization Rate (%)", 5.0, 99.0, 50.0)),
                }

        with col2:
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            st.markdown("<h4 style='color: white; margin-top: 0;'>Run Prediction</h4>", unsafe_allow_html=True)

            mode = st.selectbox(
                "Prediction Mode",
                ["ML Model (if trained)", "Heuristic (always available)"],
                index=0,
                disabled=not using_model,
            )
            st.markdown(
                "<p class='tiny-note' style='color:white;'>Tip: ML mode needs training once (tab ⚡ Model Training).</p>",
                unsafe_allow_html=True,
            )

            if st.button("🚀 PREDICT", use_container_width=True, type="primary"):
                try:
                    if country_data is None:
                        st.error("No input selected.")
                    else:
                        # risk label from heuristic, probability from selected mode
                        heur = heuristic_need_probability(country_data)

                        if (mode.startswith("ML") and using_model):
                            p = ml_predict_proba(model, country_data)
                            heur["needed"] = p
                            heur["sufficient"] = 100.0 - p
                            # confidence from model meta if available
                            heur["confidence"] = float(meta.get("confidence_hint", heur["confidence"]))

                        # update risk badges based on final probability
                        pfinal = heur["needed"]
                        if pfinal >= 70:
                            heur["risk_level"], heur["risk_color"] = "HIGH", "risk-high"
                        elif pfinal >= 40:
                            heur["risk_level"], heur["risk_color"] = "MEDIUM", "risk-medium"
                        else:
                            heur["risk_level"], heur["risk_color"] = "LOW", "risk-low"

                        st.session_state.prediction = heur

                except Exception as e:
                    st.error(f"Prediction failed: {e}")

            if st.session_state.prediction:
                pr = st.session_state.prediction
                st.markdown(
                    f"""
                    <div style="text-align:center; padding: 8px 0;">
                        <h2 style="color:white; margin: 0;">{pr["needed"]:.1f}%</h2>
                        <p style="color:white; margin: 6px 0 10px 0;">Infrastructure Need</p>
                        <div class="{pr["risk_color"]}">{pr["risk_level"]} RISK</div>
                        <p style="color:white; margin-top: 10px;">Confidence: {pr["confidence"]:.1f}%</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("</div>", unsafe_allow_html=True)

        # Detailed results
        if st.session_state.prediction:
            pr = st.session_state.prediction
            st.markdown("---")
            st.markdown('<div class="sub-header">Detailed Analysis</div>', unsafe_allow_html=True)

            colA, colB = st.columns(2)

            with colA:
                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=pr["needed"],
                        domain={"x": [0, 1], "y": [0, 1]},
                        title={"text": "Infrastructure Need"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "darkred"},
                            "steps": [
                                {"range": [0, 30], "color": "green"},
                                {"range": [30, 70], "color": "yellow"},
                                {"range": [70, 100], "color": "red"},
                            ],
                        },
                    )
                )
                fig.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)

            with colB:
                p = pr["needed"]
                if p >= 70:
                    st.warning("🚨 **High Priority Investment Needed**")
                    st.write("- Urgent infrastructure development required")
                    st.write("- Focus on transportation, power, water, sanitation")
                    st.write("- Suggested planning horizon: 3–7 years")
                elif p >= 40:
                    st.info("⚠️ **Moderate Investment Recommended**")
                    st.write("- Strategic infrastructure planning needed")
                    st.write("- Focus on sustainable + resilient upgrades")
                    st.write("- Suggested planning horizon: 5–10 years")
                else:
                    st.success("✅ **Maintenance & Optimization**")
                    st.write("- Maintain & optimize existing infrastructure")
                    st.write("- Focus on smart city initiatives, efficiency, renewals")
                    st.write("- Suggested planning horizon: continuous")

    # ---------------- Training
    with tab3:
        st.markdown('<div class="sub-header">Model Training (Cloud-safe)</div>', unsafe_allow_html=True)

        st.markdown(
            "<p class='tiny-note'>Streamlit Cloud pe 3 million samples recommended nahi. Default lightweight training use karo.</p>",
            unsafe_allow_html=True,
        )

        colL, colR = st.columns([2, 1])

        with colL:
            st.markdown("### ⚙️ Training Settings")
            n_samples = st.slider("Synthetic Samples", min_value=2000, max_value=200000, value=25000, step=1000)
            seed = st.number_input("Random Seed", min_value=1, max_value=10_000_000, value=RANDOM_SEED_DEFAULT, step=1)

            if st.button("🔥 TRAIN MODEL", type="primary", use_container_width=True):
                try:
                    progress = st.progress(0)
                    status = st.empty()

                    status.write("Generating synthetic data…")
                    # chunk generation so UI updates (but still fast)
                    chunk = min(50000, n_samples)
                    parts = []
                    created = 0
                    while created < n_samples:
                        take = min(chunk, n_samples - created)
                        parts.append(make_synthetic_training_data(base_df, take, seed + created))
                        created += take
                        progress.progress(min(int(created / n_samples * 40), 40))  # 0-40%
                    train_df = pd.concat(parts, ignore_index=True)

                    status.write("Training GradientBoosting model…")
                    progress.progress(55)

                    model, metrics = train_gb_model(train_df, seed)
                    progress.progress(85)

                    meta = {
                        "trained_at": datetime.utcnow().isoformat() + "Z",
                        "seed": int(seed),
                        "n_samples": int(n_samples),
                        "metrics": metrics,
                        "confidence_hint": float(np.clip(70 + metrics["accuracy"] * 30, 75, 99)),
                        "feature_cols": FEATURE_COLS,
                    }
                    save_model(model, meta)
                    progress.progress(100)
                    status.success("Training complete ✅ Model saved.")

                    st.session_state.model_meta = meta

                except Exception as e:
                    st.error(f"Training failed: {e}")

        with colR:
            st.markdown("### 📦 Model Status")
            model, meta = load_model_if_exists()
            if model is None:
                st.warning("⚠️ No saved model yet")
                st.info("Click **TRAIN MODEL** to create one.")
            else:
                m = meta.get("metrics", {})
                st.success("✅ Model Ready")
                st.metric("Samples", f"{meta.get('n_samples', '—')}")
                st.metric("Accuracy", f"{m.get('accuracy', 0)*100:.1f}%")
                st.metric("F1", f"{m.get('f1', 0):.3f}")

        # Explainability
        model, meta = load_model_if_exists()
        if model is not None:
            st.markdown("---")
            st.markdown("### 🔎 Explainability (Feature Importance)")
            try:
                fi = pd.DataFrame(
                    {"feature": FEATURE_COLS, "importance": model.feature_importances_}
                ).sort_values("importance", ascending=False)

                fig_fi = px.bar(fi, x="feature", y="importance", title="Feature Importance (GradientBoosting)")
                st.plotly_chart(fig_fi, use_container_width=True)
                st.dataframe(fi, use_container_width=True)
            except Exception as e:
                st.error(f"Explainability failed: {e}")

        st.markdown("---")
        st.markdown("### 📈 Model Information")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.write("**Algorithm:** Gradient Boosting (Classifier)")
        with c2:
            st.write(f"**Features:** {len(FEATURE_COLS)}")
        with c3:
            st.write("**Cloud-safe:** Yes (lightweight defaults)")

    # Footer
    st.markdown("---")
    st.markdown(
        f"""
        <div style="text-align:center; color:#666; padding: 10px;">
            <p>🌍 Global Infrastructure AI | Offline demo dataset | Updated: {datetime.now().strftime('%Y-%m-%d')}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
