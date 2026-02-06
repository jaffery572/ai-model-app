import os
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="🌍 Global Infrastructure AI",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# CSS
# -----------------------------
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        background: linear-gradient(90deg, #1a2980, #26d0ce);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.25rem;
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

# -----------------------------
# Paths / constants
# -----------------------------
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "infrastructure_gb.joblib")

FEATURE_COLS = ["Population_Millions", "GDP_per_capita_USD", "HDI_Index", "Urbanization_Rate"]
TARGET_COL = "Infrastructure_Need"


# -----------------------------
# Safe session defaults
# -----------------------------
if "prediction" not in st.session_state:
    st.session_state.prediction = None


# -----------------------------
# Data loader (cached)
# -----------------------------
@st.cache_data(show_spinner=False)
def load_global_data() -> pd.DataFrame:
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

    def calc_need(row) -> int:
        score = 0
        if row["GDP_per_capita_USD"] < 5000:
            score += 2
        if row["HDI_Index"] < 0.7:
            score += 2
        if row["Urbanization_Rate"] < 50 and row["Population_Millions"] > 50:
            score += 1
        return 1 if score >= 3 else 0

    df[TARGET_COL] = df.apply(calc_need, axis=1)
    return df


def validate_input_df(df: pd.DataFrame) -> tuple[bool, str]:
    needed = set(["Country"] + FEATURE_COLS)
    missing = needed - set(df.columns)
    if missing:
        return False, f"Missing columns: {sorted(list(missing))}"
    for c in FEATURE_COLS:
        if not pd.api.types.is_numeric_dtype(df[c]):
            return False, f"Column '{c}' must be numeric."
    if (df["GDP_per_capita_USD"] <= 0).any():
        return False, "GDP_per_capita_USD must be > 0."
    if (df["HDI_Index"] < 0).any() or (df["HDI_Index"] > 1).any():
        return False, "HDI_Index must be between 0 and 1."
    if (df["Urbanization_Rate"] < 0).any() or (df["Urbanization_Rate"] > 100).any():
        return False, "Urbanization_Rate must be between 0 and 100."
    if (df["Population_Millions"] <= 0).any():
        return False, "Population_Millions must be > 0."
    return True, ""


# -----------------------------
# Heuristic predictor (always available)
# -----------------------------
def heuristic_prediction(country: dict) -> dict:
    gdp = float(country["GDP_per_capita_USD"])
    hdi = float(country["HDI_Index"])
    urb = float(country["Urbanization_Rate"])
    pop = float(country["Population_Millions"])

    need_score = (
        (1 - min(gdp / 50000.0, 1.0)) * 40.0
        + (1 - hdi) * 30.0
        + ((urb / 100.0) * 20.0 if gdp < 10000 else 0.0)
        + (10.0 if (pop > 100 and gdp < 5000) else 0.0)
    )
    p = float(np.clip(need_score, 0, 100))

    if p >= 70:
        rl, rc = "HIGH", "risk-high"
    elif p >= 40:
        rl, rc = "MEDIUM", "risk-medium"
    else:
        rl, rc = "LOW", "risk-low"

    return {
        "needed": p,
        "sufficient": 100.0 - p,
        "confidence": float(95.0 - abs(50.0 - p) / 2.0),
        "risk_level": rl,
        "risk_color": rc,
    }


# -----------------------------
# ML: synthetic training, save/load, predict
# -----------------------------
def make_synthetic_training_data(base_df: pd.DataFrame, n_samples: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(base_df), size=n_samples)
    core = base_df.iloc[idx][FEATURE_COLS].copy()

    core["Population_Millions"] = np.clip(core["Population_Millions"] * rng.normal(1.0, 0.12, n_samples), 0.5, 2000)
    core["GDP_per_capita_USD"] = np.clip(core["GDP_per_capita_USD"] * rng.normal(1.0, 0.18, n_samples), 300, 200000)
    core["HDI_Index"] = np.clip(core["HDI_Index"] + rng.normal(0.0, 0.03, n_samples), 0.3, 0.99)
    core["Urbanization_Rate"] = np.clip(core["Urbanization_Rate"] + rng.normal(0.0, 6.0, n_samples), 5, 99)

    df = core
    score = (
        (df["GDP_per_capita_USD"] < 5000).astype(int) * 2
        + (df["HDI_Index"] < 0.7).astype(int) * 2
        + ((df["Urbanization_Rate"] < 50) & (df["Population_Millions"] > 50)).astype(int)
    )
    df[TARGET_COL] = (score >= 3).astype(int)
    return df


def train_model(train_df: pd.DataFrame, seed: int) -> tuple[GradientBoostingClassifier, dict]:
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
        return payload.get("model"), payload.get("meta", {})
    return None, None


def ml_predict_proba(model: GradientBoostingClassifier, country: dict) -> float:
    X = np.array([[float(country[c]) for c in FEATURE_COLS]], dtype=float)
    proba = model.predict_proba(X)[0, 1]
    return float(np.clip(proba * 100.0, 0, 100))


# -----------------------------
# App
# -----------------------------
def main():
    st.markdown('<div class="main-header">🌍 GLOBAL INFRASTRUCTURE AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align:center; color:#666; margin-top:-3px;">Production-grade demo (Cloud-safe) + Real ML pipeline</p>',
        unsafe_allow_html=True,
    )

    base_df = load_global_data()

    # Sidebar CSV upload
    with st.sidebar:
        st.markdown("### 📥 Optional CSV Upload")
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        st.markdown('<p class="tiny-note">Required columns: Country + Population_Millions, GDP_per_capita_USD, HDI_Index, Urbanization_Rate</p>', unsafe_allow_html=True)

    df = base_df
    if uploaded is not None:
        try:
            user_df = pd.read_csv(uploaded)
            ok, msg = validate_input_df(user_df)
            if not ok:
                st.sidebar.error(msg)
            else:
                if TARGET_COL not in user_df.columns:
                    tmp = user_df.copy()
                    score = (
                        (tmp["GDP_per_capita_USD"] < 5000).astype(int) * 2
                        + (tmp["HDI_Index"] < 0.7).astype(int) * 2
                        + ((tmp["Urbanization_Rate"] < 50) & (tmp["Population_Millions"] > 50)).astype(int)
                    )
                    tmp[TARGET_COL] = (score >= 3).astype(int)
                    user_df = tmp
                df = user_df
                st.sidebar.success(f"Loaded {len(df)} rows ✅")
        except Exception as e:
            st.sidebar.error(f"CSV error: {e}")

    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🤖 AI Predictor", "⚡ Model Training"])

    # ---------------- Dashboard
    with tab1:
        model, meta = load_model_if_exists()
        acc = meta.get("metrics", {}).get("accuracy") if meta else None
        acc_str = f"{acc*100:.1f}%" if isinstance(acc, (int, float)) else "—"

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="metric-card"><h4>🌍 Countries</h4><h2>{len(df)}</h2></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><h4>🏗️ Needs</h4><h2>{int(df[TARGET_COL].sum())}</h2></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card"><h4>💰 Avg GDP</h4><h2>${df["GDP_per_capita_USD"].mean():,.0f}</h2></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="metric-card"><h4>📈 Model Acc</h4><h2>{acc_str}</h2></div>', unsafe_allow_html=True)

        left, right = st.columns(2)
        with left:
            top10 = df.sort_values("GDP_per_capita_USD", ascending=False).head(10).copy()
            top10["Need_Label"] = top10[TARGET_COL].map({0: "No", 1: "Yes"}).astype(str)

            fig1 = px.bar(
                top10,
                x="Country",
                y="GDP_per_capita_USD",
                title="Top 10 by GDP per Capita",
                color="Need_Label",
                color_discrete_map={"Yes": "#ff4d4d", "No": "#00cc66"},
            )
            st.plotly_chart(fig1, use_container_width=True)

        with right:
            safe_df = df[df["GDP_per_capita_USD"] > 0].copy()
            fig2 = px.scatter(
                safe_df,
                x="GDP_per_capita_USD",
                y="HDI_Index",
                size="Population_Millions",
                color="Continent" if "Continent" in safe_df.columns else None,
                hover_name="Country" if "Country" in safe_df.columns else None,
                title="GDP per Capita vs HDI",
                log_x=True,
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="sub-header">Country Data</div>', unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True)

    # ---------------- Predictor
    with tab2:
        st.markdown('<div class="sub-header">AI Predictor</div>', unsafe_allow_html=True)
        model, meta = load_model_if_exists()
        has_model = model is not None

        colA, colB = st.columns([2, 1])
        with colA:
            option = st.radio("Input Method:", ["Select Country", "Custom Input"], horizontal=True)

            country = None
            if option == "Select Country":
                selected = st.selectbox("Choose Country:", df["Country"].tolist() if "Country" in df.columns else [])
                if selected:
                    row = df[df["Country"] == selected].iloc[0]
                    country = {c: float(row[c]) for c in FEATURE_COLS}
            else:
                country = {
                    "Population_Millions": float(st.number_input("Population (Millions)", 0.5, 2500.0, 100.0)),
                    "GDP_per_capita_USD": float(st.number_input("GDP per Capita (USD)", 300.0, 250000.0, 5000.0)),
                    "HDI_Index": float(st.slider("HDI Index", 0.3, 0.99, 0.70)),
                    "Urbanization_Rate": float(st.slider("Urbanization Rate (%)", 5.0, 99.0, 50.0)),
                }

        with colB:
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            st.markdown("<h4 style='color: white; margin-top: 0;'>Run Prediction</h4>", unsafe_allow_html=True)

            mode = st.selectbox(
                "Mode",
                ["ML Model (if trained)", "Heuristic (always)"],
                index=0,
                disabled=not has_model,
            )

            if st.button("🚀 PREDICT", use_container_width=True, type="primary"):
                try:
                    if country is None:
                        st.error("No input provided.")
                    else:
                        pr = heuristic_prediction(country)

                        if mode.startswith("ML") and has_model:
                            p = ml_predict_proba(model, country)
                            pr["needed"] = p
                            pr["sufficient"] = 100.0 - p
                            pr["confidence"] = float(meta.get("confidence_hint", pr["confidence"]))

                            # update risk badge based on ML prob
                            if p >= 70:
                                pr["risk_level"], pr["risk_color"] = "HIGH", "risk-high"
                            elif p >= 40:
                                pr["risk_level"], pr["risk_color"] = "MEDIUM", "risk-medium"
                            else:
                                pr["risk_level"], pr["risk_color"] = "LOW", "risk-low"

                        st.session_state.prediction = pr
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

        if st.session_state.prediction:
            pr = st.session_state.prediction
            st.markdown("---")
            st.markdown('<div class="sub-header">Detailed Analysis</div>', unsafe_allow_html=True)

            l, r = st.columns(2)
            with l:
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

            with r:
                p = pr["needed"]
                if p >= 70:
                    st.warning("🚨 High Priority Investment Needed")
                    st.write("- Transportation, power, water & sanitation")
                    st.write("- 3–7 year roadmap")
                elif p >= 40:
                    st.info("⚠️ Moderate Investment Recommended")
                    st.write("- Resilient upgrades + sustainable planning")
                    st.write("- 5–10 year roadmap")
                else:
                    st.success("✅ Maintenance & Optimization")
                    st.write("- Maintain & optimize infrastructure")
                    st.write("- Smart city + efficiency initiatives")

    # ---------------- Training
    with tab3:
        st.markdown('<div class="sub-header">Model Training (Cloud-safe)</div>', unsafe_allow_html=True)
        st.markdown("<p class='tiny-note'>Heavy training avoid. Default lightweight training for Streamlit Cloud.</p>", unsafe_allow_html=True)

        left, right = st.columns([2, 1])

        with left:
            n_samples = st.slider("Synthetic Samples", 2000, 200000, 25000, 1000)
            seed = st.number_input("Random Seed", min_value=1, max_value=10_000_000, value=42, step=1)

            if st.button("🔥 TRAIN & SAVE MODEL", type="primary", use_container_width=True):
                try:
                    prog = st.progress(0)
                    status = st.empty()

                    status.write("Generating synthetic data…")
                    chunk = min(50000, n_samples)
                    parts = []
                    created = 0
                    while created < n_samples:
                        take = min(chunk, n_samples - created)
                        parts.append(make_synthetic_training_data(base_df, take, seed + created))
                        created += take
                        prog.progress(min(int(created / n_samples * 45), 45))
                    train_df = pd.concat(parts, ignore_index=True)

                    status.write("Training model…")
                    prog.progress(65)
                    model, metrics = train_model(train_df, seed)

                    meta = {
                        "trained_at": datetime.utcnow().isoformat() + "Z",
                        "seed": int(seed),
                        "n_samples": int(n_samples),
                        "metrics": metrics,
                        "confidence_hint": float(np.clip(70 + metrics["accuracy"] * 30, 75, 99)),
                        "feature_cols": FEATURE_COLS,
                    }

                    status.write("Saving model…")
                    os.makedirs(MODEL_DIR, exist_ok=True)
                    save_model(model, meta)
                    prog.progress(100)
                    status.success("✅ Model trained and saved!")
                except Exception as e:
                    st.error(f"Training failed: {e}")

        with right:
            model, meta = load_model_if_exists()
            if model is None:
                st.warning("⚠️ No saved model yet")
                st.info("Train once to enable ML mode.")
            else:
                m = meta.get("metrics", {})
                st.success("✅ Model Ready")
                st.metric("Samples", f"{meta.get('n_samples', '—')}")
                st.metric("Accuracy", f"{m.get('accuracy', 0)*100:.1f}%")
                st.metric("F1", f"{m.get('f1', 0):.3f}")

        model, meta = load_model_if_exists()
        if model is not None:
            st.markdown("---")
            st.markdown("### 🔎 Explainability (Feature Importance)")
            try:
                fi = pd.DataFrame({"feature": FEATURE_COLS, "importance": model.feature_importances_}).sort_values(
                    "importance", ascending=False
                )
                fig_fi = px.bar(fi, x="feature", y="importance", title="Feature Importance (GradientBoosting)")
                st.plotly_chart(fig_fi, use_container_width=True)
                st.dataframe(fi, use_container_width=True)
            except Exception as e:
                st.error(f"Explainability error: {e}")

    st.markdown("---")
    st.markdown(
        f"""
        <div style="text-align:center; color:#666; padding: 10px;">
            <p>🌍 Global Infrastructure AI | Streamlit Cloud Ready | Updated: {datetime.now().strftime('%Y-%m-%d')}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
