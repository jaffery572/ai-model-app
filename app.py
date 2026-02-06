# app.py - STREAMLIT CLOUD SAFE (ONNX MODEL)
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import onnxruntime as ort
from datetime import datetime
import requests

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="🌍 Global Infrastructure AI",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
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
        font-size: 1.2rem;
        color: #cbd5e1;
        border-left: 5px solid #3498db;
        padding-left: 12px;
        margin: 0.8rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 12px;
        color: white;
        box-shadow: 0 5px 18px rgba(0,0,0,0.2);
        margin: 0.5rem 0;
    }
    .prediction-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        box-shadow: 0 5px 18px rgba(0,0,0,0.2);
    }
    .risk-high {
        background: #ff4d4d;
        color: white;
        padding: 6px 12px;
        border-radius: 8px;
        font-weight: 800;
        display: inline-block;
    }
    .risk-medium {
        background: #ffb020;
        color: black;
        padding: 6px 12px;
        border-radius: 8px;
        font-weight: 800;
        display: inline-block;
    }
    .risk-low {
        background: #00cc66;
        color: white;
        padding: 6px 12px;
        border-radius: 8px;
        font-weight: 800;
        display: inline-block;
    }
    .small-note {
        color: #94a3b8;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Constants
# -----------------------------
MODEL_PATH = os.path.join("models", "infra_model.onnx")
FEATURES = ["Population_Millions", "GDP_per_capita_USD", "HDI_Index", "Urbanization_Rate"]

# -----------------------------
# Caching
# -----------------------------
@st.cache_resource
def load_onnx_model():
    """Load ONNX model (cloud safe)"""
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
        return sess
    except Exception as e:
        st.error(f"Model load failed: {e}")
        return None


@st.cache_data
def load_default_country_data():
    """Fallback country dataset (small, safe)"""
    data = {
        "Country": ["USA","CHN","IND","DEU","GBR","JPN","BRA","RUS","FRA","ITA","CAN","AUS","KOR","MEX","IDN",
                    "TUR","SAU","CHE","NLD","ESP","PAK","BGD","NGA","EGY","VNM","THA","ZAF","ARG","COL","MYS"],
        "Population_Millions": [331,1412,1408,83,68,125,215,144,67,59,38,26,51,129,278,85,36,8.6,17,47,240]()_
