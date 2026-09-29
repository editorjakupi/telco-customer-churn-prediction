#!/usr/bin/env python3
"""
Telco Customer Churn Prediction - Fixad Streamlit App
"""

import streamlit as st
import pandas as pd
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime

from google_translate import render_translate_sidebar
from streamlit_parent_inject import inject_parent_css, inject_parent_js, inject_react_dom_patch
from streamlit_theme_force import inject_theme_force
from streamlit_theme_widgets import inject_widget_theme, streamlit_widget_theme_css

# Feature engineering funktioner; dessa funktioner måste skapas för att matcha modellens förväntningar
def tenure_group(tenure):
    if tenure <= 12:
        return '0-12 months'
    elif tenure <= 24:
        return '12-24 months'
    elif tenure <= 36:
        return '24-36 months'
    elif tenure <= 48:
        return '36-48 months'
    else:
        return '48+ months'

def charges_group(charges):
    if charges <= 35:
        return 'Low'
    elif charges <= 70:
        return 'Medium'
    else:
        return 'High'

# Konfiguration
st.set_page_config(
    page_title="Telco Churn Prediction",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

THEME_KEY = "telco_ui_theme"
APP_BUILD = "lux-2026-09-29b"
UPLOAD_DF_KEY = "telco_uploaded_df"
REQUIRED_UPLOAD_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "Churn",
]

SEO_TITLE = "Telco Churn Prediction | Retention risk atelier"
SEO_DESCRIPTION = (
    "Score individual telecom churn risk with a Random Forest model, "
    "upload your own Telco-schema dataset, and explore high-risk customers."
)
SEO_CANONICAL = "https://churn.editorjakupi.com/"


def inject_seo_meta() -> None:
    """Set title/description/canonical on the parent Streamlit document for crawlers."""
    title = json.dumps(SEO_TITLE)
    desc = json.dumps(SEO_DESCRIPTION)
    canon = json.dumps(SEO_CANONICAL)
    inject_parent_js(
        f"""
var win = window.parent || window;
var doc = win.document;
if (!doc || !doc.head) return;
doc.title = {title};
function setMeta(attr, key, content) {{
  var sel = 'meta[' + attr + '="' + key + '"]';
  var el = doc.querySelector(sel);
  if (!el) {{
    el = doc.createElement('meta');
    el.setAttribute(attr, key);
    doc.head.appendChild(el);
  }}
  el.setAttribute('content', content);
}}
function setLink(rel, href) {{
  var el = doc.querySelector('link[rel="' + rel + '"]');
  if (!el) {{
    el = doc.createElement('link');
    el.setAttribute('rel', rel);
    doc.head.appendChild(el);
  }}
  el.setAttribute('href', href);
}}
setMeta('name', 'description', {desc});
setMeta('name', 'robots', 'index, follow');
setMeta('property', 'og:title', {title});
setMeta('property', 'og:description', {desc});
setMeta('property', 'og:url', {canon});
setMeta('property', 'og:type', 'website');
setLink('canonical', {canon});
"""
    )


def inject_telco_theme(theme: str) -> None:
    """Luxury light/dark theme — parent DOM inject (not sandboxed st.html)."""
    if theme == "dark":
        vars_block = """
:root, .stApp, [data-testid="stAppViewContainer"] {
  --telco-bg-1: #05080f;
  --telco-bg-2: #0b1220;
  --telco-bg-3: #121a2b;
  --telco-surface: rgba(18, 26, 43, 0.94);
  --telco-surface-2: #1a2438;
  --telco-surface-border: rgba(94, 234, 212, 0.18);
  --telco-text: #eef3fb;
  --telco-text-muted: #9aabbf;
  --telco-accent: #5eead4;
  --telco-accent-2: #38bdf8;
  --telco-accent-soft: rgba(94, 234, 212, 0.12);
  --telco-champagne: #e8d5a3;
  --telco-shadow: 0 24px 60px rgba(0, 0, 0, 0.55);
  --telco-hero-glow: rgba(94, 234, 212, 0.18);
  --telco-field-bg: #161f33;
  --telco-field-border: rgba(148, 163, 184, 0.35);
}
"""
    else:
        vars_block = """
:root, .stApp, [data-testid="stAppViewContainer"] {
  --telco-bg-1: #f3f7fb;
  --telco-bg-2: #e7eef6;
  --telco-bg-3: #d9e4f0;
  --telco-surface: rgba(255, 255, 255, 0.92);
  --telco-surface-2: #ffffff;
  --telco-surface-border: rgba(15, 55, 90, 0.12);
  --telco-text: #0b1b2e;
  --telco-text-muted: #4a6078;
  --telco-accent: #0f766e;
  --telco-accent-2: #0369a1;
  --telco-accent-soft: rgba(15, 118, 110, 0.12);
  --telco-champagne: #8a6d2f;
  --telco-shadow: 0 18px 48px rgba(12, 40, 72, 0.12);
  --telco-hero-glow: rgba(15, 118, 110, 0.16);
  --telco-field-bg: #ffffff;
  --telco-field-border: rgba(15, 55, 90, 0.18);
}
"""

    base = """
.stApp {
  background:
    radial-gradient(ellipse 70% 40% at 10% 0%, var(--telco-hero-glow), transparent 55%),
    radial-gradient(ellipse 55% 35% at 95% 5%, rgba(56,189,248,0.12), transparent 50%),
    linear-gradient(165deg, var(--telco-bg-1) 0%, var(--telco-bg-2) 48%, var(--telco-bg-3) 100%);
  color: var(--telco-text) !important;
  color-scheme: inherit;
}
.block-container { padding-top: 0.55rem; padding-bottom: 1.25rem; max-width: 1280px; }
/* Compact Customer Prediction: less vertical chrome */
.pred-compact .telco-hero { padding: 1rem 1.25rem 1.1rem; margin-bottom: 0.75rem; }
.pred-compact .main-header { font-size: clamp(1.65rem, 3vw, 2.15rem) !important; margin-bottom: 0.25rem !important; }
.pred-compact .hero-subtitle { font-size: 0.9rem !important; line-height: 1.35 !important; }
.pred-compact .hero-meta { margin-top: 0.65rem !important; }
.pred-compact .section-header { font-size: 1.25rem !important; margin: 0.35rem 0 0.45rem !important; padding-bottom: 0.3rem !important; }
.pred-compact .form-panel { padding: 0.45rem 0.75rem 0.15rem !important; margin: 0.35rem 0 0.45rem !important; border-radius: 12px !important; }
.pred-compact .form-panel-title { margin: 0 0 0.35rem !important; font-size: 0.68rem !important; }
.pred-compact .prediction-card { padding: 0.85rem 1rem !important; margin: 0.5rem 0 !important; }
.pred-compact [data-testid="stVerticalBlock"] > div { gap: 0.25rem; }
.pred-compact div[data-testid="stSlider"] { padding-bottom: 0.15rem; }
.pred-compact .stCaption { margin-bottom: 0.25rem !important; }
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stHeader"] button, [data-testid="stHeader"] span,
[data-testid="stSidebarCollapsedControl"] button, [data-testid="stSidebarCollapsedControl"] span,
[data-testid="stSidebarCollapseButton"] button, [data-testid="stSidebarCollapseButton"] span {
  color: var(--telco-text) !important; -webkit-text-fill-color: var(--telco-text) !important;
}
[data-testid="stSidebar"] {
  background: linear-gradient(185deg, var(--telco-bg-3) 0%, var(--telco-bg-1) 100%);
  border-right: 1px solid var(--telco-surface-border);
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown li,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {
  color: var(--telco-text) !important;
  -webkit-text-fill-color: var(--telco-text) !important;
  font-family: 'Outfit', system-ui, sans-serif;
  opacity: 1 !important;
}
[data-testid="stIconMaterial"],
.material-symbols-rounded,
span[class*="material-symbols"],
[data-testid="stSidebarCollapseButton"] *,
[data-testid="stSidebarCollapsedControl"] *,
[data-testid="stExpanderToggleIcon"] {
  font-family: 'Material Symbols Rounded' !important;
  font-weight: normal !important;
  letter-spacing: normal !important;
  text-transform: none !important;
  font-feature-settings: 'liga' !important;
}

/* —— Luxury hero —— */
.telco-hero {
  position: relative; overflow: hidden;
  text-align: left; padding: 2.4rem 2.2rem 2.5rem; margin-bottom: 1.6rem;
  border-radius: 18px; background: var(--telco-surface);
  border: 1px solid var(--telco-surface-border); box-shadow: var(--telco-shadow);
}
.telco-hero::before {
  content: ""; position: absolute; inset: -20% -10% auto auto; width: 52%; height: 140%;
  background:
    radial-gradient(circle at 70% 40%, var(--telco-hero-glow), transparent 62%),
    conic-gradient(from 210deg at 60% 50%, transparent 0deg, rgba(94,234,212,0.08) 60deg, transparent 140deg);
  pointer-events: none;
}
.telco-kicker {
  font-family: 'Outfit', system-ui, sans-serif; font-size: 0.72rem; font-weight: 600;
  letter-spacing: 0.22em; text-transform: uppercase; color: var(--telco-accent); margin: 0 0 0.7rem;
  position: relative;
}
.main-header {
  font-family: 'Cormorant Garamond', Georgia, serif; font-size: clamp(2.4rem, 4.5vw, 3.35rem);
  font-weight: 600; letter-spacing: -0.02em; color: var(--telco-text); margin: 0 0 0.55rem; position: relative;
  line-height: 1.05;
}
.main-header em { font-style: italic; color: var(--telco-champagne); font-weight: 500; }
.hero-subtitle {
  font-family: 'Outfit', system-ui, sans-serif; font-size: 1.05rem; max-width: 38rem;
  color: var(--telco-text-muted); margin: 0; position: relative; line-height: 1.55;
}
.hero-meta {
  display: flex; flex-wrap: wrap; gap: 0.55rem; margin-top: 1.25rem; position: relative;
}
.hero-chip {
  font-family: 'Outfit', system-ui, sans-serif; font-size: 0.75rem; font-weight: 500;
  padding: 0.35rem 0.75rem; border-radius: 999px;
  border: 1px solid var(--telco-surface-border); color: var(--telco-text-muted);
  background: var(--telco-accent-soft);
}
.section-header {
  font-family: 'Cormorant Garamond', Georgia, serif; font-size: 1.75rem; font-weight: 600;
  color: var(--telco-text) !important; margin: 1.6rem 0 0.85rem 0; padding-bottom: 0.55rem;
  border-bottom: 1px solid var(--telco-surface-border); position: relative;
}
.section-header::after {
  content: ""; position: absolute; left: 0; bottom: -1px; width: 64px; height: 2px;
  background: linear-gradient(90deg, var(--telco-accent), var(--telco-accent-2));
}
.form-panel {
  background: var(--telco-surface); border: 1px solid var(--telco-surface-border);
  border-radius: 16px; padding: 1.15rem 1.25rem 0.85rem; margin: 0.75rem 0 1.1rem;
  box-shadow: var(--telco-shadow);
}
.form-panel-title {
  font-family: 'Outfit', system-ui, sans-serif; font-size: 0.78rem; font-weight: 600;
  letter-spacing: 0.14em; text-transform: uppercase; color: var(--telco-accent); margin: 0 0 0.85rem;
}
.prediction-card {
  background: var(--telco-surface); padding: 2rem 2rem 1.5rem; border-radius: 16px;
  box-shadow: var(--telco-shadow); margin: 1.25rem 0; border: 1px solid var(--telco-surface-border);
  color: var(--telco-text) !important;
}
.sidebar-brand { padding: 0.85rem 0.15rem 1.1rem; margin-bottom: 0.35rem; border-bottom: 1px solid var(--telco-surface-border); }
.sidebar-brand-title { font-family: 'Cormorant Garamond', Georgia, serif; font-size: 1.45rem; color: var(--telco-text); margin: 0; }
.sidebar-brand-tag { font-family: 'Outfit', system-ui, sans-serif; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.16em; color: var(--telco-accent); margin: 0.35rem 0 0; }
.sidebar-build { font-family: 'Outfit', system-ui, sans-serif; font-size: 0.65rem; color: var(--telco-text-muted); margin: 0.45rem 0 0; opacity: 0.75; }
.risk-critical, .risk-high, .risk-medium, .risk-low {
  color: #fff; padding: 1rem 1.25rem; border-radius: 12px; text-align: center; font-weight: 700;
  letter-spacing: 0.06em; font-family: 'Outfit', system-ui, sans-serif;
}
.risk-critical { background: linear-gradient(135deg, #b91c1c, #7f1d1d); }
.risk-high { background: linear-gradient(135deg, #c2410c, #9a3412); }
.risk-medium { background: linear-gradient(135deg, #0e7490, #155e75); }
.risk-low { background: linear-gradient(135deg, #0d9488, #0f766e); }
div[data-testid="stMetric"] {
  background: var(--telco-surface); border: 1px solid var(--telco-surface-border);
  border-radius: 14px; padding: 0.85rem 1rem; box-shadow: var(--telco-shadow);
}
div[data-testid="stMetric"] label { color: var(--telco-accent) !important; font-family: 'Outfit', system-ui, sans-serif !important; }
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #0f766e, #0e7490) !important; border: none !important;
  border-radius: 12px !important; font-family: 'Outfit', system-ui, sans-serif !important;
  font-weight: 650 !important; min-height: 48px !important; letter-spacing: 0.02em;
  box-shadow: 0 10px 28px rgba(15, 118, 110, 0.28) !important;
}

/* —— Nuclear widget contrast (fixes white fields / missing labels) —— */
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] *,
.stSelectbox label, .stSelectbox label *,
.stSlider label, .stSlider label *,
.stNumberInput label, .stNumberInput label *,
.stRadio label, .stRadio label *,
.stMultiSelect label, .stMultiSelect label * {
  color: var(--telco-text) !important;
  -webkit-text-fill-color: var(--telco-text) !important;
  opacity: 1 !important;
  visibility: visible !important;
}
.stTextInput input, .stNumberInput input, .stTextArea textarea,
div[data-baseweb="select"] > div,
div[data-baseweb="base-input"],
div[data-baseweb="input"],
div[data-baseweb="input"] > div,
[data-baseweb="input"] input,
.stApp input, .stApp textarea, .stApp select {
  background-color: var(--telco-field-bg) !important;
  background-image: none !important;
  color: var(--telco-text) !important;
  -webkit-text-fill-color: var(--telco-text) !important;
  border-color: var(--telco-field-border) !important;
  caret-color: var(--telco-text) !important;
}
div[data-baseweb="select"] span,
div[data-baseweb="select"] div {
  color: var(--telco-text) !important;
  -webkit-text-fill-color: var(--telco-text) !important;
}
[data-baseweb="popover"], [data-baseweb="menu"], [role="listbox"], [role="option"],
body > div[data-baseweb="popover"], body > div[data-baseweb="popover"] * {
  background-color: var(--telco-surface-2) !important;
  color: var(--telco-text) !important;
  -webkit-text-fill-color: var(--telco-text) !important;
}
@media (max-width: 768px) {
  .block-container { padding-left: 0.85rem !important; padding-right: 0.85rem !important; max-width: 100% !important; }
  .telco-hero { padding: 1.4rem 1.15rem 1.55rem; border-radius: 14px; }
  .main-header { font-size: clamp(1.85rem, 8vw, 2.4rem); }
  .stButton > button { min-height: 44px !important; width: 100%; }
  div[data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; }
}
"""
    css = (
        "@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,500;0,600;1,500&family=Outfit:wght@400;500;600;700&display=swap');\n"
        + vars_block
        + base
        + streamlit_widget_theme_css(theme, prefix="telco")
    )
    inject_parent_css(css, style_id="telco-theme-css")



def tt(key: str) -> str:
    """English UI strings — Google Translate handles other languages."""
    en = {
        "theme": "Theme",
        "appearance": "Appearance",
        "light": "Light",
        "dark": "Dark",
        "nav": "Navigation",
        "page_pred": "Customer Prediction",
        "page_risk": "Risk Explorer",
        "page_upload": "Upload dataset",
        "hero": "Telco Churn <em>Atelier</em>",
        "hero_sub": "A calm workspace to score individual churn risk and explore who needs attention next.",
        "predict": "Predict churn risk",
        "about": "About the App",
        "model_info": "Model Information",
        "section_customer": "Customer profile",
        "translate": "Translate",
        "section_upload": "Your dataset",
    }
    return en.get(key, key)

@st.cache_data
def load_model_and_info():
    """Load trained model and metadata"""
    try:
        model = joblib.load('best_churn_model.pkl')
        with open('model_info.json', 'r') as f:
            model_info = json.load(f)
        return model, model_info
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def themed_ink() -> str:
    return "#eef3fb" if st.session_state.get(THEME_KEY) == "dark" else "#0b1b2e"


def create_complete_input_form():
    """Compact customer form — denser grid so prediction fits one viewport."""
    st.markdown(f'<div class="section-header">{tt("section_customer")}</div>', unsafe_allow_html=True)
    st.caption("Same feature schema as model training. Fill the profile, then predict.")

    st.markdown('<div class="form-panel"><p class="form-panel-title">Profile · services · billing</p></div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        gender = st.selectbox("Gender", ["Male", "Female"], key="telco_gender")
        senior_citizen = st.selectbox("Senior Citizen", ["Yes", "No"], key="telco_senior")
        partner = st.selectbox("Partner", ["Yes", "No"], key="telco_partner")
        dependents = st.selectbox("Dependents", ["Yes", "No"], key="telco_dependents")
        tenure = st.slider("Tenure (months)", 0, 72, 12, key="telco_tenure")
    with c2:
        phone_service = st.selectbox("Phone Service", ["Yes", "No"], key="telco_phone")
        multiple_lines = st.selectbox(
            "Multiple Lines", ["Yes", "No", "No phone service"], key="telco_multi"
        )
        internet_service = st.selectbox(
            "Internet Service", ["DSL", "Fiber optic", "No"], key="telco_internet"
        )
        online_security = st.selectbox(
            "Online Security", ["Yes", "No", "No internet service"], key="telco_os"
        )
        online_backup = st.selectbox(
            "Online Backup", ["Yes", "No", "No internet service"], key="telco_ob"
        )
    with c3:
        device_protection = st.selectbox(
            "Device Protection", ["Yes", "No", "No internet service"], key="telco_dp"
        )
        tech_support = st.selectbox(
            "Tech Support", ["Yes", "No", "No internet service"], key="telco_ts"
        )
        streaming_tv = st.selectbox(
            "Streaming TV", ["Yes", "No", "No internet service"], key="telco_stv"
        )
        streaming_movies = st.selectbox(
            "Streaming Movies", ["Yes", "No", "No internet service"], key="telco_sm"
        )
        contract = st.selectbox(
            "Contract", ["Month-to-month", "One year", "Two year"], key="telco_contract"
        )
    with c4:
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"], key="telco_paper")
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            key="telco_pay",
        )
        monthly_charges = st.slider(
            "Monthly Charges ($)", 0.0, 200.0, 50.0, 1.0, key="telco_monthly"
        )
        total_charges = st.number_input(
            "Total Charges ($)", 0.0, 10000.0, 1000.0, 10.0, key="telco_total"
        )

    senior_citizen_numeric = 1 if senior_citizen == "Yes" else 0
    customer_data = pd.DataFrame({
        'gender': [gender],
        'SeniorCitizen': [senior_citizen_numeric],
        'Partner': [partner],
        'Dependents': [dependents],
        'tenure': [tenure],
        'PhoneService': [phone_service],
        'MultipleLines': [multiple_lines],
        'InternetService': [internet_service],
        'OnlineSecurity': [online_security],
        'OnlineBackup': [online_backup],
        'DeviceProtection': [device_protection],
        'TechSupport': [tech_support],
        'StreamingTV': [streaming_tv],
        'StreamingMovies': [streaming_movies],
        'Contract': [contract],
        'PaperlessBilling': [paperless_billing],
        'PaymentMethod': [payment_method],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges]
    })
    customer_data['TenureGroup'] = customer_data['tenure'].apply(tenure_group)
    customer_data['ChargesGroup'] = customer_data['MonthlyCharges'].apply(charges_group)
    return customer_data

def predict_churn(model, customer_data):
    """Predict churn with the model"""
    try:
        prediction = model.predict(customer_data)[0]
        probability = model.predict_proba(customer_data)[0]
        return prediction, probability
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None

def display_prediction(prediction, probability):
    """Display prediction — compact gauge + metrics for one-viewport fit."""
    if prediction is None:
        return
    
    churn_prob = probability[1]  # Probability for "Yes"
    
    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
    
    # Risk level based on probability
    if churn_prob >= 0.8:
        risk_class = "risk-critical"
        risk_text = "CRITICAL RISK"
        action = "Contact immediately"
    elif churn_prob >= 0.6:
        risk_class = "risk-high"
        risk_text = "HIGH RISK"
        action = "Offer special deal"
    elif churn_prob >= 0.3:
        risk_class = "risk-medium"
        risk_text = "MEDIUM RISK"
        action = "Monitor closely"
    else:
        risk_class = "risk-low"
        risk_text = "LOW RISK"
        action = "Standard retention"
    
    left, right = st.columns([1.15, 1])
    with left:
        st.markdown(f'<div class="{risk_class}">{risk_text}</div>', unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Churn", f"{churn_prob:.1%}")
        m2.metric("Retention", f"{1-churn_prob:.1%}")
        m3.metric("Action", action)
    with right:
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = churn_prob * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Churn risk %", 'font': {'size': 14}},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "#0d9488"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 60], 'color': "yellow"},
                    {'range': [60, 80], 'color': "orange"},
                    {'range': [80, 100], 'color': "red"}
                ],
            }
        ))
        fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def _prepare_telco_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize TotalCharges, drop empty rows, engineer Tenure/Charges groups."""
    out = df.copy()
    if "customerID" in out.columns:
        out = out.drop(columns=["customerID"])
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
    out = out.dropna()
    if "TenureGroup" not in out.columns:
        out["TenureGroup"] = out["tenure"].apply(tenure_group)
    if "ChargesGroup" not in out.columns:
        out["ChargesGroup"] = out["MonthlyCharges"].apply(charges_group)
    return out


@st.cache_data
def load_real_dataset():
    """Load the bundled Telco CSV (fallback when no upload)."""
    try:
        df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
        return _prepare_telco_frame(df)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None


def active_risk_dataset() -> tuple[pd.DataFrame | None, str]:
    """Prefer uploaded session CSV; else bundled training set."""
    uploaded = st.session_state.get(UPLOAD_DF_KEY)
    if isinstance(uploaded, pd.DataFrame) and not uploaded.empty:
        return uploaded, "uploaded"
    return load_real_dataset(), "bundled"


def validate_and_store_upload(raw: pd.DataFrame) -> pd.DataFrame | None:
    """Validate Telco schema and keep a prepared frame in session."""
    missing = [c for c in REQUIRED_UPLOAD_COLS if c not in raw.columns]
    if missing:
        st.error(
            "Missing required columns: "
            + ", ".join(missing)
            + ". CSV must match the Telco churn schema (including Churn)."
        )
        return None
    prepared = _prepare_telco_frame(raw)
    keep = [c for c in REQUIRED_UPLOAD_COLS if c in prepared.columns]
    for eng in ("TenureGroup", "ChargesGroup"):
        if eng in prepared.columns and eng not in keep:
            keep.append(eng)
    prepared = prepared[keep]
    if "Churn" not in prepared.columns:
        st.error("Column Churn is required for Risk Explorer scoring.")
        return None
    st.session_state[UPLOAD_DF_KEY] = prepared
    return prepared


def display_upload_section(model):
    """Upload own Telco-schema CSV for Risk Explorer scoring."""
    st.markdown(f'<div class="section-header">{tt("section_upload")}</div>', unsafe_allow_html=True)
    st.caption(
        "Upload a CSV with the same columns as the IBM Telco Customer Churn dataset "
        "(including Churn). Risk Explorer will score your rows with the trained model."
    )
    with st.expander("Required columns", expanded=False):
        st.code(", ".join(REQUIRED_UPLOAD_COLS), language=None)

    uploaded = st.file_uploader("Choose CSV", type=["csv"], key="telco_csv_uploader")
    if uploaded is not None:
        try:
            raw = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            raw = None
        if raw is not None:
            prepared = validate_and_store_upload(raw)
            if prepared is not None:
                st.success(
                    f"Loaded **{len(prepared):,}** rows · **{len(prepared.columns)}** columns "
                    f"from `{uploaded.name}`. Use **Risk Explorer** to score them."
                )
                st.dataframe(prepared.head(50), use_container_width=True)
                c1, c2, c3 = st.columns(3)
                c1.metric("Rows", f"{len(prepared):,}")
                churn_rate = (
                    prepared["Churn"].astype(str).str.lower().isin(["yes", "1", "true"]).mean()
                    if "Churn" in prepared.columns
                    else 0
                )
                c2.metric("Churn rate (label)", f"{churn_rate:.1%}")
                c3.metric("Source", "Your upload")

    current = st.session_state.get(UPLOAD_DF_KEY)
    if isinstance(current, pd.DataFrame) and not current.empty:
        st.info(f"Active upload in session: **{len(current):,}** customers.")
        if st.button("Clear uploaded dataset", key="telco_clear_upload"):
            del st.session_state[UPLOAD_DF_KEY]
            st.rerun()
    else:
        st.info("No upload yet — Risk Explorer uses the bundled 7,043-customer set.")

    if model is None:
        st.warning("Model not loaded; upload is stored but scoring needs the model.")


def get_high_risk_customers(model):
    """Score active dataset (upload or bundled) for churn risk."""
    df, _source = active_risk_dataset()

    if df is None:
        st.error("Could not load dataset")
        return pd.DataFrame()

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    try:
        churn_probabilities = model.predict_proba(X)[:, 1]
        results = pd.DataFrame({
            "Customer_ID": [f"C{i:04d}" for i in range(len(X))],
            "Churn_Probability": churn_probabilities,
            "Actual_Churn": y,
            "Risk_Level": pd.cut(
                churn_probabilities,
                bins=[0, 0.3, 0.6, 0.8, 1.0],
                labels=["Low", "Medium", "High", "Critical"],
            ),
        })
        return results.sort_values("Churn_Probability", ascending=False)
    except Exception as e:
        st.error(f"Error during risk analysis: {e}")
        return pd.DataFrame()


def display_risk_explorer(model):
    """Risk Explorer with working filtering"""
    st.markdown('<div class="section-header">Churn Risk Explorer</div>', unsafe_allow_html=True)
    st.markdown("**Creative Feature:** Identify customers with highest churn risk")
    df_active, source = active_risk_dataset()
    n = len(df_active) if df_active is not None else 0
    if source == "uploaded":
        st.info(f"**Using your upload:** Analyzing **{n:,}** customers with model predictions")
    else:
        st.info(f"**Using bundled dataset:** Analyzing **{n:,}** customers with actual churn predictions")

    # Controls
    col1, col2 = st.columns(2)
    with col1:
        top_n = st.selectbox("Number of customers to show", [10, 25, 50, 100, "All"], index=0)
    with col2:
        risk_threshold = st.slider("Risk threshold", 0.0, 1.0, 0.7, 0.05)

    # Analyze risk
    if st.button("Analyze Risk", type="primary"):
        with st.spinner("Loading dataset and analyzing customers..."):
            risk_data = get_high_risk_customers(model)
            
            if not risk_data.empty:
                # Filter by threshold
                high_risk = risk_data[risk_data['Churn_Probability'] >= risk_threshold]
                
                # Debug information
                st.write(f"Number of customers with risk >= {risk_threshold:.1%}: {len(high_risk)}")
                
                # Explanation
                st.info(f"**Filtering:** Showing only customers with churn risk >= {risk_threshold:.1%}")
                
                # Show summary based on filtered data
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total customers", len(risk_data))
                with col2:
                    critical = len(high_risk[high_risk['Risk_Level'] == 'Critical'])
                    st.metric("Critical (filtered)", critical)
                with col3:
                    high = len(high_risk[high_risk['Risk_Level'] == 'High'])
                    st.metric("High risk (filtered)", high)
                with col4:
                    avg_risk = high_risk['Churn_Probability'].mean() if len(high_risk) > 0 else 0
                    st.metric("Average (filtered)", f"{avg_risk:.1%}")
                
                # Show filtered results
                st.markdown("### High-Risk Customers")
                st.markdown(f"**Showing customers with risk >= {risk_threshold:.1%}**")
                
                if len(high_risk) > 0:
                    # Handle "All" option
                    if top_n == "All":
                        display_data = high_risk.copy()
                    else:
                        display_data = high_risk.head(top_n).copy()
                    display_data['Risk_%'] = (display_data['Churn_Probability'] * 100).round(1)
                    display_data['Action'] = display_data['Risk_Level'].map({
                        'Critical': 'Contact immediately',
                        'High': 'Special offer',
                        'Medium': 'Monitor',
                        'Low': 'Standard'
                    })
                    
                    st.dataframe(
                        display_data[['Customer_ID', 'Risk_%', 'Risk_Level', 'Action']],
                        use_container_width=True
                    )
                    
                    # Risk distribution
                    risk_dist = high_risk['Risk_Level'].value_counts()
                    if len(risk_dist) > 0:
                        fig = px.pie(
                            values=risk_dist.values,
                            names=risk_dist.index,
                            title="Risk Distribution (filtered)",
                            color_discrete_sequence=["#0f766e", "#0e7490", "#c2410c", "#b91c1c"],
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Export
                    csv = display_data.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"churn_risk_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info(f"No customers have risk >= {risk_threshold:.1%}. Try lowering the threshold.")
            else:
                st.error("Could not analyze risk data")

def main():
    """Main function"""
    if THEME_KEY not in st.session_state:
        st.session_state[THEME_KEY] = "dark"

    inject_react_dom_patch()
    inject_seo_meta()
    model, model_info = load_model_and_info()

    with st.sidebar:
        st.markdown(
            f"""
            <div class="sidebar-brand">
                <p class="sidebar-brand-title">Telco Retention</p>
                <p class="sidebar-brand-tag">Churn intelligence</p>
                <p class="sidebar-build notranslate">build {APP_BUILD}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(f"### {tt('theme')}")
        theme_choice = st.radio(
            tt("appearance"),
            ["Light", "Dark"],
            index=0 if st.session_state[THEME_KEY] == "light" else 1,
            horizontal=True,
            key="telco_theme_radio",
        )
        st.session_state[THEME_KEY] = "dark" if theme_choice == "Dark" else "light"

        render_translate_sidebar(page_language="en", theme=st.session_state[THEME_KEY])

        st.markdown(f"### {tt('nav')}")
        page = st.radio(
            "Select function",
            [tt("page_pred"), tt("page_upload"), tt("page_risk")],
            label_visibility="collapsed",
            key="telco_page_radio",
        )

        if model_info:
            with st.expander(tt("about"), expanded=False):
                st.markdown(
                    f"""
            **Telco Churn Prediction**

            Predict individual churn risk and explore high-risk customers in the full base.

            - Algorithm: Random Forest  
            - Training: 7,043 customers  
            - Features: {model_info.get('total_features', 21)}  
            - Validation: {model_info.get('best_accuracy', 0.742):.1%}  
            - Test: {model_info.get('test_accuracy', 0.728):.1%}
            """
                )
            with st.expander(tt("model_info"), expanded=False):
                st.markdown(f"**Model:** {model_info.get('best_model', 'Random Forest')}")
                st.markdown(f"**Validation Accuracy:** {model_info.get('best_accuracy', 0.742):.1%}")
                st.markdown(f"**Test Accuracy:** {model_info.get('test_accuracy', 0.728):.1%}")
                st.markdown(f"**Features:** {model_info.get('total_features', 21)}")
                st.markdown("**Training Data:** 7,043 customers")
                val_acc = model_info.get("best_accuracy", 0.742)
                test_acc = model_info.get("test_accuracy", 0.728)
                st.markdown(f"**Performance Gap:** {test_acc - val_acc:.1%}")

    theme = st.session_state[THEME_KEY]
    inject_telco_theme(theme)
    inject_widget_theme(theme, prefix="telco")
    inject_theme_force(theme, accent="#5eead4", accent_fg="#042f2e")

    is_pred = page == tt("page_pred")
    if is_pred:
        st.markdown('<div class="pred-compact">', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="telco-hero">
            <p class="telco-kicker">Retention desk</p>
            <div class="main-header">{tt("hero")}</div>
            <p class="hero-subtitle">{tt("hero_sub")}</p>
            <div class="hero-meta">
              <span class="hero-chip">Random Forest</span>
              <span class="hero-chip">7,043 customers</span>
              <span class="hero-chip">Live risk explorer</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if model is None:
        st.error("Could not load model. Check that files exist.")
        if is_pred:
            st.markdown("</div>", unsafe_allow_html=True)
        return

    if page == tt("page_pred"):
        customer_data = create_complete_input_form()
        if st.button(tt("predict"), type="primary", use_container_width=True):
            prediction, probability = predict_churn(model, customer_data)
            display_prediction(prediction, probability)
        st.markdown("</div>", unsafe_allow_html=True)
    elif page == tt("page_upload"):
        display_upload_section(model)
    elif page == tt("page_risk"):
        display_risk_explorer(model)

if __name__ == "__main__":
    main()
