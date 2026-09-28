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

from google_translate import inject_google_translate
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


def inject_telco_theme(theme: str) -> None:
    """In-app light/dark theme (same idea as SmartFood / Diamonds)."""
    if theme == "dark":
        vars_block = """
        :root, .stApp, [data-testid="stAppViewContainer"] {
            --telco-bg-1: #070d16;
            --telco-bg-2: #0f172a;
            --telco-surface: rgba(30, 41, 59, 0.78);
            --telco-surface-border: rgba(148, 163, 184, 0.14);
            --telco-text: #f1f5f9;
            --telco-text-muted: #94a3b8;
            --telco-accent: #2dd4bf;
            --telco-accent-soft: rgba(45, 212, 191, 0.14);
            --telco-navy: #e2e8f0;
            --telco-shadow: 0 18px 48px rgba(0, 0, 0, 0.5);
            --telco-hero-glow: rgba(45, 212, 191, 0.2);
        }
        """
    else:
        vars_block = """
        :root, .stApp, [data-testid="stAppViewContainer"] {
            --telco-bg-1: #eef4f8;
            --telco-bg-2: #dde7f0;
            --telco-surface: rgba(255, 255, 255, 0.9);
            --telco-surface-border: rgba(15, 23, 42, 0.09);
            --telco-text: #0f172a;
            --telco-text-muted: #475569;
            --telco-accent: #0f766e;
            --telco-accent-soft: rgba(15, 118, 110, 0.14);
            --telco-navy: #1e3a5f;
            --telco-shadow: 0 14px 40px rgba(15, 23, 42, 0.1);
            --telco-hero-glow: rgba(15, 118, 110, 0.22);
        }
        """

    markup = f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Instrument+Serif:ital@0;1&display=swap" rel="stylesheet">
<style>
{vars_block}

.stApp {{
    background:
      radial-gradient(ellipse 85% 50% at 50% -12%, var(--telco-hero-glow), transparent 58%),
      linear-gradient(155deg, var(--telco-bg-1) 0%, var(--telco-bg-2) 52%, var(--telco-bg-1) 100%);
}}

.block-container {{
    padding-top: 1.35rem;
    max-width: 1180px;
}}

[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, var(--telco-bg-2) 0%, var(--telco-bg-1) 100%);
    border-right: 1px solid var(--telco-surface-border);
}}

[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown li,
[data-testid="stSidebar"] label {{
    color: var(--telco-text) !important;
    font-family: 'DM Sans', system-ui, sans-serif;
}}

.telco-hero {{
    text-align: center;
    padding: 2.15rem 1.5rem 2.35rem;
    margin-bottom: 1.5rem;
    border-radius: 8px;
    background: var(--telco-surface);
    border: 1px solid var(--telco-surface-border);
    box-shadow: var(--telco-shadow);
    position: relative;
    overflow: hidden;
}}

.telco-hero::before {{
    content: "";
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse 80% 60% at 50% -20%, var(--telco-hero-glow), transparent 70%);
    pointer-events: none;
}}

.main-header {{
    font-family: 'Instrument Serif', Georgia, serif;
    font-size: 2.55rem;
    font-weight: 400;
    letter-spacing: -0.01em;
    color: var(--telco-text);
    margin: 0 0 0.4rem 0;
    position: relative;
}}

.hero-subtitle {{
    font-family: 'DM Sans', system-ui, sans-serif;
    font-size: 1.05rem;
    color: var(--telco-text-muted);
    font-weight: 400;
    margin: 0;
    position: relative;
}}

.hero-accent-line {{
    width: 72px;
    height: 2px;
    margin: 1.1rem auto 0;
    background: linear-gradient(90deg, transparent, var(--telco-accent), transparent);
    position: relative;
}}

.section-header {{
    font-family: 'Instrument Serif', Georgia, serif;
    font-size: 1.55rem;
    font-weight: 400;
    color: var(--telco-text);
    margin: 1.85rem 0 1rem 0;
    padding-bottom: 0.55rem;
    border-bottom: 1px solid var(--telco-accent-soft);
    position: relative;
}}

.section-header::after {{
    content: "";
    position: absolute;
    left: 0;
    bottom: -1px;
    width: 56px;
    height: 2px;
    background: var(--telco-accent);
}}

.prediction-card {{
    background: var(--telco-surface);
    padding: 2rem 2rem 1.5rem;
    border-radius: 8px;
    box-shadow: var(--telco-shadow);
    margin: 1.25rem 0;
    border: 1px solid var(--telco-surface-border);
    backdrop-filter: blur(12px);
}}

.sidebar-brand {{
    padding: 1rem 0.25rem 1.25rem;
    margin-bottom: 0.5rem;
    border-bottom: 1px solid var(--telco-surface-border);
}}

.sidebar-brand-title {{
    font-family: 'Instrument Serif', Georgia, serif;
    font-size: 1.25rem;
    font-weight: 400;
    color: var(--telco-text);
    margin: 0;
}}

.sidebar-brand-tag {{
    font-family: 'DM Sans', system-ui, sans-serif;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--telco-accent);
    margin: 0.3rem 0 0;
}}

.risk-critical {{
    background: linear-gradient(135deg, #b91c1c, #991b1b);
    color: #fff;
    padding: 1rem 1.25rem;
    border-radius: 8px;
    text-align: center;
    font-weight: 700;
    letter-spacing: 0.04em;
    font-family: 'DM Sans', system-ui, sans-serif;
    box-shadow: 0 8px 24px rgba(185, 28, 28, 0.35);
}}

.risk-high {{
    background: linear-gradient(135deg, #c2410c, #9a3412);
    color: #fff;
    padding: 1rem 1.25rem;
    border-radius: 8px;
    text-align: center;
    font-weight: 700;
    letter-spacing: 0.04em;
    font-family: 'DM Sans', system-ui, sans-serif;
    box-shadow: 0 8px 24px rgba(194, 65, 12, 0.3);
}}

.risk-medium {{
    background: linear-gradient(135deg, #0e7490, #155e75);
    color: #fff;
    padding: 1rem 1.25rem;
    border-radius: 8px;
    text-align: center;
    font-weight: 700;
    letter-spacing: 0.04em;
    font-family: 'DM Sans', system-ui, sans-serif;
    box-shadow: 0 8px 24px rgba(14, 116, 144, 0.3);
}}

.risk-low {{
    background: linear-gradient(135deg, #0d9488, #0f766e);
    color: #fff;
    padding: 1rem 1.25rem;
    border-radius: 8px;
    text-align: center;
    font-weight: 700;
    letter-spacing: 0.04em;
    font-family: 'DM Sans', system-ui, sans-serif;
    box-shadow: 0 8px 24px rgba(13, 148, 136, 0.35);
}}

div[data-testid="stMetric"] {{
    background: var(--telco-surface);
    border: 1px solid var(--telco-surface-border);
    border-radius: 8px;
    padding: 0.75rem 1rem;
    box-shadow: var(--telco-shadow);
}}

div[data-testid="stMetric"] label {{
    color: var(--telco-accent) !important;
    font-family: 'DM Sans', system-ui, sans-serif !important;
}}

.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, #0f766e, #115e59) !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', system-ui, sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em;
    box-shadow: 0 6px 20px rgba(13, 148, 136, 0.35);
}}

.prediction-card, .section-header {{
    color: var(--telco-text) !important;
}}
"""
    markup = markup + streamlit_widget_theme_css(theme, prefix="telco") + f"""
@media (max-width: 768px) {{
    .block-container {{
        padding-left: 0.85rem !important;
        padding-right: 0.85rem !important;
        max-width: 100% !important;
    }}
    .telco-hero {{
        padding: 1.35rem 1rem 1.5rem;
    }}
    .main-header {{
        font-size: clamp(1.7rem, 8vw, 2.2rem);
    }}
    .hero-subtitle {{
        font-size: 0.95rem;
    }}
    .prediction-card {{
        padding: 1.25rem 1rem;
    }}
    .stButton > button {{
        min-height: 44px !important;
        width: 100%;
    }}
    div[data-testid="stHorizontalBlock"] {{
        flex-wrap: wrap !important;
    }}
}}
</style>
        """
    inject = getattr(st, "html", None)
    if inject:
        inject(markup)
    else:
        st.warning("Upgrade Streamlit (>=1.33) so theme CSS does not leak as text.")


THEME_KEY = "telco_ui_theme"

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
        "hero": "Telco Churn Prediction",
        "hero_sub": "ML-powered retention insights for telecommunications customers",
        "predict": "Predict Churn",
        "about": "About the App",
        "model_info": "Model Information",
        "section_customer": "Customer Information",
        "translate": "Translate",
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

def create_complete_input_form():
    """Create complete input form with all features in correct order"""
    st.markdown(f'<div class="section-header">{tt("section_customer")}</div>', unsafe_allow_html=True)
    
    # Personal information
    st.markdown("**Personal Information**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["Yes", "No"])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
    
    with col2:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    
    with col3:
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    
    # Additional services
    st.markdown("**Services**")
    col4, col5, col6 = st.columns(3)
    
    with col4:
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    
    with col5:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    
    with col6:
        payment_method = st.selectbox("Payment Method", 
                                    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    
    # Charges
    st.markdown("**Charges**")
    col7, col8 = st.columns(2)
    
    with col7:
        monthly_charges = st.slider("Monthly Charges ($)", 0.0, 200.0, 50.0, 1.0)
    with col8:
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 1000.0, 10.0)
    
    # Convert Senior Citizen to numeric
    senior_citizen_numeric = 1 if senior_citizen == "Yes" else 0
    
    # Create customer data in exact same order as model expects
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
    
    # Add feature engineering
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
    """Display prediction with clean design"""
    if prediction is None:
        return
    
    churn_prob = probability[1]  # Probability for "Yes"
    
    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
    st.markdown("## Churn Prediction")
    
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
    
    # Display risk level
    st.markdown(f'<div class="{risk_class}">{risk_text}</div>', unsafe_allow_html=True)
    
    # Display probability
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Churn Probability", f"{churn_prob:.1%}")
    with col2:
        st.metric("Retention Probability", f"{1-churn_prob:.1%}")
    with col3:
        st.metric("Recommended Action", action)
    
    # Visual representation
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = churn_prob * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Churn Risk (%)"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "#0d9488"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 60], 'color': "yellow"},
                {'range': [60, 80], 'color': "orange"},
                {'range': [80, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

@st.cache_data
def load_real_dataset():
    """Load the real dataset"""
    try:
        # Load CSV file
        df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
        
        # Preprocess data (same as in notebook)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df = df.dropna()
        
        # Remove customerID
        df = df.drop('customerID', axis=1)
        
        # Add feature engineering
        df['TenureGroup'] = df['tenure'].apply(tenure_group)
        df['ChargesGroup'] = df['MonthlyCharges'].apply(charges_group)
        
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

def get_high_risk_customers(model):
    """Find high-risk customers with real data"""
    # Load real dataset
    df = load_real_dataset()
    
    if df is None:
        st.error("Could not load dataset")
        return pd.DataFrame()
    
    # Prepare data for model (same as in notebook)
    numerical_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 
                          'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                          'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
                          'Contract', 'PaperlessBilling', 'PaymentMethod']
    engineered_features = ['TenureGroup', 'ChargesGroup']
    
    # Create X and y
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    try:
        # Get churn probabilities for all customers
        churn_probabilities = model.predict_proba(X)[:, 1]
        
        # Create results
        results = pd.DataFrame({
            'Customer_ID': [f"C{i:04d}" for i in range(len(X))],
            'Churn_Probability': churn_probabilities,
            'Actual_Churn': y,
            'Risk_Level': pd.cut(churn_probabilities, 
                               bins=[0, 0.3, 0.6, 0.8, 1.0], 
                               labels=['Low', 'Medium', 'High', 'Critical'])
        })
        
        return results.sort_values('Churn_Probability', ascending=False)
    except Exception as e:
        st.error(f"Error during risk analysis: {e}")
        return pd.DataFrame()

def display_risk_explorer(model):
    """Risk Explorer with working filtering"""
    st.markdown('<div class="section-header">Churn Risk Explorer</div>', unsafe_allow_html=True)
    st.markdown("**Creative Feature:** Identify customers with highest churn risk")
    st.info("**Using real dataset:** Analyzing all 7,043 customers with actual churn predictions")
    
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
        st.session_state[THEME_KEY] = "light"

    # Load model first so sidebar can show accuracy
    model, model_info = load_model_and_info()

    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <p class="sidebar-brand-title">Telco Retention</p>
                <p class="sidebar-brand-tag">Churn intelligence</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(f"### {tt('translate')}")
        inject_google_translate(page_language="en")

        st.markdown(f"### {tt('theme')}")
        theme_choice = st.radio(
            tt("appearance"),
            ["Light", "Dark"],
            index=0 if st.session_state[THEME_KEY] == "light" else 1,
            horizontal=True,
            key="telco_theme_radio",
        )
        st.session_state[THEME_KEY] = "dark" if theme_choice == "Dark" else "light"

        st.markdown(f"### {tt('nav')}")
        page = st.radio(
            "Select function",
            [tt("page_pred"), tt("page_risk")],
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

    st.markdown(
        f"""
        <div class="telco-hero">
            <div class="main-header">{tt("hero")}</div>
            <p class="hero-subtitle">{tt("hero_sub")}</p>
            <div class="hero-accent-line"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if model is None:
        st.error("Could not load model. Check that files exist.")
        return

    if page == tt("page_pred"):
        customer_data = create_complete_input_form()
        if st.button(tt("predict"), type="primary", use_container_width=True):
            prediction, probability = predict_churn(model, customer_data)
            display_prediction(prediction, probability)
    elif page == tt("page_risk"):
        display_risk_explorer(model)

    # Inject theme last so widget CSS overrides Streamlit Emotion defaults
    inject_telco_theme(st.session_state[THEME_KEY])
    inject_widget_theme(st.session_state[THEME_KEY], prefix="telco")

if __name__ == "__main__":
    main()
