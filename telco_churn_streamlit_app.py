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

# Ren och professionell styling
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    .section-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin: 1.5rem 0 1rem 0;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    
    .prediction-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.15);
        margin: 1rem 0;
        border: 1px solid rgba(52, 152, 219, 0.2);
    }
    
    .risk-critical {
        background: linear-gradient(135deg, #ff6b6b, #ee5a52);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #ffa726, #ff9800);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #42a5f5, #2196f3);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #66bb6a, #4caf50);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }
    
    .sidebar .sidebar-content {
        background: rgba(255, 255, 255, 0.95);
    }
    
    .stMarkdown {
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

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
    st.markdown('<div class="section-header">Customer Information</div>', unsafe_allow_html=True)
    
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
            'bar': {'color': "darkblue"},
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
                        fig = px.pie(values=risk_dist.values, names=risk_dist.index, 
                                   title="Risk Distribution (filtered)", color_discrete_sequence=px.colors.qualitative.Set3)
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
    # Header
    st.markdown('<div class="main-header">Telco Churn Prediction</div>', unsafe_allow_html=True)
    
    # Load model
    model, model_info = load_model_and_info()
    
    if model is None:
        st.error("Could not load model. Check that files exist.")
        return
    
    # Sidebar
    with st.sidebar:
        # About the App section (expandable)
        with st.expander("About the App", expanded=False):
            st.markdown(f"""
            **Telco Churn Prediction App**
            
            This application uses machine learning to predict customer churn for telecommunications companies.
            
            **Features:**
            - Individual customer churn prediction
            - Risk analysis for customer base
            - Real-time probability calculations
            - Export functionality for risk data
            
            **Model Details:**
            - Algorithm: Random Forest
            - Training: 7,043 customers
            - Features: {model_info.get('total_features', 21)} customer attributes
            - Validation Accuracy: {model_info.get('best_accuracy', 0.742):.1%}
            - Test Accuracy: {model_info.get('test_accuracy', 0.728):.1%}
            
            **Creative Feature:**
            Risk Explorer identifies customers with highest churn probability, enabling proactive retention strategies.
            
            **Business Value:**
            - Reduce customer churn
            - Optimize retention campaigns
            - Improve customer lifetime value
            - Data-driven decision making
            """)
        
        # Model Information section (expandable)
        with st.expander("Model Information", expanded=False):
            st.markdown(f"**Model:** {model_info.get('best_model', 'Random Forest')}")
            st.markdown(f"**Validation Accuracy:** {model_info.get('best_accuracy', 0.742):.1%}")
            st.markdown(f"**Test Accuracy:** {model_info.get('test_accuracy', 0.728):.1%}")
            st.markdown(f"**Features:** {model_info.get('total_features', 21)}")
            st.markdown(f"**Training Data:** 7,043 customers")
            st.markdown(f"**Algorithm:** Random Forest Classifier")
            st.markdown(f"**Cross-validation:** 5-fold")
            
            # Calculate performance gap
            val_acc = model_info.get('best_accuracy', 0.742)
            test_acc = model_info.get('test_accuracy', 0.728)
            gap = test_acc - val_acc
            st.markdown(f"**Performance Gap:** {gap:.1%} (excellent stability)")
        
        st.markdown("### Navigation")
        page = st.selectbox("Select function", ["Customer Prediction", "Risk Explorer"])
    
    # Main content
    if page == "Customer Prediction":
        # Input form
        customer_data = create_complete_input_form()
        
        # Prediction
        if st.button("Predict Churn", type="primary", use_container_width=True):
            prediction, probability = predict_churn(model, customer_data)
            display_prediction(prediction, probability)
    
    elif page == "Risk Explorer":
        display_risk_explorer(model)

if __name__ == "__main__":
    main()
