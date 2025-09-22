# Telco Customer Churn Prediction

## About the Project

This project predicts customer churn for telecommunications companies using machine learning techniques. The solution includes a comprehensive Jupyter notebook for model development and a Streamlit web application for real-time predictions.

**Dataset:** [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://telco-customer-churn-prediction-editorjakupi.streamlit.app/)

Try the live application: https://telco-customer-churn-prediction-editorjakupi.streamlit.app/

## Quick Start

### 1. Train the Model

```bash
# Run Jupyter Notebook to train the model
jupyter notebook telco_customer_churn_analysis.ipynb
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Start Streamlit App

```bash
streamlit run telco_churn_streamlit_app.py
```

## Model Performance

- **Best Model:** Random Forest (all features)
- **Validation Accuracy:** 74.2%
- **Test Accuracy:** 74.0%
- **Total Features:** 21 customer attributes
- **Dataset Size:** 7,043 customers
- **Churn Rate:** 66.4%

### Model Comparison (Validation Set)

| Model               | All Features | Top 5 Features | Difference |
| ------------------- | ------------ | -------------- | ---------- |
| Logistic Regression | 0.738        | 0.740          | 0.002      |
| Decision Tree       | 0.717        | 0.720          | 0.004      |
| Random Forest       | 0.742        | 0.731          | -0.011     |
| Extra Trees         | 0.725        | 0.737          | 0.012      |

## Key Churn Factors

**High Churn Risk:**

- Month-to-month contracts
- Electronic check payment
- High monthly charges (>$70)
- Short tenure (<12 months)
- No online security

**Low Churn Risk:**

- Long-term contracts (1-2 years)
- Automatic payment methods
- Low monthly charges (<$50)
- Long tenure (>24 months)
- Online security and backup services

## Recommended Actions

**For High Churn Risk Customers:**

1. Offer discounts for longer contract periods
2. Improve customer service and support
3. Implement loyalty programs
4. Enhance online security features
5. Provide proactive customer support

## Streamlit App Features

### Customer Prediction

- Real-time churn probability assessment
- Interactive customer information form
- Risk level classification (Low, Medium, High, Critical)
- Actionable recommendations

### Risk Explorer (Creative Feature)

- Bulk analysis of customer base
- High-risk customer identification
- Interactive filtering and visualization
- Export functionality for retention campaigns

## Project Files

- `telco_customer_churn_analysis.ipynb` - Main notebook with complete ML workflow
- `telco_churn_streamlit_app.py` - Streamlit application for predictions
- `individual_report.docx` - Individual report following NBI template
- `best_churn_model.pkl` - Trained model (created after training)
- `model_info.json` - Model metadata (created after training)
- `WA_Fn-UseC_-Telco-Customer-Churn.csv` - Dataset
- `requirements.txt` - Python dependencies

## Academic Context

This project was developed as part of the Knowledge Control for "AI - Theory and Application Part 1" course at NBI Handelsakademin. The project demonstrates practical application of machine learning concepts in a real-world business scenario.

### Report Structure

- Abstract
- Introduction with research questions
- Theory (ML fundamentals)
- Methodology
- Results and Discussion
- Conclusions
- Self-Evaluation

## Deployment

The Streamlit app is ready for deployment on Streamlit Cloud. Simply connect your GitHub repository and the app will automatically deploy with the trained model.

## Business Impact

This solution enables telecommunications companies to:

- Identify at-risk customers before they churn
- Implement targeted retention strategies
- Improve customer lifetime value
- Make data-driven business decisions
- Reduce overall churn rates

## Technical Details

- **Preprocessing:** ColumnTransformer with StandardScaler and OneHotEncoder
- **Feature Engineering:** TenureGroup and ChargesGroup creation
- **Hyperparameter Tuning:** GridSearchCV with 5-fold cross-validation
- **Model Evaluation:** Accuracy, Confusion Matrix, Classification Report
- **Deployment:** Streamlit with joblib model persistence
