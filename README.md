# Telco Customer Churn Prediction

## About the Project

This project predicts customer churn in telecommunications companies using machine learning. The solution includes a comprehensive Jupyter notebook for model development and a Streamlit web application for interactive predictions and risk analysis.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model (Optional)

```bash
# Run Jupyter Notebook to train the model
jupyter notebook telco_customer_churn_analysis.ipynb
```

### 3. Start the Streamlit App

```bash
streamlit run telco_churn_streamlit_app.py
```

## Model Performance

- **Best Model:** Random Forest (all features)
- **Validation Accuracy:** 74.2%
- **Test Accuracy:** 74.0%
- **Total Features:** 21
- **Dataset:** 7,043 customers with 66.4% churn rate

## Key Churn Factors

**High Churn Risk:**

- Monthly contracts (Month-to-month)
- Electronic check payment method
- High monthly charges (>$70)
- Short customer tenure (<12 months)
- No online security services

**Low Churn Risk:**

- Long-term contracts (1-2 years)
- Automatic payment methods
- Low monthly charges (<$50)
- Long customer tenure (>24 months)
- Online security and backup services

## Business Recommendations

**For High-Risk Customers:**

1. Offer discounts for longer contract periods
2. Improve customer service and support
3. Implement loyalty programs
4. Enhance online security features
5. Provide proactive customer support

## Streamlit App Features

### Customer Prediction

- Interactive form for customer data input
- Real-time churn probability prediction
- Risk level classification (Low/Medium/High)
- Business recommendations based on risk level

### Churn Risk Explorer

- Automatic risk analysis for all customers
- Interactive filtering by demographics, services, and payment methods
- Export functionality for targeted marketing campaigns
- Visual analytics and insights

## Project Files

- `telco_customer_churn_analysis.ipynb` - Main Jupyter notebook with complete ML workflow
- `telco_churn_streamlit_app.py` - Streamlit web application for predictions
- `individual_report.docx` - Professional report (Word document) following NBI template
- `best_churn_model.pkl` - Trained Random Forest model (created after training)
- `model_info.json` - Model metadata and performance metrics
- `WA_Fn-UseC_-Telco-Customer-Churn.csv` - Telco customer churn dataset
- `requirements.txt` - Python dependencies
- `README.md` - This documentation file

## Technical Details

### Model Comparison Results

| Model               | All Features | Top 5 Features | Difference |
| ------------------- | ------------ | -------------- | ---------- |
| Logistic Regression | 0.738        | 0.740          | 0.002      |
| Decision Tree       | 0.717        | 0.720          | 0.004      |
| **Random Forest**   | **0.742**    | 0.731          | -0.011     |
| Extra Trees         | 0.725        | 0.737          | 0.012      |

### Top 5 Most Important Features

1. MonthlyCharges (0.281)
2. tenure (0.119)
3. TotalCharges (0.085)
4. ChargesGroup_Low (0.068)
5. Contract_One year (0.049)

## Academic Context

This project was developed as part of the "Knowledge Control, AI - Theory and Application Part 1" course at NBI Handelsakademin. The solution demonstrates practical application of machine learning techniques for business problem-solving in the telecommunications industry.

## Report Structure

The individual report includes:

- Abstract and Introduction
- Theory and Methodology
- Results and Discussion (with Security, Creative Features, Challenges, and Future Improvements)
- Conclusions and Self-Evaluation
- Technical Appendix and References

## Getting Started

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the Streamlit app: `streamlit run telco_churn_streamlit_app.py`
4. Open the Jupyter notebook to explore the ML workflow: `jupyter notebook telco_customer_churn_analysis.ipynb`
