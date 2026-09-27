# Telco Customer Churn Prediction

## About the Project

This project predicts customer churn for telecommunications companies using machine learning techniques. The solution includes a comprehensive Jupyter notebook for model development and a Streamlit web application for real-time predictions.

**Dataset:** [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://telco-customer-churn-prediction-editorjakupi.streamlit.app/)

Try the live application: https://telco-customer-churn-prediction-editorjakupi.streamlit.app/

The app uses a slate/navy palette with teal accents and adapts to **Streamlit light and dark mode** (Settings → Theme in the app menu). Switch themes anytime; cards, sidebar, and typography stay readable in both modes.

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

### 3. Start Streamlit App (local)

From the project root (where `best_churn_model.pkl` and the CSV live):

```bash
streamlit run telco_churn_streamlit_app.py
```

Open http://localhost:8501. Theme defaults come from `.streamlit/config.toml`; override appearance with Streamlit’s built-in light/dark toggle.

### 4. Run with Docker (optional)

```bash
docker build -t telco-churn .
docker run -p 8501:8501 -e PORT=8501 telco-churn
```

The container listens on `$PORT` (default **8501**) and binds to `0.0.0.0` for hosting platforms.

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
- `.streamlit/config.toml` - Default theme colors and server settings
- `Dockerfile` - Container image for always-on hosting
- `render.yaml` - Render Blueprint (Docker web service)
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

### Streamlit Community Cloud

1. Push this repository to GitHub (include `best_churn_model.pkl`, `model_info.json`, and `WA_Fn-UseC_-Telco-Customer-Churn.csv`).
2. Go to [share.streamlit.io](https://share.streamlit.io), connect the repo, and set the main file to `telco_churn_streamlit_app.py`.
3. Deploy. The app reads models and data from the repo root using relative paths.

### Render (always-on Docker)

1. Connect the GitHub repo in [Render](https://render.com).
2. Use **New → Blueprint** and point to `render.yaml`, or create a **Web Service** with **Docker** and this repo.
3. Render sets `PORT` automatically; the Dockerfile runs Streamlit on `$PORT` (fallback **8501**).
4. Health checks use `/_stcore/health`.

Free tiers may sleep when idle on some platforms; Docker on Render keeps the service defined for production-style hosting once upgraded or on a suitable plan.

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

## Always-on

- **Streamlit Cloud** hosts the live demo and auto-redeploys from `main`.
- GitHub Action **Keep Streamlit Awake** pings the app every 10 minutes so the free tier stays reachable.
- For dedicated always-on containers, use `Dockerfile` / `render.yaml` on Render or Railway.

