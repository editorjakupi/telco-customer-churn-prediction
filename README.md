# Telco Customer Churn Prediction

Predict telecom customer churn with a Random Forest model and an interactive Streamlit app.

**Live:** [https://churn.editorjakupi.com](https://churn.editorjakupi.com)  
**Repo:** [github.com/editorjakupi/telco-customer-churn-prediction](https://github.com/editorjakupi/telco-customer-churn-prediction)  
**Hosting:** Hetzner CX23 `apps-nbg1` (`23.88.100.144`) — Streamlit in Docker behind shared Caddy + Let’s Encrypt

---

## Current version highlights

- **In-app Light / Dark** — theme injected into the **parent DOM** (not sandboxed `st.html`), so labels, selects, and sidebar stay readable
- **Google Translate** — English UI as source/default; themed sidebar picker (not a black box in light mode)
- **Mobile-friendly** — full-width primary actions, wrapping columns, readable type
- **Pages** — Customer Prediction form + Risk Explorer on the full base

English is the only hardcoded UI language; other languages come from Google Translate.

---

## Quick start

```bash
pip install -r requirements.txt
streamlit run telco_churn_streamlit_app.py
```

Train / refresh the model via `telco_customer_churn_analysis.ipynb` (outputs `best_churn_model.pkl`, `model_info.json`).

### Docker (local)

```bash
docker build -t telco-churn .
docker run -p 8501:8501 -e PORT=8501 telco-churn
```

---

## Deploy (Hetzner)

Production path: `/opt/churn` on `apps-nbg1`, container `churn-prod-app` on Docker network `deploy_gematrior`. Caddy serves `churn.editorjakupi.com`.

```bash
# on VPS
cd /opt/churn
docker compose up -d --build
docker exec gematrior-prod-caddy caddy reload --config /etc/caddy/Caddyfile
curl -sI https://churn.editorjakupi.com/ | head -5
```

Compose service **must not** be named `app` (alias collision with Gematrior). See `docker-compose.yml`.

Streamlit Community Cloud hosting for this app has been **removed**; GitHub remains the source repo.

---

## Model

| Metric | Value |
|--------|-------|
| Algorithm | Random Forest |
| Validation accuracy | ~74.2% |
| Test accuracy | ~72.8–74% |
| Features | 21 |
| Training rows | 7,043 |

Dataset: [Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## Theme & translate internals

- `streamlit_parent_inject.py` — push CSS into `window.parent.document`
- `streamlit_theme_widgets.py` / `streamlit_theme_force.py` — widget colors + scrollbars
- `google_translate.py` — visible picker; `pageLanguage=en`; label **English** (not “Original”)

---

## License

See `LICENSE`.
