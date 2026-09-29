FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD sh -c 'python -c "import os, urllib.request; urllib.request.urlopen(\"http://127.0.0.1:\" + os.environ.get(\"PORT\", \"8501\") + \"/app/_stcore/health\")"' || exit 1

CMD ["sh", "-c", "streamlit run telco_churn_streamlit_app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true --server.baseUrlPath=app --browser.gatherUsageStats=false"]
