# ---- Build image ----
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app
EXPOSE 8080
CMD ["gunicorn", "--workers=2", "--threads=4", "--bind=0.0.0.0:8080", "app:app"]
