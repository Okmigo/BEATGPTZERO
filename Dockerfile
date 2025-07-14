# Dockerfile

# --- Build Stage ---
FROM python:3.10.13-slim-bookworm as builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN pip install --upgrade pip

# Copy only requirements first for caching
COPY requirements.txt .

RUN pip wheel --no-cache-dir --wheel-dir /app/wheels -r requirements.txt


# --- Final Stage ---
FROM python:3.10.13-slim-bookworm

WORKDIR /app

RUN adduser --system --group --no-create-home appuser
USER appuser

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt .

RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt

# Download and cache models as the appuser
RUN python -m spacy download en_core_web_lg && \
    python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('wordnet', quiet=True)"

# Copy the application source code
COPY --chown=appuser:appuser . .

EXPOSE 8080

# CORRECTED: CMD instruction format for gunicorn
CMD ["gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8080", "app:app"]
