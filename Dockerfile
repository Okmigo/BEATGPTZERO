# Dockerfile

# --- Build Stage ---
FROM python:3.10.13-slim-bookworm as builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN pip install --upgrade pip

# Copy only requirements first for caching
COPY requirements.txt .

# Build the wheels as root
RUN pip wheel --no-cache-dir --wheel-dir /app/wheels -r requirements.txt


# --- Final Stage ---
FROM python:3.10.13-slim-bookworm

WORKDIR /app

# Copy assets from the builder stage as root
COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt .

# Create the non-root user
RUN adduser --system --group --no-create-home appuser

# Install packages from wheels as root
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt

# CORRECTED: Download all required NLTK models (punkt and wordnet)
RUN python -m spacy download en_core_web_lg && \
    python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('wordnet', quiet=True)"

# Copy the application source code and change ownership to the non-root user
COPY --chown=appuser:appuser . .

# NOW, switch to the non-root user for security
USER appuser

EXPOSE 8080

# Run the application as the non-root user
CMD ["gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8080", "app:app"]
