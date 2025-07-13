# Dockerfile

# --- Build Stage ---
# Use a specific, stable base image for reproducibility.
FROM python:3.10.13-slim-bookworm as builder

WORKDIR /app

# Set environment variables to optimize Python.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# CORRECTED: Note the space between "requirements.txt" and the final "."
COPY requirements.txt.
RUN pip install --upgrade pip && \
    pip wheel --no-cache-dir --wheel-dir /app/wheels -r requirements.txt


# --- Final Stage ---
FROM python:3.10.13-slim-bookworm

WORKDIR /app

# Create a non-root user for enhanced security.
RUN useradd --create-home appuser
USER appuser

# Copy and install dependencies from the build stage's wheelhouse.
COPY --from=builder /app/wheels /wheels
# CORRECTED: Note the space between "requirements.txt" and the final "."
COPY requirements.txt.
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt

# Download NLP models as the non-root user during the build process.
# This caches the models in the image layer, dramatically speeding up container startup.
RUN python -m spacy download en_core_web_lg && \
    python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"

# CORRECTED: Note the space between "app.py" and the final "."
COPY --chown=appuser:appuser app.py.

# Expose the port the app runs on. Cloud Run uses the PORT env var.
EXPOSE 8080

# --- Corrected Command to run the application with Gunicorn ---
# -w 1: Use a single worker to prevent memory duplication of large ML models. This is critical.
# --threads 8: Allow the single worker to handle multiple concurrent requests using async I/O.
# -k uvicorn.workers.UvicornWorker: The correct worker class for a FastAPI application.
# -b 0.0.0.0:8080: Bind to the host and port. Cloud Run will map this to the external port.
CMD ["gunicorn", "-w", "1", "--threads", "8", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8080", "app:app"]
