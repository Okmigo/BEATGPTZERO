# Stage 1: Builder for dependencies
FROM python:3.10-slim as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies into wheels
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# Stage 2: Runtime container
FROM python:3.10-slim

WORKDIR /app

# Install minimal runtime system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels and install
COPY --from=builder /wheels /wheels
COPY --from=builder /app/requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt

# Pre-download NLTK corpora
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"

# Copy app code
COPY . .

# Set environment and expose
ENV PORT=8080
EXPOSE 8080

# Run with Gunicorn (single worker, multi-threaded, extended timeout)
CMD ["gunicorn", "--workers=1", "--threads=4", "--timeout=900", "--bind=0.0.0.0:8080", "app:app"]
