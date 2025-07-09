FROM python:3.11-slim-bullseye

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set cache directories
ENV TRANSFORMERS_CACHE=/app/.cache
ENV HF_HOME=/app/.cache
ENV TORCH_HOME=/app/.cache/torch
ENV PYTHONUNBUFFERED=1

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with optimizations
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download en_core_web_sm

# Copy application code
COPY . .

# Create startup script
RUN echo '#!/bin/bash\n\
port=${PORT:-8080}\n\
echo "Starting server on port $port"\n\
exec uvicorn app:app --host 0.0.0.0 --port $port --timeout-keep-alive 600' > /app/start.sh && \
    chmod +x /app/start.sh

# Create non-root user and set permissions
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/.cache && \
    chown -R appuser:appuser /app

USER appuser

# Expose port
EXPOSE 8080

# Health check (only checks web server, not model loading)
HEALTHCHECK --interval=30s --timeout=30s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

CMD ["/app/start.sh"]
