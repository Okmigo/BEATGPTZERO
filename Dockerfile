FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create cache directory for transformers
RUN mkdir -p /app/.cache

# Set environment variables for transformers cache
ENV TRANSFORMERS_CACHE=/app/.cache
ENV HF_HOME=/app/.cache

# Copy the download_models script
COPY download_models.py .

# Download the models during build
RUN python download_models.py

# Change ownership of the cache
RUN chown -R 1000:1000 /app/.cache

# Copy application code
COPY . .

# Create startup script
RUN echo '#!/bin/bash\nport=${PORT:-8080}\necho "Starting server on port $port"\nexec uvicorn app:app --host 0.0.0.0 --port $port' > /app/start.sh && \
    chmod +x /app/start.sh

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

CMD ["/app/start.sh"]
