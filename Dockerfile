FROM python:3.11-slim
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc build-essential curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download model during build
RUN python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('prithivida/parrot_paraphraser_on_T5')"

COPY . .

# Set production-optimized environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONFAULTHANDLER=1

# Use gunicorn with timeout settings
CMD ["gunicorn", "-w", "1", "-t", "300", "-b", "0.0.0.0:3000", "rewriter:app"]
