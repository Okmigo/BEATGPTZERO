FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc build-essential curl protobuf-compiler && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data and tokenizer during build
RUN python3 -c "import nltk; nltk.download('punkt', quiet=True)" && \
    python3 -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('prithivida/parrot_paraphraser_on_T5', cache_dir='./hf_cache')"

COPY . .

# Optimize for production
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TRANSFORMERS_CACHE=/app/hf_cache

# Use gunicorn with proper PORT expansion
CMD ["sh", "-c", "gunicorn -w 1 -
