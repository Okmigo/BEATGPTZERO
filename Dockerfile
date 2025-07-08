# Update your Dockerfile
FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc build-essential curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download model during build
RUN python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; \
    AutoTokenizer.from_pretrained('prithivida/parrot_paraphraser_on_T5'); \
    AutoModelForSeq2SeqLM.from_pretrained('prithivida/parrot_paraphraser_on_T5')"

COPY . .

# Add offline mode environment variable
ENV TRANSFORMERS_OFFLINE=1 \
    PYTHONUNBUFFERED=1

# Fix the CMD instruction
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:3000", "rewriter:app"]
