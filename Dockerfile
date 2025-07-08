FROM python:3.11-slim

WORKDIR /app

# Install system dependencies + protobuf compiler
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc build-essential curl protobuf-compiler && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Preload models and nltk data
RUN python3 -c "import nltk; nltk.download('punkt', quiet=True)" && \
    python3 -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; \
    tokenizer = AutoTokenizer.from_pretrained('prithivida/parrot_paraphraser_on_T5', cache_dir='./hf_cache'); \
    model = AutoModelForSeq2SeqLM.from_pretrained('prithivida/parrot_paraphraser_on_T5', cache_dir='./hf_cache')"

COPY . .

ENV PORT=3000
EXPOSE 3000

# Optimize for production
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1

CMD ["python", "rewriter.py"]
