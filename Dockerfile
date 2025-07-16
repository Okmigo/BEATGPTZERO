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

# Set HuggingFace cache directory
ENV TRANSFORMERS_CACHE=/app/cache
RUN mkdir -p /app/cache

# Pre-download NLTK corpora
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"

# Pre-download paraphraser model
RUN python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; AutoTokenizer.from_pretrained('mrm8488/t5-small-finetuned-quora-for-paraphrasing'); AutoModelForSeq2SeqLM.from_pretrained('mrm8488/t5-small-finetuned-quora-for-paraphrasing')"

# Pre-download GPT-2 model for perplexity scoring
RUN python -c "from transformers import GPT2TokenizerFast, GPT2LMHeadModel; GPT2TokenizerFast.from_pretrained('gpt2'); GPT2LMHeadModel.from_pretrained('gpt2')"

# Pre-download RoBERTa detector model
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; AutoTokenizer.from_pretrained('openai-community/roberta-base-openai-detector'); AutoModelForSequenceClassification.from_pretrained('openai-community/roberta-base-openai-detector')"

# Pre-download SentenceTransformer model
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-MiniLM-L6-v2')"

# Copy app code
COPY . .

# Set environment and expose
ENV PORT=8080
EXPOSE 8080

# Run with Gunicorn (single worker, multi-threaded, extended timeout)
CMD ["gunicorn", "--workers=1", "--threads=4", "--timeout=900", "--bind=0.0.0.0:8080", "app:app"]
