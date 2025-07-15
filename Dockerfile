# Stage 1: Install dependencies
FROM python:3.10-slim AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --user -r requirements.txt

# Stage 2: Build final image
FROM python:3.10-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
ENV PATH=/root/.local/bin:$PATH
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . /app
# Pre-download NLTK data for wordnet and punkt to avoid runtime download
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"
# Expose port and set environment
ENV PORT 8080
EXPOSE 8080
# Use Gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
