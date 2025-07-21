# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy app files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set env vars from .env if needed (Cloud Run can do this via dashboard too)

# Expose port & run
CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=8080"]
