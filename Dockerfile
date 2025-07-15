# Dockerfile

# CORRECTED: Use a newer PyTorch image with torch > v2.6 to resolve the security vulnerability
FROM pytorch/pytorch:2.6.0-cuda12.1-cudnn8-runtime

# Set the working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/models

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Copy requirements file
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create and switch to a non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

# Pre-download the model as root
RUN python3 -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; \
                model_name = 'tuner007/pegasus_paraphrase'; \
                AutoTokenizer.from_pretrained(model_name, cache_dir='/app/models'); \
                AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir='/app/models')"

# Switch to the non-root user
USER appuser

# Copy the application source code into the container
COPY --chown=appuser:appuser app.py .

# Expose the port the app will run on
EXPOSE 8080

# Command to run the application using Gunicorn
CMD ["gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8080", "app:app"]
