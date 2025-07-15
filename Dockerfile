# Dockerfile

# Use an official PyTorch image for stability and compatibility
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# Set the working directory
WORKDIR /app

# Set environment variables to prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

# Download and cache the NLP models as root before we switch user
RUN python3 -m spacy download en_core_web_lg && \
    python3 -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('wordnet', quiet=True)"

# Switch to the non-root user
USER appuser

# Copy the rest of the application source code into the container
COPY --chown=appuser:appuser . .

# Expose the port the app will run on
EXPOSE 8080

# Command to run the application using Gunicorn
CMD ["gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8080", "app:app"]
