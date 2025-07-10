# Dockerfile

# Use Python 3.11-slim as a base image.
FROM python:3.11-slim

# Set the working directory in the container.
WORKDIR /app

# Copy requirements file first to leverage Docker layer caching.
COPY requirements.txt .

# Install Python dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# --- Pre-download the AI Model ---
# This is the new, critical step. We copy and run the download script
# to bake the model into the image. This makes runtime startup fast and reliable.
COPY download_model.py .
RUN python download_model.py

# Download the 'punkt' sentence tokenizer from NLTK.
RUN python -c "import nltk; nltk.download('punkt')"

# Copy the rest of the application code into the container.
COPY . .

# Expose the port the app runs on.
EXPOSE 8080

# Command to run the application using uvicorn.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
