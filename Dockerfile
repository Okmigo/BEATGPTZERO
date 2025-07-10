# Dockerfile

# Use Python 3.11-slim as a base image.
FROM python:3.11-slim

# Set the working directory in the container.
WORKDIR /app

# Copy requirements file first to leverage Docker layer caching.
COPY requirements.txt .

# Install Python dependencies.
# Using --no-cache-dir reduces the image size.
RUN pip install --no-cache-dir -r requirements.txt

# Download the 'punkt' sentence tokenizer from NLTK. This is required
# by the humanizer.py script to split text into sentences.
RUN python -c "import nltk; nltk.download('punkt')"

# Copy the rest of the application code into the container.
COPY . .

# Expose the port the app runs on.
EXPOSE 8080

# Command to run the application using uvicorn.
# This starts the FastAPI server.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
