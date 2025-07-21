# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy everything
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for Cloud Run
EXPOSE 8080

# Run the app using uvicorn on correct port
CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=8080"]
