# Dockerfile

# --- Build Stage ---
# Use a specific, stable, and slim base image for security and reproducibility.
# 'bookworm' is a recent stable Debian release.
FROM python:3.10.13-slim-bookworm as builder

# Set the working directory inside the container.
WORKDIR /app

# Set environment variables to prevent Python from writing.pyc files and to run in unbuffered mode.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Upgrade pip to ensure access to the latest features and security patches.
RUN pip install --upgrade pip

# Copy only the requirements file first to leverage Docker's layer caching.
# This layer will only be rebuilt if requirements.txt changes.
COPY requirements.txt.

# Install dependencies into a wheelhouse. This pre-compiles the packages,
# making the installation in the final stage much faster.
RUN pip wheel --no-cache-dir --wheel-dir /app/wheels -r requirements.txt


# --- Final Stage ---
# Start from the same slim base image for the final, lean container.
FROM python:3.10.13-slim-bookworm

# Set the working directory.
WORKDIR /app

# Create a dedicated, non-root user for the application.
# Running as a non-root user is a critical security best practice.
RUN useradd --create-home appuser
USER appuser

# Set the same environment variables as in the builder stage.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy the pre-compiled wheels from the builder stage.
COPY --from=builder /app/wheels /wheels
# Copy the requirements file again for the pip install command.
COPY --from=builder /app/requirements.txt.

# Install the dependencies from the local wheelhouse.
# --no-index prevents pip from looking at PyPI, ensuring only our pre-compiled wheels are used.
# --find-links points pip to the directory containing the wheels.
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt

# Download and cache the NLP models as the non-root user during the build process.
# This ensures models are part of the image layer and don't need to be downloaded on container start.
# This improves startup time and reliability.
RUN python -m spacy download en_core_web_lg && \
    python -c "import nltk; nltk.download('wordnet', quiet=True)"

# Copy the application source code into the container.
# The 'chown' flag ensures the 'appuser' owns the files.
# The '..' syntax means: copy from the build context (first '.') to the WORKDIR (second '.').
# This is the corrected version of the user's failing command.
COPY --chown=appuser:appuser..

# Expose the port that Gunicorn will listen on. This is informational for developers.
# The actual port mapping is handled by Cloud Run.
EXPOSE 8080

# Command to run the application using Gunicorn, a production-grade WSGI server.
# '-w 2' specifies 2 worker processes. A common starting point is (2 * CPU_CORES) + 1.
# '-k uvicorn.worker.UvicornWorker' tells Gunicorn to use Uvicorn workers for ASGI apps like FastAPI.
# '-b 0.0.0.0:8080' binds the server to all network interfaces on port 8080, as required by Cloud Run.
CMD ["gunicorn", "-w", "2", "-k", "uvicorn.worker.UvicornWorker", "-b", "0.0.0.0:8080", "app:app"]
