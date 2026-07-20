# Use official slim Python runtime as base image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory inside container
WORKDIR /app

# Install system dependencies required for OpenCV and PyTorch CPU
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code, static assets, templates, and model files
COPY . /app/

# Expose server port (Render dynamically sets PORT env var)
ENV PORT=5000
EXPOSE $PORT

# Start Gunicorn server with 1 worker and 4 threads to prevent Render OOM RAM crashes
CMD sh -c "gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 180"
