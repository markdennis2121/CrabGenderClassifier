# Use official lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV KERAS_BACKEND=torch
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Install CPU-only dependencies to stay under memory limits
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port (7860 for Hugging Face, or dynamic $PORT)
EXPOSE 7860

# Ensure static/uploads has full write permissions for non-root user 1000
RUN mkdir -p static/uploads && chmod -R 777 static/uploads

# Command to run the application using Gunicorn dynamically binding to PORT
CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:${PORT:-7860} --workers 1 --threads 2 --timeout 120"]



