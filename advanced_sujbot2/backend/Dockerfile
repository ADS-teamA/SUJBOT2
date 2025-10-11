# Backend Dockerfile for SUJBOT2
# Multi-stage build for production optimization

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for ML libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libgomp1 \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with increased timeout for large packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --timeout=600 -r requirements.txt

# Copy application code
COPY . .

# Note: src/ directory is mounted via volume in docker-compose
# (not copied at build time to avoid build context issues)

# Create necessary directories including model cache
RUN mkdir -p /app/uploads /app/indexes /app/logs /app/models

# Set environment variables for HuggingFace cache
ENV HF_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models/transformers
ENV SENTENCE_TRANSFORMERS_HOME=/app/models/sentence-transformers
ENV TOKENIZERS_PARALLELISM=false

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

EXPOSE 8000

# Default command (can be overridden in docker-compose)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
