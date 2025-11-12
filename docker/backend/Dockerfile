# Multi-stage Dockerfile for SUJBOT2 Backend
# Supports both development and production builds

ARG PYTHON_VERSION=3.10

# ============================================================================
# Stage 1: Base - System dependencies and Python setup
# ============================================================================
FROM python:${PYTHON_VERSION}-slim AS base

# Install system dependencies required for Python packages
RUN apt-get update && apt-get install -y \
    # Build essentials
    gcc \
    g++ \
    make \
    # libspatialindex for rtree (Docling dependency)
    libspatialindex-dev \
    # Tesseract OCR for document processing
    tesseract-ocr \
    tesseract-ocr-ces \
    tesseract-ocr-eng \
    # Poppler for PDF processing
    poppler-utils \
    # libpq for PostgreSQL psycopg2
    libpq-dev \
    # OpenGL libraries for OpenCV (cv2)
    libgl1 \
    libglib2.0-0 \
    # Networking tools for health checks
    curl \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

# Install uv (Astral's fast Python package manager)
RUN pip install --no-cache-dir uv

WORKDIR /app

# ============================================================================
# Stage 2: Builder - Install Python dependencies
# ============================================================================
FROM base AS builder

# Copy dependency files (uv uses lockfile for reproducible builds)
COPY pyproject.toml uv.lock ./
COPY backend/requirements.txt ./backend/

# Install dependencies with uv (reproducible builds from lockfile)
# Using --system to install globally (no venv in Docker)
RUN uv pip install --system --no-cache \
    # Install all main dependencies from pyproject.toml
    -r backend/requirements.txt && \
    uv pip install --system --no-cache \
    # Core dependencies from pyproject.toml
    python-dotenv pydantic pyyaml \
    pypdf2 pymupdf \
    "unstructured[all-docs]>=0.10.0" \
    "docling>=2.57.0" "docling-core>=2.48.0" "docling-parse>=4.4.0" "docling-ibm-models>=3.9.0" \
    anthropic openai "google-genai>=0.5.0" "google-api-core>=2.0.0" \
    langchain-text-splitters langgraph "langgraph-checkpoint-postgres>=2.0.0" langsmith \
    tiktoken "sentence-transformers>=5.1.2" "faiss-cpu>=1.7.4" "rank-bm25>=0.2.2" \
    numpy scikit-learn hdbscan "umap-learn>=0.5.5" matplotlib \
    "neo4j>=5.25.0" networkx click tqdm \
    # PyTorch dependencies
    "torch>=2.0.0" "torchvision>=0.15.0" "transformers>=4.34.0" accelerate "safetensors>=0.4.3" \
    # OCR support
    "rapidocr>=3.3.0" "opencv-python>=4.5.1" \
    # Docling dependencies
    "pypdfium2>=4.30.0" "filetype>=1.2.0" "rtree>=1.3.0" \
    "python-docx>=1.1.2" "python-pptx>=1.0.2" "openpyxl>=3.1.5" \
    "beautifulsoup4>=4.12.3" "pandas>=2.1.4" "lxml>=4.0.0" "pillow>=10.0.0" \
    "pylatexenc>=2.10" "scipy>=1.6.0" "marko>=2.1.2" \
    # PostgreSQL drivers
    psycopg2-binary asyncpg pgvector \
    # Language support for BM25 hybrid search
    "spacy>=3.7.0" "langdetect>=1.0.9" "nltk>=3.8.0"

# Download spaCy English model and NLTK stopwords
RUN python -m spacy download en_core_web_sm && \
    python -c "import nltk; nltk.download('stopwords')"

# ============================================================================
# Stage 3: Development - Hot reload with source mounting
# ============================================================================
FROM base AS development

# Copy all installed packages and binaries from builder
# This includes Python packages in /usr/local/lib/python3.10.X/site-packages
# and scripts in /usr/local/bin
COPY --from=builder /usr/local /usr/local

WORKDIR /app

# Copy only essential files (source code mounted via volume)
COPY config.json ./config.json

# Development runs with hot reload (see docker-compose.override.yml)
EXPOSE 8000

# Default command (overridden in docker-compose.override.yml)
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# ============================================================================
# Stage 4: Production - Minimal runtime with copied source
# ============================================================================
FROM base AS production

# Copy all installed packages and binaries from builder
# This includes Python packages in /usr/local/lib/python3.10.X/site-packages
# and scripts in /usr/local/bin
COPY --from=builder /usr/local /usr/local

WORKDIR /app

# Copy application source
COPY ./src ./src
COPY ./backend ./backend
COPY ./prompts ./prompts
COPY ./config.json ./config.json

# Create necessary directories
RUN mkdir -p /app/logs /app/data

# Non-root user for security
RUN useradd -m -u 1000 sujbot && \
    chown -R sujbot:sujbot /app
USER sujbot

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production runs with multiple workers (no hot reload)
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
