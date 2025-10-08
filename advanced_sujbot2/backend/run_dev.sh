#!/bin/bash
# Development startup script for SUJBOT2 Backend

echo "Starting SUJBOT2 Backend (Development Mode)"
echo "===================================================="

# Check if Redis is running
if ! redis-cli ping > /dev/null 2>&1; then
    echo "❌ Redis is not running!"
    echo "Please start Redis first:"
    echo "  - macOS: brew services start redis"
    echo "  - Linux: sudo systemctl start redis"
    echo "  - Docker: docker run -d -p 6379:6379 redis:7.2-alpine"
    exit 1
fi

echo "✅ Redis is running"

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ .env file not found!"
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo "⚠️  Please edit .env and set your CLAUDE_API_KEY"
    exit 1
fi

echo "✅ Environment configuration found"

# Create directories
mkdir -p uploads indexes

# Start Celery worker in background
echo "Starting Celery worker..."
celery -A app.core.celery_app worker --loglevel=info &
CELERY_PID=$!

# Wait a bit for Celery to start
sleep 2

# Start Uvicorn
echo "Starting FastAPI server..."
echo "===================================================="
echo "📚 API Documentation: http://localhost:8000/api/docs"
echo "🌸 Flower (Celery): http://localhost:5555"
echo "❤️  Health Check: http://localhost:8000/api/v1/health"
echo "===================================================="

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Cleanup on exit
trap "kill $CELERY_PID" EXIT
