# SUJBOT2 Backend API

Production-grade FastAPI backend for legal compliance checking with RAG.

## Features

- **REST API**: Document management, compliance checking, and querying
- **WebSocket**: Real-time streaming chat responses
- **Celery + Redis**: Asynchronous task processing for heavy operations
- **OpenAPI Documentation**: Auto-generated at `/api/docs`
- **CORS Support**: Frontend integration ready
- **File Upload**: Multipart/form-data with validation and progress tracking

## Quick Start

### Local Development

1. **Install Redis** (required for Celery):
```bash
# macOS
brew install redis
brew services start redis

# Ubuntu/Debian
sudo apt-get install redis-server
sudo systemctl start redis

# Windows
# Download from https://redis.io/download
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Configure environment**:
```bash
cp .env.example .env
# Edit .env and set CLAUDE_API_KEY
```

4. **Run the API**:
```bash
# Development mode with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

5. **Run Celery worker** (in separate terminal):
```bash
celery -A app.core.celery_app worker --loglevel=info
```

6. **Optional: Run Flower** (Celery monitoring):
```bash
celery -A app.core.celery_app flower --port=5555
```

### Docker Deployment

```bash
# IMPORTANT: Before starting Docker, set required environment variables
export POSTGRES_PASSWORD="your-secure-password-here"  # REQUIRED
export CLAUDE_API_KEY="your-claude-api-key"          # REQUIRED

# For first-time setup: Fix file permissions for Docker bind mounts
# The container runs as UID 1000 (appuser), so host directories need correct ownership
mkdir -p uploads indexes logs
sudo chown -R 1000:1000 uploads indexes logs

# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

**Docker Permission Notes:**
- Container runs as non-root user (UID 1000) for security
- Bind-mounted directories (`uploads/`, `indexes/`, `logs/`) must be writable by UID 1000
- Named volumes (`model_cache`) handle permissions automatically
- If you get "Permission denied" errors, run: `sudo chown -R 1000:1000 uploads indexes logs`

## API Endpoints

### Document Management

- `POST /api/v1/documents/upload` - Upload document
- `GET /api/v1/documents/{document_id}/status` - Get document status
- `GET /api/v1/documents` - List documents
- `DELETE /api/v1/documents/{document_id}` - Delete document

### Compliance Checking

- `POST /api/v1/compliance/check` - Start compliance check
- `GET /api/v1/compliance/reports/{task_id}` - Get report status
- `GET /api/v1/compliance/reports/{task_id}/download` - Download report

### Query

- `POST /api/v1/query` - Query documents (REST)
- `WS /ws/chat` - Real-time chat (WebSocket)

### Health

- `GET /api/v1/health` - Health check

## API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/api/docs
- **ReDoc**: http://localhost:8000/api/redoc
- **OpenAPI JSON**: http://localhost:8000/api/openapi.json

## Usage Examples

### Upload Document

```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@contract.pdf" \
  -F "document_type=contract"
```

Response:
```json
{
  "document_id": "abc123def456",
  "filename": "contract.pdf",
  "filesize": 1024000,
  "document_type": "contract",
  "status": "uploaded",
  "indexing_task_id": "task_xyz789"
}
```

### Check Document Status

```bash
curl http://localhost:8000/api/v1/documents/abc123def456/status
```

### Start Compliance Check

```bash
curl -X POST http://localhost:8000/api/v1/compliance/check \
  -H "Content-Type: application/json" \
  -d '{
    "contract_document_id": "abc123",
    "law_document_ids": ["def456", "ghi789"],
    "mode": "exhaustive"
  }'
```

### Query Documents

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Jaké jsou povinnosti dodavatele?",
    "document_ids": ["abc123"],
    "language": "cs"
  }'
```

### WebSocket Chat

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/chat');

ws.onopen = () => {
  ws.send(JSON.stringify({
    type: 'chat_message',
    content: 'Jaké jsou povinnosti dodavatele?',
    document_ids: ['abc123']
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'stream_chunk') {
    console.log(data.content);
  }
};
```

## Project Structure

```
backend/
├── app/
│   ├── core/               # Configuration and dependencies
│   │   ├── config.py       # Settings
│   │   ├── celery_app.py   # Celery configuration
│   │   └── dependencies.py # Dependency injection
│   ├── models/             # Pydantic models
│   │   ├── document.py
│   │   ├── compliance.py
│   │   └── query.py
│   ├── routers/            # API endpoints
│   │   ├── documents.py
│   │   ├── compliance.py
│   │   ├── query.py
│   │   └── websocket.py
│   ├── services/           # Business logic
│   │   ├── document_service.py
│   │   ├── compliance_service.py
│   │   └── chat_service.py
│   ├── tasks/              # Celery tasks
│   │   ├── indexing.py
│   │   └── compliance.py
│   ├── middleware/         # Custom middleware
│   │   └── logging.py
│   └── main.py            # FastAPI application
├── uploads/               # Uploaded files
├── indexes/               # FAISS indices
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Configuration

Environment variables (`.env`):

```bash
# Claude API
CLAUDE_API_KEY=your-api-key

# Server
HOST=0.0.0.0
PORT=8000

# Redis
REDIS_URL=redis://localhost:6379/0

# File limits
MAX_UPLOAD_SIZE=524288000  # 500 MB
```

## Production Deployment

### Gunicorn + Uvicorn Workers

```bash
gunicorn app.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --access-logfile - \
  --error-logfile -
```

### Celery Workers

```bash
celery -A app.core.celery_app worker \
  --loglevel=info \
  --concurrency=4 \
  --max-tasks-per-child=50
```

## Monitoring

- **Flower**: http://localhost:5555 (Celery task monitoring)
- **Health Check**: http://localhost:8000/api/v1/health

## Integration with Frontend

The API is CORS-enabled for frontend integration. Add your frontend URL to `CORS_ORIGINS` in `.env`:

```bash
CORS_ORIGINS=["http://localhost:3000","http://localhost:5173"]
```

## Development

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest
```

### Code Quality

```bash
# Install linting tools
pip install black isort flake8

# Format code
black app/
isort app/

# Lint
flake8 app/
```

## Troubleshooting

### Redis Connection Error

Ensure Redis is running:
```bash
redis-cli ping  # Should return PONG
```

### Celery Workers Not Starting

Check Redis connection and ensure worker command is correct:
```bash
celery -A app.core.celery_app inspect active
```

### File Upload Size Limit

Increase `MAX_UPLOAD_SIZE` in `.env` and restart server.

## License

MIT License - See project root for details.
