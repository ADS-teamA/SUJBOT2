# SUJBOT2 Docker Setup Guide

**Status:** ✅ Complete PostgreSQL + Docker migration
**Date:** 2025-11-12
**Architecture:** Docker Compose + PostgreSQL (pgvector + Apache AGE)

---

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Prerequisites](#prerequisites)
3. [Environment Setup](#environment-setup)
4. [Docker Services](#docker-services)
5. [Migration from FAISS](#migration-from-faiss)
6. [Development Workflow](#development-workflow)
7. [Production Deployment](#production-deployment)
8. [Troubleshooting](#troubleshooting)
9. [Performance Tuning](#performance-tuning)

---

## 🚀 Quick Start

```bash
# 1. Clone repository (if not already)
cd /Users/michalprusek/PycharmProjects/MY_SUJBOT

# 2. Setup environment
cp .env.example .env
# Edit .env and add your API keys

# 3. Start Docker services
docker-compose up -d

# 4. Wait for services to be healthy
docker-compose ps

# 5. (Optional) Migrate existing FAISS data
python scripts/migrate_faiss_to_postgres.py \
    --faiss-dir vector_db/ \
    --db-url postgresql://postgres:sujbot_secure_password@localhost:5432/sujbot \
    --verify

# 6. Access application
# Frontend: http://localhost:5173
# Backend API: http://localhost:8000/docs
# PostgreSQL: localhost:5432
```

---

## 📦 Prerequisites

### Required

- **Docker**: Version 20.10+ ([Install](https://docs.docker.com/get-docker/))
- **Docker Compose**: Version 2.0+ (included with Docker Desktop)
- **16GB RAM minimum** (32GB recommended for production)
- **SSD storage** (HDD will be 100-500x slower for HNSW indexes)

### Optional

- **pgAdmin** or **DBeaver** for database management
- **k6** or **Locust** for load testing

### Verify Installation

```bash
docker --version          # Should show 20.10+
docker-compose --version  # Should show 2.0+
docker ps                 # Should show no errors
```

---

## ⚙️ Environment Setup

### 1. Create `.env` from template

```bash
cp .env.example .env
```

### 2. Edit `.env` - Add API Keys

```bash
# REQUIRED: Add your API keys
ANTHROPIC_API_KEY=sk-ant-your-key-here
OPENAI_API_KEY=sk-proj-your-key-here

# Optional
GOOGLE_API_KEY=

# Database (defaults are fine for development)
POSTGRES_PASSWORD=sujbot_secure_password  # CHANGE IN PRODUCTION!
DATABASE_URL=postgresql://postgres:sujbot_secure_password@postgres:5432/sujbot
```

### 3. Verify Configuration

```bash
# Check config.json is present
cat config.json | jq '.storage.backend'  # Should show "postgresql"

# Verify .env has API keys
grep "ANTHROPIC_API_KEY" .env
grep "OPENAI_API_KEY" .env
```

---

## 🐳 Docker Services

### Service Architecture

```
┌────────────────────────────────────────────────────┐
│  Frontend (React + Vite)      localhost:5173       │
│  - Development: Hot reload                         │
│  - Production: Nginx static serving                │
└────────────────────────────────────────────────────┘
                       ↓ HTTP/SSE
┌────────────────────────────────────────────────────┐
│  Backend (FastAPI + Python)   localhost:8000       │
│  - Multi-agent RAG system                          │
│  - 16 tools, 8 agents                              │
│  - PostgreSQL adapter with fallback                │
└────────────────────────────────────────────────────┘
                       ↓ asyncpg
┌────────────────────────────────────────────────────┐
│  PostgreSQL 16 + Extensions   localhost:5432       │
│  - pgvector (vector similarity search)             │
│  - Apache AGE (graph database)                     │
│  - 3-layer vector store (document/section/chunk)   │
│  - LangGraph checkpointing                         │
└────────────────────────────────────────────────────┘
```

### Start Services

```bash
# Start all services (development mode)
docker-compose up -d

# View logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f backend
docker-compose logs -f postgres

# Check service status
docker-compose ps
```

### Stop Services

```bash
# Stop all services (preserves data)
docker-compose down

# Stop and remove volumes (deletes data!)
docker-compose down -v
```

---

## 🔄 Migration from FAISS

### Pre-Migration Checklist

- [ ] Backup existing `vector_db/` directory
- [ ] Docker services are running (`docker-compose ps`)
- [ ] PostgreSQL is healthy (green in `docker-compose ps`)
- [ ] At least 4GB free RAM for migration process

### Run Migration

```bash
# Activate Python environment (if using uv)
source .venv/bin/activate  # or: uv venv

# Run migration script
python scripts/migrate_faiss_to_postgres.py \
    --faiss-dir vector_db/ \
    --db-url postgresql://postgres:sujbot_secure_password@localhost:5432/sujbot \
    --batch-size 500 \
    --verify

# Migration output:
# ✓ Layer 1: Migrated 150 vectors
# ✓ Layer 2: Migrated 2,345 vectors
# ✓ Layer 3: Migrated 12,789 vectors
# ✓ Verification: Test search successful
```

### Migration Performance

| Documents | Vectors | Time (IVFFlat) | Time (HNSW) |
|-----------|---------|----------------|-------------|
| 10        | ~1K     | 1-2 minutes    | 5-10 minutes|
| 100       | ~10K    | 5-10 minutes   | 30-60 minutes|
| 1000      | ~100K   | 30-60 minutes  | 2-4 hours   |

**Note:** HNSW index building is the bottleneck. Use IVFFlat for development.

### Verify Migration

```bash
# Connect to PostgreSQL
docker-compose exec postgres psql -U postgres -d sujbot

# Check vector counts
SELECT * FROM metadata.vector_store_stats;

# Test vector search
SELECT chunk_id, content, 1 - (embedding <=> '[0.1, 0.2, ...]'::vector) AS score
FROM vectors.layer3
ORDER BY embedding <=> '[0.1, 0.2, ...]'::vector
LIMIT 5;

# Exit
\q
```

---

## 💻 Development Workflow

### Hot Reload (Default)

Docker Compose automatically mounts source code and enables hot reload:

```bash
# Start in development mode
docker-compose up -d

# Edit backend code
vim src/agent/tools/search.py

# Changes auto-reload (watch backend logs)
docker-compose logs -f backend

# Edit frontend code
vim frontend/src/components/chat/ChatContainer.tsx

# Changes auto-reload in browser (Vite HMR)
```

### Rebuild After Dependency Changes

```bash
# If you added new Python packages to pyproject.toml
docker-compose build backend

# If you added new npm packages to frontend/package.json
docker-compose build frontend

# Rebuild all services
docker-compose build

# Start with rebuilt images
docker-compose up -d
```

### Access Services

- **Frontend (dev):** http://localhost:5173
- **Backend API:** http://localhost:8000
- **API Docs (Swagger):** http://localhost:8000/docs
- **PostgreSQL:** `postgresql://postgres:sujbot_secure_password@localhost:5432/sujbot`

### Database Management

```bash
# psql CLI
docker-compose exec postgres psql -U postgres -d sujbot

# pgAdmin (install separately)
# Connection: localhost:5432, user: postgres, password: sujbot_secure_password

# Backup database
docker-compose exec postgres pg_dump -U postgres sujbot > backup.sql

# Restore database
cat backup.sql | docker-compose exec -T postgres psql -U postgres -d sujbot
```

---

## 🚀 Production Deployment

### 1. Switch to Production Mode

```bash
# Edit .env
BUILD_TARGET=production

# Rebuild images
docker-compose build

# Start services
docker-compose -f docker-compose.yml up -d
```

**Note:** `docker-compose.override.yml` is automatically ignored in production.

### 2. Production Changes

```yaml
# docker-compose.yml modifications for production:

services:
  postgres:
    ports:
      # REMOVE this line (don't expose PostgreSQL publicly)
      # - "5432:5432"

  backend:
    environment:
      LOG_LEVEL: WARNING  # Less verbose
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G

  frontend:
    # Nginx serves static build on port 80
    ports:
      - "80:80"
      # - "5173:5173"  # Remove dev port
```

### 3. Security Hardening

```bash
# Change passwords in .env
POSTGRES_PASSWORD=<generate strong password>

# Use secrets management (AWS Secrets Manager, Vault, etc.)
# Don't commit .env to git!

# Enable SSL/TLS for PostgreSQL
# Add to docker/postgres/postgresql.conf:
# ssl = on
# ssl_cert_file = '/etc/ssl/certs/server.crt'
# ssl_key_file = '/etc/ssl/private/server.key'
```

### 4. HTTPS Setup (Nginx + Let's Encrypt)

```bash
# Install certbot
sudo apt-get install certbot python3-certbot-nginx

# Get certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo systemctl enable certbot.timer
```

---

## 🐛 Troubleshooting

### PostgreSQL Won't Start

```bash
# Check logs
docker-compose logs postgres

# Common issues:
# 1. Port 5432 already in use
sudo lsof -i :5432
sudo kill -9 <PID>

# 2. Data directory corruption
docker-compose down -v  # WARNING: Deletes data!
docker-compose up -d

# 3. Insufficient memory
# Edit docker/postgres/postgresql.conf:
# shared_buffers = 2GB  # Reduce from 4GB
```

### Backend Won't Connect to PostgreSQL

```bash
# Check DATABASE_URL in .env
echo $DATABASE_URL

# Test connection manually
docker-compose exec postgres psql -U postgres -d sujbot -c "SELECT 1;"

# Check backend logs
docker-compose logs backend | grep -i "postgres\|database"

# Verify pgvector extension
docker-compose exec postgres psql -U postgres -d sujbot -c "SELECT * FROM pg_extension WHERE extname='vector';"
```

### Frontend Can't Reach Backend

```bash
# Check VITE_API_BASE_URL in .env
grep VITE_API_BASE_URL .env

# Test backend health
curl http://localhost:8000/health

# Check CORS settings in backend/main.py
# Should include: http://localhost:5173
```

### Migration Fails

```bash
# Check FAISS directory exists
ls -la vector_db/

# Verify files
# Should have: faiss_layer*.index, metadata_layer*.pkl, bm25_layer*.pkl

# Run migration with verbose logging
python scripts/migrate_faiss_to_postgres.py \
    --faiss-dir vector_db/ \
    --db-url postgresql://postgres:sujbot_secure_password@localhost:5432/sujbot \
    --batch-size 100 \  # Reduce batch size
    --verify
```

### Slow Query Performance

```bash
# Check if indexes exist
docker-compose exec postgres psql -U postgres -d sujbot -c "\d+ vectors.layer3"

# Should show:
# Indexes:
#   "idx_layer3_embedding_hnsw" hnsw (embedding) WITH (m=16, ef_construction=64)

# If missing, rebuild indexes
docker-compose exec postgres psql -U postgres -d sujbot < docker/postgres/init/01-init.sql

# Set query parameters (per session)
SET hnsw.ef_search = 40;  # Higher = better recall, slower
SET ivfflat.probes = 10;
```

---

## ⚡ Performance Tuning

### PostgreSQL Configuration

Edit `docker/postgres/postgresql.conf`:

```ini
# For 16GB RAM system
shared_buffers = 4GB
effective_cache_size = 12GB
work_mem = 64MB
maintenance_work_mem = 1GB

# For 32GB RAM system
shared_buffers = 8GB
effective_cache_size = 24GB
work_mem = 128MB
maintenance_work_mem = 2GB
```

### Index Selection

```sql
-- Development: IVFFlat (fast build, good enough)
CREATE INDEX idx_layer3_embedding_ivfflat
ON vectors.layer3 USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 1000);

-- Production: HNSW (slow build, best quality)
CREATE INDEX idx_layer3_embedding_hnsw
ON vectors.layer3 USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

### Query Optimization

```python
# In application code (src/storage/postgres_adapter.py)

# Set per-query parameters
await conn.execute("SET hnsw.ef_search = 40;")  # 40-100 for good recall

# Use prepared statements (automatically cached)
await conn.prepare("SELECT ... FROM vectors.layer3 WHERE ...")

# Batch operations
async with conn.transaction():
    await conn.executemany(insert_sql, records)
```

### Monitoring

```bash
# Query performance
docker-compose exec postgres psql -U postgres -d sujbot -c "
SELECT query, calls, mean_exec_time, max_exec_time
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 10;
"

# Database size
docker-compose exec postgres psql -U postgres -d sujbot -c "
SELECT pg_size_pretty(pg_database_size('sujbot'));
"

# Index usage
docker-compose exec postgres psql -U postgres -d sujbot -c "
SELECT schemaname, tablename, indexname, idx_scan
FROM pg_stat_user_indexes
WHERE schemaname = 'vectors';
"
```

---

## 📊 Next Steps

### After Successful Setup

1. **Test search quality:** Compare PostgreSQL vs FAISS results
2. **Benchmark performance:** Use k6/Locust for load testing
3. **Monitor costs:** Track API usage (Anthropic/OpenAI)
4. **Setup backups:** Automate PostgreSQL pg_dump
5. **Configure monitoring:** Prometheus + Grafana for metrics

### Further Reading

- [pgvector Documentation](https://github.com/pgvector/pgvector)
- [Apache AGE Documentation](https://age.apache.org/)
- [Docker Compose Best Practices](https://docs.docker.com/compose/production/)
- [PostgreSQL Performance Tuning](https://wiki.postgresql.org/wiki/Performance_Optimization)

---

## 🆘 Support

### Common Commands Reference

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f [service]

# Rebuild
docker-compose build [service]

# Shell into container
docker-compose exec [service] bash

# Database shell
docker-compose exec postgres psql -U postgres -d sujbot

# Python shell (backend)
docker-compose exec backend python

# Restart single service
docker-compose restart [service]

# Remove all containers and volumes (DESTRUCTIVE!)
docker-compose down -v
```

### Files Created

- `docker-compose.yml` - Service orchestration
- `docker-compose.override.yml` - Development overrides
- `.dockerignore` - Build context exclusions
- `docker/backend/Dockerfile` - Backend multi-stage build
- `docker/frontend/Dockerfile` - Frontend multi-stage build
- `docker/postgres/Dockerfile` - PostgreSQL + extensions
- `docker/postgres/init/01-init.sql` - Database schema
- `docker/postgres/postgresql.conf` - Performance config
- `docker/nginx/nginx.conf` - Frontend production config
- `src/storage/*.py` - Abstraction layer (4 files)
- `scripts/migrate_faiss_to_postgres.py` - Migration script

---

**Last Updated:** 2025-11-12
**Version:** 1.0.0
**Architecture:** Docker Compose + PostgreSQL (pgvector + Apache AGE)
