# 15. Deployment & Docker Specification

## 1. Purpose

**Objective**: Define complete deployment architecture for SUJBOT2 full-stack application using Docker containers, orchestrated with Docker Compose for development and production environments.

**Why Docker Deployment?**
- Consistent environments across dev/staging/production
- Easy scaling of backend workers and API instances
- Isolated services (frontend, backend, Redis, Celery)
- Simple deployment with docker-compose up
- Portable across cloud providers (AWS, GCP, Azure)
- Easy rollback and version management

**Key Capabilities**:
1. **Multi-Container Architecture** - Frontend, Backend, Redis, Celery workers
2. **Nginx Reverse Proxy** - Load balancing and SSL termination
3. **Volume Persistence** - Uploaded documents, FAISS indices, Redis data
4. **Environment Management** - Separate .env files for dev/prod
5. **Health Checks** - Automatic container restart on failure
6. **Logging** - Centralized logging with ELK stack (future)
7. **Scaling** - Horizontal scaling of API and worker containers

---

## 2. Architecture Overview

### 2.1 Container Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Internet/Users                        │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
              ┌────────────────┐
              │  Nginx (80/443)│  Reverse Proxy + Static Files
              │  SSL Termination│
              └────┬───────┬───┘
                   │       │
        ┌──────────┘       └──────────┐
        │                              │
        ▼                              ▼
┌──────────────┐            ┌──────────────────┐
│   Frontend   │            │   Backend API    │
│   (React)    │            │   (FastAPI)      │
│   Port: 3000 │            │   Port: 8000     │
└──────────────┘            └──────┬───────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    │              │              │
                    ▼              ▼              ▼
            ┌──────────┐   ┌──────────┐   ┌──────────┐
            │  Redis   │   │ Celery   │   │  Celery  │
            │  (6379)  │   │ Worker 1 │   │ Worker 2 │
            └──────────┘   └──────────┘   └──────────┘
```

### 2.2 Data Flow

```
User Browser
    │
    │ (HTTPS)
    ▼
Nginx :443
    │
    ├──> /              → Frontend (SPA)
    ├──> /api/*         → Backend API
    ├──> /ws/*          → Backend WebSocket
    └──> /uploads/*     → Static file serving
```

---

## 3. Docker Images

### 3.1 Frontend Dockerfile

```dockerfile
# frontend/Dockerfile

# Build stage
FROM node:18-alpine AS builder

WORKDIR /app

# Copy package files
COPY package.json pnpm-lock.yaml ./

# Install dependencies
RUN npm install -g pnpm && \
    pnpm install --frozen-lockfile

# Copy source code
COPY . .

# Build application
ARG VITE_API_URL
ARG VITE_WS_URL
ENV VITE_API_URL=$VITE_API_URL
ENV VITE_WS_URL=$VITE_WS_URL

RUN pnpm build

# Production stage
FROM nginx:1.25-alpine

# Copy built files
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget --quiet --tries=1 --spider http://localhost/ || exit 1

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
```

```nginx
# frontend/nginx.conf
server {
    listen 80;
    server_name localhost;

    root /usr/share/nginx/html;
    index index.html;

    # SPA routing - always serve index.html
    location / {
        try_files $uri $uri/ /index.html;
    }

    # Enable gzip compression
    gzip on;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;
    gzip_vary on;
    gzip_min_length 1000;

    # Cache static assets
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # Don't cache index.html
    location = /index.html {
        add_header Cache-Control "no-store, no-cache, must-revalidate";
    }
}
```

### 3.2 Backend Dockerfile

```dockerfile
# backend/Dockerfile

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/uploads /app/indexes /app/logs

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

EXPOSE 8000

# Default command (can be overridden in docker-compose)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 3.3 Celery Worker Dockerfile

```dockerfile
# Uses same Dockerfile as backend
# Command is overridden in docker-compose.yml

FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/uploads /app/indexes /app/logs

# Health check for Celery
HEALTHCHECK --interval=60s --timeout=10s --start-period=20s --retries=3 \
    CMD celery -A app.core.celery_app inspect ping || exit 1

# Command specified in docker-compose
CMD ["celery", "-A", "app.core.celery_app", "worker", "--loglevel=info"]
```

---

## 4. Docker Compose Configuration

### 4.1 Development (docker-compose.dev.yml)

```yaml
version: '3.8'

services:
  # Redis - Message broker & cache
  redis:
    image: redis:7.2-alpine
    container_name: sujbot2_redis_dev
    ports:
      - "6379:6379"
    volumes:
      - redis_data_dev:/data
    command: redis-server --appendonly yes
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 3
    networks:
      - sujbot2_network

  # Backend API
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    container_name: sujbot2_backend_dev
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379/0
      - CELERY_BROKER_URL=redis://redis:6379/0
      - CELERY_RESULT_BACKEND=redis://redis:6379/1
      - CLAUDE_API_KEY=${CLAUDE_API_KEY}
      - ENVIRONMENT=development
      - DEBUG=true
    volumes:
      - ./backend:/app  # Hot reload
      - uploads_dev:/app/uploads
      - indexes_dev:/app/indexes
      - ./backend/logs:/app/logs
    depends_on:
      redis:
        condition: service_healthy
    networks:
      - sujbot2_network
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

  # Celery Worker
  celery_worker:
    build:
      context: ./backend
      dockerfile: Dockerfile
    container_name: sujbot2_celery_dev
    environment:
      - REDIS_URL=redis://redis:6379/0
      - CELERY_BROKER_URL=redis://redis:6379/0
      - CELERY_RESULT_BACKEND=redis://redis:6379/1
      - CLAUDE_API_KEY=${CLAUDE_API_KEY}
      - ENVIRONMENT=development
    volumes:
      - ./backend:/app
      - uploads_dev:/app/uploads
      - indexes_dev:/app/indexes
      - ./backend/logs:/app/logs
    depends_on:
      - redis
    networks:
      - sujbot2_network
    command: celery -A app.core.celery_app worker --loglevel=info --concurrency=4

  # Celery Flower (monitoring)
  flower:
    build:
      context: ./backend
      dockerfile: Dockerfile
    container_name: sujbot2_flower_dev
    ports:
      - "5555:5555"
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0
      - CELERY_RESULT_BACKEND=redis://redis:6379/1
    depends_on:
      - redis
      - celery_worker
    networks:
      - sujbot2_network
    command: celery -A app.core.celery_app flower --port=5555

  # Frontend
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile.dev  # Separate dev Dockerfile with hot reload
    container_name: sujbot2_frontend_dev
    ports:
      - "3000:3000"
    environment:
      - VITE_API_URL=http://localhost:8000/api/v1
      - VITE_WS_URL=ws://localhost:8000/ws
    volumes:
      - ./frontend:/app
      - /app/node_modules  # Prevent node_modules from being overwritten
    networks:
      - sujbot2_network
    command: pnpm dev --host 0.0.0.0

volumes:
  redis_data_dev:
  uploads_dev:
  indexes_dev:

networks:
  sujbot2_network:
    driver: bridge
```

### 4.2 Production (docker-compose.prod.yml)

```yaml
version: '3.8'

services:
  # Nginx Reverse Proxy
  nginx:
    image: nginx:1.25-alpine
    container_name: sujbot2_nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/conf.d:/etc/nginx/conf.d:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro  # SSL certificates
      - frontend_static:/usr/share/nginx/html:ro
      - uploads_prod:/usr/share/nginx/uploads:ro
    depends_on:
      - backend
      - frontend
    networks:
      - sujbot2_network
    restart: unless-stopped

  # Redis
  redis:
    image: redis:7.2-alpine
    container_name: sujbot2_redis
    volumes:
      - redis_data_prod:/data
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD}
    healthcheck:
      test: ["CMD", "redis-cli", "--auth", "${REDIS_PASSWORD}", "ping"]
      interval: 10s
      timeout: 3s
      retries: 3
    networks:
      - sujbot2_network
    restart: unless-stopped

  # Backend API (multiple instances for load balancing)
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    deploy:
      replicas: 2  # Run 2 instances
    environment:
      - REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379/0
      - CELERY_BROKER_URL=redis://:${REDIS_PASSWORD}@redis:6379/0
      - CELERY_RESULT_BACKEND=redis://:${REDIS_PASSWORD}@redis:6379/1
      - CLAUDE_API_KEY=${CLAUDE_API_KEY}
      - ENVIRONMENT=production
      - DEBUG=false
      - SECRET_KEY=${SECRET_KEY}
    volumes:
      - uploads_prod:/app/uploads
      - indexes_prod:/app/indexes
      - ./backend/logs:/app/logs
    depends_on:
      redis:
        condition: service_healthy
    networks:
      - sujbot2_network
    restart: unless-stopped
    command: gunicorn app.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

  # Celery Workers (scaled)
  celery_worker:
    build:
      context: ./backend
      dockerfile: Dockerfile
    deploy:
      replicas: 3  # Run 3 worker instances
    environment:
      - REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379/0
      - CELERY_BROKER_URL=redis://:${REDIS_PASSWORD}@redis:6379/0
      - CELERY_RESULT_BACKEND=redis://:${REDIS_PASSWORD}@redis:6379/1
      - CLAUDE_API_KEY=${CLAUDE_API_KEY}
      - ENVIRONMENT=production
    volumes:
      - uploads_prod:/app/uploads
      - indexes_prod:/app/indexes
      - ./backend/logs:/app/logs
    depends_on:
      - redis
    networks:
      - sujbot2_network
    restart: unless-stopped
    command: celery -A app.core.celery_app worker --loglevel=info --concurrency=4

  # Frontend (build only)
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
      args:
        - VITE_API_URL=${VITE_API_URL}
        - VITE_WS_URL=${VITE_WS_URL}
    volumes:
      - frontend_static:/usr/share/nginx/html
    networks:
      - sujbot2_network
    # This container only builds and exports static files
    # It can exit after building
    command: "true"

volumes:
  redis_data_prod:
  uploads_prod:
  indexes_prod:
  frontend_static:

networks:
  sujbot2_network:
    driver: bridge
```

### 4.3 Nginx Configuration (Production)

```nginx
# nginx/nginx.conf

user nginx;
worker_processes auto;
error_log /var/log/nginx/error.log warn;
pid /var/run/nginx.pid;

events {
    worker_connections 1024;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for"';

    access_log /var/log/nginx/access.log main;

    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types text/plain text/css text/xml text/javascript
               application/json application/javascript application/xml+rss
               application/rss+xml font/truetype font/opentype
               application/vnd.ms-fontobject image/svg+xml;

    # Include server blocks
    include /etc/nginx/conf.d/*.conf;
}
```

```nginx
# nginx/conf.d/sujbot2.conf

# Upstream backend servers
upstream backend_servers {
    least_conn;  # Load balancing strategy
    server backend:8000 max_fails=3 fail_timeout=30s;
}

# HTTP to HTTPS redirect
server {
    listen 80;
    server_name sujbot2.example.com;

    location /.well-known/acme-challenge/ {
        root /var/www/certbot;
    }

    location / {
        return 301 https://$host$request_uri;
    }
}

# HTTPS server
server {
    listen 443 ssl http2;
    server_name sujbot2.example.com;

    # SSL configuration
    ssl_certificate /etc/nginx/ssl/fullchain.pem;
    ssl_certificate_key /etc/nginx/ssl/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    # Client max body size (for file uploads)
    client_max_body_size 500M;

    # Frontend static files
    location / {
        root /usr/share/nginx/html;
        try_files $uri $uri/ /index.html;

        # Cache policy
        location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot)$ {
            expires 1y;
            add_header Cache-Control "public, immutable";
        }

        location = /index.html {
            add_header Cache-Control "no-store, no-cache, must-revalidate";
        }
    }

    # Backend API
    location /api/ {
        proxy_pass http://backend_servers;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts (for long-running requests)
        proxy_connect_timeout 60s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }

    # WebSocket
    location /ws/ {
        proxy_pass http://backend_servers;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket timeouts
        proxy_connect_timeout 7d;
        proxy_send_timeout 7d;
        proxy_read_timeout 7d;
    }

    # Uploaded files
    location /uploads/ {
        alias /usr/share/nginx/uploads/;
        autoindex off;

        # Security: only allow authenticated access (future)
        # Add authentication here
    }

    # Health check endpoint (no auth)
    location /api/v1/health {
        proxy_pass http://backend_servers;
        access_log off;
    }
}
```

---

## 5. Environment Variables

### 5.1 Development (.env.dev)

```bash
# .env.dev

# General
ENVIRONMENT=development
DEBUG=true

# Claude API
CLAUDE_API_KEY=your-claude-api-key-here

# Redis
REDIS_URL=redis://redis:6379/0

# Celery
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/1

# Frontend URLs
VITE_API_URL=http://localhost:8000/api/v1
VITE_WS_URL=ws://localhost:8000/ws

# File storage
MAX_UPLOAD_SIZE=524288000  # 500 MB
UPLOAD_DIR=/app/uploads
INDEX_DIR=/app/indexes
```

### 5.2 Production (.env.prod)

```bash
# .env.prod

# General
ENVIRONMENT=production
DEBUG=false
SECRET_KEY=your-super-secret-key-here-change-in-production

# Claude API
CLAUDE_API_KEY=your-claude-api-key-here

# Redis
REDIS_PASSWORD=your-redis-password-here
REDIS_URL=redis://:your-redis-password-here@redis:6379/0

# Celery
CELERY_BROKER_URL=redis://:your-redis-password-here@redis:6379/0
CELERY_RESULT_BACKEND=redis://:your-redis-password-here@redis:6379/1

# Frontend URLs (production domain)
VITE_API_URL=https://sujbot2.example.com/api/v1
VITE_WS_URL=wss://sujbot2.example.com/ws

# File storage
MAX_UPLOAD_SIZE=524288000  # 500 MB
UPLOAD_DIR=/app/uploads
INDEX_DIR=/app/indexes

# Security
CORS_ORIGINS=["https://sujbot2.example.com"]
```

---

## 6. Deployment Commands

### 6.1 Development

```bash
# Clone repository
git clone https://github.com/yourusername/advanced-sujbot2.git
cd advanced-sujbot2

# Create .env file
cp .env.example .env.dev
# Edit .env.dev with your API keys

# Build and start all services
docker-compose -f docker-compose.dev.yml up --build

# View logs
docker-compose -f docker-compose.dev.yml logs -f

# Stop all services
docker-compose -f docker-compose.dev.yml down

# Stop and remove volumes (CAUTION: deletes data)
docker-compose -f docker-compose.dev.yml down -v
```

### 6.2 Production

```bash
# Create production .env
cp .env.example .env.prod
# Edit .env.prod with production values

# Build images
docker-compose -f docker-compose.prod.yml build

# Start services (detached mode)
docker-compose -f docker-compose.prod.yml up -d

# View logs
docker-compose -f docker-compose.prod.yml logs -f backend

# Scale workers
docker-compose -f docker-compose.prod.yml up -d --scale celery_worker=5

# Update application (zero-downtime)
docker-compose -f docker-compose.prod.yml pull
docker-compose -f docker-compose.prod.yml up -d --no-deps --build backend

# Stop all services
docker-compose -f docker-compose.prod.yml down
```

### 6.3 Database Migrations (Future)

```bash
# Run database migrations
docker-compose -f docker-compose.prod.yml exec backend alembic upgrade head

# Create new migration
docker-compose -f docker-compose.prod.yml exec backend alembic revision --autogenerate -m "description"
```

---

## 7. Monitoring & Logging

### 7.1 Health Checks

```bash
# Check all services
docker-compose -f docker-compose.prod.yml ps

# Check API health
curl http://localhost:8000/api/v1/health

# Check Redis
docker-compose -f docker-compose.prod.yml exec redis redis-cli ping

# Check Celery workers
docker-compose -f docker-compose.prod.yml exec celery_worker celery -A app.core.celery_app inspect active
```

### 7.2 Logging

**View logs:**
```bash
# All services
docker-compose -f docker-compose.prod.yml logs

# Specific service
docker-compose -f docker-compose.prod.yml logs -f backend

# Last 100 lines
docker-compose -f docker-compose.prod.yml logs --tail=100 backend

# Follow logs
docker-compose -f docker-compose.prod.yml logs -f --tail=0 backend celery_worker
```

**Log rotation (docker-compose.prod.yml):**
```yaml
services:
  backend:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

### 7.3 Monitoring with Prometheus (Future)

```yaml
# docker-compose.monitoring.yml

services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    networks:
      - sujbot2_network

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3001:3000"
    volumes:
      - grafana_data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    networks:
      - sujbot2_network
```

---

## 8. Backup & Recovery

### 8.1 Backup Volumes

```bash
# Backup Redis data
docker run --rm \
  -v sujbot2_redis_data_prod:/data \
  -v $(pwd)/backups:/backup \
  alpine tar czf /backup/redis-$(date +%Y%m%d).tar.gz -C /data .

# Backup uploaded files
docker run --rm \
  -v sujbot2_uploads_prod:/uploads \
  -v $(pwd)/backups:/backup \
  alpine tar czf /backup/uploads-$(date +%Y%m%d).tar.gz -C /uploads .

# Backup FAISS indices
docker run --rm \
  -v sujbot2_indexes_prod:/indexes \
  -v $(pwd)/backups:/backup \
  alpine tar czf /backup/indexes-$(date +%Y%m%d).tar.gz -C /indexes .
```

### 8.2 Restore Volumes

```bash
# Restore Redis data
docker run --rm \
  -v sujbot2_redis_data_prod:/data \
  -v $(pwd)/backups:/backup \
  alpine sh -c "cd /data && tar xzf /backup/redis-20251008.tar.gz"

# Restore uploaded files
docker run --rm \
  -v sujbot2_uploads_prod:/uploads \
  -v $(pwd)/backups:/backup \
  alpine sh -c "cd /uploads && tar xzf /backup/uploads-20251008.tar.gz"
```

### 8.3 Automated Backup Script

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="./backups"
DATE=$(date +%Y%m%d-%H%M%S)

mkdir -p $BACKUP_DIR

# Backup Redis
docker run --rm \
  -v sujbot2_redis_data_prod:/data \
  -v $BACKUP_DIR:/backup \
  alpine tar czf /backup/redis-$DATE.tar.gz -C /data .

# Backup uploads
docker run --rm \
  -v sujbot2_uploads_prod:/uploads \
  -v $BACKUP_DIR:/backup \
  alpine tar czf /backup/uploads-$DATE.tar.gz -C /uploads .

# Backup indices
docker run --rm \
  -v sujbot2_indexes_prod:/indexes \
  -v $BACKUP_DIR:/backup \
  alpine tar czf /backup/indexes-$DATE.tar.gz -C /indexes .

# Delete backups older than 30 days
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete

echo "Backup completed: $DATE"
```

**Schedule with cron:**
```bash
# Run daily at 2 AM
0 2 * * * /path/to/backup.sh >> /var/log/sujbot2-backup.log 2>&1
```

---

## 9. Scaling Strategies

### 9.1 Horizontal Scaling

**Scale backend API:**
```bash
# Scale to 4 instances
docker-compose -f docker-compose.prod.yml up -d --scale backend=4

# Nginx automatically load balances across instances
```

**Scale Celery workers:**
```bash
# Scale to 5 worker instances
docker-compose -f docker-compose.prod.yml up -d --scale celery_worker=5
```

### 9.2 Vertical Scaling

```yaml
# docker-compose.prod.yml

services:
  backend:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G

  celery_worker:
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 8G
        reservations:
          cpus: '2.0'
          memory: 4G
```

### 9.3 Auto-Scaling with Docker Swarm (Future)

```yaml
# docker-stack.yml
version: '3.8'

services:
  backend:
    image: sujbot2/backend:latest
    deploy:
      replicas: 2
      update_config:
        parallelism: 1
        delay: 10s
      restart_policy:
        condition: on-failure
      resources:
        limits:
          cpus: '2'
          memory: 4G
```

```bash
# Deploy stack
docker stack deploy -c docker-stack.yml sujbot2

# Scale service
docker service scale sujbot2_backend=5
```

---

## 10. SSL/TLS Configuration

### 10.1 Let's Encrypt with Certbot

```yaml
# docker-compose.prod.yml (add certbot service)

services:
  certbot:
    image: certbot/certbot:latest
    volumes:
      - ./nginx/ssl:/etc/letsencrypt
      - ./nginx/certbot-webroot:/var/www/certbot
    entrypoint: "/bin/sh -c 'trap exit TERM; while :; do certbot renew; sleep 12h & wait $${!}; done;'"
```

**Initial certificate:**
```bash
# Request certificate
docker-compose -f docker-compose.prod.yml run --rm certbot certonly \
  --webroot \
  --webroot-path=/var/www/certbot \
  --email admin@example.com \
  --agree-tos \
  --no-eff-email \
  -d sujbot2.example.com

# Certificates will be in ./nginx/ssl/
```

**Auto-renewal:**
Certbot container automatically renews certificates every 12 hours.

### 10.2 Manual SSL Certificate

```bash
# Copy your certificates
cp fullchain.pem ./nginx/ssl/
cp privkey.pem ./nginx/ssl/

# Set permissions
chmod 600 ./nginx/ssl/*.pem
```

---

## 11. Troubleshooting

### 11.1 Common Issues

**Container won't start:**
```bash
# Check logs
docker-compose -f docker-compose.prod.yml logs backend

# Inspect container
docker inspect sujbot2_backend

# Enter container
docker-compose -f docker-compose.prod.yml exec backend /bin/bash
```

**Redis connection failed:**
```bash
# Check Redis is running
docker-compose -f docker-compose.prod.yml ps redis

# Test Redis connection
docker-compose -f docker-compose.prod.yml exec redis redis-cli ping
```

**Celery tasks not processing:**
```bash
# Check worker logs
docker-compose -f docker-compose.prod.yml logs celery_worker

# Inspect active tasks
docker-compose -f docker-compose.prod.yml exec celery_worker \
  celery -A app.core.celery_app inspect active

# Purge queue (CAUTION)
docker-compose -f docker-compose.prod.yml exec celery_worker \
  celery -A app.core.celery_app purge
```

**Frontend not updating:**
```bash
# Rebuild frontend
docker-compose -f docker-compose.prod.yml build --no-cache frontend

# Clear browser cache
```

### 11.2 Performance Issues

**Check resource usage:**
```bash
# Docker stats
docker stats

# Top processes in container
docker-compose -f docker-compose.prod.yml exec backend top
```

**Optimize Redis:**
```bash
# Redis memory usage
docker-compose -f docker-compose.prod.yml exec redis redis-cli info memory

# Clear Redis cache (CAUTION)
docker-compose -f docker-compose.prod.yml exec redis redis-cli FLUSHDB
```

---

## 12. CI/CD Pipeline (Future)

### 12.1 GitHub Actions

```yaml
# .github/workflows/deploy.yml

name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Build Docker images
        run: |
          docker-compose -f docker-compose.prod.yml build

      - name: Push to Registry
        run: |
          echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
          docker-compose -f docker-compose.prod.yml push

      - name: Deploy to server
        uses: appleboy/ssh-action@master
        with:
          host: ${{ secrets.SERVER_HOST }}
          username: ${{ secrets.SERVER_USER }}
          key: ${{ secrets.SSH_PRIVATE_KEY }}
          script: |
            cd /opt/sujbot2
            git pull
            docker-compose -f docker-compose.prod.yml pull
            docker-compose -f docker-compose.prod.yml up -d
```

---

## 13. Summary

Kompletní deployment strategie pro SUJBOT2:

✅ **Docker multi-container**: Frontend + Backend + Redis + Celery workers
✅ **Nginx reverse proxy**: Load balancing + SSL termination
✅ **Production-ready**: Health checks, logging, auto-restart
✅ **Scalable**: Horizontal scaling for API and workers
✅ **Persistent data**: Volumes for uploads, indices, Redis
✅ **Environment management**: Separate dev/prod configurations
✅ **Monitoring**: Health checks + Flower for Celery
✅ **Backup & recovery**: Automated backup scripts
✅ **SSL/TLS**: Let's Encrypt integration
✅ **CI/CD ready**: GitHub Actions template

**Quick Start:**
```bash
# Development
docker-compose -f docker-compose.dev.yml up

# Production
docker-compose -f docker-compose.prod.yml up -d

# Access:
# - Frontend: https://sujbot2.example.com
# - API Docs: https://sujbot2.example.com/api/docs
# - Flower: http://localhost:5555
```

---

**Page Count**: ~18 pages
**Last Updated**: 2025-10-08
**Status**: Complete ✅

---

## ALL FULL-STACK SPECIFICATIONS COMPLETE!

**Frontend + Backend + Deployment = Full-stack aplikace připravená k nasazení! 🚀**
