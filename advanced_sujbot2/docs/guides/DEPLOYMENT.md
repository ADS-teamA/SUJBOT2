# SUJBOT2 - Deployment Guide

Complete guide for deploying SUJBOT2 in development and production environments.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture](#architecture)
3. [Development Setup](#development-setup)
4. [Production Deployment](#production-deployment)
5. [SSL/TLS Configuration](#ssltls-configuration)
6. [Monitoring](#monitoring)
7. [Backup & Recovery](#backup--recovery)
8. [Troubleshooting](#troubleshooting)

## Quick Start

### Development (5 minutes)

```bash
# 1. Clone repository
git clone <repository-url>
cd advanced_sujbot2

# 2. Configure environment
cp .env.dev.example .env.dev
# Edit .env.dev and add your CLAUDE_API_KEY

# 3. Start all services
docker-compose -f docker-compose.dev.yml up --build

# 4. Access application
# - Frontend: http://localhost:3000
# - Backend API: http://localhost:8000
# - API Documentation: http://localhost:8000/docs
# - Celery Flower: http://localhost:5555
```

### Production (15 minutes)

```bash
# 1. Configure environment
cp .env.prod.example .env.prod
# Edit .env.prod and set all required variables

# 2. Setup SSL certificates
cd deployment/scripts
./ssl-setup.sh your-domain.com admin@your-domain.com

# 3. Deploy
./deploy.sh

# 4. Verify
./health-check.sh
```

## Architecture

### Container Overview

```
┌─────────────────────────────────────────────┐
│              Internet/Users                 │
└───────────────────┬─────────────────────────┘
                    │
                    ▼
            ┌───────────────┐
            │ Nginx (80/443)│  SSL Termination + Reverse Proxy
            └───────┬───────┘
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
    ┌────────┐ ┌────────┐ ┌────────┐
    │Frontend│ │Backend │ │Backend │  FastAPI + Gunicorn
    │ React  │ │  API   │ │  API   │  (2 replicas)
    └────────┘ └────┬───┘ └────┬───┘
                    │           │
            ┌───────┴───────────┴────────┐
            │                            │
            ▼                            ▼
        ┌────────┐                  ┌──────────┐
        │ Redis  │                  │  Celery  │
        │  6379  │                  │ Workers  │
        └────────┘                  └──────────┘
                                    (3 replicas)
```

### Data Flow

1. **User Request** → Nginx (HTTPS/443)
2. **Static Files** → Nginx serves React SPA
3. **API Calls** → Nginx → Backend (load balanced)
4. **WebSocket** → Nginx → Backend (upgraded connection)
5. **Background Tasks** → Backend → Celery → Workers
6. **Results** → Redis → Backend → User

### Volumes

- `redis_data_prod` - Redis persistence
- `uploads_prod` - Uploaded documents
- `indexes_prod` - FAISS vector indices
- `frontend_static` - Built React assets

## Development Setup

### Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- 8GB RAM minimum
- 20GB disk space

### Configuration

Create `.env.dev` from template:

```bash
cp .env.dev.example .env.dev
```

Required variables:

```bash
CLAUDE_API_KEY=your-api-key-here
ENVIRONMENT=development
DEBUG=true
```

### Start Services

```bash
# Build and start all containers
docker-compose -f docker-compose.dev.yml up --build

# Start in detached mode
docker-compose -f docker-compose.dev.yml up -d

# View logs
docker-compose -f docker-compose.dev.yml logs -f

# Stop all services
docker-compose -f docker-compose.dev.yml down

# Stop and remove volumes (CAUTION: deletes data)
docker-compose -f docker-compose.dev.yml down -v
```

### Hot Reload

Both frontend and backend support hot reload in development:

- **Frontend**: Changes to `frontend/src/` trigger Vite rebuild
- **Backend**: Changes to `backend/app/` trigger Uvicorn reload

### Development URLs

| Service | URL | Description |
|---------|-----|-------------|
| Frontend | http://localhost:3000 | React application |
| Backend API | http://localhost:8000 | FastAPI endpoints |
| API Docs | http://localhost:8000/docs | Swagger UI |
| Redoc | http://localhost:8000/redoc | Alternative docs |
| Flower | http://localhost:5555 | Celery monitoring |
| Redis | localhost:6379 | Direct Redis access |

## Production Deployment

### Prerequisites

- Linux server (Ubuntu 20.04+ recommended)
- Docker 20.10+
- Docker Compose 2.0+
- Domain name pointing to server IP
- 16GB RAM minimum
- 50GB disk space

### 1. Initial Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Clone repository
git clone <repository-url>
cd advanced_sujbot2
```

### 2. Configure Environment

```bash
# Create production environment file
cp .env.prod.example .env.prod

# Edit with production values
nano .env.prod
```

**Critical settings:**

```bash
# Strong passwords
REDIS_PASSWORD=generate-secure-random-password
SECRET_KEY=generate-secure-random-secret

# Production domain
VITE_API_URL=https://your-domain.com/api/v1
VITE_WS_URL=wss://your-domain.com/ws

# Claude API
CLAUDE_API_KEY=your-production-api-key

# Security
CORS_ORIGINS=["https://your-domain.com"]
DEBUG=false
ENVIRONMENT=production
```

### 3. SSL/TLS Setup

```bash
# Make scripts executable
chmod +x deployment/scripts/*.sh

# Obtain Let's Encrypt certificate
./deployment/scripts/ssl-setup.sh your-domain.com admin@your-domain.com
```

The script will:
- Update Nginx configuration with your domain
- Request certificate from Let's Encrypt
- Configure automatic renewal
- Verify HTTPS is working

### 4. Deploy Application

```bash
# Run deployment script
./deployment/scripts/deploy.sh
```

The script performs:
1. Pull latest code from git
2. Build Docker images
3. Create backup of current data
4. Stop old containers
5. Start new containers
6. Wait for services to be healthy
7. Run health checks
8. Cleanup old images

### 5. Verify Deployment

```bash
# Run health check
./deployment/scripts/health-check.sh

# Check service status
docker-compose -f docker-compose.prod.yml ps

# View logs
docker-compose -f docker-compose.prod.yml logs -f
```

### Production URLs

| Service | URL | Description |
|---------|-----|-------------|
| Application | https://your-domain.com | Main application |
| API | https://your-domain.com/api/v1 | API endpoints |
| API Docs | https://your-domain.com/api/docs | Swagger UI |

## SSL/TLS Configuration

### Automatic (Let's Encrypt)

Recommended for production. Free, automated, and trusted.

```bash
# Initial setup
./deployment/scripts/ssl-setup.sh your-domain.com admin@your-domain.com

# Manual renewal (automatic renewal runs every 12h)
docker-compose -f docker-compose.prod.yml exec certbot certbot renew

# Force renewal
docker-compose -f docker-compose.prod.yml run --rm certbot certonly \
  --webroot \
  --webroot-path=/var/www/certbot \
  --email admin@your-domain.com \
  --agree-tos \
  --force-renewal \
  -d your-domain.com
```

### Manual Certificates

If you have your own SSL certificates:

```bash
# Copy certificates
cp fullchain.pem deployment/nginx/ssl/fullchain.pem
cp privkey.pem deployment/nginx/ssl/privkey.pem

# Update Nginx config to point to your certificates
nano deployment/nginx/conf.d/sujbot2.conf

# Restart Nginx
docker-compose -f docker-compose.prod.yml restart nginx
```

## Monitoring

### Built-in Monitoring

```bash
# Service health
./deployment/scripts/health-check.sh

# View metrics
docker stats

# Check Celery tasks
docker-compose -f docker-compose.prod.yml exec celery_worker \
  celery -A app.core.celery_app inspect active
```

### Optional: Prometheus + Grafana

```bash
# Start monitoring stack
docker-compose -f deployment/docker-compose.monitoring.yml up -d

# Access:
# - Prometheus: http://your-server:9090
# - Grafana: http://your-server:3001 (admin/admin)
```

Metrics collected:
- Container resource usage (CPU, memory, network)
- Host metrics (disk, load)
- Redis metrics (connections, memory, operations)
- Application metrics (requests, latency)

## Backup & Recovery

### Automated Backups

```bash
# Manual backup
./deployment/scripts/backup.sh

# Backups stored in: deployment/backups/
# - redis-TIMESTAMP.tar.gz
# - uploads-TIMESTAMP.tar.gz
# - indexes-TIMESTAMP.tar.gz

# Setup automated daily backups
crontab -e

# Add line (runs daily at 2 AM):
0 2 * * * /path/to/advanced_sujbot2/deployment/scripts/backup.sh >> /var/log/sujbot2-backup.log 2>&1
```

### Restore from Backup

```bash
# Restore all components
./deployment/scripts/restore.sh ./deployment/backups/2025-10-08-120000

# Restore specific component
./deployment/scripts/restore.sh ./deployment/backups/2025-10-08-120000 redis
./deployment/scripts/restore.sh ./deployment/backups/2025-10-08-120000 uploads
./deployment/scripts/restore.sh ./deployment/backups/2025-10-08-120000 indexes
```

### Backup to Cloud Storage

Example for AWS S3:

```bash
# Install AWS CLI
aws configure

# Add to backup script
aws s3 cp ./deployment/backups/ s3://your-bucket/sujbot2-backups/ --recursive

# Restore from S3
aws s3 cp s3://your-bucket/sujbot2-backups/ ./deployment/backups/ --recursive
```

## Troubleshooting

### View Logs

```bash
# All services
docker-compose -f docker-compose.prod.yml logs

# Specific service
docker-compose -f docker-compose.prod.yml logs -f backend
docker-compose -f docker-compose.prod.yml logs -f celery_worker
docker-compose -f docker-compose.prod.yml logs -f nginx

# Last 100 lines
docker-compose -f docker-compose.prod.yml logs --tail=100 backend

# Follow in real-time
docker-compose -f docker-compose.prod.yml logs -f --tail=0 backend
```

### Service Management

```bash
# Check status
docker-compose -f docker-compose.prod.yml ps

# Restart service
docker-compose -f docker-compose.prod.yml restart backend

# Rebuild service
docker-compose -f docker-compose.prod.yml up -d --build backend

# Scale services
docker-compose -f docker-compose.prod.yml up -d --scale backend=4
docker-compose -f docker-compose.prod.yml up -d --scale celery_worker=5
```

### Common Issues

#### Container Won't Start

```bash
# Check logs
docker-compose -f docker-compose.prod.yml logs backend

# Enter container
docker-compose -f docker-compose.prod.yml exec backend /bin/bash

# Inspect container
docker inspect sujbot2_backend
```

#### Redis Connection Failed

```bash
# Test Redis
docker-compose -f docker-compose.prod.yml exec redis redis-cli -a YOUR_PASSWORD ping

# Should return: PONG
```

#### Celery Tasks Not Processing

```bash
# Check workers
docker-compose -f docker-compose.prod.yml exec celery_worker \
  celery -A app.core.celery_app inspect active

# Check queue
docker-compose -f docker-compose.prod.yml exec celery_worker \
  celery -A app.core.celery_app inspect stats

# Purge queue (CAUTION)
docker-compose -f docker-compose.prod.yml exec celery_worker \
  celery -A app.core.celery_app purge
```

#### High Memory Usage

```bash
# Check resource usage
docker stats

# Adjust resource limits in docker-compose.prod.yml
services:
  backend:
    deploy:
      resources:
        limits:
          memory: 4G
```

#### SSL Certificate Issues

```bash
# Verify certificate
docker-compose -f docker-compose.prod.yml exec nginx \
  openssl s_client -connect localhost:443 -servername your-domain.com

# Test Nginx config
docker-compose -f docker-compose.prod.yml exec nginx nginx -t

# Check Certbot logs
docker-compose -f docker-compose.prod.yml logs certbot
```

## Scaling

### Horizontal Scaling

```bash
# Scale backend API
docker-compose -f docker-compose.prod.yml up -d --scale backend=4

# Scale Celery workers
docker-compose -f docker-compose.prod.yml up -d --scale celery_worker=5

# Nginx automatically load balances requests
```

### Vertical Scaling

Edit resource limits in `docker-compose.prod.yml`:

```yaml
services:
  backend:
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 8G
        reservations:
          cpus: '2.0'
          memory: 4G
```

## Security Checklist

Before production deployment:

- [ ] Change all default passwords
- [ ] Set strong SECRET_KEY and REDIS_PASSWORD
- [ ] Configure SSL/TLS certificates
- [ ] Enable firewall (allow only 80, 443, SSH)
- [ ] Configure CORS_ORIGINS properly
- [ ] Disable DEBUG mode
- [ ] Setup automated backups
- [ ] Review security headers in Nginx config
- [ ] Enable rate limiting
- [ ] Setup monitoring alerts
- [ ] Document access procedures
- [ ] Test backup/restore procedure

## Performance Tuning

### Backend

```yaml
# docker-compose.prod.yml
services:
  backend:
    command: gunicorn app.main:app \
      --workers 4 \
      --worker-class uvicorn.workers.UvicornWorker \
      --bind 0.0.0.0:8000 \
      --timeout 300
```

### Celery

```yaml
services:
  celery_worker:
    command: celery -A app.core.celery_app worker \
      --loglevel=info \
      --concurrency=4 \
      --max-tasks-per-child=1000
```

### Redis

```yaml
services:
  redis:
    command: redis-server \
      --appendonly yes \
      --requirepass ${REDIS_PASSWORD} \
      --maxmemory 2gb \
      --maxmemory-policy allkeys-lru
```

## Useful Commands

```bash
# Check disk space
df -h
docker system df

# Clean up Docker
docker system prune -a
docker volume prune

# View environment variables
docker-compose -f docker-compose.prod.yml config

# Export logs
docker-compose -f docker-compose.prod.yml logs > logs-$(date +%Y%m%d).txt

# Database shell (if using PostgreSQL)
docker-compose -f docker-compose.prod.yml exec db psql -U postgres

# Python shell in backend
docker-compose -f docker-compose.prod.yml exec backend python
```

## Support & Resources

- **Logs Location**: `./backend/logs/`
- **Backups Location**: `./deployment/backups/`
- **SSL Certificates**: `./deployment/nginx/ssl/`
- **Documentation**: See `deployment/README.md`

---

**Last Updated**: 2025-10-08
**Version**: 1.0.0
