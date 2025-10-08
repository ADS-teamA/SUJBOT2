# SUJBOT2 - Deployment Guide

Complete deployment infrastructure for SUJBOT2 using Docker containers.

## 📋 Contents

- **Docker Compose Files**: Development and production configurations
- **Nginx Configuration**: Reverse proxy with SSL/TLS
- **Deployment Scripts**: Automated deployment, backup, and health checks
- **Monitoring**: Prometheus + Grafana stack (optional)

## 🚀 Quick Start

### Development

```bash
# 1. Create environment file
cp .env.dev.example .env.dev
# Edit .env.dev and add your CLAUDE_API_KEY

# 2. Start all services
docker-compose -f docker-compose.dev.yml up --build

# 3. Access services
# - Frontend: http://localhost:3000
# - Backend API: http://localhost:8000
# - API Docs: http://localhost:8000/docs
# - Flower (Celery): http://localhost:5555
```

### Production

```bash
# 1. Create environment file
cp .env.prod.example .env.prod
# Edit .env.prod and set all required variables

# 2. Setup SSL/TLS certificates
chmod +x deployment/scripts/*.sh
./deployment/scripts/ssl-setup.sh your-domain.com admin@your-domain.com

# 3. Deploy
./deployment/scripts/deploy.sh

# 4. Verify deployment
./deployment/scripts/health-check.sh
```

## 📁 Directory Structure

```
deployment/
├── nginx/
│   ├── nginx.conf              # Main Nginx config
│   ├── conf.d/
│   │   └── sujbot2.conf       # Site configuration
│   └── ssl/                    # SSL certificates (auto-generated)
├── scripts/
│   ├── backup.sh              # Automated backup
│   ├── restore.sh             # Restore from backup
│   ├── deploy.sh              # Production deployment
│   ├── health-check.sh        # Service health check
│   └── ssl-setup.sh           # SSL certificate setup
├── monitoring/
│   ├── prometheus.yml         # Prometheus config
│   └── docker-compose.monitoring.yml
└── README.md                  # This file
```

## 🏗️ Architecture

```
Internet/Users
       │
       ▼
┌─────────────┐
│   Nginx     │  SSL Termination + Reverse Proxy
│   80/443    │
└──────┬──────┘
       │
   ┌───┴────┬──────────┐
   ▼        ▼          ▼
┌────────┐ ┌────────┐ ┌────────┐
│Frontend│ │Backend │ │Backend │
│ React  │ │FastAPI │ │FastAPI │
└────────┘ └────┬───┘ └────┬───┘
                │           │
           ┌────┴───────────┴────┐
           │                     │
           ▼                     ▼
      ┌────────┐           ┌──────────┐
      │ Redis  │           │  Celery  │
      │  6379  │           │ Workers  │
      └────────┘           └──────────┘
```

## 🔧 Configuration

### Environment Variables

**Development (.env.dev)**
```bash
CLAUDE_API_KEY=your-api-key
ENVIRONMENT=development
DEBUG=true
VITE_API_URL=http://localhost:8000/api/v1
VITE_WS_URL=ws://localhost:8000/ws
```

**Production (.env.prod)**
```bash
CLAUDE_API_KEY=your-production-api-key
REDIS_PASSWORD=secure-random-password
SECRET_KEY=secure-random-secret
ENVIRONMENT=production
DEBUG=false
VITE_API_URL=https://your-domain.com/api/v1
VITE_WS_URL=wss://your-domain.com/ws
```

### Docker Compose Services

**Development (docker-compose.dev.yml)**
- Redis (port 6379)
- Backend API (port 8000, hot reload)
- Celery Worker
- Flower (port 5555, Celery monitoring)
- Frontend (port 3000, hot reload)

**Production (docker-compose.prod.yml)**
- Nginx (ports 80, 443)
- Certbot (SSL renewal)
- Redis (with password)
- Backend API (2 replicas, Gunicorn)
- Celery Workers (3 replicas)
- Frontend (built static files)

## 🛠️ Management Scripts

### Deployment

```bash
# Full production deployment
./deployment/scripts/deploy.sh

# The script will:
# 1. Pull latest code from git
# 2. Build Docker images
# 3. Create backup
# 4. Stop old containers
# 5. Start new containers
# 6. Wait for services to be healthy
# 7. Run health checks
# 8. Cleanup old images
```

### Backup & Restore

```bash
# Create backup
./deployment/scripts/backup.sh

# Backups are stored in ./backups/ directory:
# - redis-TIMESTAMP.tar.gz
# - uploads-TIMESTAMP.tar.gz
# - indexes-TIMESTAMP.tar.gz

# Restore from backup
./deployment/scripts/restore.sh ./backups/TIMESTAMP all

# Restore specific component
./deployment/scripts/restore.sh ./backups/TIMESTAMP redis
./deployment/scripts/restore.sh ./backups/TIMESTAMP uploads
./deployment/scripts/restore.sh ./backups/TIMESTAMP indexes
```

### Health Checks

```bash
# Check all services
./deployment/scripts/health-check.sh docker-compose.prod.yml

# The script checks:
# ✓ Docker container status
# ✓ Service health (health checks)
# ✓ API endpoint availability
# ✓ Redis connectivity
# ✓ Celery worker status
# ✓ Disk space usage
# ✓ Memory usage
```

### SSL/TLS Setup

```bash
# Initial setup
./deployment/scripts/ssl-setup.sh your-domain.com admin@your-domain.com

# The script will:
# 1. Request certificate from Let's Encrypt
# 2. Configure Nginx with SSL
# 3. Enable auto-renewal
# 4. Verify HTTPS is working

# Manual certificate renewal
docker-compose -f docker-compose.prod.yml exec certbot certbot renew

# Note: Auto-renewal runs every 12 hours via Certbot container
```

## 📊 Monitoring (Optional)

```bash
# Start monitoring stack
docker-compose -f deployment/docker-compose.monitoring.yml up -d

# Access:
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3001 (admin/admin)
# - Node Exporter: http://localhost:9100
# - cAdvisor: http://localhost:8080

# Services monitored:
# - Host metrics (CPU, memory, disk)
# - Container metrics (per-container resources)
# - Redis metrics (connections, memory, operations)
# - Application metrics (if instrumented)
```

## 🔍 Troubleshooting

### View Logs

```bash
# All services
docker-compose -f docker-compose.prod.yml logs

# Specific service
docker-compose -f docker-compose.prod.yml logs -f backend

# Last 100 lines
docker-compose -f docker-compose.prod.yml logs --tail=100 backend

# Follow logs in real-time
docker-compose -f docker-compose.prod.yml logs -f backend celery_worker
```

### Service Management

```bash
# Check service status
docker-compose -f docker-compose.prod.yml ps

# Restart service
docker-compose -f docker-compose.prod.yml restart backend

# Rebuild and restart
docker-compose -f docker-compose.prod.yml up -d --build backend

# Scale services
docker-compose -f docker-compose.prod.yml up -d --scale backend=4
docker-compose -f docker-compose.prod.yml up -d --scale celery_worker=5
```

### Common Issues

**Container won't start**
```bash
# Check logs
docker-compose -f docker-compose.prod.yml logs backend

# Enter container
docker-compose -f docker-compose.prod.yml exec backend /bin/bash

# Inspect container
docker inspect sujbot2_backend
```

**Redis connection failed**
```bash
# Test Redis
docker-compose -f docker-compose.prod.yml exec redis redis-cli -a YOUR_PASSWORD ping

# Should return: PONG
```

**Celery tasks not processing**
```bash
# Check worker status
docker-compose -f docker-compose.prod.yml exec celery_worker \
  celery -A app.core.celery_app inspect active

# Check queue
docker-compose -f docker-compose.prod.yml exec celery_worker \
  celery -A app.core.celery_app inspect stats
```

**SSL certificate issues**
```bash
# Check certificate
docker-compose -f docker-compose.prod.yml exec nginx \
  openssl s_client -connect localhost:443 -servername your-domain.com

# Verify Nginx config
docker-compose -f docker-compose.prod.yml exec nginx nginx -t

# Check Certbot logs
docker-compose -f docker-compose.prod.yml logs certbot
```

## 🔄 Scaling

### Horizontal Scaling

```bash
# Scale backend API to 4 instances
docker-compose -f docker-compose.prod.yml up -d --scale backend=4

# Scale Celery workers to 5 instances
docker-compose -f docker-compose.prod.yml up -d --scale celery_worker=5

# Nginx automatically load balances requests
```

### Vertical Scaling

Edit `docker-compose.prod.yml`:

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

## 🔒 Security Best Practices

1. **Change default passwords**
   - Redis password in `.env.prod`
   - Grafana admin password
   - Secret key for backend

2. **Use strong SSL/TLS**
   - Let's Encrypt certificates (auto-renewed)
   - TLS 1.2+ only
   - Strong cipher suites

3. **Limit exposed ports**
   - Production: Only 80 and 443
   - Internal services use Docker network

4. **Regular backups**
   - Automated daily backups (cron)
   - Test restore procedure regularly
   - Store backups off-site

5. **Keep images updated**
   ```bash
   docker-compose -f docker-compose.prod.yml pull
   docker-compose -f docker-compose.prod.yml up -d
   ```

## 📝 Automated Backups with Cron

```bash
# Add to crontab
crontab -e

# Run backup daily at 2 AM
0 2 * * * /path/to/advanced_sujbot2/deployment/scripts/backup.sh >> /var/log/sujbot2-backup.log 2>&1

# Upload to cloud storage (optional)
30 2 * * * /path/to/scripts/upload-to-s3.sh >> /var/log/sujbot2-upload.log 2>&1
```

## 🌐 Production Checklist

Before going to production:

- [ ] Update `.env.prod` with production values
- [ ] Set strong passwords for Redis and Grafana
- [ ] Configure domain DNS to point to server
- [ ] Setup SSL certificates with Let's Encrypt
- [ ] Configure firewall (allow only 80, 443, SSH)
- [ ] Test backup and restore procedure
- [ ] Setup automated backups with cron
- [ ] Configure monitoring alerts (optional)
- [ ] Review Nginx security headers
- [ ] Test health check endpoints
- [ ] Document custom configuration
- [ ] Setup log rotation
- [ ] Plan rollback procedure

## 📚 Additional Resources

- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Nginx Documentation](https://nginx.org/en/docs/)
- [Let's Encrypt Documentation](https://letsencrypt.org/docs/)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)

## 🆘 Support

For issues and questions:
1. Check logs: `docker-compose logs [service]`
2. Run health check: `./deployment/scripts/health-check.sh`
3. Review troubleshooting section above
4. Check project documentation

---

**Last Updated**: 2025-10-08
**Version**: 1.0.0
