# Makefile for SUJBOT2
# Simplifies common Docker and deployment commands

.PHONY: help dev-up dev-down dev-logs dev-build prod-up prod-down prod-logs prod-deploy backup restore health ssl-setup clean

# Default target
help:
	@echo "SUJBOT2 - Available Commands"
	@echo ""
	@echo "Development:"
	@echo "  make dev-up        - Start development environment"
	@echo "  make dev-down      - Stop development environment"
	@echo "  make dev-logs      - View development logs"
	@echo "  make dev-build     - Rebuild development containers"
	@echo "  make dev-shell     - Open shell in backend container"
	@echo ""
	@echo "Production:"
	@echo "  make prod-up       - Start production environment"
	@echo "  make prod-down     - Stop production environment"
	@echo "  make prod-logs     - View production logs"
	@echo "  make prod-deploy   - Deploy to production"
	@echo "  make prod-scale    - Scale production services"
	@echo ""
	@echo "Maintenance:"
	@echo "  make backup        - Create backup of all data"
	@echo "  make restore       - Restore from latest backup"
	@echo "  make health        - Run health check"
	@echo "  make ssl-setup     - Setup SSL certificates"
	@echo "  make clean         - Clean up Docker resources"
	@echo ""
	@echo "Monitoring:"
	@echo "  make monitoring-up - Start monitoring stack"
	@echo "  make stats         - Show container stats"
	@echo ""

# Development commands
dev-up:
	docker-compose -f docker-compose.dev.yml up -d
	@echo "Development environment started!"
	@echo "Frontend: http://localhost:3000"
	@echo "Backend: http://localhost:8000"
	@echo "API Docs: http://localhost:8000/docs"
	@echo "Flower: http://localhost:5555"

dev-down:
	docker-compose -f docker-compose.dev.yml down

dev-logs:
	docker-compose -f docker-compose.dev.yml logs -f

dev-build:
	docker-compose -f docker-compose.dev.yml up -d --build

dev-shell:
	docker-compose -f docker-compose.dev.yml exec backend /bin/bash

dev-restart:
	docker-compose -f docker-compose.dev.yml restart

# Production commands
prod-up:
	docker-compose -f docker-compose.prod.yml up -d
	@echo "Production environment started!"
	@sleep 5
	@make health

prod-down:
	docker-compose -f docker-compose.prod.yml down

prod-logs:
	docker-compose -f docker-compose.prod.yml logs -f

prod-deploy:
	@echo "Deploying to production..."
	@chmod +x deployment/scripts/deploy.sh
	@./deployment/scripts/deploy.sh

prod-scale:
	@echo "Scaling services..."
	docker-compose -f docker-compose.prod.yml up -d --scale backend=4 --scale celery_worker=5

# Maintenance commands
backup:
	@echo "Creating backup..."
	@chmod +x deployment/scripts/backup.sh
	@./deployment/scripts/backup.sh

restore:
	@echo "Available backups:"
	@ls -lht deployment/backups/ | head -n 5
	@echo ""
	@read -p "Enter backup directory name: " backup_dir; \
	chmod +x deployment/scripts/restore.sh; \
	./deployment/scripts/restore.sh deployment/backups/$$backup_dir

health:
	@chmod +x deployment/scripts/health-check.sh
	@./deployment/scripts/health-check.sh docker-compose.prod.yml

ssl-setup:
	@chmod +x deployment/scripts/ssl-setup.sh
	@read -p "Enter domain name: " domain; \
	read -p "Enter admin email: " email; \
	./deployment/scripts/ssl-setup.sh $$domain $$email

clean:
	@echo "Cleaning up Docker resources..."
	docker system prune -af
	docker volume prune -f
	@echo "Cleanup complete!"

# Monitoring commands
monitoring-up:
	docker-compose -f deployment/docker-compose.monitoring.yml up -d
	@echo "Monitoring stack started!"
	@echo "Prometheus: http://localhost:9090"
	@echo "Grafana: http://localhost:3001 (admin/admin)"

monitoring-down:
	docker-compose -f deployment/docker-compose.monitoring.yml down

stats:
	docker stats --no-stream

# Database commands (if needed)
db-shell:
	docker-compose -f docker-compose.prod.yml exec db psql -U postgres

redis-cli:
	docker-compose -f docker-compose.prod.yml exec redis redis-cli

# Test commands
test-backend:
	docker-compose -f docker-compose.dev.yml exec backend pytest

test-frontend:
	docker-compose -f docker-compose.dev.yml exec frontend npm test

# Utility commands
ps:
	docker-compose -f docker-compose.prod.yml ps

top:
	docker-compose -f docker-compose.prod.yml top

exec-backend:
	docker-compose -f docker-compose.prod.yml exec backend /bin/bash

exec-frontend:
	docker-compose -f docker-compose.prod.yml exec frontend /bin/sh

# Quick development workflow
dev: dev-build dev-up dev-logs

# Quick production deployment
deploy: backup prod-deploy health
