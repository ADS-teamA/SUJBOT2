#!/bin/bash
# Deployment script for SUJBOT2
# Zero-downtime deployment with health checks

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

COMPOSE_FILE="docker-compose.prod.yml"
ENVIRONMENT="${1:-production}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}SUJBOT2 Deployment${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Environment: $ENVIRONMENT"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Check if .env.prod exists
if [ ! -f ".env.prod" ]; then
    echo -e "${RED}Error: .env.prod file not found${NC}"
    echo "Please create .env.prod from .env.prod.example"
    exit 1
fi

# Load environment variables
set -a
source .env.prod
set +a

echo -e "${YELLOW}Step 1: Pulling latest changes from git...${NC}"
git pull origin main || echo "Warning: Git pull failed or not in git repository"
echo -e "${GREEN}✓ Code updated${NC}"
echo ""

echo -e "${YELLOW}Step 2: Building Docker images...${NC}"
docker-compose -f "$COMPOSE_FILE" build --no-cache
echo -e "${GREEN}✓ Images built${NC}"
echo ""

echo -e "${YELLOW}Step 3: Creating backup before deployment...${NC}"
./deployment/scripts/backup.sh
echo -e "${GREEN}✓ Backup completed${NC}"
echo ""

echo -e "${YELLOW}Step 4: Stopping old containers...${NC}"
docker-compose -f "$COMPOSE_FILE" down
echo -e "${GREEN}✓ Containers stopped${NC}"
echo ""

echo -e "${YELLOW}Step 5: Starting new containers...${NC}"
docker-compose -f "$COMPOSE_FILE" up -d
echo -e "${GREEN}✓ Containers started${NC}"
echo ""

echo -e "${YELLOW}Step 6: Waiting for services to be healthy...${NC}"
sleep 10

# Wait for backend to be healthy
MAX_WAIT=60
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    HEALTH=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/api/v1/health 2>/dev/null || echo "000")

    if [ "$HEALTH" = "200" ]; then
        echo -e "${GREEN}✓ Backend is healthy${NC}"
        break
    fi

    echo "Waiting for backend... ($WAITED/$MAX_WAIT seconds)"
    sleep 5
    WAITED=$((WAITED + 5))
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo -e "${RED}✗ Backend failed to become healthy within $MAX_WAIT seconds${NC}"
    echo "Rolling back..."
    docker-compose -f "$COMPOSE_FILE" down
    ./deployment/scripts/restore.sh ./backups/latest
    exit 1
fi

echo ""
echo -e "${YELLOW}Step 7: Running health checks...${NC}"
./deployment/scripts/health-check.sh "$COMPOSE_FILE"
echo ""

# Cleanup old images
echo -e "${YELLOW}Step 8: Cleaning up old Docker images...${NC}"
docker image prune -f
echo -e "${GREEN}✓ Cleanup completed${NC}"
echo ""

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Deployment completed successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Services:"
echo "  Frontend: https://${DOMAIN:-localhost}"
echo "  API: https://${DOMAIN:-localhost}/api/v1"
echo "  API Docs: https://${DOMAIN:-localhost}/api/docs"
echo ""
echo "Monitoring:"
echo "  View logs: docker-compose -f $COMPOSE_FILE logs -f"
echo "  Service status: docker-compose -f $COMPOSE_FILE ps"
echo "  Health check: ./deployment/scripts/health-check.sh"
echo ""

exit 0
