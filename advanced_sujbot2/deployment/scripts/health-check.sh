#!/bin/bash
# Health check script for SUJBOT2
# Monitors all services and reports status

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

COMPOSE_FILE="${1:-docker-compose.prod.yml}"
API_URL="${2:-http://localhost:8000}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}SUJBOT2 Health Check${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Compose file: $COMPOSE_FILE"
echo "API URL: $API_URL"
echo ""

# Function to check service status
check_service() {
    local service=$1
    local status=$(docker-compose -f "$COMPOSE_FILE" ps -q "$service" 2>/dev/null)

    if [ -z "$status" ]; then
        echo -e "${RED}✗ $service: NOT RUNNING${NC}"
        return 1
    fi

    local health=$(docker inspect --format='{{.State.Health.Status}}' $(docker-compose -f "$COMPOSE_FILE" ps -q "$service" 2>/dev/null) 2>/dev/null || echo "none")

    if [ "$health" = "healthy" ]; then
        echo -e "${GREEN}✓ $service: HEALTHY${NC}"
        return 0
    elif [ "$health" = "none" ]; then
        # No health check defined, check if running
        local running=$(docker inspect --format='{{.State.Running}}' $(docker-compose -f "$COMPOSE_FILE" ps -q "$service" 2>/dev/null) 2>/dev/null)
        if [ "$running" = "true" ]; then
            echo -e "${GREEN}✓ $service: RUNNING${NC}"
            return 0
        else
            echo -e "${RED}✗ $service: NOT RUNNING${NC}"
            return 1
        fi
    else
        echo -e "${YELLOW}⚠ $service: $health${NC}"
        return 1
    fi
}

# Check Docker services
echo -e "${YELLOW}Checking Docker services...${NC}"
echo ""

SERVICES=("redis" "backend" "celery_worker" "nginx" "frontend")
FAILED=0

for service in "${SERVICES[@]}"; do
    check_service "$service" || FAILED=$((FAILED + 1))
done

echo ""

# Check API health endpoint
echo -e "${YELLOW}Checking API health endpoint...${NC}"
API_HEALTH=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL/api/v1/health" 2>/dev/null || echo "000")

if [ "$API_HEALTH" = "200" ]; then
    echo -e "${GREEN}✓ API Health: OK (HTTP $API_HEALTH)${NC}"
else
    echo -e "${RED}✗ API Health: FAILED (HTTP $API_HEALTH)${NC}"
    FAILED=$((FAILED + 1))
fi

# Check Redis connectivity
echo -e "${YELLOW}Checking Redis connectivity...${NC}"
REDIS_PING=$(docker-compose -f "$COMPOSE_FILE" exec -T redis redis-cli ping 2>/dev/null || echo "FAILED")

if [ "$REDIS_PING" = "PONG" ]; then
    echo -e "${GREEN}✓ Redis: CONNECTED${NC}"
else
    echo -e "${RED}✗ Redis: DISCONNECTED${NC}"
    FAILED=$((FAILED + 1))
fi

# Check Celery workers
echo -e "${YELLOW}Checking Celery workers...${NC}"
CELERY_WORKERS=$(docker-compose -f "$COMPOSE_FILE" exec -T celery_worker celery -A app.core.celery_app inspect active 2>/dev/null || echo "FAILED")

if echo "$CELERY_WORKERS" | grep -q "OK"; then
    echo -e "${GREEN}✓ Celery Workers: ACTIVE${NC}"
else
    echo -e "${RED}✗ Celery Workers: INACTIVE${NC}"
    FAILED=$((FAILED + 1))
fi

# Check disk space
echo ""
echo -e "${YELLOW}Checking disk space...${NC}"

UPLOADS_SIZE=$(docker run --rm -v sujbot2_uploads_prod:/data alpine du -sh /data 2>/dev/null | cut -f1 || echo "N/A")
INDEXES_SIZE=$(docker run --rm -v sujbot2_indexes_prod:/data alpine du -sh /data 2>/dev/null | cut -f1 || echo "N/A")
REDIS_SIZE=$(docker run --rm -v sujbot2_redis_data_prod:/data alpine du -sh /data 2>/dev/null | cut -f1 || echo "N/A")

echo "  Uploads: $UPLOADS_SIZE"
echo "  Indexes: $INDEXES_SIZE"
echo "  Redis: $REDIS_SIZE"

# Check memory usage
echo ""
echo -e "${YELLOW}Checking memory usage...${NC}"
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" 2>/dev/null | head -n 6

# Summary
echo ""
echo -e "${BLUE}========================================${NC}"

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All systems operational! ✓${NC}"
    echo -e "${BLUE}========================================${NC}"
    exit 0
else
    echo -e "${RED}$FAILED issue(s) detected! ✗${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    echo "Troubleshooting commands:"
    echo "  View logs: docker-compose -f $COMPOSE_FILE logs -f [service]"
    echo "  Restart service: docker-compose -f $COMPOSE_FILE restart [service]"
    echo "  Rebuild service: docker-compose -f $COMPOSE_FILE up -d --build [service]"
    exit 1
fi
