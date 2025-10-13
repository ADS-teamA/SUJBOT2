#!/bin/bash
# Quick Start Script for SUJBOT2
# Sets up development environment with minimal configuration

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}SUJBOT2 - Quick Start${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    echo "Please install Docker from: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Error: Docker Compose is not installed${NC}"
    echo "Please install Docker Compose from: https://docs.docker.com/compose/install/"
    exit 1
fi

echo -e "${GREEN}✓ Docker and Docker Compose are installed${NC}"
echo ""

# Check if .env.dev exists
if [ ! -f ".env.dev" ]; then
    echo -e "${YELLOW}Creating .env.dev file...${NC}"
    cp .env.dev.example .env.dev

    # Prompt for Claude API key
    echo ""
    read -p "Enter your Claude API key: " CLAUDE_KEY

    if [ -z "$CLAUDE_KEY" ]; then
        echo -e "${RED}Error: Claude API key is required${NC}"
        exit 1
    fi

    # Update .env.dev with API key
    sed -i.bak "s/your-claude-api-key-here/$CLAUDE_KEY/g" .env.dev
    rm -f .env.dev.bak

    echo -e "${GREEN}✓ Environment file created${NC}"
else
    echo -e "${GREEN}✓ Environment file already exists${NC}"
fi

echo ""
echo -e "${YELLOW}Building Docker images...${NC}"
echo "This may take a few minutes on first run..."
echo ""

# Build and start services
docker-compose -f docker-compose.dev.yml up -d --build

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Failed to start services${NC}"
    exit 1
fi

echo ""
echo -e "${YELLOW}Waiting for services to be ready...${NC}"
sleep 10

# Check if backend is healthy
MAX_WAIT=60
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    HEALTH=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/api/v1/health 2>/dev/null || echo "000")

    if [ "$HEALTH" = "200" ]; then
        echo -e "${GREEN}✓ Backend is ready${NC}"
        break
    fi

    echo "Waiting for backend... ($WAITED/$MAX_WAIT seconds)"
    sleep 5
    WAITED=$((WAITED + 5))
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo -e "${RED}✗ Backend failed to start within $MAX_WAIT seconds${NC}"
    echo "Check logs with: docker-compose -f docker-compose.dev.yml logs backend"
    exit 1
fi

# Check if frontend is ready
FRONTEND_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:3000 2>/dev/null || echo "000")

if [ "$FRONTEND_STATUS" = "200" ]; then
    echo -e "${GREEN}✓ Frontend is ready${NC}"
else
    echo -e "${YELLOW}⚠ Frontend may still be building...${NC}"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Setup Complete! 🚀${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Your development environment is running!"
echo ""
echo "Access the application:"
echo -e "  ${BLUE}Frontend:${NC}    http://localhost:3000"
echo -e "  ${BLUE}Backend API:${NC} http://localhost:8000"
echo -e "  ${BLUE}API Docs:${NC}    http://localhost:8000/docs"
echo -e "  ${BLUE}Flower:${NC}      http://localhost:5555"
echo ""
echo "Useful commands:"
echo "  View logs:      docker-compose -f docker-compose.dev.yml logs -f"
echo "  Stop services:  docker-compose -f docker-compose.dev.yml down"
echo "  Restart:        docker-compose -f docker-compose.dev.yml restart"
echo ""
echo "Or use Make commands:"
echo "  make dev-logs   - View logs"
echo "  make dev-down   - Stop services"
echo "  make help       - See all available commands"
echo ""

exit 0
