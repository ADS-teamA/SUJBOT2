#!/bin/bash
#
# Start SUJBOT2 Web Interface
#
# This script starts both backend (FastAPI) and frontend (Vite) servers
# for the web interface.
#

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   SUJBOT2 Web Interface Startup${NC}"
echo -e "${BLUE}========================================${NC}"
echo

# ========================================
# Step 1: Check System Requirements
# ========================================
echo -e "${GREEN}Step 1/5: Checking system requirements...${NC}"

# Check Python/uv
if ! command -v uv &> /dev/null; then
    echo -e "${RED}Error: 'uv' not found${NC}"
    echo "Please install uv: https://github.com/astral-sh/uv"
    exit 1
fi
echo -e "  ✓ uv found: $(uv --version)"

# Check Node.js
if ! command -v node &> /dev/null; then
    echo -e "${RED}Error: Node.js not found${NC}"
    echo "Please install Node.js: https://nodejs.org/"
    exit 1
fi
echo -e "  ✓ Node.js found: $(node --version)"

# Check npm
if ! command -v npm &> /dev/null; then
    echo -e "${RED}Error: npm not found${NC}"
    exit 1
fi
echo -e "  ✓ npm found: $(npm --version)"
echo

# ========================================
# Step 2: Check Configuration Files
# ========================================
echo -e "${GREEN}Step 2/5: Checking configuration...${NC}"

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${RED}Error: .env file not found${NC}"
    echo "Please create .env from .env.example and configure API keys"
    exit 1
fi
echo -e "  ✓ .env file found"

# Check if vector store exists
if [ ! -d "vector_db" ]; then
    echo -e "${RED}Error: vector_db directory not found${NC}"
    echo "Please run the indexing pipeline first:"
    echo "  uv run python run_pipeline.py data/your_documents/"
    exit 1
fi
echo -e "  ✓ vector_db found"

# Check backend files
if [ ! -f "backend/requirements.txt" ]; then
    echo -e "${RED}Error: backend/requirements.txt not found${NC}"
    exit 1
fi
echo -e "  ✓ Backend files found"

# Check frontend files
if [ ! -f "frontend/package.json" ]; then
    echo -e "${RED}Error: frontend/package.json not found${NC}"
    exit 1
fi
echo -e "  ✓ Frontend files found"
echo

# ========================================
# Step 3: Install Backend Dependencies
# ========================================
echo -e "${GREEN}Step 3/5: Installing backend dependencies...${NC}"
cd backend
echo -e "${BLUE}  Installing Python packages with uv...${NC}"
uv pip install -r requirements.txt -q
if [ $? -eq 0 ]; then
    echo -e "  ✓ Backend dependencies installed"
else
    echo -e "${RED}  ✗ Failed to install backend dependencies${NC}"
    cd ..
    exit 1
fi
cd ..
echo

# ========================================
# Step 4: Install Frontend Dependencies
# ========================================
echo -e "${GREEN}Step 4/5: Installing frontend dependencies...${NC}"
cd frontend
echo -e "${BLUE}  Installing Node packages with npm...${NC}"
npm install --silent
if [ $? -eq 0 ]; then
    echo -e "  ✓ Frontend dependencies installed"
else
    echo -e "${RED}  ✗ Failed to install frontend dependencies${NC}"
    cd ..
    exit 1
fi
cd ..
echo

# ========================================
# Step 5: Cleanup Ports
# ========================================
echo -e "${GREEN}Step 5/5: Cleaning up ports...${NC}"
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:5173 | xargs kill -9 2>/dev/null || true
echo -e "  ✓ Ports 8000 and 5173 cleaned"
sleep 1
echo

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   Starting Servers...${NC}"
echo -e "${BLUE}========================================${NC}"
echo

# Start frontend in background first
echo -e "${GREEN}Starting Vite frontend on port 5173...${NC}"
cd frontend
npm run dev > /dev/null 2>&1 &
FRONTEND_PID=$!
cd ..
echo -e "${GREEN}✓ Frontend starting in background${NC}"

echo
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   SUJBOT2 Web Interface Ready!${NC}"
echo -e "${GREEN}========================================${NC}"
echo
echo -e "  Frontend: ${BLUE}http://localhost:5173${NC}"
echo -e "  Backend API: ${BLUE}http://localhost:8000${NC}"
echo -e "  API Docs: ${BLUE}http://localhost:8000/docs${NC}"
echo
echo -e "${BLUE}Starting FastAPI backend with hot reload...${NC}"
echo -e "${BLUE}All errors will be shown below:${NC}"
echo -e "${BLUE}Press Ctrl+C to stop both servers${NC}"
echo
echo -e "${GREEN}==================== Backend Logs ====================${NC}"

# Trap Ctrl+C to kill both processes
trap "echo -e '\n${RED}Stopping servers...${NC}'; kill $FRONTEND_PID 2>/dev/null; exit" INT

# Start backend in foreground (shows all logs)
# Set PYTHONPATH to include both backend/ and project root (for src/ imports)
PYTHONPATH="$(pwd)/backend:$(pwd):$PYTHONPATH" uv run python backend/main.py
