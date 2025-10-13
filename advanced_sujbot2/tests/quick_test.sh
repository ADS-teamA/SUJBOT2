#!/bin/bash
# Quick PostgreSQL Migration Test
# Tests basic connectivity and functionality

set -e

echo "======================================================================"
echo "PostgreSQL Migration - Quick Test"
echo "======================================================================"
echo

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if PostgreSQL is running
echo "1. Checking PostgreSQL connection..."
if docker ps | grep -q sujbot2_postgres_dev; then
    echo -e "${GREEN}✓${NC} PostgreSQL container is running"
else
    echo -e "${RED}✗${NC} PostgreSQL container not found"
    echo "Starting Docker Compose..."
    docker-compose -f docker-compose.dev.yml up -d postgres
    sleep 5
fi

# Test database connection
echo
echo "2. Testing database connection..."
if docker exec sujbot2_postgres_dev psql -U sujbot_app -d sujbot2 -c "SELECT 1" > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Database connection successful"
else
    echo -e "${RED}✗${NC} Cannot connect to database"
    exit 1
fi

# Check extensions
echo
echo "3. Checking PostgreSQL extensions..."
EXTENSIONS=$(docker exec sujbot2_postgres_dev psql -U sujbot_app -d sujbot2 -t -c "SELECT extname FROM pg_extension WHERE extname IN ('vector', 'pg_trgm', 'btree_gin')")

if echo "$EXTENSIONS" | grep -q "vector"; then
    echo -e "${GREEN}✓${NC} pgvector extension installed"
else
    echo -e "${RED}✗${NC} pgvector extension not found"
    exit 1
fi

if echo "$EXTENSIONS" | grep -q "pg_trgm"; then
    echo -e "${GREEN}✓${NC} pg_trgm extension installed"
else
    echo -e "${YELLOW}⚠${NC} pg_trgm extension not found"
fi

# Check tables
echo
echo "4. Checking database schema..."
TABLES=$(docker exec sujbot2_postgres_dev psql -U sujbot_app -d sujbot2 -t -c "\dt" | wc -l)

if [ "$TABLES" -gt 5 ]; then
    echo -e "${GREEN}✓${NC} Database schema created ($TABLES tables)"
else
    echo -e "${YELLOW}⚠${NC} Schema may be incomplete ($TABLES tables)"
fi

# Check specific tables
for table in documents chunks references cross_references; do
    if docker exec sujbot2_postgres_dev psql -U sujbot_app -d sujbot2 -t -c "\d $table" > /dev/null 2>&1; then
        echo -e "  ${GREEN}✓${NC} Table '$table' exists"
    else
        echo -e "  ${RED}✗${NC} Table '$table' missing"
    fi
done

# Run Python test
echo
echo "5. Running Python integration test..."
if [ -f "tests/test_postgresql_migration.py" ]; then
    echo "Executing test suite..."
    python3 tests/test_postgresql_migration.py
else
    echo -e "${YELLOW}⚠${NC} Test file not found, skipping Python tests"
fi

echo
echo "======================================================================"
echo -e "${GREEN}Quick Test Complete!${NC}"
echo "======================================================================"
echo
echo "Next steps:"
echo "  • Run full test suite: pytest tests/test_postgresql_migration.py -v"
echo "  • Run benchmarks: python tests/benchmark_postgresql.py"
echo "  • Start backend: docker-compose -f docker-compose.dev.yml up backend"
echo
