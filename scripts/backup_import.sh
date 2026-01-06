#!/bin/bash
# SUJBOT2 Backup Import Script
# Restores data from a backup archive
#
# Usage: ./scripts/backup_import.sh <backup_archive.tar.gz>
#
# Prerequisites:
#   - Docker containers must be running (docker compose up -d)
#   - Run from the SUJBOT2 project directory

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Container names
POSTGRES_CONTAINER="sujbot_postgres"
NEO4J_CONTAINER="sujbot_neo4j"
REDIS_CONTAINER="sujbot_redis"

# Check arguments
if [ -z "$1" ]; then
    echo -e "${RED}Usage: $0 <backup_archive.tar.gz>${NC}"
    echo ""
    echo "Example:"
    echo "  ./scripts/backup_import.sh sujbot2_backup_20250106_143000.tar.gz"
    exit 1
fi

ARCHIVE="$1"

# Resolve absolute path
if [[ ! "$ARCHIVE" = /* ]]; then
    ARCHIVE="$(pwd)/$ARCHIVE"
fi

if [ ! -f "$ARCHIVE" ]; then
    echo -e "${RED}Error: Archive not found: $ARCHIVE${NC}"
    exit 1
fi

echo -e "${GREEN}=== SUJBOT2 Backup Import ===${NC}"
echo "Archive: $ARCHIVE"
echo "Project: $PROJECT_DIR"
echo ""

# Function to check if container is running
container_running() {
    docker ps --format '{{.Names}}' | grep -q "^$1$"
}

# Extract archive
echo -e "${YELLOW}[1/6] Extracting archive...${NC}"
cd "$PROJECT_DIR"
tar -xzf "$ARCHIVE"

# Find the backup directory
BACKUP_DIR=$(find "$PROJECT_DIR" -maxdepth 1 -type d -name "backup_*" | head -1)
if [ -z "$BACKUP_DIR" ]; then
    echo -e "${RED}Error: Could not find backup directory in archive${NC}"
    exit 1
fi
echo -e "${GREEN}  Extracted to: $BACKUP_DIR${NC}"

# 2. Restore PostgreSQL
echo -e "${YELLOW}[2/6] Restoring PostgreSQL database...${NC}"
if [ -f "$BACKUP_DIR/postgres.sql" ]; then
    if container_running "$POSTGRES_CONTAINER"; then
        # Drop and recreate database to ensure clean state
        echo "  Dropping existing data..."
        docker exec "$POSTGRES_CONTAINER" psql -U postgres -c "DROP SCHEMA IF EXISTS vectors CASCADE;" 2>/dev/null || true
        docker exec "$POSTGRES_CONTAINER" psql -U postgres -c "DROP SCHEMA IF EXISTS graphs CASCADE;" 2>/dev/null || true
        docker exec "$POSTGRES_CONTAINER" psql -U postgres -c "DROP SCHEMA IF EXISTS checkpoints CASCADE;" 2>/dev/null || true
        docker exec "$POSTGRES_CONTAINER" psql -U postgres -c "DROP SCHEMA IF EXISTS metadata CASCADE;" 2>/dev/null || true
        docker exec "$POSTGRES_CONTAINER" psql -U postgres -c "DROP SCHEMA IF EXISTS bm25 CASCADE;" 2>/dev/null || true
        docker exec "$POSTGRES_CONTAINER" psql -U postgres -c "DROP SCHEMA IF EXISTS graph CASCADE;" 2>/dev/null || true
        docker exec "$POSTGRES_CONTAINER" psql -U postgres -c "DROP SCHEMA IF EXISTS auth CASCADE;" 2>/dev/null || true

        echo "  Importing database dump..."
        docker exec -i "$POSTGRES_CONTAINER" psql -U postgres -d sujbot < "$BACKUP_DIR/postgres.sql"
        echo -e "${GREEN}  PostgreSQL restored successfully${NC}"
    else
        echo -e "${RED}  ERROR: PostgreSQL container not running!${NC}"
        echo "  Start containers first: docker compose up -d postgres"
        exit 1
    fi
else
    echo -e "${YELLOW}  No postgres.sql found, skipping...${NC}"
fi

# 3. Restore Neo4j
echo -e "${YELLOW}[3/6] Restoring Neo4j database...${NC}"
if container_running "$NEO4J_CONTAINER"; then
    if [ -f "$BACKUP_DIR/neo4j_backup.cypher" ]; then
        # Import Cypher backup
        docker cp "$BACKUP_DIR/neo4j_backup.cypher" "$NEO4J_CONTAINER:/var/lib/neo4j/import/"
        docker exec "$NEO4J_CONTAINER" cypher-shell -u neo4j -p "${NEO4J_PASSWORD:-neo4j}" \
            "CALL apoc.cypher.runFile('/var/lib/neo4j/import/neo4j_backup.cypher')" 2>/dev/null || {
            echo -e "${YELLOW}  Cypher import failed, may need manual import${NC}"
        }
        echo -e "${GREEN}  Neo4j restored from Cypher export${NC}"
    elif [ -f "$BACKUP_DIR/neo4j_data.tar.gz" ]; then
        echo -e "${YELLOW}  Neo4j data archive found - manual restore required${NC}"
        echo "  1. Stop Neo4j: docker compose stop neo4j"
        echo "  2. Extract: docker run --rm -v sujbot2_neo4j_data:/data -v $BACKUP_DIR:/backup alpine tar -xzf /backup/neo4j_data.tar.gz -C /data"
        echo "  3. Start Neo4j: docker compose up -d neo4j"
    else
        echo -e "${YELLOW}  No Neo4j backup found, skipping...${NC}"
    fi
else
    echo -e "${RED}  WARNING: Neo4j container not running, skipping...${NC}"
fi

# 4. Restore Redis
echo -e "${YELLOW}[4/6] Restoring Redis snapshot...${NC}"
if [ -f "$BACKUP_DIR/dump.rdb" ]; then
    if container_running "$REDIS_CONTAINER"; then
        docker cp "$BACKUP_DIR/dump.rdb" "$REDIS_CONTAINER:/data/"
        docker restart "$REDIS_CONTAINER"
        echo -e "${GREEN}  Redis snapshot restored${NC}"
    else
        echo -e "${RED}  WARNING: Redis container not running, skipping...${NC}"
    fi
else
    echo -e "${YELLOW}  No Redis snapshot found, skipping...${NC}"
fi

# 5. Restore configuration files
echo -e "${YELLOW}[5/6] Restoring configuration files...${NC}"
cd "$PROJECT_DIR"

if [ -f "$BACKUP_DIR/.env" ]; then
    # Backup existing .env if present
    if [ -f ".env" ]; then
        cp ".env" ".env.backup_$(date +%Y%m%d_%H%M%S)"
        echo -e "${YELLOW}  Existing .env backed up${NC}"
    fi
    cp "$BACKUP_DIR/.env" "./"
    echo -e "${GREEN}  .env restored${NC}"
fi

if [ -f "$BACKUP_DIR/config.json" ]; then
    cp "$BACKUP_DIR/config.json" "./"
    echo -e "${GREEN}  config.json restored${NC}"
fi

if [ -f "$BACKUP_DIR/docker-compose.yml" ]; then
    cp "$BACKUP_DIR/docker-compose.yml" "./"
    echo -e "${GREEN}  docker-compose.yml restored${NC}"
fi

if [ -f "$BACKUP_DIR/docker-compose.override.yml" ]; then
    cp "$BACKUP_DIR/docker-compose.override.yml" "./"
    echo -e "${GREEN}  docker-compose.override.yml restored${NC}"
fi

# 6. Restore data directories
echo -e "${YELLOW}[6/6] Restoring data directories...${NC}"

if [ -d "$BACKUP_DIR/data" ]; then
    mkdir -p data
    cp -r "$BACKUP_DIR/data/"* data/ 2>/dev/null || true
    echo -e "${GREEN}  data/ restored${NC}"
fi

if [ -d "$BACKUP_DIR/prompts" ]; then
    mkdir -p prompts
    cp -r "$BACKUP_DIR/prompts/"* prompts/ 2>/dev/null || true
    echo -e "${GREEN}  prompts/ restored${NC}"
fi

if [ -d "$BACKUP_DIR/output" ]; then
    mkdir -p output
    cp -r "$BACKUP_DIR/output/"* output/ 2>/dev/null || true
    echo -e "${GREEN}  output/ restored${NC}"
fi

if [ -d "$BACKUP_DIR/logs" ]; then
    mkdir -p logs
    cp -r "$BACKUP_DIR/logs/"* logs/ 2>/dev/null || true
    echo -e "${GREEN}  logs/ restored${NC}"
fi

if [ -d "$BACKUP_DIR/dataset" ]; then
    mkdir -p dataset
    cp -r "$BACKUP_DIR/dataset/"* dataset/ 2>/dev/null || true
    echo -e "${GREEN}  dataset/ restored${NC}"
fi

if [ -d "$BACKUP_DIR/postgres_init" ]; then
    mkdir -p docker/postgres/init
    cp -r "$BACKUP_DIR/postgres_init/"* docker/postgres/init/ 2>/dev/null || true
    echo -e "${GREEN}  postgres_init/ restored${NC}"
fi

# Cleanup
echo ""
echo -e "${YELLOW}Cleaning up...${NC}"
rm -rf "$BACKUP_DIR"

# Summary
echo ""
echo -e "${GREEN}=== Import Complete ===${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Restart all services: docker compose restart"
echo "  2. Verify PostgreSQL: docker exec $POSTGRES_CONTAINER psql -U postgres -d sujbot -c 'SELECT count(*) FROM vectors.layer3;'"
echo "  3. Test the application: curl http://localhost:8000/health"
echo ""
echo -e "${GREEN}Done!${NC}"
