#!/bin/bash
# Restore script for SUJBOT2
# Restores Redis data, uploaded files, and FAISS indices from backup

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if backup directory is provided
if [ -z "$1" ]; then
    echo -e "${RED}Error: Backup directory not specified${NC}"
    echo "Usage: ./restore.sh <backup-directory> [component]"
    echo ""
    echo "Components: redis, uploads, indexes, all (default)"
    echo ""
    echo "Examples:"
    echo "  ./restore.sh ./backups/2025-10-08-120000"
    echo "  ./restore.sh ./backups/2025-10-08-120000 redis"
    echo "  ./restore.sh ./backups/2025-10-08-120000 all"
    exit 1
fi

BACKUP_DIR="$1"
COMPONENT="${2:-all}"

# Validate backup directory
if [ ! -d "$BACKUP_DIR" ]; then
    echo -e "${RED}Error: Backup directory '$BACKUP_DIR' does not exist${NC}"
    exit 1
fi

# Confirmation prompt
echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}WARNING: This will overwrite current data!${NC}"
echo -e "${YELLOW}========================================${NC}"
echo "Backup directory: $BACKUP_DIR"
echo "Component: $COMPONENT"
echo ""
read -p "Are you sure you want to continue? (yes/no): " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
    echo "Restore cancelled."
    exit 0
fi

echo ""
echo -e "${GREEN}Starting restore process...${NC}"

# Stop services before restore
echo -e "${YELLOW}Stopping services...${NC}"
docker-compose -f docker-compose.prod.yml down

# Restore Redis
if [ "$COMPONENT" = "redis" ] || [ "$COMPONENT" = "all" ]; then
    echo -e "${YELLOW}Restoring Redis data...${NC}"

    REDIS_BACKUP=$(ls -t "$BACKUP_DIR"/redis-*.tar.gz 2>/dev/null | head -n 1)

    if [ -z "$REDIS_BACKUP" ]; then
        echo -e "${RED}✗ No Redis backup found in $BACKUP_DIR${NC}"
    else
        # Clear existing data
        docker volume rm sujbot2_redis_data_prod 2>/dev/null || true
        docker volume create sujbot2_redis_data_prod

        # Restore backup
        docker run --rm \
          -v sujbot2_redis_data_prod:/data \
          -v "$(pwd)/$BACKUP_DIR":/backup \
          alpine sh -c "cd /data && tar xzf /backup/$(basename $REDIS_BACKUP)"

        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ Redis restore completed${NC}"
        else
            echo -e "${RED}✗ Redis restore failed${NC}"
        fi
    fi
fi

# Restore uploads
if [ "$COMPONENT" = "uploads" ] || [ "$COMPONENT" = "all" ]; then
    echo -e "${YELLOW}Restoring uploaded files...${NC}"

    UPLOADS_BACKUP=$(ls -t "$BACKUP_DIR"/uploads-*.tar.gz 2>/dev/null | head -n 1)

    if [ -z "$UPLOADS_BACKUP" ]; then
        echo -e "${RED}✗ No uploads backup found in $BACKUP_DIR${NC}"
    else
        # Clear existing data
        docker volume rm sujbot2_uploads_prod 2>/dev/null || true
        docker volume create sujbot2_uploads_prod

        # Restore backup
        docker run --rm \
          -v sujbot2_uploads_prod:/uploads \
          -v "$(pwd)/$BACKUP_DIR":/backup \
          alpine sh -c "cd /uploads && tar xzf /backup/$(basename $UPLOADS_BACKUP)"

        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ Uploads restore completed${NC}"
        else
            echo -e "${RED}✗ Uploads restore failed${NC}"
        fi
    fi
fi

# Restore indices
if [ "$COMPONENT" = "indexes" ] || [ "$COMPONENT" = "all" ]; then
    echo -e "${YELLOW}Restoring FAISS indices...${NC}"

    INDICES_BACKUP=$(ls -t "$BACKUP_DIR"/indexes-*.tar.gz 2>/dev/null | head -n 1)

    if [ -z "$INDICES_BACKUP" ]; then
        echo -e "${RED}✗ No indices backup found in $BACKUP_DIR${NC}"
    else
        # Clear existing data
        docker volume rm sujbot2_indexes_prod 2>/dev/null || true
        docker volume create sujbot2_indexes_prod

        # Restore backup
        docker run --rm \
          -v sujbot2_indexes_prod:/indexes \
          -v "$(pwd)/$BACKUP_DIR":/backup \
          alpine sh -c "cd /indexes && tar xzf /backup/$(basename $INDICES_BACKUP)"

        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ Indices restore completed${NC}"
        else
            echo -e "${RED}✗ Indices restore failed${NC}"
        fi
    fi
fi

# Restart services
echo -e "${YELLOW}Starting services...${NC}"
docker-compose -f docker-compose.prod.yml up -d

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Restore completed!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

exit 0
