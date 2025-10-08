#!/bin/bash
# Automated backup script for SUJBOT2
# Backs up Redis data, uploaded files, and FAISS indices

set -e

BACKUP_DIR="./backups"
DATE=$(date +%Y%m%d-%H%M%S)
RETENTION_DAYS=30

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting backup process...${NC}"
echo "Backup directory: $BACKUP_DIR"
echo "Timestamp: $DATE"

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

# Backup Redis data
echo -e "${YELLOW}Backing up Redis data...${NC}"
docker run --rm \
  -v sujbot2_redis_data_prod:/data \
  -v "$(pwd)/$BACKUP_DIR":/backup \
  alpine tar czf /backup/redis-$DATE.tar.gz -C /data .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Redis backup completed${NC}"
    REDIS_SIZE=$(du -h "$BACKUP_DIR/redis-$DATE.tar.gz" | cut -f1)
    echo "  Size: $REDIS_SIZE"
else
    echo -e "${RED}✗ Redis backup failed${NC}"
fi

# Backup uploaded files
echo -e "${YELLOW}Backing up uploaded files...${NC}"
docker run --rm \
  -v sujbot2_uploads_prod:/uploads \
  -v "$(pwd)/$BACKUP_DIR":/backup \
  alpine tar czf /backup/uploads-$DATE.tar.gz -C /uploads .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Uploads backup completed${NC}"
    UPLOADS_SIZE=$(du -h "$BACKUP_DIR/uploads-$DATE.tar.gz" | cut -f1)
    echo "  Size: $UPLOADS_SIZE"
else
    echo -e "${RED}✗ Uploads backup failed${NC}"
fi

# Backup FAISS indices
echo -e "${YELLOW}Backing up FAISS indices...${NC}"
docker run --rm \
  -v sujbot2_indexes_prod:/indexes \
  -v "$(pwd)/$BACKUP_DIR":/backup \
  alpine tar czf /backup/indexes-$DATE.tar.gz -C /indexes .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Indices backup completed${NC}"
    INDICES_SIZE=$(du -h "$BACKUP_DIR/indexes-$DATE.tar.gz" | cut -f1)
    echo "  Size: $INDICES_SIZE"
else
    echo -e "${RED}✗ Indices backup failed${NC}"
fi

# Delete backups older than retention period
echo -e "${YELLOW}Cleaning up old backups (older than $RETENTION_DAYS days)...${NC}"
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +$RETENTION_DAYS -delete

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Cleanup completed${NC}"
else
    echo -e "${RED}✗ Cleanup failed${NC}"
fi

# Calculate total backup size
TOTAL_SIZE=$(du -sh "$BACKUP_DIR" | cut -f1)
BACKUP_COUNT=$(ls -1 "$BACKUP_DIR"/*.tar.gz 2>/dev/null | wc -l)

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Backup completed successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo "Timestamp: $DATE"
echo "Total backups: $BACKUP_COUNT"
echo "Total size: $TOTAL_SIZE"
echo "Location: $BACKUP_DIR"
echo ""

# Optional: Upload to cloud storage
# Uncomment and configure for AWS S3, Google Cloud Storage, etc.
# aws s3 cp "$BACKUP_DIR/redis-$DATE.tar.gz" s3://your-bucket/sujbot2-backups/
# aws s3 cp "$BACKUP_DIR/uploads-$DATE.tar.gz" s3://your-bucket/sujbot2-backups/
# aws s3 cp "$BACKUP_DIR/indexes-$DATE.tar.gz" s3://your-bucket/sujbot2-backups/

exit 0
