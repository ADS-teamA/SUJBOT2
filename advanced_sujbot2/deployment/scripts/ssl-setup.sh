#!/bin/bash
# SSL/TLS Setup Script for SUJBOT2
# Obtains Let's Encrypt certificates using Certbot

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
DOMAIN="${1:-sujbot2.example.com}"
EMAIL="${2:-admin@example.com}"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}SSL/TLS Setup for SUJBOT2${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Domain: $DOMAIN"
echo "Email: $EMAIL"
echo ""

# Validate inputs
if [ "$DOMAIN" = "sujbot2.example.com" ]; then
    echo -e "${RED}Error: Please provide a valid domain name${NC}"
    echo "Usage: ./ssl-setup.sh <domain> <email>"
    echo "Example: ./ssl-setup.sh sujbot2.mycompany.com admin@mycompany.com"
    exit 1
fi

if [ "$EMAIL" = "admin@example.com" ]; then
    echo -e "${RED}Error: Please provide a valid email address${NC}"
    echo "Usage: ./ssl-setup.sh <domain> <email>"
    exit 1
fi

# Create necessary directories
echo -e "${YELLOW}Creating SSL directories...${NC}"
mkdir -p deployment/nginx/ssl
mkdir -p deployment/nginx/certbot-webroot

# Check if certificates already exist
if [ -d "deployment/nginx/ssl/live/$DOMAIN" ]; then
    echo -e "${YELLOW}Warning: Certificates already exist for $DOMAIN${NC}"
    read -p "Do you want to obtain new certificates? (yes/no): " CONFIRM
    if [ "$CONFIRM" != "yes" ]; then
        echo "SSL setup cancelled."
        exit 0
    fi
fi

# Update Nginx configuration with correct domain
echo -e "${YELLOW}Updating Nginx configuration...${NC}"
sed -i.bak "s/sujbot2.example.com/$DOMAIN/g" deployment/nginx/conf.d/sujbot2.conf
echo -e "${GREEN}✓ Nginx configuration updated${NC}"

# Start Nginx in HTTP-only mode for ACME challenge
echo -e "${YELLOW}Starting Nginx for ACME challenge...${NC}"
docker-compose -f docker-compose.prod.yml up -d nginx

# Wait for Nginx to start
sleep 5

# Obtain SSL certificate
echo -e "${YELLOW}Requesting SSL certificate from Let's Encrypt...${NC}"
docker-compose -f docker-compose.prod.yml run --rm certbot certonly \
  --webroot \
  --webroot-path=/var/www/certbot \
  --email "$EMAIL" \
  --agree-tos \
  --no-eff-email \
  --force-renewal \
  -d "$DOMAIN"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ SSL certificate obtained successfully${NC}"
else
    echo -e "${RED}✗ Failed to obtain SSL certificate${NC}"
    echo ""
    echo "Troubleshooting:"
    echo "1. Ensure $DOMAIN points to this server's IP address"
    echo "2. Ensure port 80 is accessible from the internet"
    echo "3. Check Nginx logs: docker-compose -f docker-compose.prod.yml logs nginx"
    exit 1
fi

# Restart Nginx with HTTPS enabled
echo -e "${YELLOW}Restarting Nginx with HTTPS...${NC}"
docker-compose -f docker-compose.prod.yml restart nginx

# Verify HTTPS is working
sleep 3
echo -e "${YELLOW}Verifying HTTPS configuration...${NC}"

HTTPS_STATUS=$(curl -s -o /dev/null -w "%{http_code}" https://$DOMAIN/api/v1/health 2>/dev/null || echo "000")

if [ "$HTTPS_STATUS" = "200" ]; then
    echo -e "${GREEN}✓ HTTPS is working correctly${NC}"
else
    echo -e "${YELLOW}⚠ HTTPS verification returned status: $HTTPS_STATUS${NC}"
    echo "  This might be normal if the backend is not yet running."
fi

# Display certificate information
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}SSL Setup Completed!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Certificate location: deployment/nginx/ssl/live/$DOMAIN/"
echo "Your site should now be accessible at: https://$DOMAIN"
echo ""
echo "Certificate auto-renewal is enabled via Certbot container."
echo "Certificates will be automatically renewed every 12 hours."
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Update .env.prod with your production domain"
echo "2. Start all services: docker-compose -f docker-compose.prod.yml up -d"
echo "3. Verify deployment: curl https://$DOMAIN/api/v1/health"
echo ""

exit 0
