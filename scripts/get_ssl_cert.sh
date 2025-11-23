#!/bin/bash
# Script to obtain Let's Encrypt SSL certificate for sujbot.fjfi.cvut.cz

set -e

DOMAIN="sujbot.fjfi.cvut.cz"
EMAIL="admin@fjfi.cvut.cz"  # Change if needed

echo "==> Obtaining SSL certificate for $DOMAIN"

# Run certbot in docker
docker compose run --rm certbot certonly \
  --webroot \
  --webroot-path=/var/www/certbot \
  --email "$EMAIL" \
  --agree-tos \
  --no-eff-email \
  -d "$DOMAIN"

echo "==> Certificate obtained successfully!"
echo "==> Now start nginx with: docker compose up -d nginx"
