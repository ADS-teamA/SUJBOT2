# Multi-stage Dockerfile for SUJBOT2 Frontend (React + Vite)

ARG NODE_VERSION=22

# ============================================================================
# Stage 1: Base - Node.js setup
# ============================================================================
FROM node:${NODE_VERSION}-alpine AS base

WORKDIR /app

# ============================================================================
# Stage 2: Dependencies - Install node_modules
# ============================================================================
FROM base AS dependencies

# Copy package files
COPY package.json package-lock.json ./

# Install all dependencies (including devDependencies for build)
# Use BuildKit cache for faster rebuilds
RUN --mount=type=cache,target=/root/.npm \
    npm ci --prefer-offline --no-audit

# ============================================================================
# Stage 3: Builder - Vite production build
# ============================================================================
FROM dependencies AS builder

# Copy source files
COPY . .

# Build static assets
# VITE_API_BASE_URL can be passed as build arg
ARG VITE_API_BASE_URL=http://localhost:8000
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}

# Limit Node.js memory during build (prevents OOM on low-memory systems)
ENV NODE_OPTIONS="--max-old-space-size=2048"

RUN npm run build && \
    # Cleanup to reduce image size
    rm -rf node_modules/.cache

# ============================================================================
# Stage 4: Development - Vite dev server with hot reload
# ============================================================================
FROM dependencies AS development

WORKDIR /app

# Copy all source files (will be overridden by volume mount)
COPY . .

EXPOSE 5173

# Development server with hot reload
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173"]

# ============================================================================
# Stage 5: Production - Nginx serving static build
# ============================================================================
FROM nginx:alpine AS production

# Copy built static files from builder
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy custom Nginx configuration (commented out - using default nginx config)
# COPY ../docker/nginx/nginx.conf /etc/nginx/nginx.conf

# Create nginx user for non-root execution
RUN chown -R nginx:nginx /usr/share/nginx/html && \
    chown -R nginx:nginx /var/cache/nginx && \
    chown -R nginx:nginx /var/log/nginx && \
    chown -R nginx:nginx /etc/nginx/conf.d && \
    touch /var/run/nginx.pid && \
    chown -R nginx:nginx /var/run/nginx.pid

USER nginx

EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget --quiet --tries=1 --spider http://localhost:80/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
