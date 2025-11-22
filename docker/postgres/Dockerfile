# PostgreSQL with pgvector extension
# Use official pgvector image as base (already has pgvector v0.7.4 installed)
# Apache AGE (graph database) can be added later if needed

FROM pgvector/pgvector:pg16

# Metadata
LABEL maintainer="SUJBOT2"
LABEL description="PostgreSQL 16 with pgvector (vector similarity search)"

# Note: Apache AGE installation skipped for initial deployment due to build complexity
# System will use simple graph storage (JSON files) as configured in config.json
# To add Apache AGE later, see: https://age.apache.org/getstarted/quickstart/

# Copy PostgreSQL configuration for performance tuning
COPY postgresql.conf /etc/postgresql/postgresql.conf

# Copy initialization scripts (will be executed on first start)
# Note: Scripts in /docker-entrypoint-initdb.d are executed in alphabetical order
# 01-init.sql - Creates schemas, extensions, tables, indexes
COPY init/*.sql /docker-entrypoint-initdb.d/

# Set custom config file location
# Note: This will be overridden by command in docker-compose.yml for more control
ENV POSTGRES_INITDB_ARGS="--encoding=UTF8 --locale=en_US.UTF-8"

# Expose PostgreSQL port
EXPOSE 5432

# Health check
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=5 \
    CMD pg_isready -U ${POSTGRES_USER:-postgres} -d ${POSTGRES_DB:-sujbot} || exit 1

# Use default postgres entrypoint (handles initialization)
# Command can be overridden in docker-compose.yml
CMD ["postgres", "-c", "config_file=/etc/postgresql/postgresql.conf"]
