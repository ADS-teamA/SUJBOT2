# PostgreSQL Migration - Test Suite

Kompletní testovací suite pro ověření migrace z FAISS na PostgreSQL + pgvector.

## Přehled testů

### 1. Quick Test (`quick_test.sh`)
Rychlý test základní funkčnosti - ověří připojení, extensions a schéma.

```bash
chmod +x tests/quick_test.sh
./tests/quick_test.sh
```

**Co testuje:**
- PostgreSQL je spuštěný
- Database připojení funguje
- pgvector extension je nainstalovaný
- Tabulky existují
- Základní Python test projde

**Trvání:** ~10 sekund

---

### 2. Full Test Suite (`test_postgresql_migration.py`)
Komplexní testy všech funkcí vector store.

```bash
# S pytest
pytest tests/test_postgresql_migration.py -v

# Standalone
python tests/test_postgresql_migration.py
```

**Co testuje:**

#### Connection Tests
- ✓ Connection pool creation
- ✓ Extension verification (pgvector, pg_trgm, btree_gin)

#### Document Indexing
- ✓ Add document with chunks
- ✓ Retrieve all chunks
- ✓ Document metadata

#### Vector Search
- ✓ Semantic similarity search
- ✓ Search by legal reference (§89, Článek 5)
- ✓ Metadata filtering (content_type)

#### Performance
- ✓ Batch insert (100 chunks)
- ✓ Search latency (<100ms target)

#### Data Integrity
- ✓ Chunk count consistency
- ✓ Embedding dimensions (1024)

#### Cleanup
- ✓ Document deletion (CASCADE)

**Trvání:** ~2 minuty

---

### 3. Performance Benchmark (`benchmark_postgresql.py`)
Detailní měření výkonu pro různé operace.

```bash
# Základní benchmark (1000 chunks, 50 queries)
python tests/benchmark_postgresql.py

# Vlastní parametry
python tests/benchmark_postgresql.py --chunks 10000 --queries 100
```

**Co měří:**

#### 1. Batch Insert Performance
- Throughput (chunks/sec)
- Latence per chunk (ms)
- Celkový čas

#### 2. Vector Search Latency
- Průměrná latence
- P50, P95, P99 percentily
- Min/max latence

#### 3. Concurrent Queries
- Queries per second (QPS)
- Latence při concurrent queries
- Connection pool efficiency

#### 4. Reference Lookup
- Direct lookup by §/Článek
- Průměrná latence

**Trvání:** ~5-10 minut (dle počtu chunks)

**Výstup:**
```
📊 Performance Evaluation:
  ✅ Vector search: EXCELLENT (<50ms)
  ✅ Insert throughput: EXCELLENT (>50 chunks/sec)
```

---

## Quick Start

### 1. Základní test flow

```bash
# 1. Spusť Docker Compose
docker-compose -f docker-compose.dev.yml up -d

# 2. Quick test
./tests/quick_test.sh

# 3. Full test suite
pytest tests/test_postgresql_migration.py -v

# 4. Benchmark
python tests/benchmark_postgresql.py
```

---

## Test Fixtures

### Environment Variables
Testy používají tyto environment variables (s výchozími hodnotami):

```bash
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=sujbot2
POSTGRES_USER=sujbot_app
POSTGRES_PASSWORD=sujbot2_dev_password
```

### Sample Data
Testy používají vzorové legal chunks:

```python
LegalChunk(
    content="§89 Odpovědnost za vady...",
    legal_reference="§89",
    metadata={"content_type": "obligation", ...}
)
```

---

## Očekávané výsledky

### Performance Targets

| Metrika | Target | Good | Needs Tuning |
|---------|--------|------|--------------|
| Vector search (P95) | <50ms | <100ms | >100ms |
| Insert throughput | >50 chunks/s | >20 chunks/s | <20 chunks/s |
| Reference lookup | <5ms | <10ms | >10ms |
| Concurrent QPS | >50 | >20 | <20 |

### Data Scale Expectations

| Chunks | Storage | Index Size | Search Latency |
|--------|---------|------------|----------------|
| 1K | ~10 MB | ~1 MB | <10ms |
| 10K | ~100 MB | ~10 MB | <20ms |
| 100K | ~1 GB | ~100 MB | <30ms |
| 500K | ~5 GB | ~500 MB | <50ms |

---

## Troubleshooting

### Test selhává na připojení

```bash
# Zkontroluj, že PostgreSQL běží
docker ps | grep postgres

# Zkontroluj logy
docker logs sujbot2_postgres_dev

# Restartuj služby
docker-compose -f docker-compose.dev.yml restart postgres
```

### pgvector extension chybí

```bash
# Zkontroluj extension
docker exec sujbot2_postgres_dev psql -U sujbot_app -d sujbot2 \
  -c "SELECT * FROM pg_available_extensions WHERE name = 'vector'"

# Znovu vytvoř extension
docker exec sujbot2_postgres_dev psql -U sujbot_app -d sujbot2 \
  -c "CREATE EXTENSION IF NOT EXISTS vector"
```

### Pomalé vyhledávání

```bash
# Zkontroluj indexy
docker exec sujbot2_postgres_dev psql -U sujbot_app -d sujbot2 \
  -c "\d+ chunks"

# Zkontroluj IVFFlat probes
docker exec sujbot2_postgres_dev psql -U sujbot_app -d sujbot2 \
  -c "SHOW ivfflat.probes"

# Zvyš probes pro lepší accuracy (obětuje rychlost)
docker exec sujbot2_postgres_dev psql -U sujbot_app -d sujbot2 \
  -c "SET ivfflat.probes = 20"
```

### Test data cleanup

```bash
# Smaž všechny test documents
docker exec sujbot2_postgres_dev psql -U sujbot_app -d sujbot2 \
  -c "DELETE FROM documents WHERE document_id LIKE 'test_%' OR document_id LIKE 'bench_%'"
```

---

## CI/CD Integration

### GitHub Actions Example

```yaml
name: PostgreSQL Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: pgvector/pgvector:pg15
        env:
          POSTGRES_DB: sujbot2
          POSTGRES_USER: sujbot_app
          POSTGRES_PASSWORD: test_password
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install -r backend/requirements.txt
      - name: Run tests
        env:
          POSTGRES_HOST: localhost
          POSTGRES_PORT: 5432
          POSTGRES_DB: sujbot2
          POSTGRES_USER: sujbot_app
          POSTGRES_PASSWORD: test_password
        run: pytest tests/test_postgresql_migration.py -v
```

---

## Performance Monitoring

### Dlouhodobé sledování

Pro produkční monitoring doporučuji:

1. **pg_stat_statements** - query performance tracking
2. **Prometheus + Grafana** - metrics visualization
3. **pgBadger** - PostgreSQL log analyzer

### Key Metrics

```sql
-- Query performance
SELECT query, calls, mean_exec_time, max_exec_time
FROM pg_stat_statements
WHERE query LIKE '%chunks%'
ORDER BY mean_exec_time DESC;

-- Index usage
SELECT schemaname, tablename, indexname, idx_scan, idx_tup_read
FROM pg_stat_user_indexes
WHERE schemaname = 'public'
ORDER BY idx_scan DESC;

-- Table sizes
SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

---

## FAQ

**Q: Jak dlouho trvá test suite?**
A: Quick test ~10s, Full test ~2min, Benchmark ~5-10min

**Q: Potřebuji spustit Docker Compose?**
A: Ano, testy vyžadují běžící PostgreSQL s pgvector

**Q: Mohu spustit testy proti produkční databázi?**
A: Ne! Testy vytvářejí a mažou data. Pouze dev/test databáze.

**Q: Co dělat když testy selžou?**
A: 1) Zkontroluj Docker logs, 2) Ověř environment variables, 3) Restartuj služby

**Q: Jak často spouštět benchmarky?**
A: Po každé změně v konfiguraci nebo update PostgreSQL

---

## Kontakt

Pro problémy s testy:
1. Zkontroluj logs: `docker logs sujbot2_postgres_dev`
2. Přečti MIGRATION_STATUS.md
3. Otevři issue s výstupem testu
