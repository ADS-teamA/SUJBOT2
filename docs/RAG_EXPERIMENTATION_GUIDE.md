# Průvodce experimentováním s RAG konfiguracemi

Tento dokument vysvětluje různé přístupy k verzování a testování RAG konfigurací bez rizika rozbití produkce.

---

## Obsah

1. [Přehled přístupů](#1-přehled-přístupů)
2. [Git branching strategie](#2-git-branching-strategie)
3. [Konfigurační profily](#3-konfigurační-profily)
4. [Docker izolace](#4-docker-izolace)
5. [LangSmith tracking experimentů](#5-langsmith-tracking-experimentů)
6. [Doporučený workflow](#6-doporučený-workflow)

---

## 1. Přehled přístupů

### Porovnání metod

| Metoda | Složitost | Izolace | Rychlost přepínání | Vhodné pro |
|--------|-----------|---------|-------------------|------------|
| Git branches | Nízká | Úplná | Pomalá (checkout) | Změny kódu |
| Config profily | Střední | Runtime | Rychlá (env var) | Změny parametrů |
| Docker stack | Střední | Úplná | Střední (restart) | A/B testing |
| Feature flags | Vysoká | Per-request | Okamžitá | Produkční A/B |

### Kdy co použít

**Git branches** - když měníte kód (nové retrieval algoritmy, změny v pipeline)

**Config profily** - když měníte pouze parametry (váhy, k hodnoty, boost)

**Docker izolace** - když chcete testovat změny bez rizika pro produkci

---

## 2. Git branching strategie

### Základní workflow

```bash
# 1. Vytvořte experimentální branch
git checkout -b experiment/hyde-weight-0.5

# 2. Upravte config.json
# Změňte retrieval.hyde_weight: 0.5

# 3. Testujte lokálně
docker compose up -d
uv run python scripts/langsmith_eval.py --limit 5

# 4. Pokud funguje → merge do main
git checkout main
git merge experiment/hyde-weight-0.5

# 5. Pokud nefunguje → smažte branch
git branch -D experiment/hyde-weight-0.5
```

### Pojmenování branches

```
experiment/hyde-weight-0.5      # Konkrétní parametr
experiment/no-graph-boost       # Vypnutá funkce
experiment/new-reranker         # Nový algoritmus
experiment/2024-01-15-baseline  # Datovaný snapshot
```

### Nevýhody

- Pomalé přepínání (git checkout, rebuild)
- Konflikty při merge
- Musíte trackovat který branch je "produkční"

---

## 3. Konfigurační profily

### Koncept

Místo jedné konfigurace máte pojmenované profily v `config.json`:

```json
{
  "rag_profiles": {
    "profiles": {
      "baseline": {
        "display_name": "Baseline (Production)",
        "retrieval": {
          "original_weight": 0.5,
          "hyde_weight": 0.25,
          "expansion_weight": 0.25,
          "default_k": 16
        }
      },
      "high-hyde": {
        "display_name": "High HyDE Weight",
        "retrieval": {
          "original_weight": 0.3,
          "hyde_weight": 0.5,
          "expansion_weight": 0.2,
          "default_k": 16
        }
      }
    },
    "default_profile": "baseline"
  }
}
```

### Přepínání pomocí env variable

```bash
# Spustit s baseline profilem (default)
docker compose up -d

# Spustit s experimentálním profilem
RAG_PROFILE=high-hyde docker compose up -d
```

### Implementace overlay patternu

Klíčová myšlenka: profil **překryje** base konfiguraci za běhu, ale nemodifikuje soubor.

```python
# V src/config.py
def get_config(reload: bool = False) -> RootConfig:
    global _CONFIG
    if _CONFIG is None or reload:
        _CONFIG = load_json_config()

        # Aplikuj profil overlay
        profile = os.getenv("RAG_PROFILE")
        if profile:
            _CONFIG = _apply_profile_overlay(_CONFIG, profile)

    return _CONFIG

def _apply_profile_overlay(config, profile_name):
    """Překryje base config hodnotami z profilu."""
    profile = config.rag_profiles.profiles.get(profile_name)
    if not profile:
        return config

    # Pydantic model_copy vytvoří novou instanci s upravenými hodnotami
    return config.model_copy(update={
        "retrieval": config.retrieval.model_copy(update={
            "original_weight": profile.retrieval.original_weight,
            "hyde_weight": profile.retrieval.hyde_weight,
            # ... další parametry
        })
    })
```

### Výhody

- Všechny konfigurace na jednom místě
- Snadné porovnání (diff profilů)
- Verzované v gitu
- Rychlé přepínání

### Nevýhody

- Vyžaduje změny v config loading kódu
- `config.json` může narůst

---

## 4. Docker izolace

### Základní myšlenka

Spustit **dvě Docker instance** vedle sebe:
- Produkce na portu 80/8000
- Experiment na portu 8181/8081

### Docker Compose override soubory

Docker Compose umožňuje kombinovat více souborů pomocí `-f`:

```bash
# Pouze produkce
docker compose up -d

# Produkce + experiment override
docker compose -f docker-compose.yml -f docker-compose.experiment.yml up -d
```

### Příklad docker-compose.experiment.yml

```yaml
# docker-compose.experiment.yml
# Override pro experiment stack

services:
  backend:
    container_name: sujbot_backend_experiment  # Jiný název
    environment:
      RAG_PROFILE: ${RAG_PROFILE:-baseline}    # Profil z env
      LANGSMITH_PROJECT_NAME: sujbot2-experiment
    ports:
      - "8081:8000"                             # Jiný port

  nginx:
    container_name: sujbot_nginx_experiment
    ports:
      - "8181:80"                               # Jiný port
```

### Sdílená vs. izolovaná databáze

**Sdílená databáze** (doporučeno pro RAG experimenty):
```yaml
# Experiment používá stejnou PostgreSQL jako produkce
# → Stejné vektory, férové porovnání retrieval kvality
```

**Izolovaná databáze** (pro destruktivní experimenty):
```yaml
services:
  postgres_experiment:
    image: pgvector/pgvector:pg16
    container_name: postgres_experiment
    volumes:
      - postgres_experiment_data:/var/lib/postgresql/data
    ports:
      - "5433:5432"  # Jiný port

volumes:
  postgres_experiment_data:
```

### Spuštění experiment stacku

```bash
# 1. Spustit experiment s konkrétním profilem
RAG_PROFILE=high-hyde docker compose \
  -f docker-compose.yml \
  -f docker-compose.experiment.yml \
  up -d backend nginx

# 2. Ověřit běžící kontejnery
docker ps --filter "name=experiment"

# 3. Testovat experiment API
curl http://localhost:8081/health

# 4. Spustit evaluaci proti experimentu
RAG_PROFILE=high-hyde uv run python scripts/langsmith_eval.py \
  --experiment-prefix "sujbot2-qa-40-high-hyde"

# 5. Zastavit experiment
docker compose -f docker-compose.yml -f docker-compose.experiment.yml down
```

---

## 5. LangSmith tracking experimentů

### Experiment naming convention

Používejte konzistentní pojmenování pro snadné porovnání:

```
sujbot2-qa-40-baseline          # Baseline
sujbot2-qa-40-high-hyde         # Experiment 1
sujbot2-qa-40-no-graph-boost    # Experiment 2
```

### Automatické pojmenování v eval scriptu

```python
# V scripts/langsmith_eval.py
import os

def main():
    rag_profile = os.getenv("RAG_PROFILE", "baseline")

    # Automaticky přidej profil do experiment prefix
    experiment_prefix = f"sujbot2-qa-40-{rag_profile}"

    print(f"Running evaluation with RAG profile: {rag_profile}")
    print(f"LangSmith experiment: {experiment_prefix}")
```

### Porovnání experimentů v LangSmith

1. Otevřete LangSmith UI
2. Jděte do **Datasets & Experiments**
3. Vyberte dataset `sujbot2-eval-qa-40`
4. Klikněte na **Compare** a vyberte experimenty k porovnání
5. Porovnejte metriky:
   - `semantic_correctness`
   - `factual_accuracy`
   - `completeness`
   - `latency`

### Query pomocí MCP tools

```bash
# Seznam projektů
# Použijte mcp__langsmith__list_projects

# Fetch runs z konkrétního experimentu
# Použijte mcp__langsmith__fetch_runs s project_name="sujbot2-qa-40-high-hyde"
```

---

## 6. Doporučený workflow

### Kompletní workflow pro experiment

```bash
# === FÁZE 1: Příprava ===

# 1.1 Ujistěte se, že máte aktuální baseline
git pull origin main

# 1.2 Spusťte baseline evaluaci (pokud nemáte)
uv run python scripts/langsmith_eval.py \
  --experiment-prefix "sujbot2-qa-40-baseline"


# === FÁZE 2: Experiment ===

# 2.1 Přidejte nový profil do config.json
# Editujte rag_profiles.profiles v config.json

# 2.2 Spusťte experiment stack
RAG_PROFILE=high-hyde docker compose \
  -f docker-compose.yml \
  -f docker-compose.experiment.yml \
  up -d

# 2.3 Počkejte na startup
sleep 30
curl http://localhost:8081/health

# 2.4 Spusťte evaluaci
RAG_PROFILE=high-hyde uv run python scripts/langsmith_eval.py \
  --experiment-prefix "sujbot2-qa-40-high-hyde"


# === FÁZE 3: Analýza ===

# 3.1 Porovnejte v LangSmith UI
# https://smith.langchain.com → Datasets & Experiments → Compare

# 3.2 Nebo použijte MCP tools pro programatický přístup


# === FÁZE 4: Rozhodnutí ===

# 4a) Experiment je lepší → Promítněte do produkce
#     Změňte default hodnoty v config.json.retrieval
#     git commit -m "feat: update RAG weights based on experiment high-hyde"

# 4b) Experiment je horší → Zahoďte
#     Smažte profil z config.json
#     git commit -m "chore: remove failed experiment high-hyde"


# === FÁZE 5: Cleanup ===

# 5.1 Zastavte experiment stack
docker compose -f docker-compose.yml -f docker-compose.experiment.yml down

# 5.2 (Volitelně) Smažte experiment profil z config.json
```

### Helper script (volitelný)

Můžete si vytvořit script pro zjednodušení:

```bash
#!/bin/bash
# scripts/experiment.sh

case "$1" in
  start)
    RAG_PROFILE="$2" docker compose \
      -f docker-compose.yml \
      -f docker-compose.experiment.yml \
      up -d
    echo "Experiment running on http://localhost:8181"
    ;;
  stop)
    docker compose -f docker-compose.yml -f docker-compose.experiment.yml down
    ;;
  eval)
    RAG_PROFILE="$2" uv run python scripts/langsmith_eval.py \
      --experiment-prefix "sujbot2-qa-40-$2"
    ;;
  status)
    docker ps --filter "name=experiment"
    ;;
  *)
    echo "Usage: $0 {start|stop|eval|status} [profile]"
    ;;
esac
```

Použití:
```bash
chmod +x scripts/experiment.sh
./scripts/experiment.sh start high-hyde
./scripts/experiment.sh eval high-hyde
./scripts/experiment.sh stop
```

---

## Klíčové parametry pro experimenty

### Retrieval parametry (config.json.retrieval)

| Parametr | Popis | Rozsah | Default |
|----------|-------|--------|---------|
| `original_weight` | Váha původního query | 0.0-1.0 | 0.5 |
| `hyde_weight` | Váha HyDE dokumentu | 0.0-1.0 | 0.25 |
| `expansion_weight` | Váha query expansions | 0.0-1.0 | 0.25 |
| `default_k` | Počet výsledků | 1-100 | 16 |
| `candidates_multiplier` | Násobič pro fusion | 1-10 | 3 |

**Důležité:** `original_weight + hyde_weight + expansion_weight = 1.0`

### Agent tools parametry (config.json.agent_tools)

| Parametr | Popis | Default |
|----------|-------|---------|
| `enable_graph_boost` | Zapnout graph boosting | true |
| `graph_boost_weight` | Váha grafu | 0.3 |
| `enable_reranking` | Zapnout reranking | false |
| `default_k` | K pro agent tools | 6 |

---

## Tipy a best practices

### 1. Vždy mějte baseline

Před jakýmkoliv experimentem spusťte baseline evaluaci, abyste měli referenční bod.

### 2. Jedna změna naráz

Měňte vždy jen jeden parametr, abyste věděli, co způsobilo změnu.

### 3. Statistická významnost

40 QA párů může být málo pro statisticky významné závěry. Zvažte:
- Větší dataset
- Opakování experimentu
- Confidence intervaly

### 4. Dokumentujte experimenty

```markdown
## Experiment: high-hyde (2024-01-15)

**Hypotéza:** Vyšší HyDE váha zlepší sémantické matchování

**Změny:**
- hyde_weight: 0.25 → 0.5
- original_weight: 0.5 → 0.3

**Výsledky:**
- semantic_correctness: 0.82 → 0.85 (+3%)
- factual_accuracy: 0.78 → 0.76 (-2%)
- latency: +15ms

**Závěr:** Mírné zlepšení sémantiky, ale horší faktická přesnost. Nezavádět.
```

### 5. Git tagy pro milníky

```bash
git tag -a v1.0.0-baseline "Baseline configuration before experiments"
git tag -a v1.1.0-high-hyde "After high-hyde experiment"
```

---

## Závěr

Doporučený minimální setup:

1. **Git branches** pro změny kódu
2. **Config profily** pro změny parametrů (volitelné, ale užitečné)
3. **Docker override** pro izolované testování
4. **LangSmith** pro tracking a porovnání

Začněte jednoduše - používejte git branches a manuální změny v `config.json`. Až budete mít více experimentů, implementujte config profily pro snazší správu.
