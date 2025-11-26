# Cost Tracking Guide

Kompletní návod pro sledování nákladů na API volání během indexace a RAG konverzací.

> **Tech Debt Note (2025-11-26):** Global `get_global_tracker()` singleton pattern je zachován pro zpětnou kompatibilitu. Budoucí refaktoring by měl přejít na dependency injection pro lepší testovatelnost.

## 📊 Co sledujeme

### LLM Usage (Summaries, Context, Agent)
- ✅ **Anthropic Claude** (Haiku 4.5, Sonnet 4.5, Opus 4)
- ✅ **OpenAI** (GPT-4o, GPT-5, O-series)
- ✅ Input tokens + Output tokens
- ✅ Náklady podle aktuálních cen (2025)

### Embedding Usage
- ✅ **OpenAI** (text-embedding-3-large, text-embedding-3-small)
- ✅ **Voyage AI** (voyage-3, voyage-3-large, voyage-law-2)
- ✅ **HuggingFace** (bge-m3 - FREE local)
- ✅ Total tokens embedded

---

## 💰 Aktuální ceníky (2025)

### Anthropic Claude

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|------------------------|
| Haiku 4.5 | $1.00 | $5.00 |
| Sonnet 4.5 | $3.00 | $15.00 |
| Opus 4 | $15.00 | $75.00 |

### OpenAI

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|------------------------|
| GPT-5 | $1.25 | $10.00 |
| GPT-4o | $2.50 | $10.00 |
| GPT-4o-mini | $0.15 | $0.60 |
| o1 | $15.00 | $60.00 |
| text-embedding-3-large | $0.13 | - |
| text-embedding-3-small | $0.02 | - |

### Voyage AI

| Model | Cost (per 1M tokens) | Free Tier |
|-------|---------------------|-----------|
| voyage-3 | $0.06 | First 200M tokens FREE |
| voyage-3-lite | $0.02 | First 200M tokens FREE |
| voyage-law-2 | $0.12 | First 50M tokens FREE |

### Local Models (FREE)
- **bge-m3** (HuggingFace): $0.00 ✨

---

## 🚀 Jak používat

### 1. Základní použití

```python
from src.cost_tracker import CostTracker

# Vytvořit tracker
tracker = CostTracker()

# Trackovat LLM usage (po každém API callu)
tracker.track_llm(
    provider="anthropic",
    model="claude-haiku-4-5",
    input_tokens=1000,
    output_tokens=500,
    operation="summary"  # "summary", "context", "agent", atd.
)

# Trackovat embedding usage
tracker.track_embedding(
    provider="openai",
    model="text-embedding-3-large",
    tokens=10000,
    operation="indexing"
)

# Získat celkové náklady
total = tracker.get_total_cost()
print(f"Total cost: ${total:.4f}")

# Vytisknout detailní summary
print(tracker.get_summary())
```

### 2. Global Tracker (doporučeno)

Pro automatické tracking across celou pipeline:

```python
from src.cost_tracker import get_global_tracker

# Získat global instance
tracker = get_global_tracker()

# Používat kdekoli v kódu
tracker.track_llm("anthropic", "haiku", 1000, 500, "summary")

# Na konci pipeline vytisknout summary
print(tracker.get_summary())

# Reset pro novou session
from src.cost_tracker import reset_global_tracker
reset_global_tracker()
```

---

## 🔧 Integrace do kódu

### Příklad: SummaryGenerator

```python
# src/summary_generator.py

from src.cost_tracker import get_global_tracker

class SummaryGenerator:
    def __init__(self, config):
        self.config = config
        self.tracker = get_global_tracker()  # Získat tracker
        # ... zbytek initu

    def _generate_with_anthropic(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens or self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}]
        )

        # NOVÉ: Track usage
        self.tracker.track_llm(
            provider="anthropic",
            model=self.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            operation="summary"
        )

        return response.content[0].text.strip()

    def _generate_with_openai(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=max_tokens or self.max_tokens
        )

        # NOVÉ: Track usage
        self.tracker.track_llm(
            provider="openai",
            model=self.model,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            operation="summary"
        )

        return response.choices[0].message.content.strip()
```

### Příklad: EmbeddingGenerator

```python
# src/embedding_generator.py

from src.cost_tracker import get_global_tracker

class EmbeddingGenerator:
    def __init__(self, config):
        self.config = config
        self.tracker = get_global_tracker()
        # ... zbytek initu

    def _embed_openai(self, texts: List[str]) -> np.ndarray:
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            response = self.client.embeddings.create(
                model=self.model_name,
                input=batch,
                encoding_format="float"
            )

            # NOVÉ: Track usage
            self.tracker.track_embedding(
                provider="openai",
                model=self.model_name,
                tokens=response.usage.total_tokens,
                operation="indexing"
            )

            batch_embeddings = [data.embedding for data in response.data]
            all_embeddings.extend(batch_embeddings)

        return np.array(all_embeddings, dtype=np.float32)
```

### Příklad: IndexingPipeline

```python
# src/indexing_pipeline.py

from src.cost_tracker import get_global_tracker, reset_global_tracker

class IndexingPipeline:
    def index_document(self, document_path: Path):
        # Reset tracker pro nový dokument
        reset_global_tracker()
        tracker = get_global_tracker()

        # ... provádět indexaci ...

        # Na konci vytisknout cost summary
        logger.info("\n" + tracker.get_summary())

        # Vrátit cost info ve výsledku
        result = {
            "vector_store": vector_store,
            "statistics": stats,
            "cost": {
                "total_usd": tracker.get_total_cost(),
                "total_tokens": tracker.get_total_tokens(),
                "breakdown": {
                    "by_provider": dict(tracker.cost_by_provider),
                    "by_operation": dict(tracker.cost_by_operation)
                }
            }
        }

        return result
```

---

## 📈 Příklad výstupu

```
============================================================
API COST SUMMARY
============================================================
Total tokens:  125,750
  Input:       100,000
  Output:      25,750
Total cost:    $0.2863

Cost by provider:
  anthropic      $0.1563
  openai         $0.1300

Cost by operation:
  summary        $0.1563
  indexing       $0.1300

============================================================
```

---

## 🎯 Tipy pro optimalizaci nákladů

### 1. Používejte levnější modely pro summaries
```bash
# .env
LLM_MODEL=claude-haiku-4-5    # $1/$5 per 1M tokens
# místo
LLM_MODEL=claude-sonnet-4-5   # $3/$15 per 1M tokens
```
**Úspora:** 67% na summaries!

### 2. Local embeddings když je to možné
```bash
# .env
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL=bge-m3         # FREE!
# místo
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-large  # $0.13 per 1M
```
**Úspora:** 100% na embeddings! (pokud máte GPU)

### 3. Batch API (50% sleva)
Pro non-real-time use cases použijte Anthropic Batch API.

### 4. Prompt caching (až 90% sleva)
Pro opakované kontexty použijte Anthropic prompt caching.

---

## 🔍 Monitoring v produkci

### CLI Output
Pipeline automaticky zobrazí cost summary na konci:

```bash
python run_pipeline.py data/document.pdf

# Output:
Processing document.pdf...
✅ PHASE 1 complete
✅ PHASE 2 complete (cost: $0.05)
✅ PHASE 3 complete (cost: $0.08)
✅ PHASE 4 complete (cost: $0.13)

============================================================
API COST SUMMARY
============================================================
Total cost:    $0.2600
...
```

### Logging
```python
import logging

# Aktivovat debug logging pro cost tracking
logging.getLogger("src.cost_tracker").setLevel(logging.DEBUG)

# V logu uvidíte:
# DEBUG - LLM usage tracked: anthropic/haiku - 1000 in, 500 out - $0.006500
# DEBUG - Embedding usage tracked: openai/text-embedding-3-large - 10000 tokens - $0.001300
```

---

## ⚠️ Poznámky

1. **Lokální modely jsou FREE** - bge-m3 má $0.00 cost
2. **Voyage AI free tier** - prvních 200M tokenů zdarma!
3. **Ceny se mění** - aktualizujte `PRICING` dict v `cost_tracker.py`
4. **Token counting** - používáme usage data z API responses (přesné)
5. **Estimace** - některé ceny (GPT-5 nano, o3-pro) jsou odhadnuté

---

## 📝 TODO: Budoucí vylepšení

- [ ] CSV export cost dat
- [ ] Grafické vizualizace (matplotlib)
- [ ] Cost alerts (warning pokud překročíme limit)
- [ ] Monthly/weekly aggregace
- [ ] Integration s indexing_pipeline.py
- [ ] Integration s agent CLI
- [ ] Dashboard (Streamlit/Gradio)

---

## 🤝 Contribuce

Pokud najdete chybu v cenách nebo chcete přidat nový model:

1. Aktualizujte `PRICING` dict v `src/cost_tracker.py`
2. Přidejte test do `tests/test_cost_tracker.py`
3. Vytvořte PR s popisem změny

---

**Autor:** Claude Code
**Verze:** 1.0.0
**Datum:** Leden 2025
**Zdroje cen:** https://docs.anthropic.com/pricing, https://openai.com/api/pricing/, https://docs.voyageai.com/docs/pricing
