# Document Analyzer

Pokročilý nástroj pro inteligentní analýzu rozsáhlých dokumentů pomocí hybridního vyhledávání a Claude API.

## 🚀 Vlastnosti

- **Hybridní vyhledávání** - Kombinuje sémantické (vector) a klíčové (BM25) vyhledávání pro optimální přesnost
- **Pokročilé chunkovanie** - Sémantické dělení dokumentů zachovávající strukturu a kontext
- **Reranking** - Cross-encoder model pro zvýšení relevance výsledků
- **Více vektorových databází** - Podpora FAISS, Qdrant, ChromaDB
- **Přesné odpovědi s citacemi** - Každá odpověď obsahuje odkazy na konkrétní části dokumentu
- **Podpora více formátů** - PDF, DOCX, TXT, Markdown s fallback mechanismy

## 📋 Požadavky

- Python 3.8+
- Claude API klíč (Anthropic)
- Závislosti v `requirements.txt`

## 🛠️ Instalace

```bash
# Navigace do adresáře
cd multi-agent

# Vytvoření virtuálního prostředí
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Instalace závislostí
pip install -r requirements.txt

# Nastavení prostředí
cp .env.example .env
# Edituj .env a nastav CLAUDE_API_KEY
```

## 💻 Použití

### Základní použití

```bash
# Jedna otázka přímo v příkazové řádce
python analyze.py dokument.pdf "Jaké jsou hlavní podmínky smlouvy?"

# Více otázek ze souboru
python analyze.py zakon.pdf otazky.md

# S ukládáním výsledků
python analyze.py specifikace.docx otazky.md --output vysledky.json
```

### Pokročilé možnosti

```bash
# Detailní výstup pro debugging
python analyze.py dokument.pdf otazky.md --verbose

# Vlastní konfigurace
python analyze.py dokument.pdf otazky.md --config custom_config.yaml

# Interaktivní režim (bez předem připravených otázek)
python analyze.py dokument.pdf
```

## 📁 Struktura projektu

```
multi-agent/
├── analyze.py                  # Hlavní spouštěč - Enhanced vector analyzer
├── config.yaml                # Konfigurace systému
├── requirements.txt           # Python závislosti (optimalizované)
├── setup.py                   # Instalační skript
├── .env.example              # Template pro prostředí
├── README.md                 # Dokumentace
│
├── src/                      # Zdrojové kódy
│   ├── hybrid_retriever.py        # Hybridní vyhledávání (semantic + BM25)
│   ├── indexing_pipeline.py       # Zpracování a indexování dokumentů
│   ├── vector_store.py            # Abstrakce vektorových databází
│   ├── document_reader.py         # Čtení různých formátů (pdfplumber + fallback)
│   ├── claude_sdk_wrapper.py      # Wrapper pro Anthropic SDK
│   ├── document_analyzer.py       # Legacy parallel system (reference)
│   ├── question_parser.py         # Parsování otázek
│   ├── result_aggregator.py       # Agregace výsledků
│   └── prompt_manager.py          # Správa prompt templates
│
├── prompts/                  # Prompt templates s YAML frontmatter
│   ├── structure_analyzer.md      # Analýza struktury
│   ├── question_analyzer.md       # Odpovídání na otázky
│   └── structure_merger.md        # Slučování struktur
│
├── examples/                 # Příklady použití
│   └── [sample documents]
│
└── indexes/                  # Generované indexy (auto-created)
```

## 📝 Formát otázek

### Prostý text
```text
Jaké jsou termíny dokončení projektu?
Kdo je odpovědný za realizaci?
Jaké jsou sankce za nedodržení termínů?
```

### Markdown formát
```markdown
# Otázky k analýze

## Obecné informace
- Jaký je účel dokumentu?
- Kdo jsou smluvní strany?

## Technické specifikace
- [HIGH] Jaké jsou technické požadavky?
- [CRITICAL] Jaké jsou bezpečnostní standardy?
```

### Číslované otázky
```text
1. Jaká je celková hodnota projektu?
2. Jaké jsou platební podmínky?
3. Existují nějaké záruky?
```

## 🎯 Použití v kódu

```python
import asyncio
from src.document_analyzer import DocumentAnalyzer

async def analyze_custom():
    analyzer = DocumentAnalyzer(max_parallel_agents=15)

    results = await analyzer.analyze_document(
        "path/to/document.pdf",
        "path/to/questions.md"
    )

    # Zpracování výsledků
    for result in results['results']:
        print(f"Q: {result['question']}")
        print(f"A: {result['answer']}")
        print(f"Confidence: {result['confidence']:.0%}")
        print("Sources:", result['sources'])
        print("-" * 50)

asyncio.run(analyze_custom())
```

## 🚦 Příklady use-cases

### Analýza smluv
```bash
python analyze.py smlouva.pdf examples/contract_questions.md
```

### Analýza zákonů
```bash
python analyze.py sbirka_zakonu.pdf "Jaké jsou podmínky pro získání licence?"
```

### Technické specifikace
```bash
python analyze.py specifikace_jaderne_elektrarny.pdf tech_questions.md --config production_config.yaml
```

## ⚡ Tipy pro optimalizaci

1. **Velké dokumenty**: Povolte streaming režim v `config.yaml`
2. **Lepší relevance**: Upravte `hybrid_alpha` v konfiguraci (více sémantické vs. klíčové vyhledávání)
3. **Rychlost**: Použijte FAISS pro lokální zpracování, Qdrant pro produkci
4. **Paměť**: Snižte `batch_size` a `memory_limit_gb` v konfiguraci
5. **Přesnost**: Zapněte cross-encoder reranking pro lepší výsledky

## 🔍 Retrieval Pipeline - Jak funguje vyhledávání

### Celkový tok dat

```
PDF/DOCX/TXT → Document Reader → Semantic Chunking → Indexing → Hybrid Retrieval → Reranking → Context pro LLM
```

### 1. Načtení dokumentu (`DocumentReader`)

**src/document_reader.py**

- **Primární**: `pdfplumber` (lepší OCR a komprimované PDF)
- **Fallback**: `PyPDF2` (když pdfplumber selže)
- **Formáty**: PDF, DOCX (python-docx), TXT, Markdown
- **Metadata**: Extrakce metadat, detekce jazyka, počet stran

### 2. Sémantické chunkovanie (`IndexingPipeline`)

**src/indexing_pipeline.py:DocumentChunker**

**Strategie**: `semantic` (default), `fixed`, `sliding_window`

#### Semantic Chunking (doporučeno):
```python
ChunkingStrategy:
  chunk_size: 512 tokens       # Cílová velikost
  chunk_overlap: 0.15          # 15% překryv
  min_chunk_size: 128 tokens
  max_chunk_size: 1024 tokens
```

**Princip**:
1. Split na věty pomocí regex `[.!?]+\s+`
2. Detekce topic boundaries (změny tématu)
3. Seskupování vět do chunků zachovávajících kontext
4. Respektování odstavců a nadpisů
5. Překryv pro zachování kontextu mezi chunky

**Metadata per chunk**:
- `chunk_id` (MD5 hash obsahu)
- `chunk_index` (pořadí v dokumentu)
- `start_char`, `end_char` (pozice v originále)
- `source_document`, `page_number`

### 3. Indexování (`IndexingPipeline`)

**src/indexing_pipeline.py:IndexingPipeline**

#### Embedding generování:
- **Model**: `sentence-transformers/all-MiniLM-L6-v2` (384 dim)
- **Batch processing**: 32 chunků najednou
- **Normalizace**: L2 normalizace vektorů
- **Cache**: Persistentní cache v `indexes/`

#### Paralelní indexování:
```python
- batch_size: 1000 chunků
- max_workers: 8 paralelních workerů
- Progress tracking s rich/tqdm
```

#### Vector Store:
- **Default**: FAISS (IndexFlatIP pro cosine similarity)
- **Alternatyvy**: Qdrant, ChromaDB
- **Index persistence**: Automatické ukládání do `indexes/{doc_hash}/`

### 4. Hybrid Retrieval (`HybridRetriever`)

**src/hybrid_retriever.py:HybridRetriever.retrieve()**

Kombinuje 3 techniky:

#### A. Sémantické vyhledávání (Vector Search)
```python
- Query embedding pomocí stejného modelu
- FAISS IndexFlatIP search (inner product)
- Top-K kandidátů (default: 20)
- Cosine similarity scoring
```

#### B. BM25 Keyword Search
```python
class BM25Searcher:
  - Tokenizace: lowercase + split
  - BM25Okapi algoritmus
  - TF-IDF váhování s document frequency
  - Top-K kandidátů (default: 20)
```

#### C. Hybrid Scoring
```python
hybrid_score = alpha * semantic_score + (1 - alpha) * bm25_score

kde alpha = 0.7 (config.yaml: retrieval.hybrid_alpha)
  → 70% sémantické, 30% keyword matching
```

**Deduplication**: Pokud chunk najdou obě metody, bere se vyšší skóre

### 5. Query Processing

**src/hybrid_retriever.py:QueryDecomposer**

#### Query Decomposition:
Rozklad složitých otázek na sub-queries:

```python
"Compare X vs Y" → ["What is X?", "What is Y?", "Compare X vs Y"]
"History of X"   → ["When X started", "Evolution of X", "Current state of X"]
```

**Query types**: `comparison`, `temporal`, `analytical`, `enumerative`, `factual`

#### Query Expansion:
```python
class QueryExpander:
  - Synonyma pomocí WordNet
  - Rozšíření multi-word expressions
  - Entity recognition
```

#### Keyword Extraction:
```python
- Stop words filtering
- Kapitalizovaná slova (named entities)
- Pattern matching: data, čísla, quoted text
```

### 6. Reranking (`HybridRetriever._rerank_results`)

**Cross-Encoder Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2`

```python
Proces:
1. Top-K chunků z hybrid search (20)
2. Cross-encoder: score(query, chunk) pro každý
3. Re-sort podle cross-encoder skóre
4. Top-N finálních chunků (default: 5)
```

**Confidence scoring**:
```python
# Normalizace cross-encoder scores (-15 až 10) na 0-1
confidence = (score - min_score) / (max_score - min_score)
confidence = max(0.1, min(1.0, confidence))
```

### 7. Context Assembly

**analyze.py:VectorizedDocumentAnalyzer**

```python
Finální kontext pro Claude:
- Top-5 reranked chunků
- Seřazeno podle relevance
- Metadata: source, page, confidence
- Separator: "\n\n---\n\n" mezi chunky
```

## 📊 Porovnání s původním systémem

| Aspekt | Legacy (document_analyzer.py) | Current (hybrid_retriever.py) |
|--------|-------------------------------|-------------------------------|
| Chunking | Fixed-size (50KB) | Semantic (512 tokens) |
| Search | Keyword matching | Hybrid (semantic + BM25) |
| Scoring | Keyword count | Weighted hybrid + reranking |
| Context | Top-5 by keyword overlap | Top-5 by cross-encoder |
| Embeddings | ❌ Žádné | ✅ all-MiniLM-L6-v2 |
| Vector DB | ❌ Žádná | ✅ FAISS/Qdrant/ChromaDB |
| Query processing | ❌ Raw query | ✅ Decomposition + expansion |

## 🎯 Nastavení retrieval parametrů

**config.yaml - klíčové parametry**:

```yaml
retrieval:
  hybrid_alpha: 0.7        # ↑ více sémantické, ↓ více keyword
  top_k: 20                # Počet chunků před rerankingem
  rerank_top_k: 5          # Finální počet chunků
  min_score: 0.1           # Threshold pro filtrování
  enable_reranking: true   # Zapnout cross-encoder

indexing:
  chunk_size: 512          # Větší = více kontextu, horší přesnost
  chunk_overlap: 0.15      # Překryv pro zachování kontextu
  semantic_chunking: true  # Respektuje strukturu dokumentu

embeddings:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  # Alternativy:
  # - "BAAI/bge-small-en-v1.5"  (lepší kvalita)
  # - "all-mpnet-base-v2"        (nejlepší, pomalé)
```

## 🔧 Debugging retrieval

```bash
# Verbose režim pro debugging
python analyze.py doc.pdf "otázka" --verbose

# Logování ukazuje:
# - Query decomposition (sub-queries)
# - Semantic search scores
# - BM25 scores
# - Hybrid scores
# - Cross-encoder reranking scores
# - Final top-K chunks s confidence
```
