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

## 🔧 Konfigurace

### Úprava promptů

Prompty jsou uloženy jako markdown soubory v adresáři `prompts/`. Můžete je upravit podle svých potřeb:

1. Otevřete soubor promptu (např. `prompts/question_analyzer.md`)
2. Upravte instrukce nebo formát výstupu
3. Aplikace automaticky načte změny

### Přidání vlastních promptů

1. Vytvořte nový `.md` soubor v adresáři `prompts/`
2. Přidejte YAML frontmatter s metadaty
3. Definujte prompt template s proměnnými v `{}`

Příklad:
```markdown
---
name: custom_analyzer
description: Vlastní analyzátor pro specifické účely
---

Analyzuj dokument s fokusem na {focus_area}.

Dokument: {document_content}
```

## 📊 Výstupní formát

Aplikace generuje strukturovaný JSON výstup:

```json
{
  "document_path": "dokument.pdf",
  "document_size": 150000,
  "structure": {
    "sections": [...]
  },
  "questions_count": 10,
  "results": [
    {
      "question_id": "q_1_abc123",
      "question": "Jaké jsou termíny?",
      "answer": "Projekt musí být dokončen do 31.12.2024",
      "sources": [
        {
          "reference": "Kapitola 3 (str. 15-16)",
          "quote": "Dokončení díla: 31.12.2024"
        }
      ],
      "confidence": 0.95
    }
  ],
  "processing_time": 45.2,
  "timestamp": "2024-01-20T10:30:00"
}
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

## 🐛 Řešení problémů

### Nedostatečná paměť
- Snižte `batch_size` a `memory_limit_gb` v konfiguraci
- Povolte streaming pro velké dokumenty

### Pomalé zpracování
- Použijte FAISS místo Qdrant pro lokální zpracování
- Zvyšte `max_workers` v konfiguraci
- Vypněte reranking pro rychlejší (ale méně přesné) výsledky

### Nízká přesnost odpovědí
- Zapněte cross-encoder reranking
- Upravte `hybrid_alpha` (více sémantického vyhledávání)
- Zkontrolujte kvalitu OCR u skenovaných PDF
- Zvyšte `chunk_overlap` pro lepší kontext

## 📄 Licence

MIT License - viz LICENSE soubor

## 🤝 Příspěvky

Příspěvky jsou vítány! Prosím:
1. Forkněte repozitář
2. Vytvořte feature branch
3. Commitujte změny
4. Pushněte branch
5. Otevřete Pull Request