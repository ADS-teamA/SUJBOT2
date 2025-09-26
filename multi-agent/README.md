# Document Analyzer

Pokročilý nástroj pro paralelní analýzu rozsáhlých dokumentů pomocí Claude Code SDK.

## 🚀 Vlastnosti

- **Paralelní zpracování** - Využívá více subagentů pro rychlou analýzu velkých dokumentů (10000+ stran)
- **Inteligentní extrakce struktury** - Automaticky analyzuje hierarchii dokumentu
- **Přesné odpovědi s citacemi** - Každá odpověď obsahuje přesné odkazy na zdroje
- **Podpora více formátů** - PDF, DOCX, TXT, Markdown
- **Škálovatelnost** - Konfigurovatelný počet paralelních agentů

## 📋 Požadavky

- Python 3.8+
- Claude Code SDK
- Závislosti v `requirements.txt`

## 🛠️ Instalace

```bash
# Klonování repozitáře
git clone <repository_url>
cd document-analyzer

# Instalace závislostí
pip install -r requirements.txt

# Alternativně pomocí setup.py
python setup.py install
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
# Zvýšení počtu paralelních agentů pro rychlejší zpracování
python analyze.py velky_dokument.pdf otazky.md --parallel 20

# Detailní výstup pro debugging
python analyze.py dokument.pdf otazky.md --verbose

# Vlastní adresář s prompty
python analyze.py dokument.pdf otazky.md --prompts-dir ./custom_prompts
```

## 📁 Struktura projektu

```
document-analyzer/
├── analyze.py              # Hlavní spouštěč
├── requirements.txt        # Python závislosti
├── setup.py               # Instalační skript
├── README.md              # Dokumentace
│
├── src/                   # Zdrojové kódy
│   ├── document_analyzer.py    # Hlavní orchestrátor
│   ├── document_reader.py      # Čtení různých formátů
│   ├── question_parser.py      # Parsování otázek
│   ├── result_aggregator.py    # Agregace výsledků
│   └── prompt_manager.py       # Správa prompt templates
│
├── prompts/               # Prompt templates
│   ├── structure_analyzer.md   # Analýza struktury
│   ├── question_analyzer.md    # Odpovídání na otázky
│   └── structure_merger.md     # Slučování struktur
│
├── examples/              # Příklady použití
│   ├── sample_questions.md     # Ukázkové otázky
│   └── sample_contract.pdf     # Ukázkový dokument
│
└── tests/                 # Testy
    └── test_analyzer.py

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
python analyze.py specifikace_jaderne_elektrarny.pdf tech_questions.md --parallel 30
```

## ⚡ Tipy pro optimalizaci

1. **Velké dokumenty**: Zvyšte počet paralelních agentů (`--parallel 20`)
2. **Mnoho otázek**: Použijte strukturovaný markdown formát s prioritami
3. **Opakované analýzy**: Výsledky se ukládají do cache pro rychlejší opakované dotazy
4. **Specifické domény**: Upravte prompty pro vaši doménu

## 🐛 Řešení problémů

### Nedostatečná paměť
- Snižte počet paralelních agentů
- Rozdělte dokument na menší části

### Pomalé zpracování
- Zvyšte počet paralelních agentů
- Optimalizujte otázky (méně komplexní)

### Nízká přesnost odpovědí
- Upravte prompty pro lepší instrukce
- Zkontrolujte kvalitu OCR u skenovaných PDF

## 📄 Licence

MIT License - viz LICENSE soubor

## 🤝 Příspěvky

Příspěvky jsou vítány! Prosím:
1. Forkněte repozitář
2. Vytvořte feature branch
3. Commitujte změny
4. Pushněte branch
5. Otevřete Pull Request