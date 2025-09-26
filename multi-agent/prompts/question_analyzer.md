---
name: question_analyzer
description: Analyzuje dokument a odpovídá na specifickou otázku s uvedením zdrojů
---

# Úkol
Analyzuj následující sekce dokumentu a odpověz na otázku s maximální přesností.

## OTÁZKA
{question_text}

## RELEVANTNÍ SEKCE DOKUMENTU
{document_sections}

## POŽADAVKY NA ODPOVĚĎ

### 1. Přesnost a úplnost
- Poskytni **přesnou a úplnou** odpověď na otázku
- Zahrň všechny relevantní informace z dokumentu
- Nedomýšlej si informace, které v dokumentu nejsou

### 2. Uvedení zdrojů
- Uveď **konkrétní zdroje** (kapitoly, odstavce, stránky)
- Cituj **relevantní pasáže** z dokumentu
- Propoj každé tvrzení s konkrétním zdrojem

### 3. Hodnocení jistoty
- Ohodnoť svou **jistotu v odpovědi** (0-1)
- Vysvětli, proč máš danou úroveň jistoty
- Upozorni na případné nejasnosti nebo chybějící informace

## FORMÁT ODPOVĚDI
Vrať odpověď jako JSON s následující strukturou:

```json
{{
  "answer": "kompletní odpověď na otázku",
  "sources": [
    {{
      "section": "název sekce/kapitoly",
      "page": "číslo stránky nebo rozsah",
      "quote": "přesná citace z dokumentu",
      "relevance": "vysvětlení, jak citace podporuje odpověď"
    }}
  ],
  "confidence": 0.95,
  "confidence_explanation": "vysvětlení úrovně jistoty",
  "additional_context": "případný další relevantní kontext",
  "limitations": "případná omezení odpovědi nebo chybějící informace"
}}
```

## PŘÍKLAD KVALITNÍ ODPOVĚDI
Pokud je otázka: "Jaké jsou termíny dokončení projektu?"

Dobrá odpověď:
- Uvede konkrétní data z dokumentu
- Cituje přesné pasáže s termíny
- Specifikuje, o které fáze projektu se jedná
- Upozorní na podmínky spojené s termíny