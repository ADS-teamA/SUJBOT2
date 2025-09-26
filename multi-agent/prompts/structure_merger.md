---
name: structure_merger
description: Sloučí struktury z jednotlivých částí dokumentu do kompletní hierarchie
---

# Úkol
Slouč následující struktury z různých částí dokumentu do jedné kompletní hierarchické struktury.

## STRUKTURY K SLOUČENÍ
{chunk_structures}

## POŽADAVKY NA SLOUČENÍ

### 1. Hierarchie
- Zachovej správnou hierarchii sekcí a podsekcí
- Identifikuj a spoj související sekce
- Odstraň duplicity

### 2. Kontinuita
- Zajisti kontinuitu číslování stránek
- Propoj sekce, které pokračují přes více chunků
- Zachovej logický tok dokumentu

### 3. Metadata
- Slouč metadata ze souvisejících sekcí
- Agreguj klíčová témata
- Zachovej důležité informace

## VÝSTUPNÍ FORMÁT
Vrať sloučenou strukturu jako JSON:

```json
{
  "document_structure": [
    {
      "id": "section_0",
      "title": "název hlavní sekce",
      "level": 1,
      "page_start": číslo,
      "page_end": číslo,
      "content_summary": "shrnutí obsahu",
      "subsections": [
        {
          "id": "section_0_1",
          "title": "název podsekce",
          "level": 2,
          "page_start": číslo,
          "page_end": číslo,
          "content_summary": "shrnutí"
        }
      ],
      "key_topics": ["téma1", "téma2"]
    }
  ],
  "document_metadata": {
    "total_pages": číslo,
    "total_sections": číslo,
    "main_topics": ["hlavní_téma1", "hlavní_téma2"],
    "document_type": "typ dokumentu"
  }
}
```