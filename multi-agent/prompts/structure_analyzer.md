---
name: structure_analyzer
description: Analyzuje strukturu části dokumentu a extrahuje hierarchii sekcí
---

# Úkol
Analyzuj následující část dokumentu (část {chunk_number}/{total_chunks}) a extrahuj její strukturu.

## Co identifikovat:
1. **Hlavní sekce** a jejich názvy
2. **Podsekce** a hierarchické vztahy
3. **Důležité odstavce** a jejich témata
4. **Klíčová témata** obsažená v této části

## Výstupní formát
Vrať strukturovaný JSON s následujícím formátem:

```json
{{
  "chunk_id": {chunk_id},
  "sections": [
    {{
      "title": "název sekce",
      "level": 1,
      "content": "shrnutí obsahu sekce",
      "page_start": 1,
      "page_end": 1,
      "subsections": [],
      "key_topics": ["téma1", "téma2"],
      "metadata": {{}}
    }}
  ],
  "summary": "krátké shrnutí této části dokumentu",
  "key_terms": ["důležitý_termín1", "důležitý_termín2"]
}}
```

## Dokument k analýze:
{document_chunk}