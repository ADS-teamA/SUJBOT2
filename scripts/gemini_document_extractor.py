#!/usr/bin/env python3
"""
Universal Document Hierarchy Extractor using Gemini 2.5 Flash.

Best Practices Applied:
1. Direct PDF upload via File API (not Base64)
2. Strict JSON schema with response_mime_type
3. Auto-detection of document type (legal, technical, report, etc.)
4. Full 1M token context window utilization
5. Czech diacritics preservation

Compatible with phase1_extraction.json format.
"""

import json
import os
import time
from pathlib import Path
from typing import Optional

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_ID = "gemini-2.5-flash"

genai.configure(api_key=GOOGLE_API_KEY)


# Universal JSON Schema for document extraction
DOCUMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "document": {
            "type": "object",
            "properties": {
                "type": {"type": "string", "description": "Document type: zakon, vyhlaska, narizeni, zprava, manual, smlouva, report, other"},
                "identifier": {"type": "string", "description": "Document number/identifier (e.g., '18/1997 Sb.', 'BZ-VR1-2017')"},
                "title": {"type": "string"},
                "date": {"type": "string"},
                "author": {"type": "string"},
                "language": {"type": "string", "description": "cs, en, de, etc."}
            },
            "required": ["type", "title"]
        },
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "section_id": {"type": "string", "description": "Unique ID: sec_1, sec_2, ..."},
                    "element_type": {"type": "string", "description": "cast, hlava, dil, clanek, paragraf, odstavec, pismeno, bod, kapitola, sekce, podsekce, priloha, poznamka"},
                    "number": {"type": "string", "description": "Clean number without symbols"},
                    "title": {"type": "string"},
                    "content": {"type": "string", "description": "Full text content with footnote references preserved"},
                    "level": {"type": "integer", "description": "Hierarchy level 1-6"},
                    "depth": {"type": "integer"},
                    "path": {"type": "string", "description": "Hierarchical path: 'ČÁST I > HLAVA 1 > § 32'"},
                    "parent_number": {"type": "string"},
                    "page_number": {"type": "integer"}
                },
                "required": ["section_id", "element_type", "level"]
            }
        }
    },
    "required": ["document", "sections"]
}


EXTRACTION_PROMPT = """Jsi expertní analyzátor dokumentů. Analyzuj nahraný dokument a extrahuj jeho kompletní hierarchickou strukturu.

## TVŮJ ÚKOL:
1. Automaticky rozpoznej typ dokumentu (zákon, vyhláška, technická zpráva, manuál, smlouva, report...)
2. Extrahuj metadata dokumentu (číslo, název, datum, autor)
3. Extrahuj KOMPLETNÍ hierarchickou strukturu do JSON

## TYPY ELEMENTŮ (podle typu dokumentu):

### Pro PRÁVNÍ dokumenty (zákon, vyhláška, nařízení):
- cast (ČÁST I, II, III...) - level 1
- hlava (HLAVA PRVNÍ, DRUHÁ...) - level 2
- dil (Díl 1, 2...) - level 3
- oddil (Oddíl 1, 2...) - level 3
- paragraf (§ 1, § 32...) - level 4
- clanek (Článek 1, 2...) - level 4
- odstavec ((1), (2), (3)...) - level 5
- pismeno (a), b), c)...) - level 6
- bod (1., 2., 3....) - level 6
- poznamka (poznámky pod čarou) - level 6

### Pro TECHNICKÉ dokumenty (zpráva, manuál, report):
- kapitola (1, 2, 3... nebo I, II, III...) - level 1
- sekce (1.1, 1.2...) - level 2
- podsekce (1.1.1, 1.1.2...) - level 3
- odstavec - level 4
- bod - level 5
- priloha (Příloha A, B...) - level 2

## PRAVIDLA:

1. **number** = čisté číslo BEZ symbolů
   - Správně: "32", "1", "a", "I", "PÁTÁ", "1.2.3"
   - Špatně: "§ 32", "(1)", "a)", "Kapitola 1"

2. **content** = úplný text elementu VČETNĚ:
   - Odkazů na poznámky: "smlouvy,26) kterou..."
   - Interních odkazů: "podle § 33 odst. 1"
   - České diakritiky: ěščřžýáíéůúďťňĚŠČŘŽÝÁÍÉŮÚĎŤŇ

3. **path** = hierarchická cesta od kořene oddělená " > "
   - Právní: "ČÁST I > HLAVA PÁTÁ > § 32 > (1)"
   - Technická: "1 Úvod > 1.1 Účel > 1.1.1 Rozsah"

4. **Separace elementů**:
   - Každý odstavec (1), (2), (3) = samostatná sekce
   - Každé písmeno a), b), c) = samostatná sekce
   - NESLUČUJ více odstavců do jednoho

5. **parent_number** = číslo nadřazeného elementu
   - Odstavec (1) pod § 32 → parent_number: "32"
   - Písmeno a) pod (1) → parent_number: "1"

6. **Úplnost**:
   - Extrahuj VŠECHNY elementy z CELÉHO dokumentu
   - Ignoruj záhlaví a zápatí stránek
   - Zachovej obsah i prázdných nadpisů (title bez content je OK)

## PŘÍKLAD VÝSTUPU pro právní dokument:

```json
{
  "document": {
    "type": "zakon",
    "identifier": "18/1997 Sb.",
    "title": "o mírovém využívání jaderné energie a ionizujícího záření",
    "date": "24. ledna 1997",
    "language": "cs"
  },
  "sections": [
    {"section_id": "sec_1", "element_type": "cast", "number": "I", "title": null, "content": null, "level": 1, "path": "ČÁST I"},
    {"section_id": "sec_2", "element_type": "hlava", "number": "PÁTÁ", "title": "OBČANSKOPRÁVNÍ ODPOVĚDNOST ZA JADERNÉ ŠKODY", "level": 2, "path": "ČÁST I > HLAVA PÁTÁ", "parent_number": "I"},
    {"section_id": "sec_3", "element_type": "paragraf", "number": "32", "title": null, "level": 4, "path": "ČÁST I > HLAVA PÁTÁ > § 32", "parent_number": "PÁTÁ"},
    {"section_id": "sec_4", "element_type": "odstavec", "number": "1", "content": "Pro účely občanskoprávní odpovědnosti za jaderné škody se použijí ustanovení mezinárodní smlouvy,26) kterou je Česká republika vázána.", "level": 5, "path": "ČÁST I > HLAVA PÁTÁ > § 32 > (1)", "parent_number": "32", "page_number": 1}
  ]
}
```

Vrať POUZE validní JSON odpovídající schématu. Žádný markdown, žádné komentáře."""


def upload_document(file_path: str) -> genai.types.File:
    """Upload document using File API."""
    print(f"Uploading: {file_path}")

    uploaded_file = genai.upload_file(file_path)
    print(f"  File ID: {uploaded_file.name}")

    # Wait for processing
    while uploaded_file.state.name == "PROCESSING":
        print("  Processing...")
        time.sleep(2)
        uploaded_file = genai.get_file(uploaded_file.name)

    if uploaded_file.state.name == "FAILED":
        raise RuntimeError(f"File processing failed: {uploaded_file.state.name}")

    print(f"  Ready: {uploaded_file.uri}")
    return uploaded_file


def extract_document_hierarchy(
    file_path: str,
    custom_prompt: Optional[str] = None
) -> dict:
    """
    Extract complete document hierarchy using Gemini 2.5 Flash.

    Args:
        file_path: Path to PDF document
        custom_prompt: Optional custom extraction prompt

    Returns:
        Document structure compatible with phase1_extraction.json
    """
    # Upload document
    uploaded_file = upload_document(file_path)

    try:
        # Create model with JSON output
        model = genai.GenerativeModel(
            MODEL_ID,
            generation_config={
                "temperature": 0.1,
                "max_output_tokens": 65536,  # Large output for big documents
                "response_mime_type": "application/json",
            }
        )

        prompt = custom_prompt or EXTRACTION_PROMPT

        print("Generating extraction...")
        response = model.generate_content([uploaded_file, prompt])

        # Parse JSON response
        result = json.loads(response.text)

        # Add extraction metadata
        result["_extraction"] = {
            "model": MODEL_ID,
            "source_path": str(file_path),
            "file_size_bytes": Path(file_path).stat().st_size,
            "prompt_tokens": response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else None,
            "output_tokens": response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else None,
        }

        return result

    finally:
        # Cleanup uploaded file
        try:
            genai.delete_file(uploaded_file.name)
            print("  File deleted from API")
        except Exception:
            pass


def convert_to_phase1_format(extraction: dict) -> dict:
    """Convert extraction to phase1_extraction.json format."""
    doc = extraction.get("document", {})
    sections = extraction.get("sections", [])

    # Calculate stats
    max_depth = max((s.get("depth", s.get("level", 1)) for s in sections), default=0)
    num_roots = sum(1 for s in sections if s.get("level", 1) == 1)

    return {
        "document_id": doc.get("identifier", Path(extraction["_extraction"]["source_path"]).stem),
        "source_path": extraction["_extraction"]["source_path"],
        "extraction_model": extraction["_extraction"]["model"],
        "document_type": doc.get("type", "unknown"),
        "document_title": doc.get("title"),
        "document_date": doc.get("date"),
        "num_sections": len(sections),
        "hierarchy_depth": max_depth,
        "num_roots": num_roots,
        "num_tables": 0,  # Not extracted in this version
        "sections": [
            {
                "section_id": s.get("section_id"),
                "title": s.get("title"),
                "content": s.get("content") or "",
                "level": s.get("level", 1),
                "depth": s.get("depth", s.get("level", 1)),
                "path": s.get("path", ""),
                "page_number": s.get("page_number"),
                "content_length": len(s.get("content") or ""),
                "element_type": s.get("element_type"),
                "number": s.get("number"),
                "parent_number": s.get("parent_number"),
            }
            for s in sections
        ]
    }


def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python gemini_document_extractor.py <pdf_path> [output_path]")
        print("\nExamples:")
        print("  python gemini_document_extractor.py data/zakon.pdf")
        print("  python gemini_document_extractor.py data/report.pdf output/report_hierarchy.json")
        sys.exit(1)

    pdf_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    if not Path(pdf_path).exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)

    print("=" * 70)
    print("Gemini 2.5 Flash - Universal Document Hierarchy Extractor")
    print("=" * 70)
    print(f"Model: {MODEL_ID}")
    print(f"Input: {pdf_path}")
    print(f"Size: {Path(pdf_path).stat().st_size / 1024:.1f} KB")

    # Extract
    extraction = extract_document_hierarchy(pdf_path)

    # Convert to standard format
    result = convert_to_phase1_format(extraction)

    # Statistics
    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"Document type: {result['document_type']}")
    print(f"Document title: {result['document_title']}")
    print(f"Total sections: {result['num_sections']}")
    print(f"Hierarchy depth: {result['hierarchy_depth']}")

    # Count by element type
    by_type = {}
    for sec in result["sections"]:
        t = sec.get("element_type", "unknown")
        by_type[t] = by_type.get(t, 0) + 1
    print(f"\nBy element type:")
    for t, count in sorted(by_type.items()):
        print(f"  {t}: {count}")

    # Save
    if output_path is None:
        output_path = Path(pdf_path).with_suffix('.hierarchy.json')

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to: {output_path}")

    # Show sample
    print("\n" + "=" * 70)
    print("SAMPLE SECTIONS:")
    print("=" * 70)
    for sec in result["sections"][:12]:
        el_type = sec.get("element_type", "?")
        num = sec.get("number", "")
        title = sec.get("title", "")
        content = sec.get("content", "")[:60] if sec.get("content") else ""
        path = sec.get("path", "")
        level = sec.get("level", 1)
        indent = "  " * (level - 1)

        if title:
            print(f"{indent}[{el_type} {num}] {title}")
        elif content:
            print(f"{indent}[{el_type} {num}] {content}...")
        else:
            print(f"{indent}[{el_type} {num}] {path}")


if __name__ == "__main__":
    main()
