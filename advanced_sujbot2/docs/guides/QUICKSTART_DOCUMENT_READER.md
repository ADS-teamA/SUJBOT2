# Quick Start Guide - Legal Document Reader

## Installation

```bash
cd /Users/michalprusek/PycharmProjects/SUJBOT2/advanced_sujbot2

# Install dependencies (if not already installed)
pip install pdfplumber PyPDF2 python-docx

# Or using requirements.txt
pip install -r requirements.txt
```

## Basic Usage

### Parse a Document

```python
import asyncio
from src.document_reader import LegalDocumentReader

async def parse_document():
    # Initialize reader
    reader = LegalDocumentReader()

    # Parse document (auto-detect type)
    document = await reader.read_legal_document("your_document.pdf")

    # Access results
    print(f"Type: {document.document_type}")
    print(f"Title: {document.metadata.title}")
    print(f"Sections: {len(document.structure.sections)}")

    return document

# Run
document = asyncio.run(parse_document())
```

### Parse with Explicit Type

```python
# For Czech laws
document = await reader.read_legal_document("zakon.pdf", document_type="law_code")

# For contracts
document = await reader.read_legal_document("smlouva.pdf", document_type="contract")

# For regulations
document = await reader.read_legal_document("vyhlaska.pdf", document_type="regulation")
```

## Working with Structure

### Navigate Hierarchy

```python
# Get all sections (paragraphs or articles)
for section in document.structure.sections:
    print(f"{section.legal_reference}: {section.title}")

# Get all parts (for laws)
for part in document.structure.parts:
    print(f"{part.legal_reference}: {part.title}")
    for chapter in part.chapters:
        print(f"  {chapter.legal_reference}: {chapter.title}")

# Get specific element by reference
element = document.structure.get_element_by_reference("§89")
if element:
    print(f"Found: {element.content}")

    # Get full path
    path = document.structure.get_path(element)
    print(f"Path: {path}")  # e.g., "Část I > Hlava II > §89"
```

### Access References

```python
# Get all legal references
for ref in document.references:
    print(f"Reference: {ref.target_reference}")
    print(f"Type: {ref.target_type}")
    print(f"Source: {ref.source_element_id}")
    print(f"Context: {ref.context}")
    print()

# Filter by type
paragraph_refs = [r for r in document.references if r.target_type == 'paragraph']
article_refs = [r for r in document.references if r.target_type == 'article']
law_citations = [r for r in document.references if r.target_type == 'law_citation']
```

### Content Classification

```python
# Find obligations
for section in document.structure.sections:
    if hasattr(section, 'contains_obligation') and section.contains_obligation:
        print(f"Obligation: {section.legal_reference}")
        print(f"  {section.content[:100]}...")

# Find prohibitions
prohibitions = [
    s for s in document.structure.sections
    if hasattr(s, 'contains_prohibition') and s.contains_prohibition
]

# Find definitions
definitions = [
    s for s in document.structure.sections
    if hasattr(s, 'contains_definition') and s.contains_definition
]
```

## Run Test Suite

```bash
cd /Users/michalprusek/PycharmProjects/SUJBOT2/advanced_sujbot2
python3 test_document_reader.py
```

Expected output:
```
=== Document Analysis Results ===
Document ID: 8bfdc0b0b9852fca
Document Type: law_code
Title: Zákon č. 89/2012 Sb., občanský zákoník
Total Words: 58
Total Sections: 2

=== Structure ===
Parts: 0
Chapters: 1
Sections: 2
Total Elements: 8

=== All tests completed successfully ===
```

## Supported Formats

| Format | Extension | Status |
|--------|-----------|--------|
| PDF | `.pdf` | ✅ Supported (pdfplumber + PyPDF2) |
| DOCX | `.docx`, `.doc` | ✅ Supported (python-docx) |
| TXT | `.txt`, `.md` | ✅ Supported (native) |
| XML | `.xml` | ✅ Supported (basic) |

## Supported Document Types

| Type | Czech Name | Detection Keywords |
|------|-----------|-------------------|
| Law Code | Zákon | "Zákon č.", "ČÁST", "HLAVA", "§" |
| Contract | Smlouva | "Smlouva", "Článek", "smluvní strany" |
| Regulation | Vyhláška | Auto-detected |

## Structure Elements

### Czech Law Structure
- **ČÁST** (Part) - Roman numerals: I, II, III, ...
- **HLAVA** (Chapter) - Roman numerals: I, II, III, ...
- **§** (Paragraph) - Arabic numerals: 1, 2, 89, ...
- **(1)** (Subsection) - Numbers in parentheses
- **a)** (Letter) - Lowercase letters with parenthesis

### Contract Structure
- **Článek** (Article) - Arabic numerals: 1, 2, 3, ...
- **1.1** (Point) - Decimal notation

## Reference Types

| Type | Example | Components |
|------|---------|-----------|
| Paragraph | `§89` | `{'paragraph': 89}` |
| Paragraph + Subsection | `§89 odst. 2` | `{'paragraph': 89, 'subsection': 2}` |
| Full Reference | `§89 odst. 2 písm. a` | `{'paragraph': 89, 'subsection': 2, 'letter': 'a'}` |
| Article | `Článek 5` | `{'article': 5}` |
| Article + Point | `Čl. 5.1` | `{'article': 5, 'point': '1'}` |
| Law Citation | `Zákon č. 89/2012 Sb.` | `{'law_number': '89', 'year': '2012'}` |

## Error Handling

```python
from src.document_reader import (
    DocumentReadError,
    UnsupportedFormatError,
    StructureParsingError,
    CorruptedDocumentError
)

try:
    document = await reader.read_legal_document("document.pdf")
except UnsupportedFormatError:
    print("File format not supported")
except CorruptedDocumentError:
    print("Document is corrupted")
except StructureParsingError:
    print("Failed to parse structure")
except DocumentReadError as e:
    print(f"Error reading document: {e}")
```

## Logging

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Or just info
logging.basicConfig(level=logging.INFO)

# Parse document (will show detailed logs)
document = await reader.read_legal_document("document.pdf")
```

## Integration Example

```python
async def index_legal_document(file_path: str):
    """Parse and index a legal document for RAG system"""

    # 1. Parse document
    reader = LegalDocumentReader()
    document = await reader.read_legal_document(file_path)

    # 2. Create chunks from structure
    chunks = []
    for section in document.structure.sections:
        chunk = {
            'content': section.content,
            'metadata': {
                'document_id': document.document_id,
                'document_type': document.document_type,
                'reference': section.legal_reference,
                'path': document.structure.get_path(section),
                'level': section.level,
            }
        }

        # Add classification flags
        if hasattr(section, 'contains_obligation'):
            chunk['metadata']['is_obligation'] = section.contains_obligation
            chunk['metadata']['is_prohibition'] = section.contains_prohibition
            chunk['metadata']['is_definition'] = section.contains_definition

        chunks.append(chunk)

    # 3. Index chunks (your RAG system)
    # await your_rag_system.index(chunks)

    return document, chunks
```

## Common Use Cases

### 1. Extract All Obligations from a Law

```python
document = await reader.read_legal_document("zakon.pdf", "law_code")

obligations = [
    {
        'reference': s.legal_reference,
        'path': document.structure.get_path(s),
        'content': s.content
    }
    for s in document.structure.sections
    if hasattr(s, 'contains_obligation') and s.contains_obligation
]
```

### 2. Find Contract Articles Referencing a Law

```python
document = await reader.read_legal_document("smlouva.pdf", "contract")

# Find references to §89
refs = [
    ref for ref in document.references
    if ref.target_reference.startswith("§89")
]

# Get the articles containing these references
articles = [
    document.structure.get_element_by_id(ref.source_element_id)
    for ref in refs
]
```

### 3. Extract Document Summary

```python
summary = {
    'document_id': document.document_id,
    'type': document.document_type,
    'title': document.metadata.title,
    'stats': {
        'pages': document.metadata.total_pages,
        'words': document.metadata.total_words,
        'sections': document.metadata.total_sections,
        'references': len(document.references)
    },
    'structure': {
        'parts': len(document.structure.parts),
        'chapters': len(document.structure.chapters),
        'sections': len(document.structure.sections),
        'total_elements': len(document.structure.all_elements)
    }
}
```

## Troubleshooting

### Issue: PDF not reading

**Solution**: Ensure pdfplumber and PyPDF2 are installed
```bash
pip install pdfplumber PyPDF2
```

### Issue: DOCX not reading

**Solution**: Install python-docx
```bash
pip install python-docx
```

### Issue: Structure not detected

**Solution**: Check document format and patterns. Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Issue: Missing references

**Solution**: References are extracted from element content. Ensure text is properly cleaned.

## Next Steps

1. Read full documentation: `DOCUMENT_READER_README.md`
2. See implementation details: `IMPLEMENTATION_SUMMARY.md`
3. Review source code: `src/document_reader.py`, `src/models.py`
4. Check specification: `specs/02_document_reader.md`

## Contact

For issues or questions, refer to the main project documentation.
