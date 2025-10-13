# Document Reader Specification - Legal Document Reader

## 1. Purpose

Parse legal documents (contracts, laws, regulations) with awareness of their hierarchical structure and legal semantics.

**Key Innovation**: Not just extracting text, but understanding the legal structure (§, články, odstavce) and relationships (references, citations).

---

## 2. Component Overview

```python
class LegalDocumentReader(DocumentReader):
    """
    Extends base DocumentReader with legal-specific parsing capabilities

    Responsibilities:
    1. Document type detection (contract vs law vs regulation)
    2. Structure parsing (hierarchy extraction)
    3. Reference extraction (§X, článek Y)
    4. Metadata enrichment
    5. Content classification (obligation, prohibition, definition)
    """
```

---

## 3. Supported Document Types

### 3.1 Contract (Smlouva)

**Structure**:
```
Smlouva o dílo
├── Preambule
├── Článek 1: Předmět smlouvy
│   ├── 1.1 Obecné vymezení
│   ├── 1.2 Specifikace
│   └── 1.3 Rozsah
├── Článek 2: Cena a platební podmínky
│   ├── 2.1 Cena díla
│   └── 2.2 Platební kalendář
├── ...
└── Přílohy
```

**Parsing Strategy**:
- Detect articles: `Článek \d+:`, `Čl\. \d+`, `Article \d+`
- Detect sub-points: `\d+\.\d+`, `bod \d+`
- Extract parties from preambule
- Identify annexes

### 3.2 Law Code (Zákon)

**Structure**:
```
Zákon č. 89/2012 Sb., občanský zákoník
├── ČÁST PRVNÍ: Obecná část
│   ├── HLAVA I: Základní ustanovení
│   │   ├── § 1: Základní zásady
│   │   │   ├── (1) První odstavec
│   │   │   ├── (2) Druhý odstavec
│   │   │   └── (3) Třetí odstavec
│   │   │       ├── a) písmeno a
│   │   │       └── b) písmeno b
│   │   └── § 2: Působnost
│   └── HLAVA II: ...
└── ČÁST DRUHÁ: ...
```

**Parsing Strategy**:
- Detect parts: `ČÁST [IVX]+`
- Detect chapters: `HLAVA [IVX]+`
- Detect paragraphs: `§ \d+`
- Detect subsections: `\(\d+\)`
- Detect letters: `[a-z]\)`

### 3.3 Regulation (Vyhláška)

Similar to law but typically flatter hierarchy.

---

## 4. Data Structures

### 4.1 LegalDocument

```python
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime

@dataclass
class LegalDocument:
    """Main document container"""

    # Identity
    document_id: str
    document_type: str  # 'contract' | 'law_code' | 'regulation'

    # Content
    raw_text: str
    cleaned_text: str

    # Structure
    structure: DocumentStructure

    # Metadata
    metadata: DocumentMetadata

    # References
    references: List[LegalReference]

    # Parsed at
    parsed_at: datetime

@dataclass
class DocumentMetadata:
    """Document-level metadata"""

    # File info
    file_path: str
    file_format: str  # 'pdf' | 'docx' | 'xml'
    file_size_bytes: int

    # Document info
    title: str
    document_number: Optional[str]  # e.g., "č. 89/2012 Sb."
    effective_date: Optional[datetime]

    # For contracts
    parties: Optional[List[str]]
    contract_type: Optional[str]  # e.g., "Smlouva o dílo"

    # For laws
    law_type: Optional[str]  # e.g., "Zákon"
    issuing_authority: Optional[str]

    # Statistics
    total_pages: int
    total_words: int
    total_sections: int
```

### 4.2 DocumentStructure

```python
@dataclass
class DocumentStructure:
    """Hierarchical structure representation"""

    hierarchy: List[StructuralElement]

    # Quick access indices
    parts: List[Part]
    chapters: List[Chapter]
    sections: List[Section]  # Paragraphs or Articles

    # Flat list of all elements
    all_elements: List[StructuralElement]

    def get_element_by_reference(self, ref: str) -> Optional[StructuralElement]:
        """Get element by legal reference, e.g., '§89'"""
        pass

    def get_children(self, element: StructuralElement) -> List[StructuralElement]:
        """Get child elements"""
        pass

    def get_path(self, element: StructuralElement) -> str:
        """Get full hierarchy path, e.g., 'Část II > Hlava III > §89'"""
        pass

@dataclass
class StructuralElement:
    """Base class for structural elements"""

    element_id: str
    element_type: str  # 'part' | 'chapter' | 'paragraph' | 'article' | 'subsection'

    # Content
    title: Optional[str]
    content: str

    # Hierarchy
    level: int  # 0 = top-level
    parent_id: Optional[str]
    children_ids: List[str]

    # Position
    start_char: int
    end_char: int
    start_line: int
    end_line: int

    # Legal reference
    legal_reference: str  # e.g., "§89", "Článek 5"

    # Metadata
    metadata: Dict[str, Any]
```

### 4.3 Specialized Elements

```python
@dataclass
class Part(StructuralElement):
    """Part (Část) - top-level division in laws"""

    number: str  # Roman numeral: "I", "II", etc.
    chapters: List[Chapter]

@dataclass
class Chapter(StructuralElement):
    """Chapter (Hlava) - second-level division"""

    number: str  # Roman numeral: "I", "II", etc.
    part: Optional[Part]
    sections: List['Section']

@dataclass
class Paragraph(StructuralElement):
    """Paragraph (§) - basic unit in laws"""

    number: int  # Arabic numeral: 1, 2, 89, etc.
    chapter: Optional[Chapter]
    subsections: List['Subsection']

    # Content classification
    contains_obligation: bool
    contains_prohibition: bool
    contains_definition: bool

@dataclass
class Subsection(StructuralElement):
    """Subsection (odstavec) - subdivision of paragraph"""

    number: int  # (1), (2), etc.
    paragraph: Paragraph
    letters: List['Letter']

@dataclass
class Letter(StructuralElement):
    """Letter (písmeno) - subdivision of subsection"""

    letter: str  # 'a', 'b', 'c', etc.
    subsection: Subsection

@dataclass
class Article(StructuralElement):
    """Article (Článek) - basic unit in contracts"""

    number: int
    points: List['Point']

@dataclass
class Point(StructuralElement):
    """Point (Bod) - subdivision of article"""

    number: str  # "5.1", "5.2", etc.
    article: Article
```

### 4.4 LegalReference

```python
@dataclass
class LegalReference:
    """Reference to another legal provision"""

    reference_id: str

    # Source (where the reference appears)
    source_element_id: str
    source_position: Tuple[int, int]  # (start_char, end_char)

    # Target (what is being referenced)
    target_type: str  # 'paragraph' | 'article' | 'law' | 'regulation'
    target_reference: str  # e.g., "§89", "§89 odst. 2"
    target_document: Optional[str]  # e.g., "Zákon č. 89/2012 Sb."

    # Context
    context: str  # Surrounding text
    reference_type: str  # 'direct' | 'implicit' | 'comparison'

    # Parsed reference components
    components: Dict[str, Any]  # {'paragraph': 89, 'subsection': 2, ...}

# Example:
# "podle §89 odst. 2 občanského zákoníku"
LegalReference(
    reference_id="ref_001",
    source_element_id="article_5_2",
    source_position=(150, 185),
    target_type="paragraph",
    target_reference="§89 odst. 2",
    target_document="Zákon č. 89/2012 Sb.",
    context="... podle §89 odst. 2 občanského zákoníku ...",
    reference_type="direct",
    components={
        'paragraph': 89,
        'subsection': 2,
        'law_number': '89/2012'
    }
)
```

---

## 5. Implementation

### 5.1 Class Structure

```python
# File: src/legal_document_reader.py

from abc import ABC, abstractmethod
import re
from typing import List, Dict, Any, Optional
import pdfplumber
import PyPDF2
from docx import Document as DocxDocument

class LegalDocumentReader:
    """Main reader orchestrating parsing pipeline"""

    def __init__(self):
        self.format_readers = {
            'pdf': PDFReader(),
            'docx': DOCXReader(),
            'xml': XMLReader(),
            'txt': TXTReader()
        }

        self.structure_parsers = {
            'contract': ContractStructureParser(),
            'law_code': LawStructureParser(),
            'regulation': RegulationStructureParser()
        }

        self.reference_extractor = ReferenceExtractor()
        self.content_classifier = ContentClassifier()

    async def read_legal_document(
        self,
        file_path: str,
        document_type: Optional[str] = None
    ) -> LegalDocument:
        """
        Main entry point for reading a legal document

        Args:
            file_path: Path to document
            document_type: 'contract' | 'law_code' | 'regulation' (auto-detect if None)

        Returns:
            LegalDocument with full structure and metadata
        """
        # 1. Detect format
        file_format = self._detect_format(file_path)

        # 2. Extract raw text
        reader = self.format_readers[file_format]
        raw_text, page_info = await reader.read(file_path)

        # 3. Clean text
        cleaned_text = self._clean_text(raw_text)

        # 4. Auto-detect document type if not provided
        if document_type is None:
            document_type = self._detect_document_type(cleaned_text)

        # 5. Parse structure
        parser = self.structure_parsers[document_type]
        structure = await parser.parse(cleaned_text)

        # 6. Extract references
        references = await self.reference_extractor.extract(
            cleaned_text,
            structure
        )

        # 7. Classify content
        structure = await self.content_classifier.classify(structure)

        # 8. Extract metadata
        metadata = self._extract_metadata(
            file_path,
            cleaned_text,
            structure,
            document_type
        )

        # 9. Create document
        document = LegalDocument(
            document_id=self._generate_document_id(file_path),
            document_type=document_type,
            raw_text=raw_text,
            cleaned_text=cleaned_text,
            structure=structure,
            metadata=metadata,
            references=references,
            parsed_at=datetime.now()
        )

        return document

    def _detect_format(self, file_path: str) -> str:
        """Detect file format from extension"""
        ext = Path(file_path).suffix.lower()
        format_map = {
            '.pdf': 'pdf',
            '.docx': 'docx',
            '.doc': 'docx',
            '.xml': 'xml',
            '.txt': 'txt',
            '.md': 'txt'
        }
        return format_map.get(ext, 'txt')

    def _detect_document_type(self, text: str) -> str:
        """Auto-detect document type from content"""
        # Law indicators
        law_indicators = [
            r'Zákon\s+č\.\s*\d+/\d+\s+Sb\.',
            r'ČÁST\s+[IVX]+',
            r'HLAVA\s+[IVX]+',
            r'§\s*\d+'
        ]

        # Contract indicators
        contract_indicators = [
            r'Smlouva\s+o\s+\w+',
            r'Článek\s+\d+',
            r'smluvní\s+strany',
            r'dodavatel\s+a\s+objednatel'
        ]

        law_score = sum(
            1 for pattern in law_indicators
            if re.search(pattern, text, re.IGNORECASE)
        )

        contract_score = sum(
            1 for pattern in contract_indicators
            if re.search(pattern, text, re.IGNORECASE)
        )

        if law_score > contract_score:
            return 'law_code'
        elif contract_score > 0:
            return 'contract'
        else:
            return 'regulation'  # default

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Normalize line breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)

        # Remove page numbers (heuristic)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)

        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")

        return text.strip()
```

### 5.2 Law Structure Parser

```python
class LawStructureParser:
    """Parse Czech law structure"""

    # Regex patterns for Czech laws
    PATTERNS = {
        'law_title': r'Zákon\s+č\.\s*(\d+)/(\d+)\s+Sb\.\s*,?\s*(.*)',
        'part': r'^\s*ČÁST\s+([IVX]+)\s*:?\s*(.*?)$',
        'chapter': r'^\s*HLAVA\s+([IVX]+)\s*:?\s*(.*?)$',
        'section': r'^\s*DÍL\s+([IVX]+)\s*:?\s*(.*?)$',
        'paragraph': r'^\s*§\s*(\d+[a-z]?)\s*(.*)$',
        'subsection': r'^\s*\((\d+)\)\s+(.*)$',
        'letter': r'^\s*([a-z])\)\s+(.*)$',
    }

    async def parse(self, text: str) -> DocumentStructure:
        """Parse law structure"""

        lines = text.split('\n')

        hierarchy = []
        current_part = None
        current_chapter = None
        current_section = None
        current_paragraph = None
        current_subsection = None

        for line_num, line in enumerate(lines):
            # Try to match structural patterns

            # Part
            if match := re.match(self.PATTERNS['part'], line, re.IGNORECASE):
                number, title = match.groups()
                current_part = Part(
                    element_id=f"part_{number}",
                    element_type='part',
                    number=number,
                    title=title.strip(),
                    content="",
                    level=0,
                    parent_id=None,
                    children_ids=[],
                    start_line=line_num,
                    end_line=line_num,  # Will be updated
                    start_char=0,  # Calculate from line
                    end_char=0,
                    legal_reference=f"Část {number}",
                    metadata={'type': 'part'},
                    chapters=[]
                )
                hierarchy.append(current_part)
                current_chapter = None
                current_paragraph = None

            # Chapter
            elif match := re.match(self.PATTERNS['chapter'], line, re.IGNORECASE):
                number, title = match.groups()
                current_chapter = Chapter(
                    element_id=f"chapter_{number}",
                    element_type='chapter',
                    number=number,
                    title=title.strip(),
                    content="",
                    level=1,
                    parent_id=current_part.element_id if current_part else None,
                    children_ids=[],
                    start_line=line_num,
                    end_line=line_num,
                    start_char=0,
                    end_char=0,
                    legal_reference=f"Hlava {number}",
                    metadata={'type': 'chapter'},
                    part=current_part,
                    sections=[]
                )
                if current_part:
                    current_part.chapters.append(current_chapter)
                    current_part.children_ids.append(current_chapter.element_id)
                hierarchy.append(current_chapter)
                current_paragraph = None

            # Paragraph
            elif match := re.match(self.PATTERNS['paragraph'], line):
                number, title = match.groups()
                current_paragraph = Paragraph(
                    element_id=f"paragraph_{number}",
                    element_type='paragraph',
                    number=int(re.sub(r'[a-z]', '', number)),  # Remove letter suffix
                    title=title.strip(),
                    content=title.strip(),
                    level=2,
                    parent_id=current_chapter.element_id if current_chapter else None,
                    children_ids=[],
                    start_line=line_num,
                    end_line=line_num,
                    start_char=0,
                    end_char=0,
                    legal_reference=f"§{number}",
                    metadata={'type': 'paragraph'},
                    chapter=current_chapter,
                    subsections=[],
                    contains_obligation=False,
                    contains_prohibition=False,
                    contains_definition=False
                )
                if current_chapter:
                    current_chapter.sections.append(current_paragraph)
                    current_chapter.children_ids.append(current_paragraph.element_id)
                hierarchy.append(current_paragraph)
                current_subsection = None

            # Subsection
            elif match := re.match(self.PATTERNS['subsection'], line):
                number, content = match.groups()
                current_subsection = Subsection(
                    element_id=f"subsection_{current_paragraph.number}_{number}",
                    element_type='subsection',
                    number=int(number),
                    title=None,
                    content=content.strip(),
                    level=3,
                    parent_id=current_paragraph.element_id if current_paragraph else None,
                    children_ids=[],
                    start_line=line_num,
                    end_line=line_num,
                    start_char=0,
                    end_char=0,
                    legal_reference=f"§{current_paragraph.number} odst. {number}",
                    metadata={'type': 'subsection'},
                    paragraph=current_paragraph,
                    letters=[]
                )
                if current_paragraph:
                    current_paragraph.subsections.append(current_subsection)
                    current_paragraph.children_ids.append(current_subsection.element_id)
                hierarchy.append(current_subsection)

            # Letter
            elif match := re.match(self.PATTERNS['letter'], line):
                letter, content = match.groups()
                letter_elem = Letter(
                    element_id=f"letter_{current_paragraph.number}_{current_subsection.number}_{letter}",
                    element_type='letter',
                    letter=letter,
                    title=None,
                    content=content.strip(),
                    level=4,
                    parent_id=current_subsection.element_id if current_subsection else None,
                    children_ids=[],
                    start_line=line_num,
                    end_line=line_num,
                    start_char=0,
                    end_char=0,
                    legal_reference=f"§{current_paragraph.number} odst. {current_subsection.number} písm. {letter}",
                    metadata={'type': 'letter'},
                    subsection=current_subsection
                )
                if current_subsection:
                    current_subsection.letters.append(letter_elem)
                    current_subsection.children_ids.append(letter_elem.element_id)
                hierarchy.append(letter_elem)

            # Regular content - append to current element
            else:
                if current_subsection:
                    current_subsection.content += " " + line.strip()
                elif current_paragraph:
                    current_paragraph.content += " " + line.strip()

        # Build structure object
        structure = DocumentStructure(
            hierarchy=hierarchy,
            parts=[e for e in hierarchy if isinstance(e, Part)],
            chapters=[e for e in hierarchy if isinstance(e, Chapter)],
            sections=[e for e in hierarchy if isinstance(e, (Paragraph, Article))],
            all_elements=hierarchy
        )

        return structure
```

### 5.3 Reference Extractor

```python
class ReferenceExtractor:
    """Extract legal references from text"""

    # Reference patterns
    PATTERNS = {
        'paragraph': r'§\s*(\d+[a-z]?)(?:\s+odst\.\s*(\d+))?(?:\s+písm\.\s*([a-z]))?',
        'article': r'[Čč]l(?:ánek|\.)\s*(\d+)(?:\.\s*(\d+))?',
        'law_citation': r'[Zz]ákon(?:a|u)?\s+č\.\s*(\d+)/(\d+)\s*Sb\.',
        'regulation': r'[Vv]yhláška\s+č\.\s*(\d+)/(\d+)\s*Sb\.',
    }

    async def extract(
        self,
        text: str,
        structure: DocumentStructure
    ) -> List[LegalReference]:
        """Extract all references from text"""

        references = []

        # For each structural element, find references in its content
        for element in structure.all_elements:
            element_refs = self._find_references_in_text(
                element.content,
                element.element_id,
                element.start_char
            )
            references.extend(element_refs)

        return references

    def _find_references_in_text(
        self,
        text: str,
        source_element_id: str,
        text_offset: int
    ) -> List[LegalReference]:
        """Find references in a piece of text"""

        references = []

        for ref_type, pattern in self.PATTERNS.items():
            for match in re.finditer(pattern, text):
                ref = self._parse_reference_match(
                    match,
                    ref_type,
                    source_element_id,
                    text_offset,
                    text
                )
                references.append(ref)

        return references

    def _parse_reference_match(
        self,
        match: re.Match,
        ref_type: str,
        source_element_id: str,
        text_offset: int,
        full_text: str
    ) -> LegalReference:
        """Parse a single reference match"""

        # Extract context (50 chars before/after)
        start, end = match.span()
        context_start = max(0, start - 50)
        context_end = min(len(full_text), end + 50)
        context = full_text[context_start:context_end]

        # Parse components based on type
        if ref_type == 'paragraph':
            para, subsec, letter = match.groups()
            target_ref = f"§{para}"
            if subsec:
                target_ref += f" odst. {subsec}"
            if letter:
                target_ref += f" písm. {letter}"

            components = {
                'paragraph': int(re.sub(r'[a-z]', '', para)),
                'subsection': int(subsec) if subsec else None,
                'letter': letter
            }

        elif ref_type == 'article':
            article, point = match.groups()
            target_ref = f"Článek {article}"
            if point:
                target_ref += f".{point}"

            components = {
                'article': int(article),
                'point': point
            }

        elif ref_type == 'law_citation':
            number, year = match.groups()
            target_ref = f"Zákon č. {number}/{year} Sb."
            components = {
                'law_number': number,
                'year': year
            }

        else:
            target_ref = match.group(0)
            components = {}

        return LegalReference(
            reference_id=f"ref_{source_element_id}_{start}",
            source_element_id=source_element_id,
            source_position=(text_offset + start, text_offset + end),
            target_type=ref_type,
            target_reference=target_ref,
            target_document=None,  # TODO: extract from context
            context=context,
            reference_type='direct',
            components=components
        )
```

---

## 6. Configuration

```yaml
# config.yaml
document_reader:
  # Supported formats
  formats:
    - pdf
    - docx
    - xml
    - txt

  # PDF reading strategy
  pdf:
    primary_reader: pdfplumber
    fallback_reader: pypdf2
    extract_tables: true
    extract_images: false

  # Text cleaning
  cleaning:
    normalize_whitespace: true
    remove_page_numbers: true
    normalize_quotes: true
    fix_line_breaks: true

  # Structure parsing
  structure:
    law_code:
      detect_parts: true
      detect_chapters: true
      detect_paragraphs: true
      detect_subsections: true
      detect_letters: true

    contract:
      detect_articles: true
      detect_points: true
      detect_parties: true
      detect_annexes: true

  # Reference extraction
  references:
    extract_paragraph_refs: true
    extract_article_refs: true
    extract_law_citations: true
    extract_regulation_refs: true
    context_window: 50  # characters

  # Content classification
  classification:
    detect_obligations: true
    detect_prohibitions: true
    detect_definitions: true
```

---

## 7. Error Handling

```python
class DocumentReadError(Exception):
    """Base exception for document reading errors"""
    pass

class UnsupportedFormatError(DocumentReadError):
    """Unsupported file format"""
    pass

class StructureParsingError(DocumentReadError):
    """Error parsing document structure"""
    pass

class CorruptedDocumentError(DocumentReadError):
    """Document file is corrupted"""
    pass

# Usage:
try:
    document = await reader.read_legal_document("smlouva.pdf")
except UnsupportedFormatError:
    logger.error("Unsupported file format")
except CorruptedDocumentError:
    logger.error("Document is corrupted, trying fallback reader")
    document = await reader.read_legal_document("smlouva.pdf", force_fallback=True)
```

---

## 8. Testing Strategy

### 8.1 Unit Tests

```python
# tests/test_legal_document_reader.py

def test_detect_law_structure():
    text = """
    ČÁST PRVNÍ
    HLAVA I
    § 1
    (1) První odstavec
    (2) Druhý odstavec
    """
    parser = LawStructureParser()
    structure = parser.parse(text)

    assert len(structure.parts) == 1
    assert len(structure.chapters) == 1
    assert len(structure.sections) == 1

def test_extract_paragraph_references():
    text = "podle §89 odst. 2 písm. a)"
    extractor = ReferenceExtractor()
    refs = extractor._find_references_in_text(text, "elem_1", 0)

    assert len(refs) == 1
    assert refs[0].components['paragraph'] == 89
    assert refs[0].components['subsection'] == 2
    assert refs[0].components['letter'] == 'a'
```

### 8.2 Integration Tests

- Test with real Czech law documents
- Test with real contracts
- Test with corrupted PDFs
- Test with multi-language documents

---

## 9. Performance Targets

| Document Size | Parse Time | Memory |
|--------------|-----------|--------|
| 100 pages | <5s | <50 MB |
| 1,000 pages | <30s | <200 MB |
| 10,000 pages | <3 min | <1 GB |

---

## 10. Future Enhancements

1. **Table extraction** - Parse tabular data in contracts
2. **Image OCR** - Extract text from scanned documents
3. **Multi-language support** - Slovak, Polish, English contracts
4. **Diff detection** - Compare versions of documents
5. **Signature detection** - Identify signed sections
6. **Automated party extraction** - NER for contract parties
