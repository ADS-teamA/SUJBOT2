"""
PHASE 1: Document Extraction using Gemini 2.5 Flash.

Extracts hierarchical document structure using Gemini's native PDF understanding.

Features:
1. Direct PDF upload via File API (not Base64)
2. JSON response format with response_mime_type="application/json"
3. Auto-detection of document type (legal, technical, report, etc.)
4. Full 1M token context window utilization
5. Czech diacritics preservation
6. Section and document summary generation
7. Automatic fallback to Unstructured on failure

Compatible with ExtractedDocument/DocumentSection interface.

Note:
    Requires GOOGLE_API_KEY environment variable.
    Use get_extractor() factory for automatic backend selection.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import google.generativeai as genai
from dotenv import load_dotenv

from src.unstructured_extractor import (
    DocumentSection,
    ExtractedDocument,
    ExtractionConfig,
    TableData,
)

load_dotenv()

logger = logging.getLogger(__name__)


# Default model - Gemini 2.5 Flash
DEFAULT_MODEL = "gemini-2.5-flash"


# Universal extraction prompt for legal/technical documents
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

7. **summary** = stručné shrnutí (max 200 znaků) pro sekce s obsahem
   - Piš GENERICKY srozumitelně (ne právnický žargon)
   - Pro nadpisy bez textu = null
   - Příklad: "Definuje odpovědnost provozovatele za škody způsobené jadernou havárií"

8. **document_summary** = shrnutí celého dokumentu (max 500 znaků)
   - Hlavní téma a účel dokumentu
   - Genericky srozumitelné pro laiky

## PŘÍKLAD VÝSTUPU pro právní dokument:

```json
{
  "document": {
    "type": "zakon",
    "identifier": "18/1997 Sb.",
    "title": "o mírovém využívání jaderné energie a ionizujícího záření",
    "date": "24. ledna 1997",
    "language": "cs",
    "summary": "Zákon upravuje podmínky využívání jaderné energie, ionizujícího záření a nakládání s radioaktivními odpady. Stanoví požadavky na bezpečnost, odpovědnost provozovatelů a ochranu před zářením."
  },
  "sections": [
    {"section_id": "sec_1", "element_type": "cast", "number": "I", "title": null, "content": null, "level": 1, "path": "ČÁST I", "summary": null},
    {"section_id": "sec_2", "element_type": "hlava", "number": "PÁTÁ", "title": "OBČANSKOPRÁVNÍ ODPOVĚDNOST ZA JADERNÉ ŠKODY", "level": 2, "path": "ČÁST I > HLAVA PÁTÁ", "parent_number": "I", "summary": null},
    {"section_id": "sec_3", "element_type": "paragraf", "number": "32", "title": null, "level": 4, "path": "ČÁST I > HLAVA PÁTÁ > § 32", "parent_number": "PÁTÁ", "summary": null},
    {"section_id": "sec_4", "element_type": "odstavec", "number": "1", "content": "Pro účely občanskoprávní odpovědnosti za jaderné škody se použijí ustanovení mezinárodní smlouvy,26) kterou je Česká republika vázána.", "level": 5, "path": "ČÁST I > HLAVA PÁTÁ > § 32 > (1)", "parent_number": "32", "page_number": 1, "summary": "Odkazuje na mezinárodní smlouvu pro řešení odpovědnosti za jaderné škody."}
  ]
}
```

Vrať POUZE validní JSON odpovídající schématu. Žádný markdown, žádné komentáře."""


@dataclass
class GeminiExtractionConfig:
    """
    Configuration for Gemini extraction.

    Attributes:
        model: Gemini model ID (default: gemini-2.5-flash)
        temperature: Generation temperature 0.0-1.0 (default: 0.1 for deterministic output)
        max_output_tokens: Maximum output tokens (default: 65536 for large documents)
        fallback_to_unstructured: Fall back to Unstructured on Gemini failure (default: True)
        chunk_large_pdfs: Enable chunked extraction for large PDFs (default: True)
        max_pages_per_chunk: Maximum pages per chunk when splitting (default: 50)
        file_size_threshold_mb: File size threshold for chunked extraction (default: 10.0 MB)
    """

    model: str = DEFAULT_MODEL
    temperature: float = 0.1
    max_output_tokens: int = 65536
    fallback_to_unstructured: bool = True
    chunk_large_pdfs: bool = True
    max_pages_per_chunk: int = 100
    file_size_threshold_mb: float = 10.0


class GeminiExtractor:
    """
    Document extraction using Gemini 2.5 Flash with File API.

    Provides same interface as UnstructuredExtractor for drop-in replacement.

    Example:
        >>> extractor = GeminiExtractor()
        >>> doc = extractor.extract(Path("document.pdf"))
    """

    def __init__(
        self,
        config: Optional[ExtractionConfig] = None,
        gemini_config: Optional[GeminiExtractionConfig] = None,
    ):
        """
        Initialize Gemini extractor.

        Args:
            config: Standard ExtractionConfig (for compatibility)
            gemini_config: Gemini-specific configuration
        """
        self.config = config
        self.gemini_config = gemini_config or GeminiExtractionConfig()

        # Configure Gemini API
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found in environment. "
                "Set it in .env file or export GOOGLE_API_KEY=..."
            )

        genai.configure(api_key=api_key)
        self.model_id = self.gemini_config.model

        logger.info(f"GeminiExtractor initialized with model={self.model_id}")

    def _fallback_to_unstructured(self, file_path: Path, reason: str) -> ExtractedDocument:
        """
        Fall back to Unstructured extraction with logging.

        Args:
            file_path: Path to document
            reason: Reason for fallback (for logging)

        Returns:
            ExtractedDocument from Unstructured extractor
        """
        logger.warning(f"{reason}. Falling back to Unstructured extraction.")
        from src.unstructured_extractor import UnstructuredExtractor
        return UnstructuredExtractor(self.config).extract(file_path)

    def extract(self, file_path: Path) -> ExtractedDocument:
        """
        Extract document structure from PDF using Gemini.

        Args:
            file_path: Path to PDF document

        Returns:
            ExtractedDocument with hierarchical sections

        Raises:
            RuntimeError: If extraction fails and no fallback available
        """
        logger.info(f"Starting Gemini extraction of {file_path.name}")
        start_time = time.time()

        # Only PDF supported
        if file_path.suffix.lower() != ".pdf":
            if self.gemini_config.fallback_to_unstructured:
                return self._fallback_to_unstructured(
                    file_path, f"Gemini only supports PDF, got {file_path.suffix}"
                )
            raise ValueError(f"Gemini extractor only supports PDF files, got: {file_path.suffix}")

        # Check if chunked extraction is needed for large PDFs
        if self._needs_chunking(file_path):
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            logger.info(
                f"Large PDF detected ({file_size_mb:.1f} MB > {self.gemini_config.file_size_threshold_mb} MB). "
                f"Using chunked extraction with max {self.gemini_config.max_pages_per_chunk} pages per chunk."
            )
            try:
                return self._extract_chunked(file_path)
            except Exception as e:
                logger.error(f"Chunked extraction failed: {e}")
                if self.gemini_config.fallback_to_unstructured:
                    return self._fallback_to_unstructured(file_path, f"Chunked extraction failed: {e}")
                raise

        try:
            # 1. Upload PDF via File API
            uploaded_file = self._upload_document(file_path)

            try:
                # 2. Extract hierarchy with Gemini
                raw_extraction = self._extract_with_gemini(uploaded_file)

                # 3. Convert to ExtractedDocument format
                extraction_time = time.time() - start_time
                return self._convert_to_extracted_document(
                    raw_extraction, file_path, extraction_time
                )

            finally:
                # 4. Cleanup uploaded file
                self._cleanup_file(uploaded_file)

        except json.JSONDecodeError as e:
            # JSON parse failure - Gemini returned invalid JSON
            logger.error(
                f"Gemini returned invalid JSON for {file_path.name}: {e}. "
                "This may happen with very large documents or truncated responses."
            )
            if self.gemini_config.fallback_to_unstructured:
                return self._fallback_to_unstructured(file_path, f"Gemini returned invalid JSON: {e}")
            raise RuntimeError(f"Gemini returned invalid JSON: {e}") from e

        except (RuntimeError, ValueError) as e:
            # Expected extraction errors
            logger.error(f"Gemini extraction failed for {file_path.name}: {e}")
            if self.gemini_config.fallback_to_unstructured:
                return self._fallback_to_unstructured(file_path, str(e))
            raise

        except Exception as e:
            # Unexpected error - log with traceback
            logger.error(
                f"Unexpected error in Gemini extraction for {file_path.name}: "
                f"{type(e).__name__}: {e}",
                exc_info=True
            )
            if self.gemini_config.fallback_to_unstructured:
                return self._fallback_to_unstructured(file_path, f"Unexpected error: {type(e).__name__}: {e}")
            raise RuntimeError(f"Gemini extraction failed: {e}") from e

    def _upload_document(self, file_path: Path) -> genai.types.File:
        """Upload document using Gemini File API."""
        logger.info(f"Uploading PDF to Gemini: {file_path}")

        uploaded_file = genai.upload_file(str(file_path))
        logger.debug(f"File ID: {uploaded_file.name}")

        # Wait for processing
        max_wait = 120  # 2 minutes max
        waited = 0
        while uploaded_file.state.name == "PROCESSING":
            if waited >= max_wait:
                raise RuntimeError(f"File processing timeout after {max_wait}s")
            logger.debug("Processing...")
            time.sleep(2)
            waited += 2
            uploaded_file = genai.get_file(uploaded_file.name)

        if uploaded_file.state.name == "FAILED":
            raise RuntimeError(f"File processing failed: {uploaded_file.state.name}")

        logger.info(f"File ready: {uploaded_file.uri}")
        return uploaded_file

    def _extract_with_gemini(self, uploaded_file: genai.types.File) -> dict:
        """Run extraction with Gemini model."""
        model = genai.GenerativeModel(
            self.model_id,
            generation_config={
                "temperature": self.gemini_config.temperature,
                "max_output_tokens": self.gemini_config.max_output_tokens,
                "response_mime_type": "application/json",
            },
        )

        logger.info(f"Generating extraction with {self.model_id}...")
        response = model.generate_content([uploaded_file, EXTRACTION_PROMPT])

        # Parse JSON response
        result = json.loads(response.text)

        # Log token usage
        if hasattr(response, "usage_metadata"):
            logger.info(
                f"Tokens: prompt={response.usage_metadata.prompt_token_count}, "
                f"output={response.usage_metadata.candidates_token_count}"
            )

        return result

    def _cleanup_file(self, uploaded_file: genai.types.File) -> None:
        """Delete uploaded file from Gemini API to prevent storage accumulation."""
        try:
            genai.delete_file(uploaded_file.name)
            logger.debug("File deleted from Gemini API")
        except Exception as e:
            logger.warning(
                f"Failed to delete uploaded file '{uploaded_file.name}' from Gemini API: {e}. "
                "File may remain in your Gemini storage quota."
            )

    def _needs_chunking(self, file_path: Path) -> bool:
        """Check if a PDF needs chunked extraction based on file size."""
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        return (
            self.gemini_config.chunk_large_pdfs
            and file_size_mb > self.gemini_config.file_size_threshold_mb
        )

    def _get_page_count(self, file_path: Path) -> int:
        """Get the number of pages in a PDF."""
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(file_path))
            return len(reader.pages)
        except ImportError:
            logger.warning("pypdf not installed, cannot get page count")
            return 0

    def _split_pdf(self, file_path: Path) -> List[Path]:
        """
        Split a large PDF into smaller chunks.

        Args:
            file_path: Path to the original PDF

        Returns:
            List of paths to chunk PDF files (temporary files)
        """
        try:
            from pypdf import PdfReader, PdfWriter
        except ImportError:
            raise ImportError("pypdf required for chunked extraction. Install with: pip install pypdf")

        import tempfile

        reader = PdfReader(str(file_path))
        total_pages = len(reader.pages)
        chunk_size = self.gemini_config.max_pages_per_chunk

        chunk_paths = []
        for start_page in range(0, total_pages, chunk_size):
            end_page = min(start_page + chunk_size, total_pages)

            # Create writer for this chunk
            writer = PdfWriter()
            for page_num in range(start_page, end_page):
                writer.add_page(reader.pages[page_num])

            # Save to temporary file
            chunk_path = Path(tempfile.mktemp(suffix=f"_chunk_{start_page+1}-{end_page}.pdf"))
            with open(chunk_path, "wb") as f:
                writer.write(f)

            chunk_paths.append(chunk_path)
            logger.info(f"Created chunk: pages {start_page+1}-{end_page} → {chunk_path.name}")

        logger.info(f"Split {file_path.name} into {len(chunk_paths)} chunks ({total_pages} pages total)")
        return chunk_paths

    def _extract_chunked(self, file_path: Path) -> ExtractedDocument:
        """
        Extract a large PDF by splitting into chunks, processing each, and merging results.

        Args:
            file_path: Path to large PDF

        Returns:
            ExtractedDocument with merged sections from all chunks
        """
        start_time = time.time()
        total_pages = self._get_page_count(file_path)
        logger.info(f"Starting chunked extraction of {file_path.name} ({total_pages} pages)")

        # Split PDF into chunks
        chunk_paths = self._split_pdf(file_path)

        all_sections: List[Dict] = []
        document_meta: Dict = {}

        try:
            for i, chunk_path in enumerate(chunk_paths):
                logger.info(f"Processing chunk {i+1}/{len(chunk_paths)}: {chunk_path.name}")

                try:
                    # Upload chunk
                    uploaded_file = self._upload_document(chunk_path)

                    try:
                        # Extract from chunk
                        raw_extraction = self._extract_with_gemini(uploaded_file)

                        # Store document metadata from first chunk
                        if not document_meta and raw_extraction.get("document"):
                            document_meta = raw_extraction["document"]

                        # Collect sections with chunk offset for section_id uniqueness
                        chunk_sections = raw_extraction.get("sections", [])
                        for sec in chunk_sections:
                            # Add chunk prefix to section_id to avoid collisions
                            original_id = sec.get("section_id", "")
                            sec["section_id"] = f"c{i+1}_{original_id}"
                            # Update parent_id references
                            if sec.get("parent_id"):
                                sec["parent_id"] = f"c{i+1}_{sec['parent_id']}"
                            all_sections.append(sec)

                        logger.info(f"Chunk {i+1}: {len(chunk_sections)} sections extracted")

                    finally:
                        self._cleanup_file(uploaded_file)

                except Exception as e:
                    logger.error(f"Failed to process chunk {i+1}: {e}")
                    # Continue with other chunks

        finally:
            # Cleanup temporary chunk files
            for chunk_path in chunk_paths:
                try:
                    chunk_path.unlink()
                except Exception:
                    pass

        # Merge all sections into a single extraction
        merged_raw = {
            "document": document_meta,
            "sections": all_sections
        }

        extraction_time = time.time() - start_time
        logger.info(
            f"Chunked extraction complete: {len(all_sections)} sections "
            f"from {len(chunk_paths)} chunks in {extraction_time:.1f}s"
        )

        return self._convert_to_extracted_document(merged_raw, file_path, extraction_time)

    def _convert_to_extracted_document(
        self, raw: dict, file_path: Path, extraction_time: float
    ) -> ExtractedDocument:
        """
        Convert Gemini JSON output to ExtractedDocument format.

        Performs the following transformations:
        1. Extracts document metadata (type, title, date, etc.)
        2. Converts raw sections to DocumentSection objects
        3. Resolves parent_id from parent_number using section lookup
        4. Calculates char_start/char_end offsets for each section
        5. Populates children_ids based on parent relationships
        6. Generates markdown representation

        Args:
            raw: Raw JSON dict from Gemini extraction
            file_path: Path to source document
            extraction_time: Time taken for extraction in seconds

        Returns:
            ExtractedDocument with all sections and metadata
        """
        doc_meta = raw.get("document", {})
        raw_sections = raw.get("sections", [])

        # Build section ID to index mapping for parent lookup
        section_id_to_idx: Dict[str, int] = {}
        for i, sec in enumerate(raw_sections):
            section_id_to_idx[sec.get("section_id", f"sec_{i+1}")] = i

        # Build parent_number to section_id mapping
        number_to_section_id: Dict[str, str] = {}
        for sec in raw_sections:
            num = sec.get("number")
            if num:
                number_to_section_id[num] = sec.get("section_id", "")

        # Convert sections
        sections: List[DocumentSection] = []
        char_offset = 0

        for i, sec in enumerate(raw_sections):
            section_id = sec.get("section_id", f"sec_{i+1}")
            title = sec.get("title") or ""
            content = sec.get("content") or ""
            level = sec.get("level", 1)
            path = sec.get("path", "")
            page_number = sec.get("page_number") or 0
            element_type = sec.get("element_type", "unknown")
            parent_number = sec.get("parent_number")
            summary = sec.get("summary")  # Section summary from Gemini

            # Compute parent_id from parent_number
            parent_id = None
            if parent_number and parent_number in number_to_section_id:
                parent_id = number_to_section_id[parent_number]

            # Compute depth from level (in legal docs, depth often equals level)
            depth = sec.get("depth", level)

            # Build ancestors from path
            path_parts = path.split(" > ") if path else []
            ancestors = path_parts[:-1] if len(path_parts) > 1 else []

            # Content length
            content_length = len(content)
            char_end = char_offset + content_length

            section = DocumentSection(
                section_id=section_id,
                title=title,
                content=content,
                level=level,
                depth=depth,
                parent_id=parent_id,
                children_ids=[],  # Will be populated below
                ancestors=ancestors,
                path=path,
                page_number=page_number,
                char_start=char_offset,
                char_end=char_end,
                content_length=content_length,
                element_type=element_type,
                element_category=element_type,  # Use element_type as category
                summary=summary,  # Section summary from Gemini
            )

            sections.append(section)
            char_offset = char_end + 2  # +2 for \n\n separator

        # Populate children_ids
        for section in sections:
            if section.parent_id:
                # Find parent by section_id
                for parent in sections:
                    if parent.section_id == section.parent_id:
                        parent.children_ids.append(section.section_id)
                        break

        # Build full text and markdown
        full_text = "\n\n".join(
            f"{sec.title}\n{sec.content}" if sec.title else sec.content
            for sec in sections
            if sec.content
        )

        markdown = self._generate_markdown(sections)

        # Calculate stats
        hierarchy_depth = max((s.depth for s in sections), default=0)
        num_roots = sum(1 for s in sections if s.depth == 1 or s.level == 1)

        return ExtractedDocument(
            document_id=doc_meta.get("identifier", file_path.stem),
            source_path=str(file_path),
            extraction_time=extraction_time,
            full_text=full_text,
            markdown=markdown,
            json_content=json.dumps(raw, ensure_ascii=False),
            sections=sections,
            hierarchy_depth=hierarchy_depth,
            num_roots=num_roots,
            tables=[],  # Gemini extraction doesn't extract tables separately
            num_pages=max((s.page_number for s in sections), default=0),
            num_sections=len(sections),
            num_tables=0,
            total_chars=len(full_text),
            title=doc_meta.get("title"),
            document_summary=doc_meta.get("summary"),  # Document summary from Gemini
            extraction_method=f"gemini_{self.model_id}",
            config={
                "model": self.model_id,
                "document_type": doc_meta.get("type"),
                "document_date": doc_meta.get("date"),
                "document_language": doc_meta.get("language"),
            },
        )

    def _generate_markdown(self, sections: List[DocumentSection]) -> str:
        """Generate markdown from sections."""
        lines = []

        for section in sections:
            # Add heading based on level
            if section.title:
                heading_level = min(section.level, 6)
                lines.append("#" * heading_level + " " + section.title)

            # Add content
            if section.content and section.content != section.title:
                lines.append(section.content)

            lines.append("")  # Empty line

        return "\n".join(lines)


def get_extractor(
    config: Optional[ExtractionConfig] = None, backend: str = "auto"
) -> Union["GeminiExtractor", "UnstructuredExtractor"]:
    """
    Factory function to get appropriate extractor based on backend setting.

    Args:
        config: ExtractionConfig for compatibility
        backend: Backend selection:
            - "gemini": Force Gemini extractor (requires GOOGLE_API_KEY)
            - "unstructured": Force Unstructured extractor
            - "auto": Use Gemini if GOOGLE_API_KEY available, else Unstructured

    Returns:
        GeminiExtractor or UnstructuredExtractor instance

    Raises:
        ValueError: If backend="gemini" but GOOGLE_API_KEY is not set

    Example:
        >>> extractor = get_extractor(backend="auto")
        >>> doc = extractor.extract(Path("document.pdf"))
    """
    from src.unstructured_extractor import UnstructuredExtractor

    if backend == "gemini":
        return GeminiExtractor(config)

    elif backend == "unstructured":
        return UnstructuredExtractor(config)

    else:  # "auto"
        # Check if GOOGLE_API_KEY is available
        if os.getenv("GOOGLE_API_KEY"):
            try:
                return GeminiExtractor(config)
            except (ValueError, RuntimeError) as e:
                # Expected errors (API key issues, initialization failures)
                logger.warning(f"Gemini extractor unavailable: {e}. Using Unstructured.")
            except Exception as e:
                # Unexpected error - log with more detail
                logger.warning(
                    f"Gemini extractor failed unexpectedly: {type(e).__name__}: {e}. "
                    "Using Unstructured."
                )

        return UnstructuredExtractor(config)
