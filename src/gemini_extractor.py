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
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import google.generativeai as genai
from dotenv import load_dotenv

from src.exceptions import APIKeyError
from src.unstructured_extractor import (
    DocumentSection,
    ExtractedDocument,
    ExtractionConfig,
    TableData,
)

load_dotenv()

logger = logging.getLogger(__name__)


def _repair_truncated_json(json_str: str) -> dict:
    """
    SOTA JSON repair for truncated/malformed Gemini API responses.

    Uses multi-strategy approach with json_repair library as primary method:
    1. Standard JSON parsing (fast path)
    2. json_repair library - SOTA streaming-aware repair with 30+ heuristics
    3. State machine extraction of complete section objects
    4. Progressive truncation recovery with content-aware reconstruction
    5. Bracket balancing with string-context awareness

    Returns parsed dict with recovered data, or raises if all strategies fail.
    """
    import re
    from json_repair import repair_json

    # Strategy 0: Standard parse (fast path)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        error_pos = e.pos
        error_msg = str(e)
        logger.warning(f"JSON parse failed at char {error_pos}: {error_msg[:100]}")

    # Strategy 1: json_repair library (SOTA - handles LLM output patterns)
    # Handles: unterminated strings, missing commas, trailing commas,
    # unquoted keys, single quotes, control chars, incomplete structures
    try:
        repaired_str = repair_json(json_str, return_objects=False)
        result = json.loads(repaired_str)
        logger.info("JSON repair: json_repair library succeeded")
        return result
    except Exception as e:
        logger.debug(f"json_repair library failed: {e}")

    # Strategy 2: Extract complete section objects via state machine
    # For document extraction JSON: {"document": {...}, "sections": [...]}
    sections_match = re.search(r'"sections"\s*:\s*\[', json_str)
    if sections_match:
        complete_sections = _extract_complete_json_objects(
            json_str, sections_match.end()
        )
        if complete_sections:
            logger.info(f"JSON repair: recovered {len(complete_sections)} complete sections")
            # Reconstruct with document metadata
            doc_obj = _extract_document_metadata(json_str)
            reconstructed = {
                "document": doc_obj,
                "sections": complete_sections
            }
            return reconstructed

    # Strategy 3: Progressive truncation point detection
    # Find the last valid JSON structure before truncation
    last_valid = _find_last_valid_json_point(json_str)
    if last_valid:
        try:
            result = json.loads(last_valid)
            logger.info("JSON repair: truncation point recovery succeeded")
            return result
        except json.JSONDecodeError:
            pass

    # Strategy 4: Bracket balancing with string-context awareness
    try:
        repaired = _balance_json_structure(json_str)
        result = json.loads(repaired)
        logger.info("JSON repair: bracket balancing succeeded")
        return result
    except json.JSONDecodeError as e:
        logger.error(f"All JSON repair strategies failed. Last error: {e}")
        # Log a snippet around the error for debugging
        start = max(0, error_pos - 50)
        end = min(len(json_str), error_pos + 50)
        logger.error(f"Context around error: ...{json_str[start:end]}...")
        raise ValueError(f"JSON repair failed after all strategies: {error_msg}")


def _extract_complete_json_objects(json_str: str, start_pos: int) -> List[dict]:
    """
    Extract complete JSON objects from an array using a proper state machine.

    Handles nested objects, arrays, and properly escaped strings.
    Returns list of parsed section dicts.
    """
    complete_sections = []
    obj_start = None
    brace_depth = 0
    in_string = False
    escape_next = False

    for i, char in enumerate(json_str[start_pos:], start=start_pos):
        if escape_next:
            escape_next = False
            continue
        if char == '\\' and in_string:
            escape_next = True
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue

        if char == '{':
            if brace_depth == 0:
                obj_start = i
            brace_depth += 1
        elif char == '}':
            brace_depth -= 1
            if brace_depth == 0 and obj_start is not None:
                obj_str = json_str[obj_start:i+1]
                try:
                    # Try to parse this object
                    obj = json.loads(obj_str)
                    complete_sections.append(obj)
                except json.JSONDecodeError as parse_err:
                    # Try json_repair on individual object
                    try:
                        from json_repair import repair_json
                        repaired = repair_json(obj_str, return_objects=True)
                        if isinstance(repaired, dict):
                            complete_sections.append(repaired)
                        else:
                            logger.debug(
                                f"JSON repair returned non-dict for section at pos {obj_start}: "
                                f"got {type(repaired).__name__}"
                            )
                    except (ImportError, ValueError, TypeError) as repair_err:
                        logger.debug(
                            f"Failed to repair section object at pos {obj_start}: "
                            f"{type(repair_err).__name__}: {repair_err}. Skipping section."
                        )
                obj_start = None
        elif char == ']' and brace_depth == 0:
            break

    return complete_sections


def _extract_document_metadata(json_str: str) -> dict:
    """
    Extract document metadata object from JSON string.

    Handles nested objects within document metadata.
    """
    import re
    from json_repair import repair_json

    # Try to find and extract the document object
    doc_match = re.search(r'"document"\s*:\s*\{', json_str)
    if not doc_match:
        return {}

    start_pos = doc_match.end() - 1  # Include the opening brace
    brace_depth = 0
    in_string = False
    escape_next = False

    for i, char in enumerate(json_str[start_pos:], start=start_pos):
        if escape_next:
            escape_next = False
            continue
        if char == '\\' and in_string:
            escape_next = True
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue

        if char == '{':
            brace_depth += 1
        elif char == '}':
            brace_depth -= 1
            if brace_depth == 0:
                doc_str = json_str[start_pos:i+1]
                try:
                    return json.loads(doc_str)
                except json.JSONDecodeError as e:
                    logger.debug(f"Direct JSON parse of document metadata failed: {e}")
                    try:
                        return repair_json(doc_str, return_objects=True)
                    except (ImportError, ValueError, TypeError) as repair_err:
                        logger.warning(
                            f"Failed to extract document metadata: "
                            f"{type(repair_err).__name__}: {repair_err}. "
                            "Document will use default metadata."
                        )
                break

    return {}


def _find_last_valid_json_point(json_str: str) -> Optional[str]:
    """
    Find the last point where the JSON was valid by searching for
    complete object/array boundaries before truncation.
    """
    # Try progressively shorter substrings ending at potential valid points
    potential_ends = []

    # Find all closing braces/brackets not inside strings
    in_string = False
    escape_next = False
    for i, char in enumerate(json_str):
        if escape_next:
            escape_next = False
            continue
        if char == '\\' and in_string:
            escape_next = True
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if char in '}]':
            potential_ends.append(i)

    # Try from the end, looking for valid JSON
    for end_pos in reversed(potential_ends[-20:]):  # Check last 20 potential points
        candidate = json_str[:end_pos + 1]
        # Quick bracket check
        if candidate.count('{') == candidate.count('}') and \
           candidate.count('[') == candidate.count(']'):
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                continue

    return None


def _balance_json_structure(json_str: str) -> str:
    """
    Balance JSON structure with string-context awareness.

    More sophisticated than simple bracket counting - tracks actual
    nesting context to produce valid JSON.
    """
    repaired = json_str.rstrip()

    # Track string context to find unterminated strings
    in_string = False
    escape_next = False
    last_string_start = -1
    string_depth_stack = []  # Track nested structure when string started

    brace_depth = 0
    bracket_depth = 0

    for i, char in enumerate(repaired):
        if escape_next:
            escape_next = False
            continue
        if char == '\\' and in_string:
            escape_next = True
            continue
        if char == '"':
            if in_string:
                in_string = False
            else:
                in_string = True
                last_string_start = i
                string_depth_stack = [brace_depth, bracket_depth]
            continue
        if in_string:
            continue

        if char == '{':
            brace_depth += 1
        elif char == '}':
            brace_depth -= 1
        elif char == '[':
            bracket_depth += 1
        elif char == ']':
            bracket_depth -= 1

    # Handle unterminated string
    if in_string:
        # Try to close the string at a sensible point
        # Look for newline or obvious truncation
        if last_string_start > 0:
            # Truncate at the string and close it
            repaired = repaired[:last_string_start] + '""'
            # Restore depths to when string started
            brace_depth, bracket_depth = string_depth_stack if string_depth_stack else (0, 0)
            # Recalculate needed closures
            remaining = repaired[last_string_start + 2:]
            for char in remaining:
                if char == '{':
                    brace_depth += 1
                elif char == '}':
                    brace_depth -= 1
                elif char == '[':
                    bracket_depth += 1
                elif char == ']':
                    bracket_depth -= 1

    # Close any open structures
    if bracket_depth > 0:
        repaired += ']' * bracket_depth
    if brace_depth > 0:
        repaired += '}' * brace_depth

    return repaired


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
   - **POVINNĚ extrahuj content** - každá sekce MUSÍ mít neprázdný content (text pod nadpisem)
   - Pokud nadpis nemá vlastní text, použij první odstavec pod ním jako content
   - content je KRITICKÝ pro další zpracování - prázdný content = neúplná extrakce

7. **summary** = stručné shrnutí (max 200 znaků) pro KAŽDOU sekci
   - Piš GENERICKY srozumitelně (ne právnický žargon)
   - Summary je POVINNÉ - pomáhá při vyhledávání
   - Příklad: "Definuje odpovědnost provozovatele za škody způsobené jadernou havárií"

8. **document_summary** = shrnutí celého dokumentu (max 500 znaků)
   - Hlavní téma a účel dokumentu
   - Genericky srozumitelné pro laiky

9. **page_number** = FYZICKÉ číslo stránky v PDF (1-based od začátku souboru)
   - NEPOUŽÍVEJ tištěné číslo stránky ze zápatí/záhlaví dokumentu
   - První stránka PDF souboru = page_number: 1
   - Pokud sekce začíná na stránce 5 PDF souboru → page_number: 5
   - Pokud není stránka jasná, vynech pole page_number (neuvádej null ani 0)
   - page_number je POVINNÉ pro každou sekci kde je stránka zjistitelná

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
    {"section_id": "sec_1", "element_type": "cast", "number": "I", "title": "OBECNÁ USTANOVENÍ", "content": "Tato část stanoví základní pojmy a principy pro využívání jaderné energie.", "level": 1, "path": "ČÁST I", "page_number": 3, "summary": "Definuje základní pojmy a principy atomového zákona."},
    {"section_id": "sec_2", "element_type": "hlava", "number": "PÁTÁ", "title": "OBČANSKOPRÁVNÍ ODPOVĚDNOST ZA JADERNÉ ŠKODY", "content": "Tato hlava upravuje odpovědnost za škody způsobené jadernou havárií.", "level": 2, "path": "ČÁST I > HLAVA PÁTÁ", "parent_number": "I", "page_number": 15, "summary": "Definuje pravidla odpovědnosti za jaderné škody."},
    {"section_id": "sec_3", "element_type": "paragraf", "number": "32", "title": "Odpovědnost provozovatele", "content": "Provozovatel jaderného zařízení odpovídá za škody způsobené jadernou havárií.", "level": 4, "path": "ČÁST I > HLAVA PÁTÁ > § 32", "parent_number": "PÁTÁ", "page_number": 15, "summary": "Stanoví odpovědnost provozovatele."},
    {"section_id": "sec_4", "element_type": "odstavec", "number": "1", "content": "Pro účely občanskoprávní odpovědnosti za jaderné škody se použijí ustanovení mezinárodní smlouvy,26) kterou je Česká republika vázána.", "level": 5, "path": "ČÁST I > HLAVA PÁTÁ > § 32 > (1)", "parent_number": "32", "page_number": 15, "summary": "Odkazuje na mezinárodní smlouvu pro řešení odpovědnosti za jaderné škody."}
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
        temperature: Generation temperature 0.0-2.0 (default: 0.1 for deterministic output)
        max_output_tokens: Maximum output tokens (default: 65536 for large documents)
        fallback_to_unstructured: Fall back to Unstructured on Gemini failure (default: True)
        chunk_large_pdfs: Enable chunked extraction for large PDFs (default: True)
        max_pages_per_chunk: Maximum pages per chunk when splitting (default: 10)
        file_size_threshold_mb: File size threshold for chunked extraction (default: 10.0 MB)
        parallel_chunks: Number of chunks to process in parallel (default: 6)
        rpm_limit: Requests per minute limit for rate limiting (default: 6)
        batch_wait_seconds: Seconds to wait between batches for rate limiting (default: 60)
    """

    model: str = DEFAULT_MODEL
    temperature: float = 0.1
    max_output_tokens: int = 65536
    fallback_to_unstructured: bool = True
    chunk_large_pdfs: bool = True
    max_pages_per_chunk: int = 10
    file_size_threshold_mb: float = 10.0
    parallel_chunks: int = 6
    rpm_limit: int = 6
    batch_wait_seconds: int = 60

    def __post_init__(self):
        """Validate configuration parameters."""
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"temperature must be in [0.0, 2.0], got {self.temperature}")
        if self.max_output_tokens <= 0:
            raise ValueError(f"max_output_tokens must be positive, got {self.max_output_tokens}")
        if self.max_pages_per_chunk <= 0:
            raise ValueError(f"max_pages_per_chunk must be positive, got {self.max_pages_per_chunk}")
        if self.file_size_threshold_mb <= 0:
            raise ValueError(f"file_size_threshold_mb must be positive, got {self.file_size_threshold_mb}")
        if self.parallel_chunks <= 0:
            raise ValueError(f"parallel_chunks must be positive, got {self.parallel_chunks}")
        if self.rpm_limit <= 0:
            raise ValueError(f"rpm_limit must be positive, got {self.rpm_limit}")
        if self.batch_wait_seconds < 0:
            raise ValueError(f"batch_wait_seconds must be non-negative, got {self.batch_wait_seconds}")


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

        # Create GeminiExtractionConfig from ExtractionConfig if not provided
        if gemini_config is not None:
            self.gemini_config = gemini_config
        elif config is not None:
            # Read gemini-specific settings from ExtractionConfig
            self.gemini_config = GeminiExtractionConfig(
                model=getattr(config, 'gemini_model', DEFAULT_MODEL),
                max_output_tokens=getattr(config, 'gemini_max_output_tokens', 65536),
                file_size_threshold_mb=getattr(config, 'gemini_file_size_threshold_mb', 10.0),
                fallback_to_unstructured=getattr(config, 'gemini_fallback_to_unstructured', True),
            )
        else:
            self.gemini_config = GeminiExtractionConfig()

        # Configure Gemini API
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise APIKeyError(
                "GOOGLE_API_KEY not found in environment. "
                "Set it in .env file or export GOOGLE_API_KEY=...",
                details={"component": "GeminiExtractor"}
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

    def _normalize_sections_list(self, sections: list) -> List[Dict[str, Any]]:
        """
        Flatten and filter sections to ensure all items are dicts.

        Gemini sometimes returns nested lists instead of flat dicts:
        - Expected: [{"section_id": "1"}, {"section_id": "2"}]
        - Received: [[{"section_id": "1"}], [{"section_id": "2"}]]

        This method handles up to 3 levels of nesting.

        Args:
            sections: Raw sections list from Gemini output

        Returns:
            Flat list of section dicts
        """
        normalized = []
        for item in sections:
            if isinstance(item, dict):
                normalized.append(item)
            elif isinstance(item, list):
                for subitem in item:
                    if isinstance(subitem, dict):
                        normalized.append(subitem)
                    elif isinstance(subitem, list):
                        # Double-nested - rare but handle it
                        for subsubitem in subitem:
                            if isinstance(subsubitem, dict):
                                normalized.append(subsubitem)
            # Skip non-dict, non-list items (strings, numbers, etc.)
        return normalized

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

                # 3. Get page count for validation
                total_pages = self._get_page_count(file_path)

                # 4. Convert to ExtractedDocument format
                extraction_time = time.time() - start_time
                return self._convert_to_extracted_document(
                    raw_extraction, file_path, extraction_time, total_pages
                )

            finally:
                # 5. Cleanup uploaded file
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

    def _extract_with_gemini(
        self, uploaded_file: genai.types.File, prompt: Optional[str] = None
    ) -> dict:
        """Run extraction with Gemini model.

        Args:
            uploaded_file: The uploaded file to extract from
            prompt: Optional custom prompt (defaults to EXTRACTION_PROMPT)
        """
        model = genai.GenerativeModel(
            self.model_id,
            generation_config={
                "temperature": self.gemini_config.temperature,
                "max_output_tokens": self.gemini_config.max_output_tokens,
                "response_mime_type": "application/json",
            },
        )

        extraction_prompt = prompt if prompt is not None else EXTRACTION_PROMPT
        logger.info(f"Generating extraction with {self.model_id}...")
        response = model.generate_content([uploaded_file, extraction_prompt])

        # Parse JSON response with SOTA repair for truncated/malformed output
        result = _repair_truncated_json(response.text)

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

            # Save to temporary file (use NamedTemporaryFile to avoid race conditions)
            with tempfile.NamedTemporaryFile(
                suffix=f"_chunk_{start_page+1}-{end_page}.pdf",
                delete=False
            ) as tmp:
                chunk_path = Path(tmp.name)
                writer.write(tmp)

            chunk_paths.append(chunk_path)
            logger.info(f"Created chunk: pages {start_page+1}-{end_page} → {chunk_path.name}")

        logger.info(f"Split {file_path.name} into {len(chunk_paths)} chunks ({total_pages} pages total)")
        return chunk_paths

    def _process_single_chunk(
        self, chunk_path: Path, chunk_index: int, page_offset: int = 0
    ) -> Tuple[int, Dict, List[Dict]]:
        """
        Process a single PDF chunk and return extracted sections.

        Args:
            chunk_path: Path to the temporary chunk PDF
            chunk_index: Index of the chunk (for logging and section_id prefixing)
            page_offset: Page number offset to add to all page_number values
                        (e.g., if chunk contains pages 11-20, offset should be 10)

        Returns:
            Tuple of (chunk_index, document_meta, sections_list)
        """
        logger.info(f"Processing chunk {chunk_index + 1}: {chunk_path.name} (page_offset={page_offset})")

        try:
            # Upload chunk
            uploaded_file = self._upload_document(chunk_path)

            try:
                # Extract from chunk with chunk-specific prompt
                # Tell Gemini this is a chunk and to use 1-based page numbers within the chunk
                max_pages = self.gemini_config.max_pages_per_chunk
                chunk_prompt = f"""DŮLEŽITÉ: Zpracováváš ČÁST většího dokumentu (chunk {chunk_index + 1}).
Tento chunk obsahuje stránky {page_offset + 1} až {page_offset + max_pages} původního dokumentu.

Pro page_number uvádějte čísla RELATIVNĚ k tomuto chunku:
- První stránka tohoto chunku = page_number: 1
- Druhá stránka tohoto chunku = page_number: 2
- atd.

{EXTRACTION_PROMPT}"""
                raw_extraction = self._extract_with_gemini(uploaded_file, prompt=chunk_prompt)

                # Handle case where Gemini returns a list instead of dict
                if isinstance(raw_extraction, list):
                    # Wrap list in standard structure - assume it's sections
                    raw_extraction = {"document": {}, "sections": raw_extraction}

                document_meta = raw_extraction.get("document", {})

                # Collect sections with chunk offset for section_id uniqueness
                chunk_sections = raw_extraction.get("sections", [])

                # Normalize sections - Gemini may return nested lists
                chunk_sections = self._normalize_sections_list(chunk_sections)

                # Add chunk prefix to section_id and fix page_number offset
                for sec in chunk_sections:
                    original_id = sec.get("section_id", "")
                    sec["section_id"] = f"c{chunk_index + 1}_{original_id}"
                    # Update parent_id references
                    if sec.get("parent_id"):
                        sec["parent_id"] = f"c{chunk_index + 1}_{sec['parent_id']}"

                    # Fix page_number by adding offset
                    # Gemini returns page_number relative to the chunk (1-based)
                    # We need to convert to absolute page number in original document
                    if "page_number" in sec and sec["page_number"] is not None:
                        sec["page_number"] = sec["page_number"] + page_offset

                logger.info(f"Chunk {chunk_index + 1}: {len(chunk_sections)} sections extracted")
                return (chunk_index, document_meta, chunk_sections)

            finally:
                self._cleanup_file(uploaded_file)

        except Exception as e:
            logger.error(f"Failed to process chunk {chunk_index + 1}: {e}")
            return (chunk_index, {}, [])

    def _extract_chunked(self, file_path: Path) -> ExtractedDocument:
        """
        Extract a large PDF by splitting into chunks, processing in parallel batches.

        Uses batch-based parallelism: processes `parallel_chunks` chunks at once,
        waits for all to complete, then waits `batch_wait_seconds` before next batch
        to respect Gemini API rate limits.

        Args:
            file_path: Path to large PDF

        Returns:
            ExtractedDocument with merged sections from all chunks

        Raises:
            RuntimeError: If more than 50% of chunks fail to extract
        """
        start_time = time.time()
        total_pages = self._get_page_count(file_path)
        logger.info(f"Starting chunked extraction of {file_path.name} ({total_pages} pages)")

        # Split PDF into chunks
        chunk_paths = self._split_pdf(file_path)
        total_chunks = len(chunk_paths)
        batch_size = self.gemini_config.parallel_chunks

        # Results storage: index -> (document_meta, sections)
        results: Dict[int, Tuple[Dict, List[Dict]]] = {}
        document_meta: Dict = {}
        failed_chunks: List[int] = []  # Track failed chunk indices

        # Thread lock for document_meta access
        meta_lock = threading.Lock()

        try:
            # Process chunks in batches
            for batch_start in range(0, total_chunks, batch_size):
                batch_end = min(batch_start + batch_size, total_chunks)
                batch_chunks = chunk_paths[batch_start:batch_end]
                batch_num = (batch_start // batch_size) + 1
                total_batches = (total_chunks + batch_size - 1) // batch_size

                logger.info(
                    f"Processing batch {batch_num}/{total_batches}: "
                    f"chunks {batch_start + 1}-{batch_end} of {total_chunks}"
                )

                # Process batch in parallel using ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=batch_size) as executor:
                    # Submit all chunks in this batch
                    # Calculate page_offset for each chunk:
                    # chunk 0: pages 1-10, offset=0
                    # chunk 1: pages 11-20, offset=10
                    # chunk N: pages (N*max_pages+1)-(N+1)*max_pages, offset=N*max_pages
                    max_pages = self.gemini_config.max_pages_per_chunk
                    futures = {
                        executor.submit(
                            self._process_single_chunk,
                            chunk_path,
                            batch_start + i,
                            (batch_start + i) * max_pages  # page_offset
                        ): batch_start + i
                        for i, chunk_path in enumerate(batch_chunks)
                    }

                    # Collect results as they complete
                    for future in as_completed(futures):
                        chunk_idx = futures[future]
                        try:
                            idx, meta, sections = future.result()
                            results[idx] = (meta, sections)

                            # Thread-safe document metadata update
                            with meta_lock:
                                if not document_meta and meta:
                                    document_meta = meta

                            # Track chunks that returned empty results
                            if not sections:
                                failed_chunks.append(chunk_idx)
                                logger.warning(f"Chunk {chunk_idx + 1} returned no sections")

                        except Exception as e:
                            logger.error(f"Chunk {chunk_idx + 1} future failed: {e}")
                            results[chunk_idx] = ({}, [])
                            failed_chunks.append(chunk_idx)

                # Wait before next batch (unless this is the last batch)
                if batch_end < total_chunks:
                    wait_time = self.gemini_config.batch_wait_seconds
                    logger.info(f"Batch {batch_num} complete. Waiting {wait_time}s for rate limit...")
                    time.sleep(wait_time)

        finally:
            # Cleanup temporary chunk files with logging for failures
            cleanup_failures = []
            for chunk_path in chunk_paths:
                try:
                    chunk_path.unlink()
                except OSError as e:
                    cleanup_failures.append((chunk_path, e))

            if cleanup_failures:
                logger.warning(
                    f"Failed to cleanup {len(cleanup_failures)} temporary chunk files. "
                    f"Manual cleanup may be needed: {[str(p) for p, _ in cleanup_failures[:3]]}"
                )

        # Check for excessive failures
        if failed_chunks:
            failure_rate = len(failed_chunks) / total_chunks
            logger.error(
                f"CHUNK EXTRACTION ISSUES: {len(failed_chunks)}/{total_chunks} chunks "
                f"failed or returned empty results. Failed chunks: {failed_chunks}"
            )
            if failure_rate > 0.5:
                raise RuntimeError(
                    f"Chunked extraction failed: {len(failed_chunks)}/{total_chunks} chunks "
                    f"({failure_rate:.0%}) failed. Document extraction is incomplete."
                )

        # Merge all sections in order (by chunk index)
        all_sections: List[Dict] = []
        for i in range(total_chunks):
            if i in results:
                _, sections = results[i]
                all_sections.extend(sections)

        # Deduplicate sections with same path (TOC vs content pages)
        # When chunked extraction processes TOC pages and content pages separately,
        # we get duplicate sections - TOC has only titles, content has full text.
        # Keep the section with the longest content for each path.
        deduplicated_sections = self._deduplicate_sections_by_path(all_sections)

        merged_raw = {
            "document": document_meta,
            "sections": deduplicated_sections
        }

        extraction_time = time.time() - start_time
        success_rate = (total_chunks - len(failed_chunks)) / total_chunks if total_chunks > 0 else 0
        logger.info(
            f"Chunked extraction complete: {len(deduplicated_sections)} sections "
            f"(deduped from {len(all_sections)}) from {total_chunks} chunks in {extraction_time:.1f}s "
            f"(success rate: {success_rate:.0%}, parallel batches of {batch_size})"
        )

        return self._convert_to_extracted_document(merged_raw, file_path, extraction_time, total_pages)

    def _deduplicate_sections_by_path(self, sections: List[Dict]) -> List[Dict]:
        """
        Deduplicate sections with the same hierarchical path.

        When processing large PDFs in chunks, TOC pages (early chunks) produce sections
        with only titles as content, while content pages (later chunks) produce sections
        with the actual text. Both have the same hierarchical path.

        This method keeps the section with the longest content for each unique path,
        effectively merging TOC-only sections with their content counterparts.

        Args:
            sections: List of section dicts from all chunks

        Returns:
            Deduplicated list of sections, keeping longest content for each path
        """
        # Group sections by path
        by_path: Dict[str, List[Dict]] = {}
        for sec in sections:
            path = sec.get("path", "")
            if not path:
                # Keep sections without path as-is
                if "" not in by_path:
                    by_path[""] = []
                by_path[""].append(sec)
                continue

            if path not in by_path:
                by_path[path] = []
            by_path[path].append(sec)

        # Select best section for each path (longest content, but preserve highest page_number)
        deduplicated = []
        for path, secs in by_path.items():
            if len(secs) == 1:
                deduplicated.append(secs[0])
            else:
                # Multiple sections with same path - keep the one with longest content
                best = max(secs, key=lambda s: len(s.get("content", "")))

                # But use page_number from the section with highest page_number
                # (TOC sections have low page numbers, content sections have correct ones)
                max_page = max(s.get("page_number", 0) or 0 for s in secs)
                if max_page > 0:
                    best["page_number"] = max_page

                deduplicated.append(best)
                if len(secs) > 1:
                    logger.debug(
                        f"Deduplicated path '{path}': kept {len(best.get('content', ''))} chars, "
                        f"page={max_page}, removed {len(secs) - 1} duplicate(s)"
                    )

        # Sort by section_id to maintain order (c1_sec_1, c1_sec_2, c2_sec_1, etc.)
        def sort_key(sec: Dict) -> Tuple[int, int]:
            sid = sec.get("section_id", "")
            # Parse c{chunk}_sec_{num} format
            try:
                parts = sid.replace("c", "").replace("sec", "").split("_")
                parts = [p for p in parts if p]
                if len(parts) >= 2:
                    return (int(parts[0]), int(parts[1]))
                elif len(parts) == 1:
                    return (0, int(parts[0]))
            except (ValueError, IndexError):
                pass
            return (999, 0)

        deduplicated.sort(key=sort_key)

        # Renumber section_ids to be sequential
        for i, sec in enumerate(deduplicated, 1):
            sec["section_id"] = f"sec_{i}"

        return deduplicated

    def _convert_to_extracted_document(
        self, raw: dict, file_path: Path, extraction_time: float, total_pages: int = 0
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
        7. Validates and interpolates page numbers

        Args:
            raw: Raw JSON dict from Gemini extraction
            file_path: Path to source document
            extraction_time: Time taken for extraction in seconds
            total_pages: Total number of pages in the PDF (for validation)

        Returns:
            ExtractedDocument with all sections and metadata
        """
        doc_meta = raw.get("document", {})
        raw_sections = raw.get("sections", [])

        # Normalize sections - Gemini may return nested lists instead of dicts
        raw_sections = self._normalize_sections_list(raw_sections)

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

            # Robust type coercion - Gemini may return lists instead of strings
            def _to_str(val) -> str:
                """Convert value to string, joining lists with newlines."""
                if val is None:
                    return ""
                if isinstance(val, list):
                    return "\n".join(str(item) for item in val)
                return str(val)

            title = _to_str(sec.get("title"))
            content = _to_str(sec.get("content"))
            level = sec.get("level", 1)
            path = _to_str(sec.get("path"))

            # Validate page_number: must be positive and within PDF page count
            raw_page = sec.get("page_number")
            if raw_page is not None:
                try:
                    page_number = int(raw_page)
                    # Validate against total_pages if available
                    if total_pages > 0 and page_number > total_pages:
                        logger.warning(
                            f"Section {section_id} has invalid page_number {page_number} > {total_pages}, "
                            "will attempt interpolation"
                        )
                        page_number = None  # Mark for interpolation
                    elif page_number <= 0:
                        page_number = None  # Invalid, mark for interpolation
                except (ValueError, TypeError):
                    page_number = None  # Invalid value
            else:
                page_number = None  # Missing

            element_type = sec.get("element_type", "unknown")
            parent_number = sec.get("parent_number")
            summary = _to_str(sec.get("summary")) or None  # Section summary from Gemini

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

        # Interpolate missing page numbers from neighbors
        self._interpolate_page_numbers(sections, total_pages)

        # Build full text (markdown generation removed - not used in pipeline)
        full_text = "\n\n".join(
            f"{sec.title}\n{sec.content}" if sec.title else sec.content
            for sec in sections
            if sec.content
        )

        # Calculate stats
        hierarchy_depth = max((s.depth for s in sections), default=0)
        num_roots = sum(1 for s in sections if s.depth == 1 or s.level == 1)

        return ExtractedDocument(
            document_id=doc_meta.get("identifier", file_path.stem),
            source_path=str(file_path),
            extraction_time=extraction_time,
            full_text=full_text,
            markdown="",  # Not used in pipeline
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

    def _interpolate_page_numbers(
        self, sections: List[DocumentSection], total_pages: int
    ) -> None:
        """
        Interpolate missing page numbers based on surrounding sections.

        For sections with page_number=None, attempts to estimate the page number
        by looking at neighboring sections with valid page numbers.

        Args:
            sections: List of DocumentSection objects (modified in place)
            total_pages: Total number of pages in the PDF (for validation)
        """
        missing_count = sum(1 for s in sections if s.page_number is None)
        if missing_count == 0:
            return

        logger.info(f"Interpolating {missing_count} missing page numbers")

        for i, section in enumerate(sections):
            if section.page_number is not None:
                continue

            # Look for nearest neighbors with valid page numbers
            prev_page = None
            next_page = None

            # Search backwards for previous valid page
            for j in range(i - 1, -1, -1):
                if sections[j].page_number is not None:
                    prev_page = sections[j].page_number
                    break

            # Search forwards for next valid page
            for j in range(i + 1, len(sections)):
                if sections[j].page_number is not None:
                    next_page = sections[j].page_number
                    break

            # Interpolate based on available neighbors
            if prev_page is not None and next_page is not None:
                # Use average of neighbors
                section.page_number = (prev_page + next_page) // 2
            elif prev_page is not None:
                # Use previous page (assume same page or next)
                section.page_number = prev_page
            elif next_page is not None:
                # Use next page (assume same page or previous)
                section.page_number = max(1, next_page)
            else:
                # No neighbors with valid pages - use 1 as fallback
                section.page_number = 1

            # Final validation against total_pages
            if total_pages > 0 and section.page_number > total_pages:
                section.page_number = total_pages

        interpolated = sum(1 for s in sections if s.page_number is not None)
        logger.info(f"After interpolation: {interpolated}/{len(sections)} sections have page numbers")


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
            except (ImportError, ModuleNotFoundError) as e:
                # Missing dependencies
                logger.warning(f"Gemini SDK not available: {e}. Using Unstructured.")
            except (ValueError, RuntimeError) as e:
                # Expected errors (API key issues, initialization failures)
                logger.warning(f"Gemini extractor unavailable: {e}. Using Unstructured.")
            except (AttributeError, TypeError) as e:
                # Code bugs - log as error with traceback
                logger.error(
                    f"Code error in Gemini extractor initialization: "
                    f"{type(e).__name__}: {e}. Using Unstructured.",
                    exc_info=True
                )
            except Exception as e:
                # Unexpected error - log with traceback for investigation
                logger.error(
                    f"Unexpected error initializing Gemini extractor: "
                    f"{type(e).__name__}: {e}. This may indicate a bug. Using Unstructured.",
                    exc_info=True
                )

        return UnstructuredExtractor(config)
