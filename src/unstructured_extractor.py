"""
PHASE 1: Document Extraction using Unstructured.io

Supported Formats (tested):
- PDF (.pdf) - detectron2_mask_rcnn (Mask R-CNN X_101_32x8d_FPN_3x - most accurate)
- PowerPoint (.pptx, .ppt) - presentation structure and speaker notes
- Word (.docx, .doc) - document structure and track changes
- HTML (.html, .htm) - web content with semantic tags
- Plain text (.txt) - basic text files
- LaTeX (.tex, .latex) - scientific documents with math

Additional formats supported via partition_auto() fallback - see Unstructured.io docs.

Features:
- Explicit hierarchy via parent_id relationships (not font-size inference)
- Rotated text filtering (25-65° diagonal watermarks)
- Per-element language detection (Czech/English)
- Fallback to universal partitioner for unknown formats
- PHASE 2: Hierarchical document summary from section summaries

Architecture:
- Element types: Title, ListItem, NarrativeText, Table, Header, Footer
- Hierarchy detection: parent_id chains + page break continuity
- Table detection: ✅ Tables found, ⚠️ Cell structure not yet parsed (TODO)

Known Limitations:
- Tables: Detected but cell data not parsed (num_rows=0, data=[])
- Deep nesting: >10 levels may indicate detection errors
- Czech-specific: Legal keywords ("ZÁKON", "§") hardcoded in type scoring
- Memory: Large PDFs (>100 pages) may require >4GB RAM
- Paragraph detection: 100% on test document (Sb_1997_18), not validated on larger corpus yet

See Also:
- .env.example: Configuration options and defaults
- CLAUDE.md: Integration with 7-phase pipeline
- tests/test_unstructured_*.py: Test suite for extraction validation
"""

import logging
import re
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import math
from typing import Any

import numpy as np

# Unstructured imports
from unstructured.partition.auto import partition  # Universal partitioner
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.pptx import partition_pptx
from unstructured.partition.docx import partition_docx
from unstructured.partition.html import partition_html
from unstructured.partition.text import partition_text
from unstructured.documents.elements import (
    Element,
    Title,
    NarrativeText,
    ListItem,
    Table,
    Header,
    Footer,
    PageBreak,
)

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES (Previously compatible with Docling, now standalone)
# ============================================================================


@dataclass
class DocumentSection:
    """
    Hierarchical document section for multi-layer chunking.

    Data structure used throughout the indexing pipeline (Phases 1-7).
    Designed for compatibility with downstream chunking and embedding tools.

    Populated by Unstructured.io (v2.x).
    Field structure remains stable to preserve existing vector stores and tests.
    """

    section_id: str
    title: str
    content: str
    level: int  # Semantic hierarchy level (0=root, 1=major, 2=chapter, 3=section, etc.)
    depth: int  # Depth in hierarchy tree (1=root, 2=child of root, etc.)
    parent_id: Optional[str]
    children_ids: List[str]
    ancestors: List[str]  # List of parent titles (used by chunker for context)
    path: str  # Full path: "Chapter 1 > Section 1.1 > Subsection 1.1.1"
    page_number: int
    char_start: int
    char_end: int
    content_length: int

    # Unstructured.io metadata (used for type-based filtering)
    element_type: Optional[str] = None  # Element type: Title, NarrativeText, ListItem, etc.
    element_category: Optional[str] = None  # Element category from Unstructured

    # PHASE 2: Summaries
    summary: Optional[str] = None  # 150-char generic summary

    def __post_init__(self) -> None:
        """Validate invariants after initialization."""
        # Validate character range consistency
        if self.char_end < self.char_start:
            raise ValueError(
                f"DocumentSection '{self.section_id}': char_end ({self.char_end}) "
                f"< char_start ({self.char_start})"
            )
        # Validate depth (must be >= 1 for all sections)
        if self.depth < 1:
            raise ValueError(
                f"DocumentSection '{self.section_id}': depth ({self.depth}) must be >= 1"
            )
        # Validate level (must be >= 0)
        if self.level < 0:
            raise ValueError(
                f"DocumentSection '{self.section_id}': level ({self.level}) must be >= 0"
            )

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "section_id": self.section_id,
            "title": self.title,
            "content": self.content,
            "level": self.level,
            "depth": self.depth,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "ancestors": self.ancestors,
            "path": self.path,
            "page_number": self.page_number,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "content_length": self.content_length,
            "element_category": self.element_category,
            "summary": self.summary,
        }


@dataclass
class TableData:
    """Table metadata and content."""

    table_id: str
    caption: Optional[str]
    num_rows: int
    num_cols: int
    data: List[List[str]]  # 2D array of cell contents
    bbox: Optional[Dict]  # Bounding box coordinates
    page_number: int

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "table_id": self.table_id,
            "caption": self.caption,
            "num_rows": self.num_rows,
            "num_cols": self.num_cols,
            "data": self.data,
            "bbox": self.bbox,
            "page_number": self.page_number,
        }


@dataclass
class ExtractedDocument:
    """
    Complete extracted document with hierarchical sections.

    Output of PHASE 1+2: Document extraction and summary generation.
    Used as input for PHASE 3 (multi-layer chunking) and downstream pipeline.

    Populated by Unstructured.io (v2.x).
    """

    # Identification
    document_id: str
    source_path: str
    extraction_time: float  # seconds

    # Content
    full_text: str
    markdown: str
    json_content: str

    # Hierarchical structure
    sections: List[DocumentSection]
    hierarchy_depth: int  # Max depth of hierarchy tree
    num_roots: int  # Number of root sections

    # Tables
    tables: List[TableData]

    # Metadata
    num_pages: int
    num_sections: int
    num_tables: int
    total_chars: int

    # PHASE 2: Document-level summary
    document_summary: Optional[str] = None

    # Document title
    title: Optional[str] = None

    # Extraction metadata
    extraction_method: str = "unstructured_yolox"
    config: Optional[Dict] = None

    def __post_init__(self) -> None:
        """Validate invariants after initialization."""
        # Validate num_sections matches actual sections list
        if self.num_sections != len(self.sections):
            raise ValueError(
                f"ExtractedDocument '{self.document_id}': num_sections ({self.num_sections}) "
                f"!= len(sections) ({len(self.sections)})"
            )
        # Validate num_tables matches actual tables list
        if self.num_tables != len(self.tables):
            raise ValueError(
                f"ExtractedDocument '{self.document_id}': num_tables ({self.num_tables}) "
                f"!= len(tables) ({len(self.tables)})"
            )
        # Validate hierarchy_depth is positive if sections exist
        if self.sections and self.hierarchy_depth < 1:
            raise ValueError(
                f"ExtractedDocument '{self.document_id}': hierarchy_depth ({self.hierarchy_depth}) "
                f"must be >= 1 when sections exist"
            )
        # Validate total_chars is non-negative
        if self.total_chars < 0:
            raise ValueError(
                f"ExtractedDocument '{self.document_id}': total_chars ({self.total_chars}) "
                f"must be >= 0"
            )

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "metadata": {
                "document_id": self.document_id,
                "source_path": self.source_path,
                "extraction_time_seconds": self.extraction_time,
                "extraction_method": self.extraction_method,
                "num_pages": self.num_pages,
                "num_sections": self.num_sections,
                "num_tables": self.num_tables,
                "total_chars": self.total_chars,
                "hierarchy_depth": self.hierarchy_depth,
                "num_roots": self.num_roots,
                "title": self.title,
                "document_summary": self.document_summary,
            },
            "sections": [section.to_dict() for section in self.sections],
            "tables": [table.to_dict() for table in self.tables],
            "full_text": self.full_text,
            "markdown": self.markdown,
            "json_content": self.json_content,
            "config": self.config,
        }


# ============================================================================
# CONFIGURATION
# ============================================================================


@dataclass
class ExtractionConfig:
    """Configuration for document extraction (Unstructured or Gemini)."""

    # Backend selection: "auto" checks GOOGLE_API_KEY, falls back to unstructured
    extraction_backend: str = "auto"  # "gemini", "unstructured", "auto"
    gemini_model: str = "gemini-2.5-flash"  # Gemini model to use
    gemini_fallback_to_unstructured: bool = True  # Fall back to Unstructured on Gemini failure
    gemini_max_output_tokens: int = 65536  # Max output tokens for Gemini
    gemini_file_size_threshold_mb: float = 10.0  # File size threshold for chunked extraction

    # Unstructured model configuration
    strategy: str = "hi_res"  # "hi_res", "fast", "ocr_only"
    model: str = "yolox"  # "detectron2_mask_rcnn" (most accurate), "yolox" (faster)

    # Language detection
    languages: List[str] = field(default_factory=lambda: ["ces", "eng"])
    detect_language_per_element: bool = True

    # Table extraction
    infer_table_structure: bool = True
    extract_images: bool = False
    include_page_breaks: bool = False  # PageBreaks don't contain structural content

    # Rotated text filtering
    filter_rotated_text: bool = True
    rotation_method: str = "bbox_orientation"  # "bbox_orientation", "pymupdf"
    rotation_min_angle: float = 25.0  # degrees
    rotation_max_angle: float = 65.0  # degrees

    # Hierarchy detection
    enable_generic_hierarchy: bool = True
    hierarchy_signals: List[str] = field(
        default_factory=lambda: ["type", "font_size", "spacing", "numbering", "parent_id"]
    )
    hierarchy_clustering_eps: float = 0.15
    hierarchy_clustering_min_samples: int = 2

    # Output formats
    generate_markdown: bool = True
    generate_json: bool = True

    # PHASE 2: Summary generation
    generate_summaries: bool = False
    summary_model: str = "gpt-4o-mini"
    summary_max_chars: int = 150
    summary_style: str = "generic"
    extract_tables: bool = True

    @classmethod
    def from_env(cls) -> "ExtractionConfig":
        """Load configuration from environment variables with validation."""
        import os

        def get_float_env(key: str, default: float) -> float:
            """Get float from env with validation."""
            try:
                value = os.getenv(key)
                return float(value) if value else default
            except ValueError as e:
                raise ValueError(
                    f"Invalid environment variable {key}='{value}': must be a number. "
                    f"Example: {key}={default}"
                ) from e

        def get_int_env(key: str, default: int) -> int:
            """Get int from env with validation."""
            try:
                value = os.getenv(key)
                return int(value) if value else default
            except ValueError as e:
                raise ValueError(
                    f"Invalid environment variable {key}='{value}': must be an integer. "
                    f"Example: {key}={default}"
                ) from e

        def get_bool_env(key: str, default: bool) -> bool:
            """Get bool from env with validation."""
            value = os.getenv(key, str(default)).lower()
            if value not in {"true", "false"}:
                raise ValueError(
                    f"Invalid environment variable {key}='{value}': must be 'true' or 'false'"
                )
            return value == "true"

        # Load and validate config values
        min_angle = get_float_env("ROTATION_MIN_ANGLE", 25.0)
        max_angle = get_float_env("ROTATION_MAX_ANGLE", 65.0)

        # Validate relationships
        if min_angle >= max_angle:
            raise ValueError(
                f"ROTATION_MIN_ANGLE ({min_angle}) must be < ROTATION_MAX_ANGLE ({max_angle})"
            )
        if min_angle < 0 or max_angle > 90:
            raise ValueError(
                f"Rotation angles must be in range [0, 90]. Got min={min_angle}, max={max_angle}"
            )

        return cls(
            # Backend selection
            extraction_backend=os.getenv("EXTRACTION_BACKEND", "gemini"),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            gemini_fallback_to_unstructured=get_bool_env("GEMINI_FALLBACK_TO_UNSTRUCTURED", True),
            gemini_max_output_tokens=get_int_env("GEMINI_MAX_OUTPUT_TOKENS", 65536),
            gemini_file_size_threshold_mb=get_float_env("GEMINI_FILE_SIZE_THRESHOLD_MB", 10.0),
            # Unstructured settings
            strategy=os.getenv("UNSTRUCTURED_STRATEGY", "hi_res"),
            model=os.getenv("UNSTRUCTURED_MODEL", "yolox"),
            languages=os.getenv("UNSTRUCTURED_LANGUAGES", "ces,eng").split(","),
            detect_language_per_element=get_bool_env("UNSTRUCTURED_DETECT_LANGUAGE_PER_ELEMENT", True),
            infer_table_structure=get_bool_env("UNSTRUCTURED_INFER_TABLE_STRUCTURE", True),
            extract_images=get_bool_env("UNSTRUCTURED_EXTRACT_IMAGES", False),
            filter_rotated_text=get_bool_env("FILTER_ROTATED_TEXT", True),
            rotation_method=os.getenv("ROTATION_METHOD", "bbox_orientation"),
            rotation_min_angle=min_angle,
            rotation_max_angle=max_angle,
            enable_generic_hierarchy=get_bool_env("ENABLE_GENERIC_HIERARCHY", True),
            hierarchy_signals=os.getenv(
                "HIERARCHY_SIGNALS", "type,font_size,spacing,numbering,parent_id"
            ).split(","),
            hierarchy_clustering_eps=get_float_env("HIERARCHY_CLUSTERING_EPS", 0.15),
            hierarchy_clustering_min_samples=get_int_env("HIERARCHY_CLUSTERING_MIN_SAMPLES", 2),
            generate_markdown=get_bool_env("GENERATE_MARKDOWN", True),
            generate_json=get_bool_env("GENERATE_JSON", True),
        )


# ============================================================================
# TEXT NORMALIZATION UTILITIES
# ============================================================================


def _normalize_text_diacritics(text: str) -> str:
    """
    Normalize Czech/Slovak diacritics in extracted text.

    Fixes common OCR/extraction issues from Unstructured.io:
    1. Combining diacritical marks → precomposed characters (Unicode NFC)
    2. Standalone caron (ˇ U+02C7) before letters → correct diacritics
    3. Common typos: "mimiřádná" → "mimořádná"

    Args:
        text: Raw text from Unstructured.io element

    Returns:
        Normalized text with correct Czech/Slovak diacritics

    Examples:
        >>> _normalize_text_diacritics("zaˇrízení")  # ˇ before r
        "zařízení"
        >>> _normalize_text_diacritics("mimiřádná událost")
        "mimořádná událost"

    Note:
        Applied automatically in _build_sections()
        to ensure consistent text quality throughout the extraction pipeline.
    """
    # 1. NFC normalization (combining marks → precomposed)
    text = unicodedata.normalize('NFC', text)

    # 2. Standalone caron (U+02C7) - reconstruct diacritics
    # Pattern: [letter]ˇ OR ˇ[letter] → letter with caron
    # Háček může být před NEBO za písmenem
    caron_map = {
        # Háček ZA písmenem (méně časté)
        'rˇ': 'ř', 'Rˇ': 'Ř',
        'cˇ': 'č', 'Cˇ': 'Č',
        'sˇ': 'š', 'Sˇ': 'Š',
        'zˇ': 'ž', 'Zˇ': 'Ž',
        'eˇ': 'ě', 'Eˇ': 'Ě',
        'nˇ': 'ň', 'Nˇ': 'Ň',
        'dˇ': 'ď', 'Dˇ': 'Ď',
        'tˇ': 'ť', 'Tˇ': 'Ť',
        # Háček PŘED písmenem (časté v BZ_VR1)
        'ˇr': 'ř', 'ˇR': 'Ř',
        'ˇc': 'č', 'ˇC': 'Č',
        'ˇs': 'š', 'ˇS': 'Š',
        'ˇz': 'ž', 'ˇZ': 'Ž',
        'ˇe': 'ě', 'ˇE': 'Ě',
        'ˇn': 'ň', 'ˇN': 'Ň',
        'ˇd': 'ď', 'ˇD': 'Ď',
        'ˇt': 'ť', 'ˇT': 'Ť',
    }
    for wrong, correct in caron_map.items():
        text = text.replace(wrong, correct)

    # Remove orphaned carons (likely OCR errors)
    text = text.replace('\u02c7', '')

    # 3. Common OCR typos in Czech legal documents
    text = text.replace('mimiřád', 'mimořád')  # mimořádná (extraordinary)
    text = text.replace('Mimiřád', 'Mimořád')

    return text


# ============================================================================
# BBOX ORIENTATION ANALYSIS
# ============================================================================


def analyze_bbox_orientation(element: Element) -> Tuple[Optional[float], bool]:
    """
    Analyze bbox orientation from Unstructured metadata.

    Detects rotated text by analyzing bounding box coordinates.

    Args:
        element: Unstructured element with metadata

    Returns:
        Tuple of (angle_degrees, is_rotated)
        - angle_degrees: Rotation angle in degrees (0-360), or None if cannot detect
        - is_rotated: True if text is rotated beyond threshold

    Algorithm:
    1. Extract bbox points from element.metadata.coordinates
    2. Calculate angle using vector between first two points
    3. Determine if angle is outside horizontal/vertical margin
    4. Additional signals: aspect ratio, text patterns
    """
    # Check if coordinates metadata exists
    if not hasattr(element.metadata, 'coordinates') or element.metadata.coordinates is None:
        return (None, False)

    coords = element.metadata.coordinates

    # Extract points from coordinates dict
    if isinstance(coords, dict) and 'points' in coords:
        points = coords['points']

        # Need at least 2 points to calculate angle
        if len(points) < 2:
            return (None, False)

        # Calculate vector from first to second point
        try:
            p1 = points[0]
            p2 = points[1]

            # Handle numpy arrays
            if hasattr(p1, '__iter__'):
                x1, y1 = float(p1[0]), float(p1[1])
                x2, y2 = float(p2[0]), float(p2[1])
            else:
                return (None, False)

            # Calculate angle in degrees
            dx = x2 - x1
            dy = y2 - y1

            # Avoid division by zero
            if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                return (0.0, False)

            angle_rad = math.atan2(dy, dx)
            angle_deg = math.degrees(angle_rad)

            # Normalize to 0-360
            angle_deg = angle_deg % 360

            # Check if rotated (outside horizontal/vertical margin)
            # Horizontal: 0° ± margin or 180° ± margin
            # Vertical: 90° ± margin or 270° ± margin
            margin = 25.0  # degrees

            # Define acceptable ranges
            horizontal_ranges = [
                (0 - margin, 0 + margin),
                (180 - margin, 180 + margin),
                (360 - margin, 360),  # Wrap around
            ]

            vertical_ranges = [
                (90 - margin, 90 + margin),
                (270 - margin, 270 + margin),
            ]

            # Check if angle falls within acceptable ranges
            is_horizontal = any(low <= angle_deg <= high for low, high in horizontal_ranges)
            is_vertical = any(low <= angle_deg <= high for low, high in vertical_ranges)

            is_rotated = not (is_horizontal or is_vertical)

            return (angle_deg, is_rotated)

        except (TypeError, ValueError, AttributeError) as e:
            logger.debug(f"Failed to calculate bbox orientation: {e}")
            return (None, False)

    return (None, False)


def filter_rotated_elements(
    elements: List[Element],
    min_angle: float = 25.0,
    max_angle: float = 65.0
) -> List[Element]:
    """
    Filter out rotated text elements.

    Args:
        elements: List of Unstructured elements
        min_angle: Minimum rotation angle to filter (degrees)
        max_angle: Maximum rotation angle to filter (degrees)

    Returns:
        Filtered list of elements (rotated elements removed)
    """
    filtered = []
    num_filtered = 0

    for element in elements:
        angle, is_rotated = analyze_bbox_orientation(element)

        if is_rotated and angle is not None:
            # Check if angle is in the problematic range
            # (diagonal watermarks typically 25-65 degrees)
            if min_angle <= angle <= max_angle or min_angle <= (360 - angle) <= max_angle:
                num_filtered += 1
                logger.debug(
                    f"Filtered rotated element (angle={angle:.1f}°): "
                    f"{str(element)[:50]}..."
                )
                continue

        filtered.append(element)

    if num_filtered > 0:
        logger.info(f"Filtered {num_filtered} rotated text elements")

    return filtered


# ============================================================================
# GENERIC HIERARCHY DETECTION
# ============================================================================

def _get_structure_signature(text: str) -> str:
    """
    Generate a 'signature' for a title to detect if two titles are siblings.
    """
    text = text.strip().lower()
    tokens = text.split()
    if not tokens:
        return ""

    # 1. Structural Markers: Only care about the marker word.
    # Note: Keys are normalized (lowercase, no dots) to match stripped token.
    markers = {
        '§': 'section',
        'section': 'section',
        'article': 'article',
        'článek': 'article',
        'čl': 'article',  # Cleaned 'čl.' matches this
        'chapter': 'chapter',
        'hlava': 'chapter',
        'part': 'part',
        'část': 'part',
        'oddíl': 'division',
        'díl': 'division',
        'zakon': 'law',
        'zákon': 'law'
    }

    # Strip punctuation from first token (e.g. "Čl." -> "čl")
    first_token_clean = re.sub(r'[^\w§]', '', tokens[0])

    if first_token_clean in markers:
        return markers[first_token_clean]

        # 2. Numbered items
    if re.match(r'^[\d\.]+$', tokens[0]):
        return "numbered_item"

    # 3. Lettered items (a), b))
    if re.match(r'^[a-z]\)$', tokens[0]):
        return "lettered_item"

    return text[:20]


def _get_numbering_depth(text: str) -> int:
    """
    Calculates depth based on numbering pattern (e.g., '1.2.3' -> 3).
    Returns 0 if no numbering detected.
    Handles '1.', '1.2', 'A.1' patterns.
    """
    text = text.strip()
    match = re.match(r'^((?:[A-Za-z0-9]+\.)+[A-Za-z0-9]*)\s', text + " ")
    if match:
        numbering = match.group(1).rstrip('.')
        if numbering:
            return len(numbering.split('.'))
    return 0


def _get_list_style(text: str) -> str:
    """
    Determine list style to match continuations across pages.
    e.g. '1.', 'a)', 'A.', '(1)'
    """
    text = text.strip()
    if re.match(r'^\d+\.', text): return "decimal_dot"  # 1.
    if re.match(r'^[a-z]\)', text): return "alpha_paren"  # a)
    if re.match(r'^\([a-z]\)', text): return "paren_alpha_paren"  # (a)
    if re.match(r'^\(\d+\)', text): return "paren_decimal_paren"  # (1)
    if re.match(r'^[A-Z]\.', text): return "upper_alpha_dot"  # A.
    return "unknown"


def build_robust_hierarchy(elements: List[Element], config: ExtractionConfig) -> List[Dict]:
    """
    Builds a robust hierarchy by cleaning Unstructured.io output.

    Logic:
    0. Classify UncategorizedText that looks like Titles/Headers (Multi-signal similarity).
    1. Filter Headers/Footers.
    2. Flatten NarrativeText.
    3. Detect Siblings & Cousins using signatures AND numbering depth.
    4. Fix Page Orphans by adopting grandparent if sibling conflict exists (Local only).
    """
    logger.info(f"Building robust hierarchy for {len(elements)} elements...")
    if not elements:
        return []

    # --- Pass 0: Statistics & Classification Correction ---
    # Collect data for similarity comparison
    title_sizes = []
    title_x_positions = []  # Track left-alignment

    # NEW: Track known header/footer text content for exact matching
    known_header_footer_texts = set()
    header_sizes = []  # Track header/footer sizes

    # NEW: Track spatial zones for headers and footers
    header_y_ranges = []  # List of (y_min, y_max)
    footer_y_ranges = []

    for elem in elements:
        meta = getattr(elem, 'metadata', None)
        size = 0.0
        x_pos = 0.0
        y_min = 0.0
        y_max = 0.0

        if meta and hasattr(meta, 'to_dict'):
            meta_dict = meta.to_dict()
            size = meta_dict.get('font_size', 0.0) or 0.0

            # Extract coordinates
            coords = meta_dict.get('coordinates', {})
            if coords and 'points' in coords:
                points = coords['points']
                ys = [p[1] for p in points]
                xs = [p[0] for p in points]
                if ys: y_min, y_max = min(ys), max(ys)
                if xs: x_pos = min(xs)

        cat = getattr(elem, 'category', 'Unknown')
        if cat == 'Title' and size > 0:
            title_sizes.append(size)
            title_x_positions.append(x_pos)
        elif cat == 'Header':
            if size > 0: header_sizes.append(size)
            if y_max > 0: header_y_ranges.append((y_min, y_max))
            text_content = _normalize_text_diacritics(str(elem)).strip().lower()
            if text_content: known_header_footer_texts.add(text_content)
        elif cat == 'Footer':
            if size > 0: header_sizes.append(size)
            if y_max > 0: footer_y_ranges.append((y_min, y_max))
            text_content = _normalize_text_diacritics(str(elem)).strip().lower()
            if text_content: known_header_footer_texts.add(text_content)

    avg_title_size = np.mean(title_sizes) if title_sizes else 0.0
    avg_title_x = np.mean(title_x_positions) if title_x_positions else 0.0
    avg_header_size = np.mean(header_sizes) if header_sizes else 0.0

    # Correction Loop
    for elem in elements:
        cat = getattr(elem, 'category', 'Unknown')
        meta = getattr(elem, 'metadata', None)
        size = 0.0
        x_pos = 0.0
        y_min = 0.0
        y_max = 0.0
        page_height = 0.0

        if meta and hasattr(meta, 'to_dict'):
            meta_dict = meta.to_dict()
            size = meta_dict.get('font_size', 0.0) or 0.0
            coords = meta_dict.get('coordinates', {})
            if coords and 'points' in coords:
                points = coords['points']
                ys = [p[1] for p in points]
                xs = [p[0] for p in points]
                if ys: y_min, y_max = min(ys), max(ys)
                if xs: x_pos = min(xs)
                page_height = coords.get('layout_height', 0.0)
                if not page_height and coords.get('system') == 'PixelSpace':
                    page_height = 842.0

        text_raw = str(elem).strip()
        text_norm = _normalize_text_diacritics(text_raw).strip().lower()

        if cat == 'UncategorizedText':

            # --- Rule 0: Content-Based Header/Footer Downgrade ---
            if text_norm in known_header_footer_texts:
                new_cat = 'Header' if (page_height > 0 and y_min < page_height / 2) else 'Footer'
                elem.category = new_cat
                logger.debug(f"Downgraded UncategorizedText to {new_cat} (content match)")
                continue

                # --- Rule 0b: Positional Header/Footer Downgrade ---
            y_center = (y_min + y_max) / 2
            is_in_header_zone = False
            is_in_footer_zone = False

            for h_min, h_max in header_y_ranges:
                if h_min - 5 <= y_center <= h_max + 5:
                    is_in_header_zone = True;
                    break
            for f_min, f_max in footer_y_ranges:
                if f_min - 5 <= y_center <= f_max + 5:
                    is_in_footer_zone = True;
                    break

            if is_in_header_zone:
                elem.category = 'Header';
                continue
            if is_in_footer_zone:
                elem.category = 'Footer';
                continue

            # Upgrade Title candidates
            title_score = 0.0
            if avg_title_size > 0 and abs(size - avg_title_size) < 1.5:
                title_score += 0.5
            elif avg_title_size > 0 and size > avg_title_size:
                title_score += 0.5

            if _get_structure_signature(text_raw) != text_raw[:20]:
                title_score += 0.3
            elif _get_numbering_depth(text_raw) > 0:
                title_score += 0.3
            if avg_title_x > 0 and abs(x_pos - avg_title_x) < 50: title_score += 0.2

            if title_score >= 0.5:
                elem.category = 'Title'
                continue

            # Visual Downgrade fallback
            if page_height > 0:
                is_top = y_min < (page_height * 0.10)
                is_bottom = y_min > (page_height * 0.90)
                is_header_size = (avg_header_size > 0 and abs(size - avg_header_size) < 1.5)
                is_small = size < avg_title_size * 0.8 if avg_title_size > 0 else False

                if (is_top or is_bottom) and (is_header_size or is_small):
                    new_cat = 'Header' if is_top else 'Footer'
                    elem.category = new_cat

    # --- Pass 1: Feature Extraction ---
    features = []
    id_to_index = {}
    header_footer_ids = set()

    for i, elem in enumerate(elements):
        element_id = getattr(elem, 'id', f"elem_{i}")
        metadata = getattr(elem, 'metadata', None)
        parent_id = getattr(metadata, 'parent_id', None)
        page_number = getattr(metadata, 'page_number', None)
        category = getattr(elem, 'category', 'Unknown')

        font_size = 0.0
        if metadata and hasattr(metadata, 'to_dict'):
            font_size = metadata.to_dict().get('font_size', 0.0) or 0.0

        if category in ["Header", "Footer"]:
            header_footer_ids.add(element_id)

        signature = None
        numbering_depth = 0
        list_style = "unknown"

        if category == "Title" or category == "ListItem":  # Check lists too for siblings
            signature = _get_structure_signature(str(elem))
            numbering_depth = _get_numbering_depth(str(elem))

        if category == "ListItem":
            list_style = _get_list_style(str(elem))

        feat = {
            "element": elem,
            "index": i,
            "element_id": element_id,
            "original_parent_id": parent_id,
            "page_number": page_number,
            "category": category,
            "font_size": font_size,
            "signature": signature,
            "numbering_depth": numbering_depth,
            "list_style": list_style,
            "true_parent_id": None,
            "level": 0
        }
        features.append(feat)
        id_to_index[element_id] = i

    # --- Pass 2: Resolve Parents ---
    def find_valid_parent(current_parent_id: Optional[str], child_feat: Dict) -> Optional[str]:
        if not current_parent_id or current_parent_id not in id_to_index:
            return None

        parent_idx = id_to_index[current_parent_id]
        parent_feat = features[parent_idx]

        # RULE 1: Skip Headers/Footers
        if parent_feat["element_id"] in header_footer_ids:
            return find_valid_parent(parent_feat["original_parent_id"], child_feat)

        # RULE 2: Flatten Content (Text cannot be a parent)
        if parent_feat["category"] not in ["Title", "ListItem"]:
            return find_valid_parent(parent_feat["original_parent_id"], child_feat)

        # RULE 3: Title Hierarchy Checks
        if (child_feat["category"] == "Title" and parent_feat["category"] == "Title") or \
                (child_feat["category"] == "ListItem" and parent_feat["category"] == "ListItem"):

            # A. Numbering Hierarchy Check
            if child_feat["numbering_depth"] > 0 and parent_feat["numbering_depth"] > 0:
                if child_feat["numbering_depth"] <= parent_feat["numbering_depth"]:
                    return find_valid_parent(parent_feat["original_parent_id"], child_feat)

            # B. Immediate Sibling Detection (Signature)
            if child_feat["signature"] and child_feat["signature"] == parent_feat["signature"]:
                return find_valid_parent(parent_feat["original_parent_id"], child_feat)

            # C. Font Size Check
            if parent_feat["font_size"] > 0 and child_feat["font_size"] >= (parent_feat["font_size"] - 0.5):
                return find_valid_parent(parent_feat["original_parent_id"], child_feat)

            # D. Ancestor Scan (Cousin Detection)
            if child_feat["signature"]:
                # Use current true_parent (which is resolved so far) to traverse up
                ancestor_id = parent_feat.get("true_parent_id")
                # Also fall back to original parent structure if true parent is not set or we want to scan raw structure
                if not ancestor_id:
                    ancestor_id = parent_feat["original_parent_id"]

                depth_limit = 20

                while ancestor_id and ancestor_id in id_to_index and depth_limit > 0:
                    ancestor_idx = id_to_index[ancestor_id]
                    ancestor_feat = features[ancestor_idx]

                    if (ancestor_feat["category"] == child_feat["category"] and  # Same type (Title/List)
                            ancestor_feat["signature"] == child_feat["signature"]):
                        # Match found! This ancestor is our "cousin".
                        # We should be siblings with them, so we take their parent.
                        return find_valid_parent(ancestor_feat["original_parent_id"], child_feat)

                    # Move up using original_parent for raw structure scan
                    ancestor_id = ancestor_feat["original_parent_id"]
                    depth_limit -= 1

        return parent_feat["element_id"]

    for feat in features:
        if feat["element_id"] not in header_footer_ids:
            feat["true_parent_id"] = find_valid_parent(feat["original_parent_id"], feat)

    # --- Pass 3: Assign Levels & Fix Page Orphans ---
    final_features = []
    last_title_by_page = {}
    last_list_item_by_page = {}  # Track list items for cross-page lists
    current_page = None

    def calculate_level(feat_id: str, visited: set) -> int:
        if not feat_id or feat_id not in id_to_index: return 0
        if feat_id in visited: return 0
        visited.add(feat_id)
        parent_feat = features[id_to_index[feat_id]]
        return calculate_level(parent_feat["true_parent_id"], visited) + 1

    for i, feat in enumerate(features):
        if feat["element_id"] in header_footer_ids:
            continue

        if feat["page_number"] is not None:
            current_page = feat["page_number"]

        # Fix Orphans
        if feat["true_parent_id"] is None and current_page is not None:
            potential_parent_id = None

            # --- NarrativeText Bridging ---
            # If NarrativeText is orphaned, try to attach to previous element (if List Item)
            if feat["category"] in ["NarrativeText", "UncategorizedText"]:
                if i > 0:
                    prev_feat = features[i - 1]
                    # If previous was a ListItem, we likely belong to its parent (sibling of item)
                    # or to the item itself? Usually text follows item as description.
                    # Let's attach to the ListItem's parent to keep flow correct.
                    if prev_feat["category"] == "ListItem":
                        # Check if prev item has a parent, otherwise we attach to prev item (risky?)
                        # Safer: Attach to prev item's parent so we are indentation-aligned
                        grandparent_id = prev_feat.get("true_parent_id")
                        if grandparent_id:
                            feat["true_parent_id"] = grandparent_id
                            potential_parent_id = None
                    # If previous was Title, we are likely content of that Title
                    elif prev_feat["category"] == "Title":
                        feat["true_parent_id"] = prev_feat["element_id"]
                        potential_parent_id = None

            # --- ListItem Continuation Check (Simplified) ---
            # If this is a ListItem, try to attach to parent of last ListItem on previous page
            if feat["category"] == "ListItem":
                if (current_page - 1) in last_list_item_by_page:
                    prev_item_id = last_list_item_by_page[current_page - 1]
                    prev_item = features[id_to_index[prev_item_id]]

                    # Adopt previous item's parent to become its sibling
                    grandparent_id = prev_item.get("true_parent_id")
                    if grandparent_id:
                        feat["true_parent_id"] = grandparent_id
                        potential_parent_id = None  # Skip title check
                    else:
                        # Previous item was root? Attach to same root (None)
                        potential_parent_id = None

                        # --- Title/Standard Orphan Check ---
            if feat["true_parent_id"] is None:
                if current_page in last_title_by_page:
                    potential_parent_id = last_title_by_page[current_page]
                elif (current_page - 1) in last_title_by_page:
                    potential_parent_id = last_title_by_page[current_page - 1]

            if potential_parent_id and potential_parent_id in id_to_index:
                parent_feat = features[id_to_index[potential_parent_id]]
                should_adopt = True

                if feat["category"] == "Title" and parent_feat["category"] == "Title":

                    # 1. Numbering Check for Orphans
                    if feat["numbering_depth"] > 0 and parent_feat["numbering_depth"] > 0:
                        if feat["numbering_depth"] <= parent_feat["numbering_depth"]:
                            # Adopt GRANDPARENT instead of remaining Orphan
                            grandparent_id = parent_feat.get("true_parent_id")
                            if grandparent_id:
                                feat["true_parent_id"] = grandparent_id
                                should_adopt = False
                            else:
                                should_adopt = False

                    # 2. Signature Check
                    elif feat["signature"] and feat["signature"] == parent_feat["signature"]:
                        # Adopt GRANDPARENT
                        grandparent_id = parent_feat.get("true_parent_id")
                        if grandparent_id:
                            feat["true_parent_id"] = grandparent_id
                            should_adopt = False
                        else:
                            should_adopt = False

                    # 3. Font Check
                    elif feat["font_size"] > (parent_feat["font_size"] + 0.5):
                        should_adopt = False

                    # 4. Ancestor Scan (Cousin Detection for Orphans)
                    elif feat["signature"]:
                        # Check up the hierarchy of the potential parent (last title)
                        ancestor_id = parent_feat.get("true_parent_id")  # Use resolved parent
                        depth_limit = 20
                        while ancestor_id and ancestor_id in id_to_index and depth_limit > 0:
                            ancestor_idx = id_to_index[ancestor_id]
                            ancestor_feat = features[ancestor_idx]

                            if (ancestor_feat["category"] == "Title" and
                                    ancestor_feat["signature"] == feat["signature"]):
                                # We matched an ancestor of the potential parent!
                                # So we should adopt THAT ancestor's parent.
                                grandparent_id = ancestor_feat.get("true_parent_id")
                                if grandparent_id:
                                    feat["true_parent_id"] = grandparent_id
                                should_adopt = False  # Handled manually
                                break

                            ancestor_id = ancestor_feat.get("true_parent_id")  # Keep going up resolved chain
                            depth_limit -= 1

                if should_adopt:
                    feat["true_parent_id"] = potential_parent_id

        # Update Trackers
        if feat["category"] == "Title" and current_page is not None:
            last_title_by_page[current_page] = feat["element_id"]
        elif feat["category"] == "ListItem" and current_page is not None:
            last_list_item_by_page[current_page] = feat["element_id"]

        feat["level"] = calculate_level(feat["true_parent_id"], set())
        final_features.append(feat)

    logger.info(f"Hierarchy built: {len(final_features)} valid elements.")
    return final_features

# ============================================================================
# UNSTRUCTURED EXTRACTOR
# ============================================================================


class UnstructuredExtractor:
    """
    Multi-format document extractor using Unstructured.io.

    Supports: PDF, PPTX, DOCX, HTML, TXT, LaTeX
    Replaces DoclingExtractorV2 with backward-compatible interface.

    Example:
        >>> config = ExtractionConfig.from_env()
        >>> extractor = UnstructuredExtractor(config)
        >>> doc = extractor.extract(Path("document.pdf"))
        >>> doc = extractor.extract(Path("presentation.pptx"))
        >>> doc = extractor.extract(Path("report.docx"))
    """

    def __init__(self, config: Optional[ExtractionConfig] = None):
        """
        Initialize extractor.

        Args:
            config: Extraction configuration (defaults to env vars)
        """
        self.config = config or ExtractionConfig.from_env()

        # Handle both ExtractionConfig types (Unstructured.io vs Pipeline)
        # Unstructured.io config: has 'model', 'strategy' attributes
        # Pipeline config: has 'ocr_engine', 'extract_hierarchy' attributes
        self._is_unstructured_config = hasattr(self.config, 'model')

        # Log config attributes
        if self._is_unstructured_config:
            logger.info(f"UnstructuredExtractor initialized: model={self.config.model}, strategy={self.config.strategy}")
        else:
            logger.info(f"UnstructuredExtractor initialized: ocr_engine={getattr(self.config, 'ocr_engine', 'tesseract')}")

    def extract(self, file_path: Path) -> ExtractedDocument:
        """
        Extract document structure from any supported file format.

        Supported formats:
        - PDF (.pdf) - with optional hi_res OCR models
        - PowerPoint (.pptx, .ppt)
        - Word (.docx, .doc)
        - HTML (.html, .htm)
        - Plain text (.txt)
        - LaTeX (.tex, .latex)

        Args:
            file_path: Path to document file

        Returns:
            ExtractedDocument with hierarchical sections
        """
        logger.info(f"Starting extraction of {file_path.name} ({file_path.suffix})")
        start_time = time.time()

        # Extract with Unstructured
        elements = self._partition_document(file_path)

        # Filter rotated text (watermark removal)
        if self.config.filter_rotated_text:
            before_count = len(elements)
            elements = filter_rotated_elements(
                elements,
                self.config.rotation_min_angle,
                self.config.rotation_max_angle
            )
            after_count = len(elements)
            removed = before_count - after_count

            if removed == 0:
                logger.warning(
                    f"Rotated text filtering enabled but removed 0/{before_count} elements. "
                    f"Document may not have bbox metadata or all text is horizontal/vertical. "
                    f"Rotation thresholds: [{self.config.rotation_min_angle}°, {self.config.rotation_max_angle}°]"
                )
            else:
                logger.info(
                    f"Filtered {removed}/{before_count} rotated text elements "
                    f"(angle range: [{self.config.rotation_min_angle}°, {self.config.rotation_max_angle}°])"
                )

        # Detect hierarchy
        if getattr(self.config, 'enable_generic_hierarchy', True):
            hierarchy_features = build_robust_hierarchy(elements, self.config)
        else:
            # Fallback: simple type-based hierarchy
            hierarchy_features = [
                {
                    "element": elem,
                    "index": i,
                    "level": 3,  # Default level
                    "type_name": elem.category if hasattr(elem, 'category') else "Unknown",
                }
                for i, elem in enumerate(elements)
            ]

        # Build sections
        sections = self._build_sections(hierarchy_features)

        # Extract tables
        tables = self._extract_tables(elements)

        # PHASE 2: Generate summaries (hierarchical document summary from section summaries)
        if self.config.generate_summaries:
            from src.summary_generator import SummaryGenerator
            from src.config import SummarizationConfig, get_config

            # Load summarization config from config.json
            root_config = get_config()
            summary_config = SummarizationConfig.from_config(root_config.summarization)

            summary_gen = SummaryGenerator(config=summary_config)

            # Generate section summaries
            section_summaries = []
            for section in sections:
                if section.content and len(section.content.strip()) > 50:  # Min length threshold
                    try:
                        section.summary = summary_gen.generate_section_summary(
                            section.content, section.title or ""
                        )
                        section_summaries.append(section.summary)
                    except Exception as e:
                        logger.warning(
                            f"Failed to generate summary for section '{section.title}': {e}"
                        )
                        section.summary = None

            # Generate hierarchical document summary from section summaries (NOT full text)
            # This follows CLAUDE.md constraint: "ALWAYS generate from section summaries"
            if section_summaries:
                try:
                    document_summary = summary_gen.generate_document_summary(
                        section_summaries=section_summaries
                    )
                except Exception as e:
                    logger.warning(f"Failed to generate document summary: {e}")
                    document_summary = "(Document summary unavailable)"
            else:
                document_summary = "(No section summaries available)"
        else:
            document_summary = None

        # Generate outputs
        full_text = "\n\n".join(str(elem) for elem in elements if hasattr(elem, '__str__'))
        markdown = self._generate_markdown(sections) if self.config.generate_markdown else ""
        json_content = "" # Will be generated during serialization

        extraction_time = time.time() - start_time

        # Build ExtractedDocument
        document_id = file_path.stem
        extracted_doc = ExtractedDocument(
            document_id=document_id,
            source_path=str(file_path),
            extraction_time=extraction_time,
            full_text=full_text,
            markdown=markdown,
            json_content=json_content,
            sections=sections,
            hierarchy_depth=max((s.depth for s in sections), default=0),
            num_roots=sum(1 for s in sections if s.depth == 1),
            tables=tables,
            num_pages=self._count_pages(elements),
            num_sections=len(sections),
            num_tables=len(tables),
            total_chars=len(full_text),
            title=self._extract_title(sections),
            document_summary=document_summary,  # PHASE 2: Hierarchical summary
            extraction_method=f"unstructured_{getattr(self.config, 'model', 'yolox')}",
            config=self.config.__dict__ if hasattr(self.config, '__dict__') else {},
        )

        logger.info(
            f"Extraction completed in {extraction_time:.2f}s: "
            f"{len(sections)} sections (depth={extracted_doc.hierarchy_depth}), "
            f"{len(tables)} tables"
        )

        return extracted_doc

    def _partition_document(self, file_path: Path) -> List[Element]:
        """
        Run Unstructured partition on any supported document format.

        Supported formats:
        - PDF (.pdf)
        - PowerPoint (.pptx, .ppt)
        - Word (.docx, .doc)
        - HTML (.html, .htm)
        - Plain text (.txt)
        - LaTeX (.tex, .latex)

        Args:
            file_path: Path to document file

        Returns:
            List of Unstructured elements
        """
        file_suffix = file_path.suffix.lower()

        # Get config values with fallbacks for both config types
        strategy = getattr(self.config, 'strategy', 'hi_res')
        model = getattr(self.config, 'model', 'yolox')
        languages = getattr(self.config, 'languages', ['ces', 'eng'])
        include_page_breaks = getattr(self.config, 'include_page_breaks', False)
        infer_table_structure = getattr(self.config, 'infer_table_structure', True)
        extract_images = getattr(self.config, 'extract_images', False)

        logger.info(f"Partitioning {file_suffix} document with strategy={strategy}")

        # Common parameters for all formats
        common_params = {
            "filename": str(file_path),
            "languages": languages,
            "include_page_breaks": include_page_breaks,
        }

        try:
            # PDF - use specialized function with hi_res models
            if file_suffix == ".pdf":
                logger.info(f"Using partition_pdf with model={model}")
                # Available Unstructured models:
                # - "yolox" (default, fast)
                # - "detectron2_onnx" (Faster R-CNN R_50_FPN_3x)
                # - "detectron2_mask_rcnn" (Mask R-CNN X_101_32x8d_FPN_3x, MOST ACCURATE)
                # - "detectron2_quantized" (quantized for speed)
                elements = partition_pdf(
                    **common_params,
                    strategy=strategy,
                    hi_res_model_name=model if strategy == "hi_res" else None,
                    infer_table_structure=infer_table_structure,
                    extract_images_in_pdf=extract_images,
                )

            # PowerPoint - use specialized function
            elif file_suffix in [".pptx", ".ppt"]:
                logger.info("Using partition_pptx")
                elements = partition_pptx(
                    **common_params,
                    infer_table_structure=self.config.infer_table_structure if hasattr(self.config, 'infer_table_structure') else True,
                )

            # Word - use specialized function
            elif file_suffix in [".docx", ".doc"]:
                logger.info("Using partition_docx")
                elements = partition_docx(
                    **common_params,
                    infer_table_structure=self.config.infer_table_structure if hasattr(self.config, 'infer_table_structure') else True,
                )

            # HTML - use specialized function
            elif file_suffix in [".html", ".htm"]:
                logger.info("Using partition_html")
                # HTML doesn't support all common params
                elements = partition_html(
                    filename=str(file_path),
                    include_page_breaks=False,  # HTML doesn't have pages
                )

            # Plain text - use text partitioner
            elif file_suffix == ".txt":
                logger.info("Using partition_text")
                # Text partitioner has minimal parameters
                elements = partition_text(
                    filename=str(file_path),
                    languages=common_params["languages"],
                )

            # LaTeX - use custom parser for better hierarchy
            elif file_suffix in [".tex", ".latex"]:
                logger.info("Using custom LaTeX parser for better hierarchy")
                try:
                    from src.latex_parser import parse_latex_document, clean_latex_text
                    from unstructured.documents.elements import Title as UnstrTitle, NarrativeText

                    latex_data = parse_latex_document(file_path)

                    if latex_data:
                        # Convert parsed sections to Unstructured elements
                        elements = []
                        for section in latex_data['sections']:
                            # Create Title element for section heading
                            title_elem = UnstrTitle(section['title_clean'])
                            if hasattr(title_elem, 'metadata'):
                                title_elem.metadata.page_number = 1  # LaTeX doesn't have pages
                                # Store hierarchy info in metadata
                                title_elem.metadata.parent_id = section.get('parent_id')
                                title_elem.metadata.section_level = section['level']
                                title_elem.metadata.section_depth = section['depth']
                            # Store section_id as element_id for hierarchy tracking
                            title_elem.element_id = section['section_id']
                            elements.append(title_elem)

                            # Create NarrativeText for content
                            if section['content_clean'].strip():
                                content_elem = NarrativeText(section['content_clean'])
                                if hasattr(content_elem, 'metadata'):
                                    content_elem.metadata.page_number = 1
                                    content_elem.metadata.parent_id = section['section_id']
                                content_elem.element_id = f"{section['section_id']}_content"
                                elements.append(content_elem)

                        logger.info(f"LaTeX parser created {len(elements)} elements with proper hierarchy")
                    else:
                        # Fallback to text partitioner
                        logger.warning("LaTeX parser returned no sections, falling back to partition_text")
                        elements = partition_text(
                            filename=str(file_path),
                            languages=common_params["languages"],
                        )
                except Exception as latex_error:
                    logger.warning(f"LaTeX parser failed: {latex_error}, falling back to partition_text")
                    elements = partition_text(
                        filename=str(file_path),
                        languages=common_params["languages"],
                    )

            # Unsupported format - try universal partitioner
            else:
                logger.warning(f"Unknown file format {file_suffix}, trying universal partitioner")
                elements = partition(
                    **common_params,
                    strategy=self.config.strategy if file_suffix == ".pdf" else "auto",
                )

            logger.info(f"Partitioned {file_suffix} document into {len(elements)} elements")
            return elements

        except FileNotFoundError as e:
            # User-fixable error - fail fast with clear message
            raise RuntimeError(
                f"Cannot access file {file_path}: file not found. "
                "Check the file path is correct."
            ) from e

        except PermissionError as e:
            # User-fixable error - fail fast with clear message
            raise RuntimeError(
                f"Cannot access file {file_path}: permission denied. "
                "Check you have read permissions for this file."
            ) from e

        except (ImportError, RuntimeError) as e:
            # Expected errors - fallback to universal partitioner
            logger.warning(f"Format-specific partition failed: {e}")
            logger.info("Attempting fallback with universal partitioner (may have lower quality)")
            try:
                elements = partition(filename=str(file_path))
                logger.info(f"Fallback successful: {len(elements)} elements extracted")
                logger.warning(
                    "Used universal partitioner fallback. Results may have degraded quality "
                    "(e.g., no hierarchy detection, simplified text extraction). "
                    f"Consider fixing the error: {e}"
                )
                return elements
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                raise RuntimeError(
                    f"Could not partition document {file_path.name}. "
                    f"Format-specific error: {e}. Fallback error: {fallback_error}"
                ) from fallback_error

        except Exception as e:
            # Unexpected error - log and re-raise
            logger.error(f"Unexpected error during partition: {e}", exc_info=True)
            raise

    def _partition_pdf(self, pdf_path: Path) -> List[Element]:
        """
        Run Unstructured partition_pdf (deprecated - use _partition_document instead).

        Kept for backward compatibility.
        """
        logger.warning("_partition_pdf is deprecated, use _partition_document instead")
        return self._partition_document(pdf_path)

    def _validate_page_boundary_hierarchy(self, sections: List[DocumentSection]) -> None:
        """
        Validate parent-child relationships across page boundaries.

        Checks that page breaks don't break the document hierarchy structure.
        Logs warnings for suspicious patterns (e.g., child on page N+2 but parent on page N).

        Args:
            sections: List of DocumentSection objects

        Raises:
            Warning logs if hierarchy inconsistencies detected
        """
        broken_relationships = []
        suspicious_gaps = []

        # Build section lookup
        section_by_id = {s.section_id: s for s in sections}

        for section in sections:
            if not section.parent_id:
                continue  # Root sections have no parent

            parent = section_by_id.get(section.parent_id)
            if not parent:
                broken_relationships.append({
                    "child": section.section_id,
                    "child_page": section.page_number,
                    "parent_id": section.parent_id,
                    "error": "Parent not found"
                })
                continue

            # Check page gap between parent and child
            page_gap = abs(section.page_number - parent.page_number)

            # Warning: Child is more than 1 page away from parent (suspicious)
            # Exception: Multi-page chapters are OK if level difference >= 2
            level_diff = section.level - parent.level
            if page_gap > 1 and level_diff < 2:
                suspicious_gaps.append({
                    "child": section.section_id,
                    "child_page": section.page_number,
                    "parent": parent.section_id,
                    "parent_page": parent.page_number,
                    "page_gap": page_gap,
                    "level_diff": level_diff,
                })

        # Log warnings
        if broken_relationships:
            logger.warning(
                f"Found {len(broken_relationships)} sections with missing parents:\n" +
                "\n".join(
                    f"  - Section {rel['child']} (page {rel['child_page']}) → "
                    f"parent {rel['parent_id']} not found"
                    for rel in broken_relationships[:5]  # Show first 5
                )
            )

        if suspicious_gaps:
            logger.warning(
                f"Found {len(suspicious_gaps)} suspicious parent-child page gaps:\n" +
                "\n".join(
                    f"  - Section {gap['child']} (page {gap['child_page']}) → "
                    f"parent {gap['parent']} (page {gap['parent_page']}), "
                    f"gap={gap['page_gap']} pages, level_diff={gap['level_diff']}"
                    for gap in suspicious_gaps[:5]  # Show first 5
                )
            )

        # Log summary
        total_cross_page = sum(
            1 for s in sections
            if s.parent_id and section_by_id.get(s.parent_id)
            and s.page_number != section_by_id[s.parent_id].page_number
        )

        if total_cross_page > 0:
            logger.info(
                f"Hierarchy validation complete: {total_cross_page} parent-child relationships "
                f"cross page boundaries (OK if validated)"
            )

    def _build_sections(self, hierarchy_features: List[Dict]) -> List[DocumentSection]:
        """
        Build DocumentSection objects from hierarchy_features produced by
        `build_robust_hierarchy`.

        Assumptions about each feature dict:
            - feat["element"]: Unstructured element
            - feat["element_id"]: unique id
            - feat["true_parent_id"]: cleaned parent element_id or None
            - feat["level"]: integer depth (0 for root)
            - feat["page_number"], feat["category"] may be present
        """
        sections: List[DocumentSection] = []
        char_offset = 0

        # ------------------------------------------------------------------ #
        # 1. Create all sections and map element_id -> section_id
        # ------------------------------------------------------------------ #
        elemid_to_sectionid: Dict[str, str] = {}

        for i, feat in enumerate(hierarchy_features):
            elem = feat["element"]
            text = _normalize_text_diacritics(str(elem))
            elem_category = getattr(elem, 'category', None)

            # Title elements get their text as title, others get content only
            if elem_category == "Title":
                title = text.strip()
            else:
                # NarrativeText and other content - no title, only content
                title = ""

            section_id = f"sec_{i + 1}"
            elemid_to_sectionid[feat["element_id"]] = section_id

            section = DocumentSection(
                section_id=section_id,
                title=title,
                content=text,
                level=feat["level"],  # level from robust hierarchy
                depth=1,  # filled later
                parent_id=None,  # filled later using true_parent_id
                children_ids=[],
                ancestors=[],
                path="",  # filled later
                page_number=getattr(elem.metadata, 'page_number', 0)
                if hasattr(elem, 'metadata') else 0,
                char_start=char_offset,
                char_end=char_offset + len(text),
                content_length=len(text),
                element_category=elem_category,
            )

            sections.append(section)
            char_offset += len(text) + 2  # +2 for '\n\n' separator

        # ------------------------------------------------------------------ #
        # 2. Wire parent_id from true_parent_id
        # ------------------------------------------------------------------ #
        for feat, section in zip(hierarchy_features, sections):
            true_parent_elem_id = feat.get("true_parent_id")
            if true_parent_elem_id is None:
                section.parent_id = None
                continue

            parent_section_id = elemid_to_sectionid.get(true_parent_elem_id)
            section.parent_id = parent_section_id

        # ------------------------------------------------------------------ #
        # 3. Compute children_ids
        # ------------------------------------------------------------------ #
        section_by_id: Dict[str, DocumentSection] = {s.section_id: s for s in sections}

        for section in sections:
            if section.parent_id and section.parent_id in section_by_id:
                parent = section_by_id[section.parent_id]
                parent.children_ids.append(section.section_id)

        # ------------------------------------------------------------------ #
        # 4. Compute ancestors, path, and depth
        # ------------------------------------------------------------------ #
        def compute_ancestors_and_depth(sec: DocumentSection) -> None:
            if sec.ancestors:
                # Already computed
                return

            ancestors: List[str] = []
            current = sec
            # Walk up until no parent
            while current.parent_id:
                parent = section_by_id.get(current.parent_id)
                if not parent:
                    break
                # Use parent.title if available, otherwise a short prefix of content
                label = (parent.title or parent.content[:100]).strip()
                if label:
                    ancestors.insert(0, label)
                current = parent

            sec.ancestors = ancestors
            # Depth: number of ancestors + 1 (so roots are depth=1)
            sec.depth = len(ancestors) + 1
            path_parts = ancestors + ([sec.title] if sec.title else [])
            sec.path = " > ".join(path_parts) if path_parts else "Untitled"

        for section in sections:
            compute_ancestors_and_depth(section)

        # ------------------------------------------------------------------ #
        # 5. Validate cross-page relationships
        # ------------------------------------------------------------------ #
        self._validate_page_boundary_hierarchy(sections)

        return sections

    def _extract_tables(self, elements: List[Element]) -> List[TableData]:
        """Extract tables from elements."""
        tables = []

        for i, elem in enumerate(elements):
            if hasattr(elem, 'category') and elem.category == "Table":
                # Extract table data
                # Note: Unstructured table format may vary
                table_data = TableData(
                    table_id=f"table_{i+1}",
                    caption=getattr(elem.metadata, 'caption', None) if hasattr(elem, 'metadata') else None,
                    num_rows=0,  # TODO: Extract from element
                    num_cols=0,  # TODO: Extract from element
                    data=[],  # TODO: Parse table structure
                    bbox=getattr(elem.metadata, 'coordinates', None) if hasattr(elem, 'metadata') else None,
                    page_number=getattr(elem.metadata, 'page_number', 0) if hasattr(elem, 'metadata') else 0,
                )
                tables.append(table_data)

        return tables

    def _generate_markdown(self, sections: List[DocumentSection]) -> str:
        """Generate markdown from sections."""
        lines = []

        for section in sections:
            # Add heading based on level
            if section.title:
                heading_level = min(section.level + 1, 6)
                lines.append("#" * heading_level + " " + section.title)

            # Add content
            if section.content and section.content != section.title:
                lines.append(section.content)

            lines.append("")  # Empty line

        return "\n".join(lines)

    def _count_pages(self, elements: List[Element]) -> int:
        """Count number of pages."""
        pages = set()
        for elem in elements:
            if hasattr(elem, 'metadata') and hasattr(elem.metadata, 'page_number'):
                pages.add(elem.metadata.page_number)
        return len(pages)

    def _extract_title(self, sections: List[DocumentSection]) -> Optional[str]:
        """Extract document title from sections."""
        for section in sections:
            if section.level == 0 and section.title:
                return section.title
        return None
