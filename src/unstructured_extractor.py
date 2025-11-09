"""
PHASE 1: Document Extraction using Unstructured.io

Multi-format document extraction replacing Docling (v1.x → v2.x).

Supported Formats (tested):
- PDF (.pdf) - detectron2_mask_rcnn (Mask R-CNN X_101_32x8d_FPN_3x - most accurate)
- PowerPoint (.pptx, .ppt) - presentation structure and speaker notes
- Word (.docx, .doc) - document structure and track changes
- HTML (.html, .htm) - web content with semantic tags
- Plain text (.txt) - basic text files
- LaTeX (.tex, .latex) - scientific documents with math

Additional formats supported via partition_auto() fallback - see Unstructured.io docs.

Migration from Docling:
- Reason: Improved § paragraph detection in legal documents (10/10 vs 0/10 on test doc Sb_1997_18)
- Breaking changes: Configuration env vars, data structure fields extended
- See README.md "Migration Guide" for upgrade instructions

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

    Previously used with Docling (v1.x), now populated by Unstructured.io (v2.x).
    Field structure remains stable to preserve existing vector stores and tests.
    """

    section_id: str
    title: str
    content: str
    level: int  # Hierarchy level (0=root, 1=major, 2=chapter, 3=section, etc.)
    depth: int  # Depth in hierarchy tree (1=root, 2=child of root, etc.)
    parent_id: Optional[str]
    children_ids: List[str]
    ancestors: List[str]  # List of parent titles
    path: str  # Full path: "Chapter 1 > Section 1.1 > Subsection 1.1.1"
    page_number: int
    char_start: int
    char_end: int
    content_length: int

    # Unstructured.io element tracking
    element_id: Optional[str] = None  # Original element.id from Unstructured
    unstructured_parent_id: Optional[str] = None  # Original parent_id from Unstructured
    element_type: Optional[str] = None  # Element type: Title, NarrativeText, ListItem, etc.
    element_category: Optional[str] = None  # Element category from Unstructured

    # PHASE 2: Summaries
    summary: Optional[str] = None  # 150-char generic summary

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
            "element_id": self.element_id,
            "unstructured_parent_id": self.unstructured_parent_id,
            "element_type": self.element_type,
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

    Previously populated by Docling (v1.x), now by Unstructured.io (v2.x).
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
    extraction_method: str = "unstructured_detectron2"
    config: Optional[Dict] = None

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
    """Configuration for Unstructured extraction."""

    # Model configuration
    strategy: str = "hi_res"  # "hi_res", "fast", "ocr_only"
    model: str = "detectron2_mask_rcnn"  # "detectron2_mask_rcnn" (most accurate), "yolox" (faster)

    # Language detection
    languages: List[str] = field(default_factory=lambda: ["ces", "eng"])
    detect_language_per_element: bool = True

    # Table extraction
    infer_table_structure: bool = True
    extract_images: bool = False
    include_page_breaks: bool = True

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
            strategy=os.getenv("UNSTRUCTURED_STRATEGY", "hi_res"),
            model=os.getenv("UNSTRUCTURED_MODEL", "detectron2_mask_rcnn"),
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
        Applied automatically in _build_sections() and detect_element_type_score()
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


def detect_element_type_score(element: Element) -> Tuple[str, float]:
    """
    Detect element type and assign hierarchy score.

    Returns:
        Tuple of (type_name, score)
        - score: 0.0 (low hierarchy) to 1.0 (high hierarchy)
    """
    category = element.category if hasattr(element, 'category') else 'Unknown'
    text = _normalize_text_diacritics(str(element)).strip()

    # Title elements are high priority
    if category == "Title":
        # Check for document title patterns
        if any(kw in text.upper() for kw in ["ZÁKON", "VYHLÁŠKA", "NAŘÍZENÍ", "SMĚRNICE"]):
            return ("Document Title", 1.0)
        # Check for major parts
        if re.search(r"^(ČÁST|PART|CHAPTER|HLAVA|ARTICLE)\s+[IVX\d]+", text, re.IGNORECASE):
            return ("Major Heading", 0.9)
        # Check for paragraphs
        if re.match(r"^§\s*\d+", text):
            return ("Paragraph", 0.8)
        # Generic title
        return ("Title", 0.7)

    # ListItem indicates subsections
    if category == "ListItem":
        # Numbered subsections
        if re.match(r"^\(\d+\)", text):
            return ("Subsection", 0.5)
        # Lettered items
        if re.match(r"^[a-z]\)", text):
            return ("Item", 0.4)
        # Numbered items
        if re.match(r"^\d+\.", text):
            return ("Sub-item", 0.3)
        return ("ListItem", 0.5)

    # Narrative text is low priority
    if category == "NarrativeText":
        return ("Narrative Text", 0.2)

    # Headers/footers are outside hierarchy
    if category in ["Header", "Footer", "PageBreak"]:
        return (category, 0.0)

    # Tables
    if category == "Table":
        return ("Table", 0.6)

    return ("Unknown", 0.3)


def detect_hierarchy_generic(elements: List[Element], config: ExtractionConfig) -> List[Dict]:
    """
    Generic hierarchy detection based on parent_id relationships.

    Uses Unstructured.io's element.id and parent_id to build clean hierarchy tree.
    NOT language or document-type specific.

    Args:
        elements: List of Unstructured elements
        config: Extraction configuration

    Returns:
        List of dicts with hierarchy metadata for each element
    """
    logger.info(f"Detecting hierarchy for {len(elements)} elements (parent-based)")

    # First pass: Extract element.id and parent_id, build lookup table
    features = []
    id_to_index = {}  # Map element.id -> index in features list

    # Track last valid structural parent per page (for page break continuity)
    last_structural_by_page = {}
    current_page = None

    for i, elem in enumerate(elements):
        # Get element's unique ID (Unstructured provides elem.id, or elem.element_id for custom)
        element_id = None
        if hasattr(elem, 'id') and elem.id:
            element_id = elem.id
        elif hasattr(elem, 'element_id') and elem.element_id:
            element_id = elem.element_id
        else:
            element_id = f"elem_{i}"

        # Get parent_id from metadata
        parent_id = getattr(elem.metadata, 'parent_id', None) if hasattr(elem, 'metadata') else None

        # Check for pre-computed level/depth from custom parsers (e.g., LaTeX)
        preset_level = getattr(elem.metadata, 'section_level', None) if hasattr(elem, 'metadata') else None
        preset_depth = getattr(elem.metadata, 'section_depth', None) if hasattr(elem, 'metadata') else None

        # Track page number
        page_number = getattr(elem.metadata, 'page_number', None) if hasattr(elem, 'metadata') else None

        type_name, type_score = detect_element_type_score(elem)

        # Check if element is structural (Title, ListItem) - potential parent
        elem_category = elem.category if hasattr(elem, 'category') else None
        is_structural = elem_category in ["Title", "ListItem"]

        # Page break continuity: if no parent but page changed, use last structural from previous page
        if parent_id is None and page_number is not None and current_page is not None:
            if page_number != current_page and current_page in last_structural_by_page:
                # Inherit parent from last structural element on previous page
                parent_id = last_structural_by_page[current_page]
                logger.debug(f"Page break detected (page {current_page} -> {page_number}): "
                           f"element {i} inherits parent from previous page")

        # Update current page tracking
        if page_number is not None:
            current_page = page_number
            if is_structural and parent_id is not None:
                # Track this as potential parent for next page
                last_structural_by_page[current_page] = element_id

        features.append({
            "element": elem,
            "index": i,
            "element_id": element_id,
            "parent_id": parent_id,
            "type_name": type_name,
            "type_score": type_score,
            "level": preset_level,  # Use preset if available, otherwise compute in second pass
            "depth": preset_depth,  # Use preset if available
            "page_number": page_number,
        })

        # Build lookup: element ID -> index
        id_to_index[element_id] = i

    # Second pass: Calculate level and depth based on parent hierarchy (skip if preset)
    def get_level(feat_index: int, visited: set = None) -> int:
        """
        Recursively calculate level based on parent chain.

        Args:
            feat_index: Index of feature in features list
            visited: Set of visited indices (to detect cycles)

        Returns:
            Level (0 = root, 1 = child of root, etc.)
        """
        if visited is None:
            visited = set()

        # Cycle detection
        if feat_index in visited:
            logger.warning(f"Cycle detected in parent hierarchy at index {feat_index}")
            return 0

        visited.add(feat_index)
        feat = features[feat_index]

        # If level already computed or preset, return it (memoization)
        if feat["level"] is not None:
            return feat["level"]

        # No parent → root level
        if feat["parent_id"] is None:
            feat["level"] = 0
            return 0

        # Has parent → find parent by ID
        parent_index = id_to_index.get(feat["parent_id"])

        if parent_index is None:
            # Parent not found (shouldn't happen with valid Unstructured output)
            logger.debug(f"Parent ID {feat['parent_id']} not found for element {feat_index}")
            feat["level"] = 0
            return 0

        # Recursive: get parent's level and add 1
        parent_level = get_level(parent_index, visited.copy())
        feat["level"] = parent_level + 1
        return feat["level"]

    # Calculate levels for all elements
    for i in range(len(features)):
        if features[i]["level"] is None:
            get_level(i)

    # Log level distribution
    level_counts = {}
    for feat in features:
        level = feat["level"]
        level_counts[level] = level_counts.get(level, 0) + 1

    logger.info(f"Hierarchy detection complete: {len(features)} elements")
    logger.info(f"Level distribution: {dict(sorted(level_counts.items()))}")

    return features


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
        logger.info(f"UnstructuredExtractor initialized: model={self.config.model}, strategy={self.config.strategy}")

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
        if self.config.enable_generic_hierarchy:
            hierarchy_features = detect_hierarchy_generic(elements, self.config)
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
            from summary_generator import SummaryGenerator
            from config import SummarizationConfig

            # Create summarization config
            summary_config = SummarizationConfig(
                model=self.config.summary_model,
                max_chars=self.config.summary_max_chars,
                style=self.config.summary_style,
            )

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
            extraction_method=f"unstructured_{self.config.model}",
            config=self.config.__dict__,
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
        logger.info(f"Partitioning {file_suffix} document with strategy={self.config.strategy}")

        # Common parameters for all formats
        common_params = {
            "filename": str(file_path),
            "languages": self.config.languages if hasattr(self.config, 'languages') else ["ces", "eng"],
            "include_page_breaks": self.config.include_page_breaks if hasattr(self.config, 'include_page_breaks') else True,
        }

        try:
            # PDF - use specialized function with hi_res models
            if file_suffix == ".pdf":
                logger.info(f"Using partition_pdf with model={self.config.model}")
                # Available Unstructured models:
                # - "yolox" (default, fast)
                # - "detectron2_onnx" (Faster R-CNN R_50_FPN_3x)
                # - "detectron2_mask_rcnn" (Mask R-CNN X_101_32x8d_FPN_3x, MOST ACCURATE)
                # - "detectron2_quantized" (quantized for speed)
                elements = partition_pdf(
                    **common_params,
                    strategy=self.config.strategy,
                    hi_res_model_name=self.config.model if self.config.strategy == "hi_res" else None,
                    infer_table_structure=self.config.infer_table_structure,
                    extract_images_in_pdf=self.config.extract_images,
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

    def _build_sections(self, hierarchy_features: List[Dict]) -> List[DocumentSection]:
        """Build DocumentSection objects from hierarchy features."""
        sections = []
        char_offset = 0

        # Build parent-child relationships
        for i, feat in enumerate(hierarchy_features):
            elem = feat["element"]
            text = _normalize_text_diacritics(str(elem))

            # Extract title from element
            # Use element category directly - more reliable than type_name
            elem_category = elem.category if hasattr(elem, 'category') else None

            if elem_category in ["Title", "ListItem"]:
                # Title or ListItem - use as section title
                title = text.strip()
            elif elem_category == "UncategorizedText" and (
                # Check if it looks like a structural heading
                re.match(r'^§\s*\d+', text) or  # § paragraph
                re.match(r'^(ČÁST|HLAVA|ČLÁNEK|CHAPTER|ARTICLE)\s+[IVX\d]+', text, re.IGNORECASE) or  # Major heading
                re.match(r'^\(\d+\)', text)  # Subsection (1), (2), etc.
            ):
                # UncategorizedText that looks structural - use as title
                title = text.strip()
            elif feat["type_name"] in ["Document Title", "Major Heading", "Paragraph"]:
                # Fallback: check type_name for custom-detected types
                title = text.strip()
            else:
                # NarrativeText and other content - no title, only content
                title = ""

            # Find parent
            parent_id = None
            ancestors = []
            depth = 1

            # Look backwards for parent (higher hierarchy level)
            current_level = feat["level"]
            for j in range(i - 1, -1, -1):
                if hierarchy_features[j]["level"] < current_level:
                    parent_id = f"sec_{j+1}"
                    depth = hierarchy_features[j].get("depth", 1) + 1

                    # Collect ancestors - walk up parent chain collecting only progressively higher levels
                    k = j
                    prev_level = current_level  # Track previous level to ensure we only go up
                    while k >= 0:
                        elem_k = hierarchy_features[k]["element"]
                        elem_k_level = hierarchy_features[k]["level"]
                        elem_k_category = elem_k.category if hasattr(elem_k, 'category') else None

                        # Only add if this is at a higher level (lower number) than previous
                        if elem_k_level < prev_level:
                            # Skip Headers, Footers, PageBreaks - they're not structural content
                            if elem_k_category not in ["Header", "Footer", "PageBreak"]:
                                ancestor_title = str(elem_k).strip()[:100]
                                if ancestor_title:  # Only add non-empty titles
                                    ancestors.insert(0, ancestor_title)
                            prev_level = elem_k_level  # Update for next iteration

                        k -= 1
                        if elem_k_level == 0:  # Reached root
                            break
                    break

            # Store depth for parent lookup
            feat["depth"] = depth

            # Build path
            path_parts = ancestors + ([title] if title else [])
            path = " > ".join(path_parts) if path_parts else "Untitled"

            section = DocumentSection(
                section_id=f"sec_{i+1}",
                title=title,
                content=text,
                level=feat["level"],
                depth=depth,
                parent_id=parent_id,
                children_ids=[],  # Will be populated below
                ancestors=ancestors,
                path=path,
                page_number=getattr(elem.metadata, 'page_number', 0) if hasattr(elem, 'metadata') else 0,
                char_start=char_offset,
                char_end=char_offset + len(text),
                content_length=len(text),
                element_id=feat.get("element_id"),  # Track Unstructured element ID
                unstructured_parent_id=feat.get("parent_id"),  # Track original parent_id
                element_type=type(elem).__name__,  # e.g., "Title", "NarrativeText", "ListItem"
                element_category=elem.category if hasattr(elem, 'category') else None,
            )

            sections.append(section)
            char_offset += len(text) + 2  # +2 for \n\n separator

        # Populate children_ids
        for section in sections:
            if section.parent_id:
                parent_idx = int(section.parent_id.split("_")[1]) - 1
                if 0 <= parent_idx < len(sections):
                    sections[parent_idx].children_ids.append(section.section_id)

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
