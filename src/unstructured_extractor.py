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
            strategy=os.getenv("UNSTRUCTURED_STRATEGY", "hi_res"),
            model=os.getenv("UNSTRUCTURED_MODEL", "detectron2_mask_rcnn"),
            
            # --- OPRAVA 1: Pevný seznam jazyků ---
            languages=os.getenv("UNSTRUCTURED_LANGUAGES", "ces,eng").split(","),
            
            detect_language_per_element=get_bool_env("UNSTRUCTURED_DETECT_LANGUAGE_PER_ELEMENT", True),
            infer_table_structure=get_bool_env("UNSTRUCTURED_INFER_TABLE_STRUCTURE", True),
            extract_images=get_bool_env("UNSTRUCTURED_EXTRACT_IMAGES", False),
            filter_rotated_text=get_bool_env("FILTER_ROTATED_TEXT", True),
            rotation_method=os.getenv("ROTATION_METHOD", "bbox_orientation"),
            rotation_min_angle=min_angle,
            rotation_max_angle=max_angle,
            enable_generic_hierarchy=get_bool_env("ENABLE_GENERIC_HIERARCHY", True),
            
            # --- OPRAVA 2: Pevný seznam signálů hierarchie ---
            # Původně: ",".join(cls.hierarchy_signals) -> Způsobovalo AttributeError
            hierarchy_signals=os.getenv("HIERARCHY_SIGNALS", "type,font_size,spacing,numbering,parent_id").split(","),
            
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


def filter_rotated_elements(elements: List[Element], min_angle: float = 25.0,
                            max_angle: float = 65.0) -> List[Element]:
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
def build_robust_hierarchy(elements: List[Element], config: ExtractionConfig) -> List[Dict]:
    """
    Rebuild a robust element hierarchy, solving three key problems:

    1. Filters Headers/Footers (nuisance elements).
    2. Connects orphaned elements (page breaks) to the last known major parent.
    3. Detects siblings: prevents nesting of elements with similar structure
       (e.g., `§ 32` -> `§ 33`) using pattern signature matching.

    Returns:
        List[Dict] – one dict per *content* element with keys:
            - element: original Unstructured element
            - element_id: unique id
            - parent_id: original unstructured parent_id (for reference)
            - true_parent_id: cleaned parent id (None for roots)
            - level: integer depth in cleaned tree (0 = root)
            - page_number, category, signature (for titles), ...
    """
    logger.info(f"Building robust hierarchy for {len(elements)} elements...")

    if not elements:
        return []

    # ------------------------------------------------------------------ #
    # Helper: Structure Pattern Signature (for sibling detection)
    # ------------------------------------------------------------------ #
    def get_structure_signature(text: str) -> str:
        """
        Generate a 'signature' for a title to detect if two titles are siblings.

        Examples:
            '§ 34'                    -> '§ <num>'
            '§ 35'                    -> '§ <num>'          (same signature)
            'Chapter 1'               -> 'chapter <num>'
            '1.2.3 Title'             -> '<num>.<num>.<num>'
            'Article V'               -> 'article <rom>'
            'HLAVA V – Ustanovení...' -> 'hlava <rom>'
        """
        text = text.strip().lower()

        # Replace digits with <num>
        text = re.sub(r'\d+', '<num>', text)

        # Replace simple Roman numerals with <rom>
        text = re.sub(r'\b[ivx]+\b', '<rom>', text)

        tokens = text.split()
        if not tokens:
            return ""

        # High-signal structural words at the beginning
        markers = ['§', 'article', 'chapter', 'part', 'hlava', 'část', 'oddíl', 'článek']
        if tokens[0] in markers:
            return " ".join(tokens[:2]) if len(tokens) > 1 else tokens[0]

        # Numbering prefix like "1.", "1.2.", "(1)" etc.
        if '<num>' in tokens[0]:
            return tokens[0]

        # Fallback: first ~20 chars
        return text[:20]

    # ------------------------------------------------------------------ #
    # Pass 1: Collect raw features and mark headers/footers
    # ------------------------------------------------------------------ #
    features: List[Dict[str, Any]] = []
    id_to_index: Dict[str, int] = {}
    header_footer_ids: set[str] = set()

    for i, elem in enumerate(elements):
        element_id = getattr(elem, 'id', None) or getattr(elem, 'element_id', None) or f"elem_{i}"
        parent_id = getattr(elem.metadata, 'parent_id', None)
        page_number = getattr(elem.metadata, 'page_number', None)
        category = getattr(elem, 'category', 'Unknown')

        if category in ["Header", "Footer"]:
            header_footer_ids.add(element_id)

        feat = {
            "element": elem,
            "index": i,
            "element_id": element_id,
            "parent_id": parent_id,
            "page_number": page_number,
            "category": category,
            "true_parent_id": None,   # filled later
            "level": None,            # filled later
            "signature": get_structure_signature(str(elem)) if category == "Title" else None,
        }

        features.append(feat)
        id_to_index[element_id] = i

    logger.info(f"Pass 1 complete: found {len(header_footer_ids)} Header/Footer elements.")

    # ------------------------------------------------------------------ #
    # Pass 2: Resolve true structural parents (skip header/footer + siblings)
    # ------------------------------------------------------------------ #
    def get_true_structural_parent(elem_id: Optional[str],
                                   child_signature: Optional[str] = None) -> Optional[str]:
        """
        Recursively finds the first valid structural parent for a child.

        Rules:
        - Skip Header/Footer nodes.
        - If parent is a Title and shares the same signature as child Title
          (e.g. '§ <num>' vs '§ <num>'), treat them as siblings and skip
          to the parent’s parent.
        """
        if elem_id is None:
            return None

        if elem_id not in id_to_index:
            # Broken link, consider root
            return None

        parent_feat = features[id_to_index[elem_id]]

        # Case 1: parent is Header/Footer → skip it
        if parent_feat["element_id"] in header_footer_ids:
            return get_true_structural_parent(parent_feat["parent_id"], child_signature)

        # Case 2: parent is Title and we are Title → sibling detection
        if child_signature and parent_feat["category"] == "Title":
            parent_sig = parent_feat["signature"]
            if parent_sig and parent_sig == child_signature:
                # Same structural pattern → siblings → climb one level higher
                return get_true_structural_parent(parent_feat["parent_id"], child_signature)

        # Case 3: acceptable parent
        return parent_feat["element_id"]

    for feat in features:
        if feat["element_id"] in header_footer_ids:
            # We don't care about parent for header/footer in final content
            continue

        feat["true_parent_id"] = get_true_structural_parent(
            feat["parent_id"],
            child_signature=feat["signature"]
        )

    logger.info("Pass 2 complete: resolved true structural parents (with sibling detection).")

    # ------------------------------------------------------------------ #
    # Pass 3: Assign levels and fix page-break orphans
    # ------------------------------------------------------------------ #
    content_features: List[Dict[str, Any]] = []
    last_major_structural_parent_per_page: Dict[int, str] = {}

    # Index of all non-header/footer feats by id (for level recursion)
    id_to_clean_feat: Dict[str, Dict[str, Any]] = {
        feat["element_id"]: feat
        for feat in features
        if feat["element_id"] not in header_footer_ids
    }

    def compute_level(elem_id: Optional[str], seen: set[str]) -> int:
        """
        Recursively compute depth (0 = root) using cleaned parents.
        """
        if elem_id is None:
            return 0

        if elem_id in seen:
            logger.warning(f"Hierarchy cycle detected at {elem_id}")
            return 0

        feat = id_to_clean_feat.get(elem_id)
        if not feat:
            return 0

        if feat["level"] is not None:
            return feat["level"]

        seen.add(elem_id)
        parent_level = compute_level(feat["true_parent_id"], seen)
        feat["level"] = parent_level + 1
        return feat["level"]

    current_page: Optional[int] = None

    for feat in features:
        # Skip headers/footers entirely in final list
        if feat["element_id"] in header_footer_ids:
            continue

        page = feat["page_number"]
        if page is not None:
            current_page = page

        # Fix page-break orphans: if no parent, but we know a major title on this/prev page
        if feat["true_parent_id"] is None and current_page is not None:
            if current_page in last_major_structural_parent_per_page:
                feat["true_parent_id"] = last_major_structural_parent_per_page[current_page]
                logger.debug(
                    f"Page-break orphan {feat['element_id']} (p{current_page}) "
                    f"adopted by {feat['true_parent_id']} (same page)."
                )
            elif (current_page - 1) in last_major_structural_parent_per_page:
                feat["true_parent_id"] = last_major_structural_parent_per_page[current_page - 1]
                logger.debug(
                    f"Page-break orphan {feat['element_id']} (p{current_page}) "
                    f"adopted by {feat['true_parent_id']} (prev page)."
                )

        # Track last major title per page for orphan adoption
        if feat["category"] == "Title" and current_page is not None:
            last_major_structural_parent_per_page[current_page] = feat["element_id"]

        content_features.append(feat)

    # Compute levels for all remaining feats
    for feat in content_features:
        if feat["level"] is None:
            feat["level"] = compute_level(feat["element_id"], seen=set())

    logger.info(
        f"Pass 3 complete: built final hierarchy with {len(content_features)} content elements."
    )

    # Log simple level distribution for debugging
    level_counts: Dict[int, int] = {}
    for feat in content_features:
        lvl = feat["level"]
        level_counts[lvl] = level_counts.get(lvl, 0) + 1
    logger.info(f"Final level distribution: {dict(sorted(level_counts.items()))}")

    return content_features

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
            document_summary=None,  # PHASE 2: Hierarchical summary
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
        include_page_breaks = getattr(self.config, 'include_page_breaks', True)
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
                title = ""

            section_id = f"sec_{i + 1}"
            elemid_to_sectionid[feat["element_id"]] = section_id

            section = DocumentSection(
                section_id=section_id,
                title=title,
                content=text,
                level=feat["level"],  # level from robust hierarchy
                depth=0,  # filled later
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