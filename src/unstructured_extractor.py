"""
PHASE 1: Document Extraction using Unstructured.io

Replaces Docling with Unstructured.io for document extraction.

Features:
- detectron2 backend (most accurate hi_res model)
- Per-element language detection
- Bbox orientation analysis for rotated text filtering
- Generic hierarchy detection (not language-specific)
- Backward compatible with existing pipeline

Based on research:
- Unstructured.io: 100% § paragraph detection (vs 0% Docling)
- Parent ID relationships for explicit hierarchy
- Element categorization (Title, ListItem, NarrativeText)
"""

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import math

import numpy as np

# Unstructured imports
from unstructured.partition.pdf import partition_pdf
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
# DATA STRUCTURES (Backward Compatible with Docling)
# ============================================================================


@dataclass
class DocumentSection:
    """
    Hierarchical document section.

    Backward compatible with Docling's DocumentSection.
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

    Backward compatible with Docling's ExtractedDocument.
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

    @classmethod
    def from_env(cls) -> "ExtractionConfig":
        """Load configuration from environment variables."""
        import os

        return cls(
            strategy=os.getenv("UNSTRUCTURED_STRATEGY", "hi_res"),
            model=os.getenv("UNSTRUCTURED_MODEL", "detectron2_mask_rcnn"),
            languages=os.getenv("UNSTRUCTURED_LANGUAGES", "ces,eng").split(","),
            detect_language_per_element=os.getenv(
                "UNSTRUCTURED_DETECT_LANGUAGE_PER_ELEMENT", "true"
            ).lower() == "true",
            infer_table_structure=os.getenv(
                "UNSTRUCTURED_INFER_TABLE_STRUCTURE", "true"
            ).lower() == "true",
            extract_images=os.getenv(
                "UNSTRUCTURED_EXTRACT_IMAGES", "false"
            ).lower() == "true",
            filter_rotated_text=os.getenv(
                "FILTER_ROTATED_TEXT", "true"
            ).lower() == "true",
            rotation_method=os.getenv("ROTATION_METHOD", "bbox_orientation"),
            rotation_min_angle=float(os.getenv("ROTATION_MIN_ANGLE", "25.0")),
            rotation_max_angle=float(os.getenv("ROTATION_MAX_ANGLE", "65.0")),
            enable_generic_hierarchy=os.getenv(
                "ENABLE_GENERIC_HIERARCHY", "true"
            ).lower() == "true",
            hierarchy_signals=os.getenv(
                "HIERARCHY_SIGNALS", "type,font_size,spacing,numbering,parent_id"
            ).split(","),
            hierarchy_clustering_eps=float(os.getenv("HIERARCHY_CLUSTERING_EPS", "0.15")),
            hierarchy_clustering_min_samples=int(
                os.getenv("HIERARCHY_CLUSTERING_MIN_SAMPLES", "2")
            ),
            generate_markdown=os.getenv("GENERATE_MARKDOWN", "true").lower() == "true",
            generate_json=os.getenv("GENERATE_JSON", "true").lower() == "true",
        )


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
    text = str(element).strip()

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
        # Get element's unique ID (Unstructured provides elem.id)
        element_id = elem.id if hasattr(elem, 'id') else f"elem_{i}"

        # Get parent_id from metadata
        parent_id = getattr(elem.metadata, 'parent_id', None) if hasattr(elem, 'metadata') else None

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
            "level": None,  # Will be computed in second pass
            "page_number": page_number,
        })

        # Build lookup: element ID -> index
        id_to_index[element_id] = i

    # Second pass: Calculate level based on parent hierarchy
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

        # If level already computed, return it (memoization)
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
    Document extractor using Unstructured.io.

    Replaces DoclingExtractorV2 with backward-compatible interface.
    """

    def __init__(self, config: Optional[ExtractionConfig] = None):
        """
        Initialize extractor.

        Args:
            config: Extraction configuration (defaults to env vars)
        """
        self.config = config or ExtractionConfig.from_env()
        logger.info(f"UnstructuredExtractor initialized: model={self.config.model}, strategy={self.config.strategy}")

    def extract(self, pdf_path: Path) -> ExtractedDocument:
        """
        Extract document structure from PDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            ExtractedDocument with hierarchical sections
        """
        logger.info(f"Starting extraction of {pdf_path.name}")
        start_time = time.time()

        # Extract with Unstructured
        elements = self._partition_pdf(pdf_path)

        # Filter rotated text
        if self.config.filter_rotated_text:
            elements = filter_rotated_elements(
                elements,
                self.config.rotation_min_angle,
                self.config.rotation_max_angle
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

        # Generate outputs
        full_text = "\n\n".join(str(elem) for elem in elements if hasattr(elem, '__str__'))
        markdown = self._generate_markdown(sections) if self.config.generate_markdown else ""
        json_content = "" # Will be generated during serialization

        extraction_time = time.time() - start_time

        # Build ExtractedDocument
        document_id = pdf_path.stem
        extracted_doc = ExtractedDocument(
            document_id=document_id,
            source_path=str(pdf_path),
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
            extraction_method=f"unstructured_{self.config.model}",
            config=self.config.__dict__,
        )

        logger.info(
            f"Extraction completed in {extraction_time:.2f}s: "
            f"{len(sections)} sections (depth={extracted_doc.hierarchy_depth}), "
            f"{len(tables)} tables"
        )

        return extracted_doc

    def _partition_pdf(self, pdf_path: Path) -> List[Element]:
        """Run Unstructured partition_pdf."""
        logger.info(f"Partitioning PDF with strategy={self.config.strategy}")

        # Available Unstructured models:
        # - "yolox" (default, fast)
        # - "detectron2_onnx" (Faster R-CNN R_50_FPN_3x)
        # - "detectron2_mask_rcnn" (Mask R-CNN X_101_32x8d_FPN_3x, MOST ACCURATE)
        # - "detectron2_quantized" (quantized for speed)
        elements = partition_pdf(
            filename=str(pdf_path),
            strategy=self.config.strategy,
            hi_res_model_name=self.config.model if self.config.strategy == "hi_res" else None,
            languages=self.config.languages,
            infer_table_structure=self.config.infer_table_structure,
            extract_images_in_pdf=self.config.extract_images,
            include_page_breaks=self.config.include_page_breaks,
        )

        logger.info(f"Partitioned PDF into {len(elements)} elements")
        return elements

    def _build_sections(self, hierarchy_features: List[Dict]) -> List[DocumentSection]:
        """Build DocumentSection objects from hierarchy features."""
        sections = []
        char_offset = 0

        # Build parent-child relationships
        for i, feat in enumerate(hierarchy_features):
            elem = feat["element"]
            text = str(elem)

            # Extract title from element
            if feat["type_name"] in ["Document Title", "Major Heading", "Paragraph", "Title"]:
                title = text.strip()
            else:
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
