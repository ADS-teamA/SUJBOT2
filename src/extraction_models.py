"""
Extraction models re-exported for cleaner imports.

These models are the canonical data structures used throughout the indexing pipeline.
They are defined in unstructured_extractor.py but re-exported here for cleaner imports.

SSOT: The actual implementation is in unstructured_extractor.py.
This module only provides a clean import path.

Usage:
    from src.extraction_models import ExtractedDocument, DocumentSection
"""

from src.unstructured_extractor import (
    DocumentSection,
    ExtractedDocument,
)

__all__ = [
    "DocumentSection",
    "ExtractedDocument",
]
