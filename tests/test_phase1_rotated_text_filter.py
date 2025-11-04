"""
Integration test: verifies that rotated watermark text is removed during phase-1 extraction.

We convert the first 10 pages of the real BZ_VR1 PDF into a temporary document and ensure
that the extracted sections no longer contain the diagonal "NEPLATNÉ" watermark.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest
from PyPDF2 import PdfReader, PdfWriter

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
for logger_name in ("docling_extractor_v2", "src.docling_extractor_v2"):
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(handler)

from config import ExtractionConfig
from docling_extractor_v2 import DoclingExtractorV2

DATA_ROOT = Path(__file__).parent.parent / "data"
SOURCE_PDF = DATA_ROOT / "BZ_VR1.pdf"
MAX_PAGES = 10


@pytest.mark.slow
def test_rotated_watermark_removed(tmp_path):
    if not SOURCE_PDF.exists():
        pytest.skip(f"Test document not available: {SOURCE_PDF}")

    truncated_pdf = tmp_path / "BZ_VR1_first10.pdf"

    reader = PdfReader(str(SOURCE_PDF))
    writer = PdfWriter()

    pages_to_copy = min(MAX_PAGES, len(reader.pages))
    for idx in range(pages_to_copy):
        writer.add_page(reader.pages[idx])

    with truncated_pdf.open("wb") as output_file:
        writer.write(output_file)

    config = ExtractionConfig(
        enable_smart_hierarchy=True,
        generate_summaries=False,
        extract_tables=False,
        generate_markdown=False,
        generate_json=False,
        filter_rotated_text=True,
        rotation_min_angle=5.0,
        rotation_max_angle=85.0,
    )

    extractor = DoclingExtractorV2(config)
    result = extractor.extract(truncated_pdf)

    relevant_sections = [
        section for section in result.sections if 0 < section.page_number <= pages_to_copy
    ]
    assert relevant_sections, "Expected extracted sections within the first 10 pages"

    offending_sections = []
    target_tokens = ("NEPLATNÉ", "NEPLATNE", "NEPLATN")

    for section in relevant_sections:
        combined_text = " ".join(
            [section.title or "", section.path or "", section.content or ""]
        ).upper()
        if any(token in combined_text for token in target_tokens):
            offending_sections.append(section.section_id)

    assert not offending_sections, (
        "Watermark text should be removed from the first 10 pages; "
        f"found in sections: {offending_sections}"
    )
