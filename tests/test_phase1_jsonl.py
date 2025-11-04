#!/usr/bin/env python3
"""
Test PHASE 1 extraction with JSONL hierarchical output.

PHASE 1: Smart hierarchy extraction (font-size based)
Output: JSONL format - one JSON object per line (each section as a separate line)

Usage:
    python tests/test_phase1_jsonl.py <pdf_path>
    python tests/test_phase1_jsonl.py "data/regulace/GRI/GRI 306_ Effluents and Waste 2016.pdf"
"""

import json
import logging
import sys
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from config import ExtractionConfig
from docling_extractor_v2 import DoclingExtractorV2


def test_phase1_jsonl(pdf_path: Path, output_dir: Path = None):
    """
    Test PHASE 1: Smart hierarchy extraction with JSONL output.

    Args:
        pdf_path: Path to PDF document
        output_dir: Output directory (default: output/phase1_jsonl/)
    """

    print("=" * 80)
    print("PHASE 1: Smart Hierarchy Extraction (JSONL Output)")
    print("=" * 80)
    print()

    # Setup output directory
    if output_dir is None:
        output_dir = project_root / "output" / "phase1_jsonl"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure extraction
    config = ExtractionConfig(
        enable_smart_hierarchy=True,
        generate_summaries=False,  # PHASE 1 only, no summaries
        extract_tables=True,
        ocr_engine="rapidocr"
    )

    # Extract document
    print(f"Extracting: {pdf_path.name}")
    print()

    extractor = DoclingExtractorV2(config)
    result = extractor.extract(pdf_path)

    # Print extraction stats
    print("Extraction Results:")
    print(f"  Document ID: {result.document_id}")
    print(f"  Sections: {result.num_sections}")
    print(f"  Hierarchy depth: {result.hierarchy_depth}")
    print(f"  Root sections: {result.num_roots}")
    print(f"  Tables: {result.num_tables}")
    print(f"  Total chars: {result.total_chars}")
    print()

    # Generate output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{pdf_path.stem}_{timestamp}.jsonl"

    # Write JSONL output (one section per line)
    with open(output_file, "w", encoding="utf-8") as f:
        for section in result.sections:
            section_data = {
                "document_id": result.document_id,
                "section_id": section.section_id,
                "title": section.title,
                "level": section.level,
                "depth": section.depth,
                "path": section.path,
                "page_number": section.page_number,
                "content": section.content,
                "content_length": len(section.content),
                "parent_id": section.parent_id,
                "children_ids": section.children_ids,
            }
            # Write one JSON object per line
            f.write(json.dumps(section_data, ensure_ascii=False) + "\n")

    print(f"✓ JSONL output saved: {output_file}")
    print(f"  Total lines: {result.num_sections}")
    print()

    # Print hierarchy preview
    print("Hierarchy Preview (first 15 sections):")
    print()
    for i, section in enumerate(result.sections[:15], 1):
        indent = "  " * section.depth
        content_preview = section.content[:60].replace("\n", " ") if section.content else ""
        print(f"{i:3}. {indent}[L{section.level}, D{section.depth}] {section.title}")
        print(f"     {indent}Page: {section.page_number}, Chars: {len(section.content)}")
        if content_preview:
            print(f"     {indent}Content: {content_preview}...")
        print()

    # Print JSONL format info
    print()
    print("=" * 80)
    print("JSONL Format Information")
    print("=" * 80)
    print()
    print("Each line in the JSONL file contains one section as a JSON object with:")
    print("  - document_id: Unique document identifier")
    print("  - section_id: Unique section identifier")
    print("  - title: Section title")
    print("  - level: Heading level (1-6)")
    print("  - depth: Hierarchical depth (0 = root)")
    print("  - path: Full hierarchical path")
    print("  - page_number: Page number in PDF")
    print("  - content: Full section text content")
    print("  - content_length: Number of characters")
    print("  - parent_id: Parent section ID (null for roots)")
    print("  - children_ids: List of child section IDs")
    print()
    print("Reading JSONL files:")
    print("  Python: with open('file.jsonl') as f: for line in f: data = json.loads(line)")
    print("  Command line: cat file.jsonl | jq '.'")
    print()

    return result, output_file


def main():
    """Main entry point."""

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    if len(sys.argv) < 2:
        print("Usage: python tests/test_phase1_jsonl.py <pdf_path>")
        print()
        print("Example:")
        print(
            '  python tests/test_phase1_jsonl.py "data/regulace/GRI/GRI 306_ Effluents and Waste 2016.pdf"'
        )
        print()
        sys.exit(1)

    pdf_path = Path(sys.argv[1])

    if not pdf_path.exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)

    # Run test
    result, output_file = test_phase1_jsonl(pdf_path)

    print()
    print("=" * 80)
    print("✓ TEST COMPLETED")
    print("=" * 80)
    print()
    print(f"Output file: {output_file}")
    print()
    print("Next steps:")
    print("  1. Review the JSONL output file")
    print("  2. Process it line-by-line for streaming or large-scale processing")
    print("  3. Use jq or similar tools to filter/query the hierarchical data")
    print()


if __name__ == "__main__":
    main()
