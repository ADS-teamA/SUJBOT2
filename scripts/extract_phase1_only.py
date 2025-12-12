#!/usr/bin/env python3
"""
Phase 1 Extraction Only - Extract document structure and hierarchy.

This script runs ONLY Phase 1 extraction (no summaries, chunking, or embeddings)
and saves the results to a specified output directory.

Usage:
    uv run python scripts/extract_phase1_only.py                    # All docs from data/
    uv run python scripts/extract_phase1_only.py --output dataset/  # Custom output dir
    uv run python scripts/extract_phase1_only.py data/single.pdf    # Single document
"""

import sys
import json
import logging
import argparse
from pathlib import Path

# Apply nest_asyncio early (required for some extraction backends)
import nest_asyncio
nest_asyncio.apply()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import extraction pipeline
from src.unified_extraction_pipeline import UnifiedDocumentPipeline
from src.unstructured_extractor import ExtractionConfig


def extract_phase1(document_path: Path, output_dir: Path) -> Path:
    """
    Extract Phase 1 (structure & hierarchy) for a single document.

    Args:
        document_path: Path to the document
        output_dir: Directory to save the output JSON

    Returns:
        Path to the saved JSON file
    """
    doc_name = document_path.stem.replace(" ", "_").replace("(", "").replace(")", "")

    logger.info(f"Extracting Phase 1 for: {document_path.name}")

    # Initialize extractor with default config
    config = ExtractionConfig.from_env()
    extractor = UnifiedDocumentPipeline(config)

    # Run extraction (Phase 1 only)
    result = extractor.process_document(document_path)

    # Prepare Phase 1 export data
    phase1_export = {
        "document_id": result.document_id,
        "source_path": str(document_path),
        "num_sections": result.num_sections,
        "hierarchy_depth": result.hierarchy_depth,
        "num_roots": result.num_roots,
        "num_tables": result.num_tables,
        "sections": [
            {
                "section_id": s.section_id,
                "title": s.title,
                "content": s.content,
                "level": s.level,
                "depth": s.depth,
                "path": s.path,
                "page_number": s.page_number,
                "content_length": len(s.content),
            }
            for s in result.sections
        ],
    }

    # Save to output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{doc_name}_phase1.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(phase1_export, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved: {output_path} ({result.num_sections} sections)")
    return output_path


def get_supported_documents(directory: Path) -> list:
    """Get list of all supported documents in directory."""
    supported_formats = [".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm", ".txt"]
    documents = []

    for ext in supported_formats:
        documents.extend(directory.glob(f"*{ext}"))
        documents.extend(directory.glob(f"*{ext.upper()}"))

    return sorted(set(documents))


def main():
    parser = argparse.ArgumentParser(
        description="Extract Phase 1 (document structure) only",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "input_path",
        type=str,
        nargs="?",
        default="data/",
        help="Path to document file or directory (default: data/)"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default="dataset/",
        help="Output directory for Phase 1 JSON files (default: dataset/)"
    )

    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_dir = Path(args.output)

    if not input_path.exists():
        print(f"Error: Path not found: {input_path}")
        sys.exit(1)

    # Collect documents to process
    if input_path.is_dir():
        documents = get_supported_documents(input_path)
        if not documents:
            print(f"Error: No supported documents found in {input_path}")
            sys.exit(1)
        print(f"\n{'='*60}")
        print(f"PHASE 1 EXTRACTION - {len(documents)} documents")
        print(f"{'='*60}")
        print(f"Input:  {input_path}")
        print(f"Output: {output_dir}")
        print(f"Documents:")
        for doc in documents:
            print(f"  - {doc.name}")
        print()
    else:
        documents = [input_path]

    # Process each document
    results = []
    for i, doc_path in enumerate(documents, 1):
        print(f"\n[{i}/{len(documents)}] Processing: {doc_path.name}")
        print("-" * 60)

        try:
            output_path = extract_phase1(doc_path, output_dir)
            results.append((doc_path.name, "OK", output_path))
        except Exception as e:
            logger.error(f"Failed to extract {doc_path.name}: {e}", exc_info=True)
            results.append((doc_path.name, "FAILED", str(e)))

    # Summary
    print(f"\n{'='*60}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*60}")

    success = sum(1 for _, status, _ in results if status == "OK")
    print(f"Success: {success}/{len(results)}")
    print(f"\nResults:")
    for name, status, path_or_error in results:
        if status == "OK":
            print(f"  [OK] {name} -> {path_or_error}")
        else:
            print(f"  [FAIL] {name}: {path_or_error}")

    print(f"\nOutput saved to: {output_dir}")


if __name__ == "__main__":
    main()
