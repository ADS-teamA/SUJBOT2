#!/usr/bin/env python3
"""
Test PHASE 1 + PHASE 2 extraction pipeline.

PHASE 1: Smart hierarchy extraction (font-size based)
PHASE 2: Generic summary generation (gpt-4o-mini, 150 chars)

Based on PIPELINE.md research findings.
"""

import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from unstructured_extractor import UnstructuredExtractor, ExtractionConfig


def test_phase1_only(pdf_path: Path, output_dir: Path):
    """Test PHASE 1: Smart hierarchy extraction WITHOUT summaries."""

    print("=" * 80)
    print("TEST 1: PHASE 1 ONLY (Smart Hierarchy)")
    print("=" * 80)
    print()

    config = ExtractionConfig(
        strategy="hi_res",  # Use hi_res strategy for accuracy
        model="detectron2_mask_rcnn",  # Most accurate model
        generate_summaries=False,  # Disable PHASE 2
        extract_tables=True,
    )

    extractor = UnstructuredExtractor(config)

    print(f"Extracting: {pdf_path.name}")
    result = extractor.extract(pdf_path)

    print()
    print("Results:")
    print(f"  Sections: {result.num_sections}")
    print(f"  Hierarchy depth: {result.hierarchy_depth}")
    print(f"  Root sections: {result.num_roots}")
    print(f"  Tables: {result.num_tables}")
    print(f"  Total chars: {result.total_chars}")

    # Save to JSON
    output_path = output_dir / f"{pdf_path.stem}_phase1_only.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    print(f"\n✓ Saved to: {output_path}")

    # Print hierarchy sample
    print()
    print("Hierarchy sample (first 10 sections):")
    for section in result.sections[:10]:
        indent = "  " * section.depth
        print(f"  {indent}[L{section.level}, D{section.depth}] {section.title[:50]}")

    return result


def test_phase1_phase2(pdf_path: Path, output_dir: Path):
    """Test PHASE 1 + PHASE 2: Smart hierarchy + Generic summaries."""

    print()
    print("=" * 80)
    print("TEST 2: PHASE 1 + PHASE 2 (Smart Hierarchy + Summaries)")
    print("=" * 80)
    print()

    config = ExtractionConfig(
        strategy="hi_res",  # Use hi_res strategy for accuracy
        model="detectron2_mask_rcnn",  # Most accurate model
        generate_summaries=True,  # Enable PHASE 2
        summary_model="gpt-4o-mini",
        summary_max_chars=150,
        summary_style="generic",
        extract_tables=True,
    )

    # NOTE: Requires OPENAI_API_KEY environment variable
    try:
        extractor = UnstructuredExtractor(config)
    except ValueError as e:
        print(f"⚠️  Skipping PHASE 2 test: {e}")
        print("   Set OPENAI_API_KEY environment variable to enable summary generation")
        return None

    print(f"Extracting: {pdf_path.name}")
    result = extractor.extract(pdf_path)

    print()
    print("Results:")
    print(f"  Sections: {result.num_sections}")
    print(f"  Hierarchy depth: {result.hierarchy_depth}")
    print(f"  Root sections: {result.num_roots}")
    print(f"  Tables: {result.num_tables}")
    print(
        f"  Document summary: {result.document_summary[:80] if result.document_summary else 'N/A'}..."
    )

    # Save to JSON
    output_path = output_dir / f"{pdf_path.stem}_phase1_phase2.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    print(f"\n✓ Saved to: {output_path}")

    # Print sample sections with summaries
    print()
    print("Section summaries (first 5):")
    for i, section in enumerate(result.sections[:5], 1):
        print(f"\n{i}. [{section.level}] {section.title[:50]}")
        print(f"   Summary: {section.summary}")
        print(f"   Length: {len(section.summary)} chars" if section.summary else "   No summary")

    return result


def compare_results(phase1_result, phase2_result):
    """Compare PHASE 1 vs PHASE 1+2 results."""

    if not phase2_result:
        return

    print()
    print("=" * 80)
    print("COMPARISON: PHASE 1 vs PHASE 1+2")
    print("=" * 80)
    print()

    print(f"{'Metric':<30} {'PHASE 1 Only':<20} {'PHASE 1+2':<20}")
    print("-" * 80)
    print(f"{'Sections':<30} {phase1_result.num_sections:<20} {phase2_result.num_sections:<20}")
    print(
        f"{'Hierarchy depth':<30} {phase1_result.hierarchy_depth:<20} {phase2_result.hierarchy_depth:<20}"
    )
    print(
        f"{'Document summary':<30} {'No':<20} {'Yes' if phase2_result.document_summary else 'No':<20}"
    )

    # Count sections with summaries
    sections_with_summaries = sum(1 for s in phase2_result.sections if s.summary)
    print(f"{'Sections with summaries':<30} {'0':<20} {sections_with_summaries:<20}")


def main():
    # Test document
    pdf_path = Path("data/regulace/GRI/GRI 306_ Effluents and Waste 2016.pdf")

    if not pdf_path.exists():
        print(f"Error: {pdf_path} not found")
        print("Please add a test PDF to test the extraction")
        sys.exit(1)

    output_dir = Path("output/phase1_phase2_tests")

    # Run tests
    phase1_result = test_phase1_only(pdf_path, output_dir)
    phase2_result = test_phase1_phase2(pdf_path, output_dir)

    # Compare
    compare_results(phase1_result, phase2_result)

    print()
    print("=" * 80)
    print("✓ ALL TESTS COMPLETED")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Review extracted hierarchy in JSON files")
    print("  2. Test on your nuclear documentation PDFs")
    print("  3. Proceed to PHASE 3: Multi-layer chunking with SAC")


if __name__ == "__main__":
    main()
