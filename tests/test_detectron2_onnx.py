#!/usr/bin/env python3
"""
Extract PDF and save output using detectron2_onnx model.
"""

import sys
import json
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.unstructured_extractor import UnstructuredExtractor, ExtractionConfig

if __name__ == "__main__":
    # PDF to extract
    pdf_path = project_root / "data" / "Sb_1997_18_2017-01-01_IZ.pdf"

    if not pdf_path.exists():
        print(f"❌ PDF not found: {pdf_path}")
        sys.exit(1)

    print(f"🧪 Extracting: {pdf_path.name}")
    print(f"⚙️  Model: detectron2_onnx (hi_res strategy)\n")

    # Initialize with detectron2_onnx
    config = ExtractionConfig(
        strategy="hi_res",
        model="detectron2_onnx",  # Faster R-CNN R_50_FPN_3x
        languages=["ces", "eng"],
        detect_language_per_element=True,
        filter_rotated_text=True,
    )

    extractor = UnstructuredExtractor(config)

    # Extract
    print("🔄 Extracting...")
    extracted_doc = extractor.extract(pdf_path)
    print(f"✅ Extraction completed in {extracted_doc.extraction_time:.2f}s\n")

    # Prepare output directory
    output_dir = project_root / "phase1_output"
    output_dir.mkdir(exist_ok=True)

    # Save detailed hierarchy with levels
    output_file = output_dir / "unstructured_detectron2_onnx_detailed.txt"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write(f"UNSTRUCTURED.IO EXTRACTION OUTPUT (detectron2_onnx)\n")
        f.write(f"Document: {pdf_path.name}\n")
        f.write("="*80 + "\n\n")

        f.write(f"Model: detectron2_onnx (Faster R-CNN R_50_FPN_3x)\n")
        f.write(f"Strategy: hi_res\n")
        f.write(f"Extraction Time: {extracted_doc.extraction_time:.2f}s\n")
        f.write(f"Sections: {extracted_doc.num_sections}\n")
        f.write(f"Pages: {extracted_doc.num_pages}\n")
        f.write(f"Total Characters: {extracted_doc.total_chars:,}\n")
        f.write(f"Hierarchy Depth: {extracted_doc.hierarchy_depth}\n")
        f.write(f"Root Sections: {extracted_doc.num_roots}\n")
        f.write(f"Tables: {extracted_doc.num_tables}\n")

        # Count § paragraphs
        paragraph_sections = [s for s in extracted_doc.sections if s.title and s.title.startswith("§")]
        f.write(f"§ Paragraphs: {len(paragraph_sections)}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("HIERARCHY STRUCTURE\n")
        f.write("="*80 + "\n\n")

        for i, section in enumerate(extracted_doc.sections, 1):
            indent = "  " * section.depth
            f.write(f"{indent}[{i}] Level {section.level}: {section.title or '(no title)'}\n")
            f.write(f"{indent}    Path: {section.path}\n")
            f.write(f"{indent}    Page: {section.page_number}, Length: {section.content_length} chars\n")

            # Show content preview for § paragraphs
            if section.title and section.title.startswith("§"):
                preview = section.content[:200].replace("\n", " ")
                f.write(f"{indent}    Content preview: {preview}...\n")

            f.write("\n")

        f.write("="*80 + "\n")
        f.write("§ PARAGRAPHS SUMMARY\n")
        f.write("="*80 + "\n\n")

        for s in paragraph_sections:
            f.write(f"{s.title} (Level {s.level}, Page {s.page_number})\n")
            f.write(f"  Path: {s.path}\n")
            f.write(f"  Content: {s.content_length} chars\n\n")

    print(f"💾 Saved detailed output to:")
    print(f"   {output_file}")

    # Also save as JSON for programmatic access
    json_file = output_dir / "unstructured_detectron2_onnx_extraction.json"

    json_output = {
        "document_id": extracted_doc.document_id,
        "source_path": str(pdf_path),
        "model": "detectron2_onnx",
        "strategy": "hi_res",
        "extraction_time": extracted_doc.extraction_time,
        "num_sections": extracted_doc.num_sections,
        "num_pages": extracted_doc.num_pages,
        "total_chars": extracted_doc.total_chars,
        "hierarchy_depth": extracted_doc.hierarchy_depth,
        "num_roots": extracted_doc.num_roots,
        "num_tables": extracted_doc.num_tables,
        "num_paragraphs": len(paragraph_sections),
        "sections": [s.to_dict() for s in extracted_doc.sections]
    }

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)

    print(f"   {json_file}")
    print(f"\n✅ All outputs saved successfully!")

    # Print comparison summary
    print(f"\n📊 Quick Stats:")
    print(f"   ⏱️  Extraction time: {extracted_doc.extraction_time:.2f}s")
    print(f"   📊 Sections: {extracted_doc.num_sections}")
    print(f"   📝 § paragraphs detected: {len(paragraph_sections)}/10")
    print(f"   🌲 Hierarchy depth: {extracted_doc.hierarchy_depth}")
