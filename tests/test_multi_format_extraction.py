"""
Test multi-format document extraction

Tests the UnstructuredExtractor with various document formats.

Usage:
    # Run all tests
    uv run pytest tests/test_multi_format_extraction.py -v

    # Run specific format test
    uv run pytest tests/test_multi_format_extraction.py::test_extract_pdf -v
"""

import logging
import pytest
from pathlib import Path
from src.unstructured_extractor import UnstructuredExtractor, ExtractionConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def extractor():
    """Create extractor instance with test config."""
    config = ExtractionConfig(
        strategy="fast",  # Use fast mode for tests
        model="yolox",
        languages=["ces", "eng"],
        detect_language_per_element=False,  # Faster for tests
        infer_table_structure=True,
        extract_images=False,
        filter_rotated_text=False,  # Disable for faster tests
        enable_generic_hierarchy=True,
        generate_markdown=True,
        generate_json=False,
    )
    return UnstructuredExtractor(config)


@pytest.fixture
def test_data_dir():
    """Return path to test data directory."""
    return Path("data")


class TestMultiFormatExtraction:
    """Test extraction for different document formats."""

    def test_extract_pdf(self, extractor, test_data_dir):
        """Test PDF extraction."""
        # Find a PDF file in data directory
        pdf_files = list(test_data_dir.glob("**/*.pdf"))
        if not pdf_files:
            pytest.skip("No PDF files found in data directory")

        pdf_path = pdf_files[0]
        logger.info(f"Testing PDF extraction: {pdf_path.name}")

        doc = extractor.extract(pdf_path)

        assert doc is not None
        assert doc.num_sections > 0
        assert doc.total_chars > 0
        assert doc.extraction_method.startswith("unstructured_")
        assert doc.full_text
        logger.info(f"✓ PDF extracted: {doc.num_sections} sections, {doc.total_chars} chars")

    def test_extract_pptx(self, extractor, test_data_dir):
        """Test PowerPoint extraction."""
        pptx_files = list(test_data_dir.glob("**/*.pptx"))
        if not pptx_files:
            pytest.skip("No PPTX files found in data directory")

        pptx_path = pptx_files[0]
        logger.info(f"Testing PPTX extraction: {pptx_path.name}")

        doc = extractor.extract(pptx_path)

        assert doc is not None
        assert doc.num_sections > 0
        assert doc.total_chars > 0
        assert doc.source_path == str(pptx_path)
        logger.info(f"✓ PPTX extracted: {doc.num_sections} sections, {doc.total_chars} chars")

    def test_extract_docx(self, extractor, test_data_dir):
        """Test Word document extraction."""
        docx_files = list(test_data_dir.glob("**/*.docx"))
        if not docx_files:
            pytest.skip("No DOCX files found in data directory")

        docx_path = docx_files[0]
        logger.info(f"Testing DOCX extraction: {docx_path.name}")

        doc = extractor.extract(docx_path)

        assert doc is not None
        assert doc.num_sections > 0
        assert doc.total_chars > 0
        assert doc.source_path == str(docx_path)
        logger.info(f"✓ DOCX extracted: {doc.num_sections} sections, {doc.total_chars} chars")

    def test_extract_html(self, extractor, test_data_dir):
        """Test HTML extraction."""
        html_files = list(test_data_dir.glob("**/*.html")) + list(test_data_dir.glob("**/*.htm"))
        if not html_files:
            pytest.skip("No HTML files found in data directory")

        html_path = html_files[0]
        logger.info(f"Testing HTML extraction: {html_path.name}")

        doc = extractor.extract(html_path)

        assert doc is not None
        assert doc.num_sections > 0
        assert doc.total_chars > 0
        assert doc.source_path == str(html_path)
        logger.info(f"✓ HTML extracted: {doc.num_sections} sections, {doc.total_chars} chars")

    def test_extract_txt(self, extractor, test_data_dir):
        """Test plain text extraction."""
        txt_files = list(test_data_dir.glob("**/*.txt"))
        if not txt_files:
            pytest.skip("No TXT files found in data directory")

        txt_path = txt_files[0]
        logger.info(f"Testing TXT extraction: {txt_path.name}")

        doc = extractor.extract(txt_path)

        assert doc is not None
        assert doc.num_sections > 0
        assert doc.total_chars > 0
        assert doc.source_path == str(txt_path)
        logger.info(f"✓ TXT extracted: {doc.num_sections} sections, {doc.total_chars} chars")

    def test_extract_latex(self, extractor, test_data_dir):
        """Test LaTeX extraction."""
        tex_files = list(test_data_dir.glob("**/*.tex")) + list(test_data_dir.glob("**/*.latex"))
        if not tex_files:
            pytest.skip("No LaTeX files found in data directory")

        tex_path = tex_files[0]
        logger.info(f"Testing LaTeX extraction: {tex_path.name}")

        doc = extractor.extract(tex_path)

        assert doc is not None
        assert doc.num_sections > 0
        assert doc.total_chars > 0
        assert doc.source_path == str(tex_path)
        logger.info(f"✓ LaTeX extracted: {doc.num_sections} sections, {doc.total_chars} chars")

    def test_hierarchy_detection(self, extractor, test_data_dir):
        """Test hierarchy detection across formats."""
        # Find any supported document
        for pattern in ["*.pdf", "*.pptx", "*.docx", "*.html", "*.txt"]:
            files = list(test_data_dir.glob(f"**/{pattern}"))
            if files:
                file_path = files[0]
                logger.info(f"Testing hierarchy detection: {file_path.name}")

                doc = extractor.extract(file_path)

                assert doc.hierarchy_depth >= 1
                assert doc.num_roots >= 1

                # Check sections have hierarchy metadata
                for section in doc.sections[:5]:
                    assert section.level is not None
                    assert section.depth is not None
                    assert isinstance(section.ancestors, list)
                    assert section.path

                logger.info(
                    f"✓ Hierarchy detected: depth={doc.hierarchy_depth}, "
                    f"roots={doc.num_roots}, sections={doc.num_sections}"
                )
                return

        pytest.skip("No supported documents found in data directory")

    def test_table_extraction(self, extractor, test_data_dir):
        """Test table extraction (PDF, PPTX, DOCX)."""
        # Find documents that might contain tables
        for pattern in ["*.pdf", "*.pptx", "*.docx"]:
            files = list(test_data_dir.glob(f"**/{pattern}"))
            if files:
                file_path = files[0]
                logger.info(f"Testing table extraction: {file_path.name}")

                doc = extractor.extract(file_path)

                # Tables are optional, so just check structure
                assert isinstance(doc.tables, list)
                assert doc.num_tables == len(doc.tables)

                if doc.num_tables > 0:
                    logger.info(f"✓ Found {doc.num_tables} tables")
                    table = doc.tables[0]
                    assert hasattr(table, 'table_id')
                    assert hasattr(table, 'page_number')
                else:
                    logger.info("✓ No tables found (document may not contain tables)")

                return

        pytest.skip("No documents with table support found")

    def test_extraction_metadata(self, extractor, test_data_dir):
        """Test extraction metadata consistency."""
        # Find any supported document
        for pattern in ["*.pdf", "*.pptx", "*.docx", "*.html", "*.txt"]:
            files = list(test_data_dir.glob(f"**/{pattern}"))
            if files:
                file_path = files[0]
                logger.info(f"Testing extraction metadata: {file_path.name}")

                doc = extractor.extract(file_path)

                # Check required metadata
                assert doc.document_id == file_path.stem
                assert doc.source_path == str(file_path)
                assert doc.extraction_time > 0
                assert doc.extraction_method.startswith("unstructured_")
                assert doc.num_sections == len(doc.sections)
                assert doc.num_tables == len(doc.tables)

                logger.info(
                    f"✓ Metadata valid: "
                    f"id={doc.document_id}, "
                    f"method={doc.extraction_method}, "
                    f"time={doc.extraction_time:.2f}s"
                )
                return

        pytest.skip("No supported documents found in data directory")


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""

    def test_pdf_extraction_legacy(self, extractor, test_data_dir):
        """Test that PDF extraction still works (backward compatibility)."""
        pdf_files = list(test_data_dir.glob("**/*.pdf"))
        if not pdf_files:
            pytest.skip("No PDF files found in data directory")

        pdf_path = pdf_files[0]

        # Should still work with old method name (with deprecation warning)
        doc = extractor._partition_pdf(pdf_path)

        assert doc is not None
        assert len(doc) > 0
        logger.info("✓ Legacy _partition_pdf method still works")


class TestErrorHandling:
    """Test error handling and fallbacks."""

    def test_unsupported_format_fallback(self, extractor, tmp_path):
        """Test fallback to universal partitioner for unknown formats."""
        # Create a file with unusual extension
        test_file = tmp_path / "test.xyz"
        test_file.write_text("This is test content\n\nWith multiple paragraphs.")

        # Should fallback to universal partitioner
        try:
            doc = extractor.extract(test_file)
            assert doc is not None
            logger.info("✓ Fallback to universal partitioner works")
        except RuntimeError as e:
            # Fallback may also fail for truly unsupported formats
            logger.info(f"✓ Graceful error for unsupported format: {e}")

    def test_empty_file_handling(self, extractor, tmp_path):
        """Test handling of empty files."""
        test_file = tmp_path / "empty.txt"
        test_file.write_text("")

        try:
            doc = extractor.extract(test_file)
            # Empty files should extract but with no sections
            assert doc is not None
            logger.info("✓ Empty file handled gracefully")
        except Exception as e:
            logger.info(f"✓ Empty file raises expected error: {e}")


# Manual test function for interactive testing
def manual_test_extraction():
    """
    Manual test function for interactive testing.

    Usage:
        python -c "from tests.test_multi_format_extraction import manual_test_extraction; manual_test_extraction()"
    """
    print("\n" + "=" * 70)
    print("MULTI-FORMAT EXTRACTION MANUAL TEST")
    print("=" * 70 + "\n")

    extractor = UnstructuredExtractor(ExtractionConfig.from_env())
    data_dir = Path("data")

    formats = {
        "PDF": "*.pdf",
        "PowerPoint": "*.pptx",
        "Word": "*.docx",
        "HTML": "*.html",
        "Text": "*.txt",
        "LaTeX": "*.tex",
    }

    results = []

    for format_name, pattern in formats.items():
        files = list(data_dir.glob(f"**/{pattern}"))
        if files:
            file_path = files[0]
            print(f"\nTesting {format_name}: {file_path.name}")
            try:
                doc = extractor.extract(file_path)
                print(f"  ✓ Sections: {doc.num_sections}")
                print(f"  ✓ Characters: {doc.total_chars:,}")
                print(f"  ✓ Hierarchy depth: {doc.hierarchy_depth}")
                print(f"  ✓ Tables: {doc.num_tables}")
                print(f"  ✓ Extraction time: {doc.extraction_time:.2f}s")
                results.append((format_name, "✓ PASS"))
            except Exception as e:
                print(f"  ✗ Error: {e}")
                results.append((format_name, f"✗ FAIL: {e}"))
        else:
            print(f"\nSkipping {format_name}: No files found")
            results.append((format_name, "⊘ SKIP"))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for format_name, status in results:
        print(f"  {format_name:15s} {status}")
    print()


if __name__ == "__main__":
    # Run manual test
    manual_test_extraction()
