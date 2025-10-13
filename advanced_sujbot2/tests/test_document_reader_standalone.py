"""
Standalone test script for LegalDocumentReader (no dependencies on other modules)

This script demonstrates the usage of the LegalDocumentReader system
with sample legal documents.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Import only what we need
from src.document_reader import LegalDocumentReader
from src.models import LegalDocument

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def test_law_parsing():
    """Test parsing a Czech law structure"""

    sample_law = """
Zákon č. 89/2012 Sb., občanský zákoník

ČÁST PRVNÍ: OBECNÁ ČÁST

HLAVA I: ZÁKLADNÍ USTANOVENÍ

§ 1 Základní zásady soukromého práva

(1) Tento zákon upravuje právní vztahy fyzických a právnických osob.

(2) Každý má právo na ochranu života podle §89 odst. 2.

a) První možnost
b) Druhá možnost

§ 2 Působnost

(1) Zákon se použije na všechny právní vztahy.
"""

    # Create a temporary test file
    test_file = Path("test_law.txt")
    test_file.write_text(sample_law, encoding='utf-8')

    try:
        # Initialize reader
        reader = LegalDocumentReader()

        # Parse document
        logger.info("Parsing test law document...")
        document = await reader.read_legal_document(str(test_file))

        # Display results
        logger.info(f"\n=== Document Analysis Results ===")
        logger.info(f"Document ID: {document.document_id}")
        logger.info(f"Document Type: {document.document_type}")
        logger.info(f"Title: {document.metadata.title}")
        logger.info(f"Total Words: {document.metadata.total_words}")
        logger.info(f"Total Sections: {document.metadata.total_sections}")

        logger.info(f"\n=== Structure ===")
        logger.info(f"Parts: {len(document.structure.parts)}")
        logger.info(f"Chapters: {len(document.structure.chapters)}")
        logger.info(f"Sections: {len(document.structure.sections)}")
        logger.info(f"Total Elements: {len(document.structure.all_elements)}")

        logger.info(f"\n=== Structural Elements ===")
        for element in document.structure.all_elements[:10]:  # Show first 10
            logger.info(f"  {element.legal_reference}: {element.title or element.content[:50]}")

        logger.info(f"\n=== References ===")
        logger.info(f"Total References: {len(document.references)}")
        for ref in document.references[:5]:  # Show first 5
            logger.info(f"  {ref.target_reference} in {ref.source_element_id}")
            logger.info(f"    Context: ...{ref.context[:60]}...")

        logger.info(f"\n=== Content Classification ===")
        for section in document.structure.sections:
            if hasattr(section, 'contains_obligation'):
                if section.contains_obligation:
                    logger.info(f"  {section.legal_reference} contains OBLIGATION")
                if section.contains_prohibition:
                    logger.info(f"  {section.legal_reference} contains PROHIBITION")
                if section.contains_definition:
                    logger.info(f"  {section.legal_reference} contains DEFINITION")

        return True

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return False

    finally:
        # Clean up
        if test_file.exists():
            test_file.unlink()


async def test_contract_parsing():
    """Test parsing a contract structure"""

    sample_contract = """
Smlouva o dílo

mezi smluvními stranami:
Dodavatel: ABC s.r.o.
Objednatel: XYZ a.s.

Článek 1: Předmět smlouvy

1.1 Dodavatel se zavazuje provést dílo podle čl. 5.

1.2 Objednatel se zavazuje dílo převzít a zaplatit.

Článek 2: Cena a platební podmínky

2.1 Celková cena činí 1.000.000 Kč.

2.2 Platba bude provedena podle Článek 2.1.

Článek 3: Termíny

Dílo bude dokončeno do 31.12.2025 podle §89 občanského zákoníku.
"""

    # Create a temporary test file
    test_file = Path("test_contract.txt")
    test_file.write_text(sample_contract, encoding='utf-8')

    try:
        # Initialize reader
        reader = LegalDocumentReader()

        # Parse document
        logger.info("\n\n=== Testing Contract Parsing ===\n")
        logger.info("Parsing test contract...")
        document = await reader.read_legal_document(str(test_file))

        # Display results
        logger.info(f"\n=== Document Analysis Results ===")
        logger.info(f"Document ID: {document.document_id}")
        logger.info(f"Document Type: {document.document_type}")
        logger.info(f"Title: {document.metadata.title}")

        logger.info(f"\n=== Structure ===")
        logger.info(f"Articles: {len(document.structure.sections)}")
        logger.info(f"Total Elements: {len(document.structure.all_elements)}")

        logger.info(f"\n=== Structural Elements ===")
        for element in document.structure.all_elements:
            logger.info(f"  {element.legal_reference}: {element.title or element.content[:50]}")

        logger.info(f"\n=== References ===")
        logger.info(f"Total References: {len(document.references)}")
        for ref in document.references:
            logger.info(f"  {ref.target_reference} in {ref.source_element_id}")

        return True

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return False

    finally:
        # Clean up
        if test_file.exists():
            test_file.unlink()


async def main():
    """Run all tests"""
    logger.info("Starting LegalDocumentReader tests...")

    try:
        result1 = await test_law_parsing()
        result2 = await test_contract_parsing()

        if result1 and result2:
            logger.info("\n\n=== All tests completed successfully ===")
            return 0
        else:
            logger.error("\n\n=== Some tests failed ===")
            return 1

    except Exception as e:
        logger.error(f"Test suite failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
