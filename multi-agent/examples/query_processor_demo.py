#!/usr/bin/env python3
"""
Query Processor Demo

This script demonstrates the query processor capabilities with real examples.
"""

import asyncio
import yaml
import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir / "src"))

from query_processor import QueryProcessor, print_processed_query


async def demo_gap_analysis():
    """Demo: Gap analysis query."""
    print("\n" + "="*80)
    print("DEMO 1: GAP ANALYSIS")
    print("="*80)
    print("\nScenario: Finding missing requirements in a contract")
    print("Query: 'Které povinné body ze zákona č. 89/2012 Sb. chybí v této smlouvě?'")

    # Load config
    config_path = parent_dir / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config['query_processing']['verbose_logging'] = True

    # Initialize processor
    processor = QueryProcessor(config)

    # Process query
    query = "Které povinné body ze zákona č. 89/2012 Sb. chybí v této smlouvě?"
    processed = await processor.process(query)

    # Print results
    print_processed_query(processed)

    # Explain what happened
    print("\nWhat the processor did:")
    print("1. Classified intent as GAP_ANALYSIS")
    print("2. Extracted legal reference: Zákon č. 89/2012 Sb.")
    print("3. Decomposed into sub-questions:")
    print("   - What are the legal requirements?")
    print("   - What does the contract specify?")
    print("   - Which requirements are missing?")
    print("4. Selected cross_document retrieval strategy")


async def demo_conflict_detection():
    """Demo: Conflict detection query."""
    print("\n" + "="*80)
    print("DEMO 2: CONFLICT DETECTION")
    print("="*80)
    print("\nScenario: Finding contradictions between contract and law")
    print("Query: 'Najdi všechny konflikty mezi smlouvou a §89 odst. 2.'")

    config_path = parent_dir / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    processor = QueryProcessor(config)

    query = "Najdi všechny konflikty mezi smlouvou a §89 odst. 2."
    processed = await processor.process(query)

    print_processed_query(processed)

    print("\nWhat the processor did:")
    print("1. Classified intent as CONFLICT_DETECTION")
    print("2. Extracted legal reference: §89 odst. 2")
    print("3. Used provision_pairing decomposition strategy")
    print("4. Selected cross_document retrieval for comparison")


async def demo_simple_factual():
    """Demo: Simple factual query (no decomposition)."""
    print("\n" + "="*80)
    print("DEMO 3: SIMPLE FACTUAL QUERY")
    print("="*80)
    print("\nScenario: Simple fact retrieval")
    print("Query: 'Co je termín dokončení projektu?'")

    config_path = parent_dir / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    processor = QueryProcessor(config)

    query = "Co je termín dokončení projektu?"
    processed = await processor.process(query)

    print_processed_query(processed)

    print("\nWhat the processor did:")
    print("1. Classified intent as FACTUAL")
    print("2. Assessed complexity as SIMPLE")
    print("3. Extracted temporal entity: 'termín'")
    print("4. Expanded with synonyms: lhůta, deadline")
    print("5. No decomposition needed (simple query)")
    print("6. Selected hybrid retrieval strategy")


async def demo_comparison():
    """Demo: Comparison query."""
    print("\n" + "="*80)
    print("DEMO 4: COMPARISON QUERY")
    print("="*80)
    print("\nScenario: Comparing two legal provisions")
    print("Query: 'Porovnej ustanovení smlouvy o odpovědnosti s §45 zákona.'")

    config_path = parent_dir / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    processor = QueryProcessor(config)

    query = "Porovnej ustanovení smlouvy o odpovědnosti s §45 zákona."
    processed = await processor.process(query)

    print_processed_query(processed)

    print("\nWhat the processor did:")
    print("1. Classified intent as COMPARISON")
    print("2. Extracted legal reference: §45")
    print("3. Used entity_separation strategy:")
    print("   - Describe contract provisions")
    print("   - Describe law provisions")
    print("   - Compare them")
    print("4. Expanded 'odpovědnost' with synonyms: liability, ručení")


async def demo_compliance_check():
    """Demo: Compliance check query."""
    print("\n" + "="*80)
    print("DEMO 5: COMPLIANCE CHECK")
    print("="*80)
    print("\nScenario: Verifying contract compliance")
    print("Query: 'Je smlouva v souladu s požadavky zákona č. 89/2012 Sb.?'")

    config_path = parent_dir / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    processor = QueryProcessor(config)

    query = "Je smlouva v souladu s požadavky zákona č. 89/2012 Sb.?"
    processed = await processor.process(query)

    print_processed_query(processed)

    print("\nWhat the processor did:")
    print("1. Classified intent as COMPLIANCE_CHECK")
    print("2. Extracted legal reference: Zákon č. 89/2012 Sb.")
    print("3. Used clause_by_clause strategy:")
    print("   - What are the legal requirements?")
    print("   - What are the contract clauses?")
    print("   - Do they meet requirements?")
    print("4. Expanded 'soulad' with: compliance, shoda")


async def demo_entity_extraction():
    """Demo: Entity extraction capabilities."""
    print("\n" + "="*80)
    print("DEMO 6: ENTITY EXTRACTION")
    print("="*80)
    print("\nScenario: Complex query with multiple entities")
    print("Query: 'Podle §89 odst. 2 písm. a) zákona č. 89/2012 Sb. musí dodavatel")
    print("        splnit povinnost do 30 dní od 15.10.2024.'")

    config_path = parent_dir / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    processor = QueryProcessor(config)

    query = "Podle §89 odst. 2 písm. a) zákona č. 89/2012 Sb. musí dodavatel splnit povinnost do 30 dní od 15.10.2024."
    processed = await processor.process(query)

    print_processed_query(processed)

    print("\nExtracted entities breakdown:")
    for entity in processed.entities:
        print(f"\n  Entity: {entity.entity_type}")
        print(f"  Value: '{entity.value}'")
        print(f"  Normalized: '{entity.normalized}'")
        print(f"  Position: {entity.span}")
        print(f"  Confidence: {entity.confidence:.2f}")


async def main():
    """Run all demos."""
    print("\n" + "="*80)
    print("QUERY PROCESSOR DEMONSTRATION")
    print("Advanced SUJBOT2 - Legal Compliance Analysis")
    print("="*80)

    demos = [
        demo_gap_analysis,
        demo_conflict_detection,
        demo_simple_factual,
        demo_comparison,
        demo_compliance_check,
        demo_entity_extraction
    ]

    for demo in demos:
        await demo()
        input("\nPress Enter to continue to next demo...")

    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)
    print("\nFor more information, see QUERY_PROCESSOR.md")
    print("To run tests: python test_query_processor.py")
    print("To run interactively: python test_query_processor.py --mode interactive")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nError running demo: {e}")
        import traceback
        traceback.print_exc()
