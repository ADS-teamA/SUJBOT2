#!/usr/bin/env python3
"""
Test script for Query Processor

Demonstrates:
- Intent classification
- Entity extraction
- Query decomposition
- Query expansion
- Strategy selection
"""

import asyncio
import yaml
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from query_processor import (
    QueryProcessor,
    print_processed_query
)


async def test_query_processor():
    """Test the query processor with various queries."""

    # Load config
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Enable verbose logging
    config['query_processing']['verbose_logging'] = True

    # Initialize processor
    processor = QueryProcessor(config)

    # Test queries
    test_queries = [
        # Gap analysis
        "Které povinné body ze zákona č. 89/2012 Sb. chybí v této smlouvě?",

        # Conflict detection
        "Najdi všechny konflikty mezi smlouvou a §89 odst. 2 zákona.",

        # Compliance check
        "Je smlouva v souladu s požadavky zákona č. 89/2012 Sb.?",

        # Comparison
        "Porovnej ustanovení smlouvy o odpovědnosti s §45 zákona.",

        # Simple factual
        "Co je termín dokončení projektu?",

        # Risk assessment
        "Jaká jsou rizika vyplývající ze smlouvy podle zákona?",

        # Enumeration
        "Vyjmenuj všechny sankce uvedené ve smlouvě."
    ]

    print("\n" + "="*80)
    print("QUERY PROCESSOR TEST SUITE")
    print("="*80 + "\n")

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"Test {i}/{len(test_queries)}")
        print(f"{'='*80}")

        try:
            # Process query
            processed = await processor.process(query)

            # Print results
            print_processed_query(processed)

        except Exception as e:
            print(f"ERROR processing query: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print("TEST SUITE COMPLETE")
    print("="*80 + "\n")


async def test_specific_features():
    """Test specific features individually."""

    # Load config
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    processor = QueryProcessor(config)

    print("\n" + "="*80)
    print("FEATURE-SPECIFIC TESTS")
    print("="*80 + "\n")

    # Test 1: Entity Extraction
    print("\n--- Test 1: Entity Extraction ---")
    query1 = "Podle §89 odst. 2 písm. a) zákona č. 89/2012 Sb. musí dodavatel termín do 30 dní."
    entities = processor.entity_extractor.extract(query1)
    print(f"Query: {query1}")
    print(f"Extracted {len(entities)} entities:")
    for entity in entities:
        print(f"  - Type: {entity.entity_type}, Value: '{entity.value}', Normalized: '{entity.normalized}'")

    # Test 2: Query Expansion
    print("\n--- Test 2: Query Expansion ---")
    query2 = "Jaké jsou povinnosti dodavatele podle smlouvy?"
    entities2 = processor.entity_extractor.extract(query2)
    expanded = processor.expander.expand(query2, entities2)
    print(f"Query: {query2}")
    print(f"Expanded {len(expanded)} terms:")
    for term, synonyms in expanded.items():
        print(f"  - {term}: {', '.join(synonyms)}")

    # Test 3: Complexity Assessment
    print("\n--- Test 3: Complexity Assessment ---")
    test_complexities = [
        ("Co je §89?", "SIMPLE"),
        ("Jaké jsou povinnosti podle §89 a §45?", "MODERATE"),
        ("Porovnej ustanovení smlouvy o odpovědnosti s §45 a najdi všechny konflikty a rizika.", "COMPLEX")
    ]
    for query, expected in test_complexities:
        _, complexity = await processor.classifier.classify(query)
        print(f"  Query: {query}")
        print(f"  Complexity: {complexity.value} (expected: {expected})")

    print("\n" + "="*80 + "\n")


async def interactive_test():
    """Interactive testing mode."""

    # Load config
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config['query_processing']['verbose_logging'] = True

    processor = QueryProcessor(config)

    print("\n" + "="*80)
    print("INTERACTIVE QUERY PROCESSOR TEST")
    print("="*80)
    print("\nEnter queries to process (or 'quit' to exit)\n")

    while True:
        try:
            query = input("\nQuery> ").strip()

            if not query:
                continue

            if query.lower() in ['quit', 'exit', 'q']:
                break

            # Process query
            processed = await processor.process(query)

            # Print results
            print_processed_query(processed)

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Test Query Processor")
    parser.add_argument(
        '--mode',
        choices=['full', 'features', 'interactive'],
        default='full',
        help='Test mode to run'
    )

    args = parser.parse_args()

    if args.mode == 'full':
        asyncio.run(test_query_processor())
    elif args.mode == 'features':
        asyncio.run(test_specific_features())
    elif args.mode == 'interactive':
        asyncio.run(interactive_test())


if __name__ == "__main__":
    main()
