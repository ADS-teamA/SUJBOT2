#!/usr/bin/env python
"""
Export Knowledge Graph from Neo4j to Gephi-compatible formats.

Supports:
- GEXF (Graph Exchange XML Format) - Native Gephi format with full metadata
- GraphML - Standard XML-based graph format
- CSV (nodes + edges) - For custom processing

Usage:
    uv run python scripts/export_to_gephi.py                    # Export to GEXF (default)
    uv run python scripts/export_to_gephi.py --format graphml   # Export to GraphML
    uv run python scripts/export_to_gephi.py --format csv       # Export to CSV files
    uv run python scripts/export_to_gephi.py --output my_graph  # Custom output name

Output:
    output/kg_graph.gexf     (or .graphml, or _nodes.csv + _edges.csv)

Gephi Usage Tips:
    1. Open Gephi and go to File → Open
    2. Select the exported .gexf file
    3. In the Overview tab, run Layout → ForceAtlas 2 for best results
    4. Use Partition (Appearance) to color nodes by 'entity_type'
    5. Use Ranking (Appearance) to size nodes by 'confidence'
    6. Go to Preview tab for export-ready visualization
"""

import argparse
import csv
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import networkx as nx


def sanitize_text(text: str | None) -> str:
    """Remove control characters that break XML/GEXF parsing."""
    if not text:
        return ""
    # Remove control characters (0x00-0x1F except tab, newline, carriage return)
    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', str(text))
    return cleaned

from src.graph.config import Neo4jConfig
from src.graph.entity_display import (
    ENTITY_CATEGORIES,
    CATEGORY_COLORS_RGB,
    get_entity_category,
    get_category_rgb,
)
from src.graph.neo4j_manager import Neo4jManager

# Alias for backwards compatibility
CATEGORY_COLORS = CATEGORY_COLORS_RGB


def fetch_graph_data(manager: Neo4jManager) -> tuple[list[dict], list[dict]]:
    """Fetch all entities and relationships from Neo4j.

    Supports both Graphiti schema (name, uuid, summary) and legacy schema (id, type, value).
    """

    print("📊 Loading entities from Neo4j...")
    # Graphiti schema uses: uuid, name, summary, group_id
    entities_query = """
    MATCH (e:Entity)
    RETURN e.uuid as id,
           e.name as value,
           e.summary as summary,
           e.group_id as document_id,
           labels(e) as labels
    """
    entities = manager.execute(entities_query)
    print(f"  → Found {len(entities)} entities")

    print("🔗 Loading relationships from Neo4j...")
    # Graphiti relationships
    rels_query = """
    MATCH (source:Entity)-[r]->(target:Entity)
    RETURN source.uuid as source_id, target.uuid as target_id,
           type(r) as rel_type, r.fact as evidence
    """
    relationships = manager.execute(rels_query)
    print(f"  → Found {len(relationships)} relationships")

    return entities, relationships


def export_gexf(entities: list[dict], relationships: list[dict], output_path: str) -> None:
    """Export to GEXF format (Gephi native format with full metadata support)."""

    G = nx.DiGraph()

    # Add nodes with all attributes
    for entity in entities:
        if not entity.get("id") or not entity.get("value"):
            continue  # Skip invalid entities

        # Infer entity type from labels or use generic
        labels = entity.get("labels", [])
        entity_type = "entity"
        for label in labels:
            if label != "Entity":
                entity_type = label.lower()
                break

        category = ENTITY_CATEGORIES.get(entity_type, "other")
        color = CATEGORY_COLORS.get(category, (150, 150, 150))

        # Truncate value for display and sanitize
        value = sanitize_text(entity["value"])
        label = value[:100] if value else entity["id"][:20]

        G.add_node(
            entity["id"],
            label=sanitize_text(label),
            entity_type=entity_type,
            category=category,
            document_id=sanitize_text(entity.get("document_id", "")),
            summary=sanitize_text((entity.get("summary", "") or "")[:500]),
            # Gephi visualization attributes
            viz={
                "color": {"r": color[0], "g": color[1], "b": color[2], "a": 1.0},
                "size": 15,
            },
        )

    # Add edges with attributes
    for rel in relationships:
        if not rel.get("source_id") or not rel.get("target_id"):
            continue

        # Only add edge if both nodes exist
        if rel["source_id"] not in G.nodes or rel["target_id"] not in G.nodes:
            continue

        G.add_edge(
            rel["source_id"],
            rel["target_id"],
            label=sanitize_text(rel.get("rel_type", "RELATED")),
            rel_type=sanitize_text(rel.get("rel_type", "RELATED")),
            weight=1.0,
            evidence=sanitize_text((rel.get("evidence", "") or "")[:200]),
        )

    # Write GEXF
    nx.write_gexf(G, output_path)
    print(f"✅ GEXF exported to: {Path(output_path).absolute()}")


def export_graphml(entities: list[dict], relationships: list[dict], output_path: str) -> None:
    """Export to GraphML format (standard XML graph format)."""

    G = nx.DiGraph()

    # Add nodes
    for entity in entities:
        if not entity.get("id") or not entity.get("value"):
            continue

        labels = entity.get("labels", [])
        entity_type = "entity"
        for label in labels:
            if label != "Entity":
                entity_type = label.lower()
                break

        value = entity["value"] or ""
        G.add_node(
            entity["id"],
            label=value[:100] if value else entity["id"][:20],
            entity_type=entity_type,
            category=ENTITY_CATEGORIES.get(entity_type, "other"),
            document_id=entity.get("document_id", ""),
        )

    # Add edges
    for rel in relationships:
        if not rel.get("source_id") or not rel.get("target_id"):
            continue
        if rel["source_id"] not in G.nodes or rel["target_id"] not in G.nodes:
            continue

        G.add_edge(
            rel["source_id"],
            rel["target_id"],
            label=rel.get("rel_type", "RELATED"),
            weight="1.0",
        )

    # Write GraphML
    nx.write_graphml(G, output_path)
    print(f"✅ GraphML exported to: {Path(output_path).absolute()}")


def export_csv(
    entities: list[dict], relationships: list[dict], output_base: str
) -> None:
    """Export to CSV format (nodes + edges files for Gephi import)."""

    nodes_path = f"{output_base}_nodes.csv"
    edges_path = f"{output_base}_edges.csv"

    # Build valid node set
    valid_nodes = set()

    # Export nodes
    with open(nodes_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Id",
                "Label",
                "entity_type",
                "category",
                "document_id",
            ],
        )
        writer.writeheader()
        for entity in entities:
            if not entity.get("id") or not entity.get("value"):
                continue

            labels = entity.get("labels", [])
            entity_type = "entity"
            for label in labels:
                if label != "Entity":
                    entity_type = label.lower()
                    break

            value = entity["value"] or ""
            valid_nodes.add(entity["id"])

            writer.writerow(
                {
                    "Id": entity["id"],
                    "Label": value[:100] if value else entity["id"][:20],
                    "entity_type": entity_type,
                    "category": ENTITY_CATEGORIES.get(entity_type, "other"),
                    "document_id": entity.get("document_id", ""),
                }
            )

    print(f"✅ Nodes CSV exported to: {Path(nodes_path).absolute()}")

    # Export edges
    with open(edges_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Source", "Target", "Type", "Label", "Weight"],
        )
        writer.writeheader()
        for rel in relationships:
            if not rel.get("source_id") or not rel.get("target_id"):
                continue
            if rel["source_id"] not in valid_nodes or rel["target_id"] not in valid_nodes:
                continue

            writer.writerow(
                {
                    "Source": rel["source_id"],
                    "Target": rel["target_id"],
                    "Type": "Directed",
                    "Label": rel.get("rel_type", "RELATED"),
                    "Weight": 1.0,
                }
            )

    print(f"✅ Edges CSV exported to: {Path(edges_path).absolute()}")


def print_gephi_tips(export_format: str) -> None:
    """Print tips for using the export in Gephi."""

    print("\n" + "=" * 60)
    print("📖 GEPHI USAGE TIPS")
    print("=" * 60)

    if export_format == "csv":
        print("""
1. Open Gephi
2. Go to File → Import Spreadsheet
3. First import the _nodes.csv file:
   - Select "Nodes table"
   - Set Id as "Id" column
   - Set Label as "Label" column
4. Then import the _edges.csv file:
   - Select "Edges table"
   - Set Source and Target columns
""")
    else:
        print("""
1. Open Gephi
2. Go to File → Open → Select the exported file
3. In the Import Report dialog, click "OK"
""")

    print("""
RECOMMENDED WORKFLOW:
─────────────────────
a) LAYOUT:
   • Go to Layout panel (bottom left)
   • Select "ForceAtlas 2" algorithm
   • Check "Prevent Overlap" and "LinLog mode"
   • Click "Run" and wait for stabilization

b) APPEARANCE (coloring):
   • Go to Appearance panel (top left)
   • Click "Nodes" → "Partition"
   • Select "entity_type" or "category"
   • Click "Apply"

c) APPEARANCE (sizing):
   • Click "Nodes" → "Ranking"
   • Select "confidence"
   • Set Min/Max size (e.g., 10-50)
   • Click "Apply"

d) LABELS:
   • In Graph window, click the "T" button to show labels
   • Right-click → "Text..." to configure label size

e) STATISTICS (optional):
   • Run "Modularity" for community detection
   • Run "PageRank" for importance ranking
   • Run "Betweenness Centrality" for bridge nodes

f) EXPORT:
   • Go to Preview tab
   • Adjust settings and click "Refresh"
   • Click "Export SVG/PDF/PNG"
""")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Export Knowledge Graph from Neo4j to Gephi formats"
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["gexf", "graphml", "csv"],
        default="gexf",
        help="Output format (default: gexf)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="kg_graph",
        help="Output filename without extension (default: kg_graph)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Don't print Gephi usage tips",
    )

    args = parser.parse_args()

    # Create output directory
    Path("output").mkdir(exist_ok=True)

    # Connect to Neo4j
    print("🔗 Connecting to Neo4j...")
    config = Neo4jConfig.from_env()
    manager = Neo4jManager(config)

    try:
        # Fetch data
        entities, relationships = fetch_graph_data(manager)

        if not entities:
            print("⚠️  No entities found in Neo4j. Is the graph populated?")
            return

        # Export based on format
        output_base = f"output/{args.output}"

        if args.format == "gexf":
            export_gexf(entities, relationships, f"{output_base}.gexf")
        elif args.format == "graphml":
            export_graphml(entities, relationships, f"{output_base}.graphml")
        elif args.format == "csv":
            export_csv(entities, relationships, output_base)

        # Print statistics
        valid_entities = [e for e in entities if e.get("id") and e.get("value")]
        print("\n📊 Export Statistics:")
        print(f"  Nodes: {len(valid_entities):,}")
        print(f"  Edges: {len(relationships):,}")

        # Count by document
        doc_counts: dict[str, int] = {}
        for entity in valid_entities:
            doc = entity.get("document_id") or "unknown"
            doc_counts[doc] = doc_counts.get(doc, 0) + 1

        print("\n  By document:")
        for doc, count in sorted(doc_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"    • {doc}: {count}")

        # Print tips unless quiet mode
        if not args.quiet:
            print_gephi_tips(args.format)

    finally:
        manager.close()


if __name__ == "__main__":
    main()
