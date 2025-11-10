#!/usr/bin/env python
"""
Create interactive HTML visualization of the entire Knowledge Graph from Neo4j.

This script exports all entities and relationships from Neo4j and creates
an interactive HTML graph using pyvis that can be opened in a web browser.

Usage:
    uv run python scripts/visualize_kg_graph.py

Output:
    output/kg_visualization.html (open in browser)
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyvis.network import Network
from src.graph.config import Neo4jConfig
from src.graph.neo4j_manager import Neo4jManager

# Color mapping for entity types
ENTITY_COLORS = {
    "clause": "#FF6B6B",       # Red
    "topic": "#4ECDC4",        # Teal
    "regulation": "#45B7D1",   # Blue
    "organization": "#FFA07A", # Orange
    "date": "#98D8C8",         # Light green
    "standard": "#F7DC6F",     # Yellow
}

# Relationship colors
REL_COLORS = {
    "covers_topic": "#95A5A6",
    "contains_clause": "#7F8C8D",
    "issued_by": "#E74C3C",
    "effective_date": "#3498DB",
    "superseded_by": "#9B59B6",
    "references": "#1ABC9C",
    "supersedes": "#F39C12",
}


def create_visualization():
    """Create interactive HTML visualization of Neo4j graph."""

    print("🔗 Connecting to Neo4j...")
    config = Neo4jConfig.from_env()
    manager = Neo4jManager(config)

    try:
        # Create pyvis network
        net = Network(
            height="900px",
            width="100%",
            bgcolor="#222222",
            font_color="white",
            directed=True,
            notebook=False,
        )

        # Configure physics for better layout
        net.set_options("""
        {
          "physics": {
            "enabled": true,
            "barnesHut": {
              "gravitationalConstant": -8000,
              "centralGravity": 0.3,
              "springLength": 95,
              "springConstant": 0.04,
              "damping": 0.09,
              "avoidOverlap": 0.1
            },
            "minVelocity": 0.75,
            "solver": "barnesHut"
          },
          "nodes": {
            "font": {
              "size": 14,
              "color": "#ffffff"
            }
          },
          "edges": {
            "smooth": {
              "type": "continuous"
            },
            "font": {
              "size": 10,
              "align": "middle"
            }
          }
        }
        """)

        # Fetch all entities
        print("📊 Loading entities from Neo4j...")
        entities_query = """
        MATCH (e:Entity)
        RETURN e.id as id, e.type as type, e.value as value, e.confidence as confidence
        """
        entities = manager.execute(entities_query)

        print(f"  → Found {len(entities)} entities")

        # Add nodes
        entity_counts = {}
        for entity in entities:
            entity_type = entity["type"]
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1

            # Truncate long values for display
            label = entity["value"]
            if len(label) > 40:
                label = label[:37] + "..."

            # Get color for entity type
            color = ENTITY_COLORS.get(entity_type, "#95A5A6")

            # Node size based on confidence
            size = 10 + (entity.get("confidence", 0.5) * 20)

            # Create tooltip with full info
            title = f"""
            <b>{entity['value']}</b><br>
            Type: {entity_type}<br>
            Confidence: {entity.get('confidence', 'N/A')}<br>
            ID: {entity['id']}
            """

            net.add_node(
                entity["id"],
                label=label,
                title=title,
                color=color,
                size=size,
                shape="dot",
            )

        # Fetch all relationships
        print("🔗 Loading relationships from Neo4j...")
        rels_query = """
        MATCH (source:Entity)-[r]->(target:Entity)
        RETURN source.id as source_id, target.id as target_id,
               type(r) as rel_type, r.confidence as confidence
        """
        relationships = manager.execute(rels_query)

        print(f"  → Found {len(relationships)} relationships")

        # Add edges
        rel_counts = {}
        for rel in relationships:
            rel_type = rel["rel_type"]
            rel_counts[rel_type] = rel_counts.get(rel_type, 0) + 1

            # Get color for relationship type
            color = REL_COLORS.get(rel_type.lower(), "#95A5A6")

            # Edge width based on confidence
            width = 1 + (rel.get("confidence", 0.5) * 3)

            net.add_edge(
                rel["source_id"],
                rel["target_id"],
                title=rel_type,
                color=color,
                width=width,
                label=rel_type,
            )

        # Save to HTML
        output_path = "output/kg_visualization.html"
        Path("output").mkdir(exist_ok=True)
        net.save_graph(output_path)

        print("\n" + "="*60)
        print("✅ VISUALIZATION CREATED SUCCESSFULLY!")
        print("="*60)
        print(f"\n📊 Statistics:")
        print(f"  Total entities: {len(entities)}")
        for entity_type, count in sorted(entity_counts.items(), key=lambda x: -x[1]):
            print(f"    • {entity_type:20s}: {count:5d}")

        print(f"\n  Total relationships: {len(relationships)}")
        for rel_type, count in sorted(rel_counts.items(), key=lambda x: -x[1]):
            print(f"    • {rel_type:20s}: {count:5d}")

        print(f"\n📂 Output file: {Path(output_path).absolute()}")
        print(f"\n💡 Open in browser:")
        print(f"   open {output_path}")
        print("\n" + "="*60)

        # Print legend
        print("\n🎨 Color Legend:")
        print("  Entities:")
        for entity_type, color in ENTITY_COLORS.items():
            print(f"    • {entity_type:20s}: {color}")
        print("\n  Relationships:")
        for rel_type, color in REL_COLORS.items():
            print(f"    • {rel_type:20s}: {color}")

    finally:
        manager.close()


if __name__ == "__main__":
    create_visualization()
