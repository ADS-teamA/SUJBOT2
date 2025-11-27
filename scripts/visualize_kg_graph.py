#!/usr/bin/env python
"""
Interactive Knowledge Graph Visualization from Neo4j.

Creates an interactive HTML visualization with:
- Complete color palette for all 55 entity types (organized by category)
- Interactive filtering by entity type (toggle checkboxes)
- Search bar to find entities by name
- Statistics panel with entity/relationship counts
- Physics-based layout with configurable options
- Export to GEXF format (for Gephi import)

Usage:
    uv run python scripts/visualize_kg_graph.py
    uv run python scripts/visualize_kg_graph.py --output custom_name.html
    uv run python scripts/visualize_kg_graph.py --export-gexf

Output:
    output/kg_visualization.html (open in browser)
    output/kg_graph.gexf (for Gephi import, with --export-gexf)
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import networkx as nx
from pyvis.network import Network

from src.graph.config import Neo4jConfig
from src.graph.graphiti_types import get_entity_type_categories
from src.graph.neo4j_manager import Neo4jManager

# =============================================================================
# COLOR PALETTES BY CATEGORY
# =============================================================================

# Entity colors organized by category (all 55 types)
ENTITY_COLORS = {
    # Core Entities (8) - Blues/Teals
    "standard": "#3498DB",  # Blue
    "organization": "#2980B9",  # Darker Blue
    "date": "#1ABC9C",  # Teal
    "clause": "#16A085",  # Darker Teal
    "topic": "#48C9B0",  # Light Teal
    "person": "#5DADE2",  # Light Blue
    "location": "#85C1E9",  # Very Light Blue
    "contract": "#2E86AB",  # Steel Blue
    # Regulatory Hierarchy (6) - Purples
    "regulation": "#9B59B6",  # Purple
    "decree": "#8E44AD",  # Darker Purple
    "directive": "#BB8FCE",  # Light Purple
    "treaty": "#7D3C98",  # Deep Purple
    "legal_provision": "#A569BD",  # Medium Purple
    "requirement": "#D7BDE2",  # Pale Purple
    # Authorization (2) - Oranges
    "permit": "#E67E22",  # Orange
    "license_condition": "#D35400",  # Darker Orange
    # Nuclear Technical (9) - Reds/Pinks
    "reactor": "#E74C3C",  # Red
    "facility": "#C0392B",  # Darker Red
    "system": "#F1948A",  # Light Red
    "safety_function": "#EC7063",  # Coral
    "fuel_type": "#CD6155",  # Indian Red
    "isotope": "#E57373",  # Light Coral
    "radiation_source": "#EF5350",  # Bright Red
    "waste_category": "#F44336",  # Material Red
    "dose_limit": "#FF8A80",  # Red Accent
    # Events (4) - Yellows/Ambers
    "incident": "#F39C12",  # Amber
    "emergency_classification": "#F1C40F",  # Yellow
    "inspection": "#D4AC0D",  # Dark Yellow
    "decommissioning_phase": "#F7DC6F",  # Light Yellow
    # Liability (1) - Brown
    "liability_regime": "#A0522D",  # Sienna
    # Legal Terminology (2) - Grays
    "legal_term": "#7F8C8D",  # Gray
    "definition": "#95A5A6",  # Light Gray
    # Czech Legal Types (8) - Greens
    "vyhlaska": "#27AE60",  # Green
    "narizeni": "#229954",  # Darker Green
    "sbirka_zakonu": "#58D68D",  # Light Green
    "metodicky_pokyn": "#82E0AA",  # Pale Green
    "sujb_rozhodnuti": "#1E8449",  # Forest Green
    "bezpecnostni_dokumentace": "#2ECC71",  # Emerald
    "limitni_stav": "#52BE80",  # Medium Green
    "mezni_hodnota": "#73C6B6",  # Sea Green
    # Technical Parameters (7) - Cyans
    "numeric_threshold": "#00BCD4",  # Cyan
    "measurement_unit": "#00ACC1",  # Darker Cyan
    "time_period": "#4DD0E1",  # Light Cyan
    "frequency": "#26C6DA",  # Medium Cyan
    "percentage": "#80DEEA",  # Pale Cyan
    "temperature": "#FF7043",  # Deep Orange
    "pressure": "#FFAB91",  # Light Orange
    # Process Types (5) - Indigos
    "radiation_activity": "#5C6BC0",  # Indigo
    "maintenance_activity": "#7986CB",  # Light Indigo
    "emergency_procedure": "#3F51B5",  # Material Indigo
    "training_requirement": "#9FA8DA",  # Pale Indigo
    "documentation_requirement": "#C5CAE9",  # Very Light Indigo
    # Compliance Types (3) - Deep Colors
    "compliance_gap": "#D32F2F",  # Deep Red (problems)
    "risk_factor": "#FFA000",  # Amber (warnings)
    "mitigation_measure": "#388E3C",  # Deep Green (solutions)
}

# Relationship colors by semantic category
RELATIONSHIP_COLORS = {
    # Compliance
    "complies_with": "#27AE60",  # Green
    "contradicts": "#E74C3C",  # Red
    "partially_satisfies": "#F39C12",  # Amber
    "specifies_requirement": "#9B59B6",  # Purple
    "requires_clause": "#8E44AD",  # Dark Purple
    # Regulatory
    "implements": "#3498DB",  # Blue
    "transposes": "#2980B9",  # Dark Blue
    "superseded_by": "#95A5A6",  # Gray
    "supersedes": "#7F8C8D",  # Dark Gray
    "amends": "#1ABC9C",  # Teal
    # Document Structure
    "contains_clause": "#E67E22",  # Orange
    "contains_provision": "#D35400",  # Dark Orange
    "contains": "#F39C12",  # Amber
    "part_of": "#D4AC0D",  # Dark Amber
    # Citations
    "references": "#5DADE2",  # Light Blue
    "referenced_by": "#85C1E9",  # Very Light Blue
    "cites_provision": "#48C9B0",  # Light Teal
    "based_on": "#16A085",  # Dark Teal
    # Authorization
    "issued_by": "#9B59B6",  # Purple
    "granted_by": "#8E44AD",  # Dark Purple
    "enforced_by": "#7D3C98",  # Deep Purple
    "subject_to_inspection": "#BB8FCE",  # Light Purple
    "supervises": "#A569BD",  # Medium Purple
    # Nuclear Technical
    "regulated_by": "#E74C3C",  # Red
    "operated_by": "#C0392B",  # Dark Red
    "has_system": "#F1948A",  # Light Red
    "performs_function": "#EC7063",  # Coral
    "uses_fuel": "#CD6155",  # Indian Red
    "contains_isotope": "#E57373",  # Light Coral
    "produces_waste": "#EF5350",  # Bright Red
    "has_dose_limit": "#FF8A80",  # Red Accent
    # Temporal
    "effective_date": "#1ABC9C",  # Teal
    "expiry_date": "#16A085",  # Dark Teal
    "signed_on": "#48C9B0",  # Light Teal
    "decommissioned_on": "#73C6B6",  # Sea Green
    # Content
    "covers_topic": "#5C6BC0",  # Indigo
    "applies_to": "#7986CB",  # Light Indigo
    # Terminology
    "definition_of": "#7F8C8D",  # Gray
    # Provenance
    "mentioned_in": "#BDC3C7",  # Silver
    "defined_in": "#95A5A6",  # Light Gray
    "documented_in": "#7F8C8D",  # Gray
}

# Category display names (Czech)
CATEGORY_NAMES = {
    "core": "Základní entity",
    "regulatory": "Regulatorní hierarchie",
    "authorization": "Autorizace",
    "nuclear_technical": "Jaderně technické",
    "events": "Události",
    "liability": "Odpovědnost",
    "legal_terminology": "Právní terminologie",
    "czech_legal": "České právní typy",
    "technical_parameters": "Technické parametry",
    "processes": "Procesní typy",
    "compliance": "Compliance typy",
}


def create_filter_html(entity_counts: dict, categories: dict) -> str:
    """Generate HTML for entity type filter panel."""
    html = """
    <div id="filter-panel" style="
        position: fixed;
        top: 10px;
        left: 10px;
        background: rgba(34, 34, 34, 0.95);
        border-radius: 8px;
        padding: 15px;
        max-height: 80vh;
        overflow-y: auto;
        z-index: 1000;
        min-width: 250px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    ">
        <h3 style="color: white; margin: 0 0 10px 0; font-size: 14px;">🔍 Filtr entit</h3>

        <div style="margin-bottom: 10px;">
            <input type="text" id="search-input" placeholder="Hledat entitu..."
                   style="width: 100%; padding: 8px; border-radius: 4px; border: none;
                          background: #333; color: white; box-sizing: border-box;">
        </div>

        <div style="margin-bottom: 10px;">
            <button onclick="toggleAll(true)" style="padding: 5px 10px; margin-right: 5px;
                    background: #27AE60; color: white; border: none; border-radius: 4px; cursor: pointer;">
                Vše</button>
            <button onclick="toggleAll(false)" style="padding: 5px 10px;
                    background: #E74C3C; color: white; border: none; border-radius: 4px; cursor: pointer;">
                Nic</button>
        </div>

        <div id="category-filters">
    """

    # Group entity types by category
    type_to_category = {}
    for cat_name, types in categories.items():
        for t in types:
            type_to_category[t.value] = cat_name

    # Generate filter checkboxes by category
    for cat_key, cat_display in CATEGORY_NAMES.items():
        cat_types = categories.get(cat_key, [])
        if not cat_types:
            continue

        # Check if any types in this category have entities
        cat_has_entities = any(entity_counts.get(t.value, 0) > 0 for t in cat_types)
        if not cat_has_entities:
            continue

        html += f"""
        <div class="category" style="margin-bottom: 15px;">
            <div style="color: #aaa; font-size: 11px; text-transform: uppercase;
                        margin-bottom: 5px; border-bottom: 1px solid #444; padding-bottom: 3px;">
                {cat_display}
            </div>
        """

        for entity_type in cat_types:
            type_str = entity_type.value
            count = entity_counts.get(type_str, 0)
            if count == 0:
                continue

            color = ENTITY_COLORS.get(type_str, "#95A5A6")
            html += f"""
            <label style="display: flex; align-items: center; color: white;
                          font-size: 12px; margin: 3px 0; cursor: pointer;">
                <input type="checkbox" checked data-type="{type_str}"
                       onchange="filterByType()" style="margin-right: 8px;">
                <span style="display: inline-block; width: 12px; height: 12px;
                             background: {color}; border-radius: 2px; margin-right: 8px;"></span>
                {type_str} <span style="color: #888; margin-left: 5px;">({count})</span>
            </label>
            """

        html += "</div>"

    html += """
        </div>
    </div>
    """

    return html


def create_stats_html(entity_counts: dict, rel_counts: dict) -> str:
    """Generate HTML for statistics panel."""
    total_entities = sum(entity_counts.values())
    total_rels = sum(rel_counts.values())

    # Top 5 entity types
    top_entities = sorted(entity_counts.items(), key=lambda x: -x[1])[:5]
    top_rels = sorted(rel_counts.items(), key=lambda x: -x[1])[:5]

    html = f"""
    <div id="stats-panel" style="
        position: fixed;
        top: 10px;
        right: 10px;
        background: rgba(34, 34, 34, 0.95);
        border-radius: 8px;
        padding: 15px;
        z-index: 1000;
        min-width: 200px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    ">
        <h3 style="color: white; margin: 0 0 10px 0; font-size: 14px;">📊 Statistiky</h3>

        <div style="color: white; font-size: 13px;">
            <div style="margin-bottom: 10px;">
                <strong>Entity:</strong> {total_entities:,}<br>
                <strong>Vztahy:</strong> {total_rels:,}
            </div>

            <div style="color: #aaa; font-size: 11px; margin-bottom: 5px;">Top 5 entit:</div>
    """

    for entity_type, count in top_entities:
        color = ENTITY_COLORS.get(entity_type, "#95A5A6")
        html += f"""
            <div style="display: flex; align-items: center; margin: 2px 0;">
                <span style="display: inline-block; width: 10px; height: 10px;
                             background: {color}; border-radius: 2px; margin-right: 6px;"></span>
                <span style="color: #ddd; font-size: 11px;">{entity_type}: {count}</span>
            </div>
        """

    html += """
            <div style="color: #aaa; font-size: 11px; margin: 10px 0 5px 0;">Top 5 vztahů:</div>
    """

    for rel_type, count in top_rels:
        color = RELATIONSHIP_COLORS.get(rel_type.lower(), "#95A5A6")
        html += f"""
            <div style="display: flex; align-items: center; margin: 2px 0;">
                <span style="display: inline-block; width: 10px; height: 10px;
                             background: {color}; border-radius: 2px; margin-right: 6px;"></span>
                <span style="color: #ddd; font-size: 11px;">{rel_type}: {count}</span>
            </div>
        """

    html += """
        </div>
    </div>
    """

    return html


def create_javascript() -> str:
    """Generate JavaScript for interactivity."""
    return """
    <script>
    // Store original node visibility
    const nodeVisibility = {};
    const allNodes = network.body.data.nodes;
    const allEdges = network.body.data.edges;

    // Initialize visibility tracking
    allNodes.forEach(function(node) {
        nodeVisibility[node.id] = true;
    });

    // Filter by entity type
    function filterByType() {
        const checkboxes = document.querySelectorAll('#category-filters input[type="checkbox"]');
        const visibleTypes = new Set();

        checkboxes.forEach(function(cb) {
            if (cb.checked) {
                visibleTypes.add(cb.dataset.type);
            }
        });

        // Update node visibility
        const updates = [];
        allNodes.forEach(function(node) {
            const isVisible = visibleTypes.has(node.group);
            nodeVisibility[node.id] = isVisible;
            updates.push({
                id: node.id,
                hidden: !isVisible
            });
        });
        allNodes.update(updates);

        // Update edge visibility
        const edgeUpdates = [];
        allEdges.forEach(function(edge) {
            const sourceVisible = nodeVisibility[edge.from];
            const targetVisible = nodeVisibility[edge.to];
            edgeUpdates.push({
                id: edge.id,
                hidden: !(sourceVisible && targetVisible)
            });
        });
        allEdges.update(edgeUpdates);
    }

    // Toggle all checkboxes
    function toggleAll(checked) {
        const checkboxes = document.querySelectorAll('#category-filters input[type="checkbox"]');
        checkboxes.forEach(function(cb) {
            cb.checked = checked;
        });
        filterByType();
    }

    // Search functionality
    document.getElementById('search-input').addEventListener('input', function(e) {
        const searchTerm = e.target.value.toLowerCase();

        if (searchTerm.length < 2) {
            // Reset to filter state
            filterByType();
            return;
        }

        const updates = [];
        allNodes.forEach(function(node) {
            const matches = node.label.toLowerCase().includes(searchTerm);
            updates.push({
                id: node.id,
                hidden: !matches,
                // Highlight matching nodes
                borderWidth: matches ? 3 : 1,
                borderWidthSelected: matches ? 5 : 2
            });
        });
        allNodes.update(updates);

        // Focus on first match
        const firstMatch = updates.find(u => !u.hidden);
        if (firstMatch) {
            network.focus(firstMatch.id, {
                scale: 1.5,
                animation: {
                    duration: 500,
                    easingFunction: 'easeInOutQuad'
                }
            });
        }
    });

    // Double-click to focus on node's neighborhood
    network.on('doubleClick', function(params) {
        if (params.nodes.length > 0) {
            const nodeId = params.nodes[0];
            const connectedNodes = network.getConnectedNodes(nodeId);
            connectedNodes.push(nodeId);

            network.fit({
                nodes: connectedNodes,
                animation: {
                    duration: 500,
                    easingFunction: 'easeInOutQuad'
                }
            });
        }
    });
    </script>
    """


def export_to_gexf(
    entities: list, relationships: list, output_path: str = "output/kg_graph.gexf"
) -> None:
    """Export graph to GEXF format for Gephi."""
    G = nx.DiGraph()

    # Add nodes
    for entity in entities:
        G.add_node(
            entity["id"],
            label=entity["value"],
            entity_type=entity["type"],
            confidence=entity.get("confidence", 0.5),
            viz={"color": {"hex": ENTITY_COLORS.get(entity["type"], "#95A5A6")}},
        )

    # Add edges
    for rel in relationships:
        G.add_edge(
            rel["source_id"],
            rel["target_id"],
            label=rel["rel_type"],
            weight=rel.get("confidence", 0.5),
        )

    # Write GEXF
    Path(output_path).parent.mkdir(exist_ok=True)
    nx.write_gexf(G, output_path)
    print(f"✅ GEXF exported to: {Path(output_path).absolute()}")


def create_visualization(output_file: str = "kg_visualization.html", export_gexf: bool = False):
    """Create interactive HTML visualization of Neo4j graph."""

    print("🔗 Connecting to Neo4j...")
    config = Neo4jConfig.from_env()
    manager = Neo4jManager(config)

    try:
        # Create pyvis network
        net = Network(
            height="100vh",
            width="100%",
            bgcolor="#1a1a2e",
            font_color="white",
            directed=True,
            notebook=False,
            cdn_resources="remote",
        )

        # Configure physics for better layout
        net.set_options(
            """
        {
          "physics": {
            "enabled": true,
            "barnesHut": {
              "gravitationalConstant": -10000,
              "centralGravity": 0.4,
              "springLength": 120,
              "springConstant": 0.05,
              "damping": 0.1,
              "avoidOverlap": 0.2
            },
            "minVelocity": 0.5,
            "solver": "barnesHut",
            "stabilization": {
              "enabled": true,
              "iterations": 500,
              "updateInterval": 50
            }
          },
          "nodes": {
            "font": {
              "size": 12,
              "color": "#ffffff",
              "face": "arial"
            },
            "borderWidth": 2,
            "borderWidthSelected": 4
          },
          "edges": {
            "smooth": {
              "type": "continuous",
              "forceDirection": "none"
            },
            "font": {
              "size": 9,
              "align": "middle",
              "color": "#aaaaaa"
            },
            "arrows": {
              "to": {
                "enabled": true,
                "scaleFactor": 0.5
              }
            }
          },
          "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "hideEdgesOnDrag": true,
            "hideEdgesOnZoom": true
          }
        }
        """
        )

        # Fetch all entities
        print("📊 Loading entities from Neo4j...")
        entities_query = """
        MATCH (e:Entity)
        RETURN e.id as id, e.type as type, e.value as value,
               e.confidence as confidence, e.document_id as document_id
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
            if len(label) > 50:
                label = label[:47] + "..."

            # Get color for entity type
            color = ENTITY_COLORS.get(entity_type, "#95A5A6")

            # Node size based on confidence
            size = 12 + (entity.get("confidence", 0.5) * 18)

            # Create tooltip with full info
            title = f"""
            <b>{entity['value']}</b><br>
            <b>Typ:</b> {entity_type}<br>
            <b>Confidence:</b> {entity.get('confidence', 'N/A'):.2f}<br>
            <b>Dokument:</b> {entity.get('document_id', 'N/A')}<br>
            <b>ID:</b> {entity['id'][:8]}...
            """

            net.add_node(
                entity["id"],
                label=label,
                title=title,
                color=color,
                size=size,
                shape="dot",
                group=entity_type,
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
            color = RELATIONSHIP_COLORS.get(rel_type.lower(), "#555555")

            # Edge width based on confidence
            width = 1 + (rel.get("confidence", 0.5) * 2)

            net.add_edge(
                rel["source_id"],
                rel["target_id"],
                title=rel_type,
                color=color,
                width=width,
            )

        # Export to GEXF if requested
        if export_gexf:
            export_to_gexf(entities, relationships)

        # Generate HTML
        output_path = f"output/{output_file}"
        Path("output").mkdir(exist_ok=True)

        # Get entity type categories
        categories = get_entity_type_categories()

        # Generate the base HTML
        net.save_graph(output_path)

        # Read the generated HTML and inject our custom panels
        with open(output_path) as f:
            html_content = f.read()

        # Inject custom CSS, filter panel, stats panel, and JavaScript
        filter_html = create_filter_html(entity_counts, categories)
        stats_html = create_stats_html(entity_counts, rel_counts)
        custom_js = create_javascript()

        # Insert before closing body tag
        injection = f"""
        {filter_html}
        {stats_html}
        {custom_js}
        </body>
        """
        html_content = html_content.replace("</body>", injection)

        # Add custom CSS
        custom_css = """
        <style>
            body { margin: 0; padding: 0; overflow: hidden; }
            #mynetwork { width: 100%; height: 100vh; }
            #filter-panel::-webkit-scrollbar { width: 6px; }
            #filter-panel::-webkit-scrollbar-track { background: #333; }
            #filter-panel::-webkit-scrollbar-thumb { background: #666; border-radius: 3px; }
            #filter-panel::-webkit-scrollbar-thumb:hover { background: #888; }
        </style>
        """
        html_content = html_content.replace("</head>", f"{custom_css}</head>")

        with open(output_path, "w") as f:
            f.write(html_content)

        print("\n" + "=" * 60)
        print("✅ VISUALIZATION CREATED SUCCESSFULLY!")
        print("=" * 60)
        print(f"\n📊 Statistics:")
        print(f"  Total entities: {len(entities):,}")
        print(f"  Total relationships: {len(relationships):,}")
        print(f"  Entity types: {len(entity_counts)}")
        print(f"  Relationship types: {len(rel_counts)}")

        print(f"\n📂 Output file: {Path(output_path).absolute()}")
        print(f"\n💡 Open in browser:")
        print(f"   xdg-open {output_path}  # Linux")
        print(f"   open {output_path}       # macOS")
        print("\n" + "=" * 60)

        print("\n🎨 Features:")
        print("  • Filter panel (left) - toggle entity types by category")
        print("  • Search bar - find entities by name")
        print("  • Stats panel (right) - entity/relationship counts")
        print("  • Double-click node - focus on neighborhood")
        print("  • Scroll to zoom, drag to pan")

    finally:
        manager.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize Knowledge Graph from Neo4j")
    parser.add_argument(
        "--output",
        "-o",
        default="kg_visualization.html",
        help="Output HTML filename (default: kg_visualization.html)",
    )
    parser.add_argument(
        "--export-gexf",
        action="store_true",
        help="Also export to GEXF format for Gephi",
    )

    args = parser.parse_args()
    create_visualization(output_file=args.output, export_gexf=args.export_gexf)


if __name__ == "__main__":
    main()
