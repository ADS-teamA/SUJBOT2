#!/usr/bin/env python
"""
Simple HTTP server for graph visualization.

Generates an interactive HTML visualization and serves it on localhost.
Forward port 8765 in VS Code to view in your browser.

Usage:
    uv run python scripts/serve_graph.py

Then in VS Code: Forward port 8765 and open http://localhost:8765
"""

import http.server
import re
import socketserver
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pyvis.network import Network

from src.graph.config import Neo4jConfig
from src.graph.neo4j_manager import Neo4jManager

PORT = 8765


def sanitize(text):
    """Remove control characters."""
    if not text:
        return ""
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', str(text))


def generate_graph_html():
    """Generate interactive graph HTML."""
    print("🔗 Connecting to Neo4j...")
    config = Neo4jConfig.from_env()
    manager = Neo4jManager(config)

    try:
        # Create pyvis network - optimized for large graphs
        net = Network(
            height="100vh",
            width="100%",
            bgcolor="#1a1a2e",
            font_color="white",
            directed=True,
            notebook=False,
            cdn_resources="remote",
        )

        # Physics optimized for CPU
        net.set_options('''
        {
          "physics": {
            "enabled": true,
            "barnesHut": {
              "gravitationalConstant": -8000,
              "centralGravity": 0.3,
              "springLength": 150,
              "springConstant": 0.04,
              "damping": 0.09
            },
            "minVelocity": 0.75,
            "solver": "barnesHut"
          },
          "nodes": {
            "font": {"size": 12, "color": "#ffffff"}
          },
          "edges": {
            "smooth": {"type": "continuous"},
            "arrows": {"to": {"enabled": true, "scaleFactor": 0.5}}
          },
          "interaction": {
            "hover": true,
            "hideEdgesOnDrag": true,
            "hideEdgesOnZoom": true
          }
        }
        ''')

        # Fetch entities (Graphiti schema)
        print("📊 Loading entities...")
        entities = manager.execute("""
            MATCH (e:Entity)
            RETURN e.uuid as id, e.name as name, e.summary as summary,
                   e.group_id as doc_id
            LIMIT 5000
        """)
        print(f"  → {len(entities)} entities")

        # Color palette by document
        doc_colors = {}
        colors = ["#E74C3C", "#3498DB", "#2ECC71", "#9B59B6", "#F39C12",
                  "#1ABC9C", "#E67E22", "#34495E", "#16A085", "#C0392B"]

        for entity in entities:
            if not entity.get("id"):
                continue

            doc_id = entity.get("doc_id") or "unknown"
            if doc_id not in doc_colors:
                doc_colors[doc_id] = colors[len(doc_colors) % len(colors)]

            name = sanitize(entity.get("name") or "")
            label = name[:40] + "..." if len(name) > 40 else name
            summary = sanitize(entity.get("summary") or "")[:200]

            net.add_node(
                entity["id"],
                label=label or entity["id"][:8],
                title=f"<b>{name}</b><br>{summary}<br><i>Doc: {doc_id}</i>",
                color=doc_colors[doc_id],
                size=15,
            )

        # Fetch relationships
        print("🔗 Loading relationships...")
        rels = manager.execute("""
            MATCH (a:Entity)-[r]->(b:Entity)
            RETURN a.uuid as source, b.uuid as target, type(r) as rel_type
            LIMIT 10000
        """)
        print(f"  → {len(rels)} relationships")

        node_ids = {e["id"] for e in entities if e.get("id")}
        for rel in rels:
            if rel.get("source") in node_ids and rel.get("target") in node_ids:
                net.add_edge(
                    rel["source"],
                    rel["target"],
                    title=rel.get("rel_type", ""),
                    color="#555555",
                )

        # Generate HTML
        output_path = Path("output/graph_server.html")
        output_path.parent.mkdir(exist_ok=True)
        net.save_graph(str(output_path))

        print(f"\n✅ Graph generated: {len(entities)} nodes, {len(rels)} edges")
        return output_path

    finally:
        manager.close()


def serve():
    """Start HTTP server."""
    html_path = generate_graph_html()

    # Change to output directory
    import os
    os.chdir(html_path.parent)

    handler = http.server.SimpleHTTPRequestHandler

    with socketserver.TCPServer(("", PORT), handler) as httpd:
        print(f"\n🌐 Server running at http://localhost:{PORT}")
        print(f"📂 Serving: {html_path.name}")
        print("\n💡 In VS Code: Forward port {PORT} and open http://localhost:{PORT}/graph_server.html")
        print("\nPress Ctrl+C to stop...")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n👋 Server stopped")


if __name__ == "__main__":
    serve()
