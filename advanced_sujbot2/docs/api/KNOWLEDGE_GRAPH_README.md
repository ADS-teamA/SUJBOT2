# Legal Knowledge Graph Implementation

This implementation provides a comprehensive knowledge graph system for legal document analysis, based on specification `10_knowledge_graph.md`.

## Overview

The knowledge graph system models legal documents as interconnected structures, capturing:
- **Hierarchical structure**: Document → Part → Chapter → Paragraph → Subsection
- **Legal references**: Explicit citations between provisions (§89, Článek 5, etc.)
- **Semantic relationships**: Similar provisions linked by content similarity
- **Compliance mappings**: Contract clauses mapped to law requirements
- **Conflicts**: Detected conflicts between provisions

## Architecture

### Core Components

1. **LegalKnowledgeGraph** (`knowledge_graph.py:131-280`)
   - NetworkX-based directed graph
   - Fast lookup indexes (by reference, by document)
   - Save/load persistence
   - Graph statistics

2. **GraphBuilder** (`knowledge_graph.py:286-484`)
   - Builds graph from parsed documents
   - Supports law codes (Part → Chapter → Paragraph)
   - Supports contracts (Article → Point)
   - Generic structure for unstructured documents

3. **ReferenceLinker** (`knowledge_graph.py:490-554`)
   - Extracts legal references from text
   - Resolves references to target nodes
   - Creates REFERENCES edges

4. **SemanticLinker** (`knowledge_graph.py:560-628`)
   - Computes semantic similarity between provisions
   - Links similar nodes with RELATED_TO edges
   - Configurable similarity threshold (default: 0.75)

5. **ComplianceLinker** (`knowledge_graph.py:634-708`)
   - Links contract clauses to law requirements
   - Creates COMPLIES_WITH edges for compliance
   - Creates CONFLICTS_WITH edges for conflicts

6. **GraphRetriever** (`knowledge_graph.py:714-823`)
   - Proximity search within N hops
   - Reference traversal (recursive and non-recursive)
   - Path finding between nodes

7. **MultiHopReasoner** (`knowledge_graph.py:829-979`)
   - Finds indirect requirements via multi-hop traversal
   - Explains relationships with natural language
   - Detects transitive conflicts

8. **GraphAnalyzer** (`knowledge_graph.py:985-1071`)
   - Computes centrality metrics (betweenness, degree, PageRank)
   - Finds hub provisions (most important nodes)
   - Community detection (Louvain algorithm)

9. **GraphVisualizer** (`knowledge_graph.py:1077-1137`)
   - Exports to JSON (for D3.js)
   - Exports to GraphML (for Gephi/Cytoscape)
   - Generates text summary reports

## Data Structures

### Node Types (NodeType enum)
- `DOCUMENT`: Entire document
- `PART`: Část (law structure)
- `CHAPTER`: Hlava (law structure)
- `SECTION`: Oddíl (law structure)
- `PARAGRAPH`: § (law provision)
- `ARTICLE`: Článek (contract provision)
- `SUBSECTION`: odstavec (sub-provision)
- `LETTER`: písmeno (enumeration)
- `POINT`: bod (enumeration)
- `CHUNK`: Indexed chunk (fallback)

### Edge Types (EdgeType enum)
- `PART_OF`: Structural hierarchy
- `REFERENCES`: Explicit citation
- `RELATED_TO`: Semantic similarity
- `CONFLICTS_WITH`: Detected conflict
- `COMPLIES_WITH`: Compliance mapping
- `REQUIRES`: Dependency relationship
- `DEFINES`: Definition relationship

### GraphNode (dataclass)
```python
@dataclass
class GraphNode:
    node_id: str
    node_type: NodeType
    label: str
    document_id: str
    content: Optional[str]
    legal_reference: Optional[str]
    hierarchy_path: Optional[str]
    hierarchy_level: int
    metadata: Dict[str, Any]
```

### GraphEdge (dataclass)
```python
@dataclass
class GraphEdge:
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float
    metadata: Dict[str, Any]
```

## Usage Examples

### Building a Knowledge Graph

```python
from knowledge_graph import GraphBuilder, LegalKnowledgeGraph

# Build from parsed documents
builder = GraphBuilder()
kg = builder.build_from_documents([law_doc, contract_doc])

# Get statistics
stats = kg.get_statistics()
print(f"Nodes: {stats['total_nodes']}, Edges: {stats['total_edges']}")
```

### Adding Reference Links

```python
from knowledge_graph import ReferenceLinker

# Link references (requires LegalReferenceExtractor)
ref_linker = ReferenceLinker(reference_extractor)
ref_linker.link_references(kg)
```

### Adding Semantic Links

```python
from knowledge_graph import SemanticLinker

# Link semantically similar provisions
semantic_linker = SemanticLinker(embedding_model, threshold=0.75)
await semantic_linker.link_semantically(kg)
```

### Adding Compliance Links

```python
from knowledge_graph import ComplianceLinker

# Add compliance edges from analysis results
compliance_linker = ComplianceLinker()
compliance_linker.link_compliance(kg, compliance_report)
```

### Retrieving by Reference

```python
# Find nodes by legal reference
para_nodes = kg.get_nodes_by_reference("§89")

for node_id in para_nodes:
    node = kg.get_node(node_id)
    print(f"{node.label}: {node.hierarchy_path}")
```

### Proximity Search

```python
from knowledge_graph import GraphRetriever, EdgeType

retriever = GraphRetriever(kg)

# Find provisions within 2 hops
nearby = retriever.get_provisions_by_proximity(
    anchor_node_ids=["para_89_id"],
    max_hops=2,
    edge_types=[EdgeType.REFERENCES, EdgeType.RELATED_TO]
)
```

### Path Finding

```python
# Find shortest path between nodes
path = retriever.find_path(
    source_id="contract_article_5",
    target_id="law_para_89",
    max_length=5
)

if path:
    print(f"Path: {' → '.join(path)}")
```

### Multi-Hop Reasoning

```python
from knowledge_graph import MultiHopReasoner

reasoner = MultiHopReasoner(kg)

# Find indirect requirements
indirect = reasoner.find_indirect_requirements(
    contract_node_id="contract_article_5",
    max_hops=3
)

for result in indirect:
    print(f"Requirement: {result['requirement_node_id']}")
    print(f"Path: {result['path']}")
    print(f"Hops: {result['hop_count']}")

# Explain relationship
explanation = reasoner.explain_relationship(
    source_id="contract_article_5",
    target_id="law_para_89"
)
print(explanation)
```

### Graph Analysis

```python
from knowledge_graph import GraphAnalyzer

analyzer = GraphAnalyzer(kg)

# Compute centrality
centrality = analyzer.compute_centrality()

# Find hub provisions
hubs = analyzer.find_hub_provisions(top_k=10)
for node_id, importance in hubs:
    node = kg.get_node(node_id)
    print(f"{node.label}: {importance:.4f}")

# Compute PageRank
pagerank = analyzer.compute_pagerank()

# Detect communities
communities = analyzer.detect_communities()
```

### Saving and Loading

```python
# Save graph
kg.save("indexes/knowledge_graph.pkl")

# Load graph
kg = LegalKnowledgeGraph.load("indexes/knowledge_graph.pkl")
```

### Visualization

```python
from knowledge_graph import GraphVisualizer

visualizer = GraphVisualizer()

# Export to JSON (for D3.js)
visualizer.export_to_json(kg, "output/graph.json")

# Export to GraphML (for Gephi)
visualizer.export_to_graphml(kg, "output/graph.graphml")

# Generate summary report
report = visualizer.generate_summary_report(kg)
print(report)
```

## Testing

Run the test suite to verify functionality:

```bash
cd /Users/michalprusek/PycharmProjects/SUJBOT2/advanced_sujbot2/src
python test_knowledge_graph.py
```

The test suite demonstrates:
- Graph building from mock documents
- Reference linking
- Proximity search
- Path finding
- Multi-hop reasoning
- Graph analysis
- Export and visualization
- Persistence (save/load)

## Integration Points

### With DocumentReader
The GraphBuilder expects documents with structure:
- Law codes: `parts` → `chapters` → `paragraphs` → `subsections`
- Contracts: `articles` → `points`
- Generic: `chunks`

### With ComplianceAnalyzer
The ComplianceLinker integrates with compliance reports:
- `clause_mappings`: Creates COMPLIES_WITH edges
- `all_issues`: Creates CONFLICTS_WITH edges

### With HybridRetriever
Graph structure can enhance retrieval:
- Proximity scoring based on graph distance
- Reference-aware reranking
- Multi-hop context expansion

### With EmbeddingModel
SemanticLinker uses embeddings for similarity:
- Compatible with any model with `.encode()` method
- Supports batch processing
- Normalized embeddings for cosine similarity

## Performance Considerations

### Memory Usage
- NetworkX stores entire graph in memory
- Large graphs (>100K nodes) may require optimization
- Consider subgraph extraction for specific documents

### Indexing
- Fast lookup by reference: O(1)
- Fast lookup by document: O(1)
- Node retrieval: O(1)

### Graph Operations
- Path finding: O(V + E) for BFS/DFS
- Centrality: O(V * E) for betweenness
- PageRank: O(V * E * iterations)

### Optimization Tips
1. **Subgraph Extraction**: Work with document-specific subgraphs
2. **Lazy Loading**: Load only needed portions of large graphs
3. **Caching**: Cache expensive metrics (centrality, PageRank)
4. **Parallel Processing**: Use multiprocessing for semantic linking
5. **Threshold Tuning**: Adjust similarity threshold to control edge density

## Limitations and Future Work

### Current Limitations
1. **In-memory storage**: All data loaded in RAM
2. **No incremental updates**: Requires full rebuild on changes
3. **Single-language**: Optimized for Czech legal texts
4. **Basic reference extraction**: Requires external LegalReferenceExtractor

### Planned Enhancements
1. **Graph database backend**: Neo4j or Neptune for scalability
2. **Incremental updates**: Add/remove nodes without full rebuild
3. **Temporal versioning**: Track document changes over time
4. **Advanced reasoning**: SPARQL-like queries, graph embeddings
5. **Interactive visualization**: Web-based graph explorer

## Dependencies

```
networkx>=3.0
scikit-learn>=1.0  # For semantic linking
pickle  # For persistence (built-in)
json  # For export (built-in)
asyncio  # For async operations (built-in)
```

## File Structure

```
src/
├── knowledge_graph.py           # Main implementation (1137 lines)
├── test_knowledge_graph.py      # Test suite with examples
└── KNOWLEDGE_GRAPH_README.md    # This file
```

## References

- Specification: `../specs/10_knowledge_graph.md`
- NetworkX Documentation: https://networkx.org/
- Graph Theory for Legal Documents: See spec section 1

## License

Same as parent project.

## Author

Implemented based on specification by Michal Prusek.

## Status

✅ **Complete** - All features from specification implemented:
- [x] LegalKnowledgeGraph with NetworkX
- [x] GraphBuilder for structure modeling
- [x] Node types (Document, Part, Chapter, Paragraph, Article, etc.)
- [x] Edge types (PART_OF, REFERENCES, RELATED_TO, CONFLICTS_WITH, etc.)
- [x] ReferenceLinker and SemanticLinker
- [x] Multi-hop reasoning algorithms
- [x] GraphRetriever for graph-based search
- [x] Path finding and proximity search
- [x] Graph analysis (centrality, hubs, communities)
- [x] Visualization export (JSON, GraphML)
- [x] Persistence (save/load)
- [x] Comprehensive test suite
