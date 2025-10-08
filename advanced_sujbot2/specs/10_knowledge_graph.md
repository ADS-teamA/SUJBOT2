# 10. Knowledge Graph Specification

## 1. Purpose

**Objective**: Build and maintain a knowledge graph representing the legal structure, relationships, and cross-references between provisions in contracts and laws to enable graph-enhanced retrieval and multi-hop reasoning.

**Why Knowledge Graph?**
- Legal documents have rich structural hierarchy (Parts > Chapters > Paragraphs)
- Legal provisions reference each other ("podle §89", "v souladu s Článkem 5")
- Understanding relationships improves retrieval accuracy
- Multi-hop reasoning enables discovering indirect dependencies
- Graph proximity can boost relevance scoring in reranking
- Conflict detection benefits from reference tracking

**Key Capabilities**:
1. **Hierarchical Structure Modeling** - Represent document hierarchy as graph
2. **Reference Tracking** - Capture explicit citations between provisions
3. **Cross-Document Edges** - Link contract clauses to law paragraphs
4. **Multi-Hop Reasoning** - Traverse graph to find indirect relationships
5. **Path Finding** - Discover connections between provisions
6. **Graph-Based Retrieval** - Use graph structure to enhance search
7. **Conflict Relationship Modeling** - Track detected conflicts as edges

---

## 2. Knowledge Graph Architecture

### High-Level Structure

```
LegalKnowledgeGraph (NetworkX DiGraph)
│
├── Nodes
│   ├── DocumentNode (entire document)
│   ├── StructuralNode (Part, Chapter, Section)
│   ├── ProvisionNode (Paragraph §, Article, Clause)
│   └── ChunkNode (actual indexed chunks)
│
└── Edges
    ├── PART_OF (structural hierarchy)
    ├── REFERENCES (explicit citations)
    ├── RELATED_TO (semantic similarity)
    ├── CONFLICTS_WITH (detected conflicts)
    ├── COMPLIES_WITH (compliance mapping)
    └── REQUIRES (dependency relationships)
```

### Graph Construction Flow

```
┌─────────────────────────────────────┐
│  Document Structure                 │
│  (from DocumentReader)              │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  GraphBuilder                       │
│  - Create nodes for each element    │
│  - Create PART_OF edges             │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  ReferenceLinker                    │
│  - Extract citations                │
│  - Resolve references               │
│  - Create REFERENCES edges          │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  SemanticLinker                     │
│  - Compute similarity               │
│  - Create RELATED_TO edges          │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  ComplianceLinker                   │
│  - Add compliance edges             │
│  - Add conflict edges               │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  LegalKnowledgeGraph                │
│  (complete graph)                   │
└─────────────────────────────────────┘
```

---

## 3. Data Structures

### 3.1 Node Types

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any
from enum import Enum
import networkx as nx

class NodeType(Enum):
    """Type of graph node."""
    DOCUMENT = "document"
    PART = "part"                   # Část
    CHAPTER = "chapter"             # Hlava
    SECTION = "section"             # Oddíl
    PARAGRAPH = "paragraph"         # §
    ARTICLE = "article"             # Článek
    SUBSECTION = "subsection"       # odstavec
    LETTER = "letter"               # písmeno
    POINT = "point"                 # bod
    CHUNK = "chunk"                 # indexed chunk

class EdgeType(Enum):
    """Type of graph edge."""
    PART_OF = "part_of"             # Structural hierarchy
    REFERENCES = "references"       # Explicit citation
    RELATED_TO = "related_to"       # Semantic similarity
    CONFLICTS_WITH = "conflicts_with"  # Detected conflict
    COMPLIES_WITH = "complies_with"    # Compliance mapping
    REQUIRES = "requires"           # Dependency
    DEFINES = "defines"             # Definition relationship

@dataclass
class GraphNode:
    """Base class for graph nodes."""
    node_id: str
    node_type: NodeType
    label: str                      # Human-readable label
    document_id: str

    # Content
    content: Optional[str] = None   # Text content
    legal_reference: Optional[str] = None  # §89, Článek 5, etc.

    # Hierarchy
    hierarchy_path: Optional[str] = None
    hierarchy_level: int = 0

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for NetworkX storage."""
        return {
            "node_type": self.node_type.value,
            "label": self.label,
            "document_id": self.document_id,
            "content": self.content,
            "legal_reference": self.legal_reference,
            "hierarchy_path": self.hierarchy_path,
            "hierarchy_level": self.hierarchy_level,
            "metadata": self.metadata
        }

@dataclass
class GraphEdge:
    """Base class for graph edges."""
    source_id: str
    target_id: str
    edge_type: EdgeType

    # Edge weight/confidence
    weight: float = 1.0

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for NetworkX storage."""
        return {
            "edge_type": self.edge_type.value,
            "weight": self.weight,
            "metadata": self.metadata
        }
```

### 3.2 Knowledge Graph Class

```python
class LegalKnowledgeGraph:
    """Knowledge graph for legal documents."""

    def __init__(self):
        # NetworkX directed graph
        self.graph = nx.DiGraph()

        # Indexes for fast lookup
        self.node_index: Dict[str, GraphNode] = {}
        self.reference_index: Dict[str, List[str]] = {}  # legal_ref → [node_ids]
        self.document_index: Dict[str, List[str]] = {}   # doc_id → [node_ids]

        self.logger = logging.getLogger(__name__)

    def add_node(self, node: GraphNode) -> None:
        """Add node to graph."""
        self.graph.add_node(node.node_id, **node.to_dict())
        self.node_index[node.node_id] = node

        # Update indexes
        if node.legal_reference:
            if node.legal_reference not in self.reference_index:
                self.reference_index[node.legal_reference] = []
            self.reference_index[node.legal_reference].append(node.node_id)

        if node.document_id not in self.document_index:
            self.document_index[node.document_id] = []
        self.document_index[node.document_id].append(node.node_id)

    def add_edge(self, edge: GraphEdge) -> None:
        """Add edge to graph."""
        self.graph.add_edge(
            edge.source_id,
            edge.target_id,
            **edge.to_dict()
        )

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Retrieve node by ID."""
        return self.node_index.get(node_id)

    def get_neighbors(self, node_id: str, edge_type: Optional[EdgeType] = None) -> List[str]:
        """Get neighboring nodes."""
        if node_id not in self.graph:
            return []

        neighbors = []
        for neighbor in self.graph.successors(node_id):
            edge_data = self.graph[node_id][neighbor]
            if edge_type is None or edge_data.get("edge_type") == edge_type.value:
                neighbors.append(neighbor)

        return neighbors

    def get_nodes_by_reference(self, legal_reference: str) -> List[str]:
        """Find nodes by legal reference."""
        return self.reference_index.get(legal_reference, [])

    def get_nodes_by_document(self, document_id: str) -> List[str]:
        """Find all nodes in a document."""
        return self.document_index.get(document_id, [])

    def get_subgraph_by_edge_type(self, edge_type: EdgeType) -> nx.DiGraph:
        """Extract subgraph with only specific edge type."""
        edges = [
            (u, v) for u, v, data in self.graph.edges(data=True)
            if data.get("edge_type") == edge_type.value
        ]
        return self.graph.edge_subgraph(edges)

    def save(self, path: str) -> None:
        """Persist graph to disk."""
        import pickle
        with open(path, "wb") as f:
            pickle.dump({
                "graph": self.graph,
                "node_index": self.node_index,
                "reference_index": self.reference_index,
                "document_index": self.document_index
            }, f)

    @classmethod
    def load(cls, path: str) -> 'LegalKnowledgeGraph':
        """Load graph from disk."""
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)

        kg = cls()
        kg.graph = data["graph"]
        kg.node_index = data["node_index"]
        kg.reference_index = data["reference_index"]
        kg.document_index = data["document_index"]

        return kg
```

---

## 4. Graph Builder

### 4.1 Building from Document Structure

```python
class GraphBuilder:
    """Build knowledge graph from document structure."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def build_from_documents(
        self,
        documents: List['LegalDocument']
    ) -> LegalKnowledgeGraph:
        """
        Build graph from parsed legal documents.

        Args:
            documents: List of LegalDocument objects (from DocumentReader)

        Returns:
            LegalKnowledgeGraph
        """
        kg = LegalKnowledgeGraph()

        for doc in documents:
            self._build_document_subgraph(doc, kg)

        self.logger.info(f"Built graph with {kg.graph.number_of_nodes()} nodes, "
                        f"{kg.graph.number_of_edges()} edges")

        return kg

    def _build_document_subgraph(
        self,
        document: 'LegalDocument',
        kg: LegalKnowledgeGraph
    ) -> None:
        """Build subgraph for a single document."""
        # Add document node
        doc_node = GraphNode(
            node_id=f"doc_{document.document_id}",
            node_type=NodeType.DOCUMENT,
            label=document.title or document.document_id,
            document_id=document.document_id,
            metadata={
                "document_type": document.document_type,
                "total_chunks": len(document.chunks) if hasattr(document, 'chunks') else 0
            }
        )
        kg.add_node(doc_node)

        # Add structural elements recursively
        if document.document_type == "law_code":
            self._build_law_structure(document, kg, doc_node.node_id)
        elif document.document_type == "contract":
            self._build_contract_structure(document, kg, doc_node.node_id)

    def _build_law_structure(
        self,
        document: 'LegalDocument',
        kg: LegalKnowledgeGraph,
        parent_id: str
    ) -> None:
        """Build structure for law document."""
        # Assume document has .parts attribute
        if not hasattr(document, 'parts'):
            return

        for part in document.parts:
            # Add Part node
            part_node = GraphNode(
                node_id=f"{document.document_id}_part_{part.number}",
                node_type=NodeType.PART,
                label=f"Část {part.number}: {part.title}",
                document_id=document.document_id,
                legal_reference=f"Část {part.number}",
                hierarchy_path=f"Část {part.number}",
                hierarchy_level=1
            )
            kg.add_node(part_node)

            # Add PART_OF edge to document
            kg.add_edge(GraphEdge(
                source_id=part_node.node_id,
                target_id=parent_id,
                edge_type=EdgeType.PART_OF
            ))

            # Add chapters
            for chapter in part.chapters:
                chapter_node = GraphNode(
                    node_id=f"{document.document_id}_chapter_{part.number}_{chapter.number}",
                    node_type=NodeType.CHAPTER,
                    label=f"Hlava {chapter.number}: {chapter.title}",
                    document_id=document.document_id,
                    legal_reference=f"Hlava {chapter.number}",
                    hierarchy_path=f"{part_node.hierarchy_path} > Hlava {chapter.number}",
                    hierarchy_level=2
                )
                kg.add_node(chapter_node)

                kg.add_edge(GraphEdge(
                    source_id=chapter_node.node_id,
                    target_id=part_node.node_id,
                    edge_type=EdgeType.PART_OF
                ))

                # Add paragraphs
                for paragraph in chapter.paragraphs:
                    para_node = GraphNode(
                        node_id=f"{document.document_id}_para_{paragraph.number}",
                        node_type=NodeType.PARAGRAPH,
                        label=f"§{paragraph.number}",
                        document_id=document.document_id,
                        content=paragraph.content,
                        legal_reference=f"§{paragraph.number}",
                        hierarchy_path=f"{chapter_node.hierarchy_path} > §{paragraph.number}",
                        hierarchy_level=3,
                        metadata={
                            "contains_obligation": paragraph.contains_obligation,
                            "contains_prohibition": paragraph.contains_prohibition
                        }
                    )
                    kg.add_node(para_node)

                    kg.add_edge(GraphEdge(
                        source_id=para_node.node_id,
                        target_id=chapter_node.node_id,
                        edge_type=EdgeType.PART_OF
                    ))

                    # Add subsections
                    for subsec in paragraph.subsections:
                        subsec_node = GraphNode(
                            node_id=f"{para_node.node_id}_subsec_{subsec.number}",
                            node_type=NodeType.SUBSECTION,
                            label=f"§{paragraph.number} odst. {subsec.number}",
                            document_id=document.document_id,
                            content=subsec.content,
                            legal_reference=f"§{paragraph.number} odst. {subsec.number}",
                            hierarchy_path=f"{para_node.hierarchy_path} > odst. {subsec.number}",
                            hierarchy_level=4
                        )
                        kg.add_node(subsec_node)

                        kg.add_edge(GraphEdge(
                            source_id=subsec_node.node_id,
                            target_id=para_node.node_id,
                            edge_type=EdgeType.PART_OF
                        ))

    def _build_contract_structure(
        self,
        document: 'LegalDocument',
        kg: LegalKnowledgeGraph,
        parent_id: str
    ) -> None:
        """Build structure for contract document."""
        if not hasattr(document, 'articles'):
            return

        for article in document.articles:
            article_node = GraphNode(
                node_id=f"{document.document_id}_article_{article.number}",
                node_type=NodeType.ARTICLE,
                label=f"Článek {article.number}: {article.title}",
                document_id=document.document_id,
                content=article.content,
                legal_reference=f"Článek {article.number}",
                hierarchy_path=f"Článek {article.number}",
                hierarchy_level=1
            )
            kg.add_node(article_node)

            kg.add_edge(GraphEdge(
                source_id=article_node.node_id,
                target_id=parent_id,
                edge_type=EdgeType.PART_OF
            ))

            # Add points
            for point in article.points:
                point_node = GraphNode(
                    node_id=f"{article_node.node_id}_point_{point.number}",
                    node_type=NodeType.POINT,
                    label=f"Článek {article.number}.{point.number}",
                    document_id=document.document_id,
                    content=point.content,
                    legal_reference=f"Článek {article.number}.{point.number}",
                    hierarchy_path=f"{article_node.hierarchy_path} > {point.number}",
                    hierarchy_level=2
                )
                kg.add_node(point_node)

                kg.add_edge(GraphEdge(
                    source_id=point_node.node_id,
                    target_id=article_node.node_id,
                    edge_type=EdgeType.PART_OF
                ))
```

---

## 5. Reference Linker

### 5.1 Extracting and Resolving References

```python
class ReferenceLinker:
    """Link nodes based on explicit legal references."""

    def __init__(self):
        self.reference_extractor = LegalReferenceExtractor()  # From DocumentReader
        self.logger = logging.getLogger(__name__)

    def link_references(self, kg: LegalKnowledgeGraph) -> None:
        """
        Find and link references between nodes.

        Modifies kg in-place.
        """
        reference_count = 0

        # Iterate over all nodes with content
        for node_id, node in kg.node_index.items():
            if not node.content:
                continue

            # Extract references from content
            references = self.reference_extractor.extract(node.content)

            for ref_entity in references:
                # Resolve reference to target node(s)
                target_nodes = kg.get_nodes_by_reference(ref_entity.normalized)

                for target_id in target_nodes:
                    # Don't self-reference
                    if target_id == node_id:
                        continue

                    # Add REFERENCES edge
                    kg.add_edge(GraphEdge(
                        source_id=node_id,
                        target_id=target_id,
                        edge_type=EdgeType.REFERENCES,
                        weight=ref_entity.confidence,
                        metadata={
                            "reference_text": ref_entity.value,
                            "normalized_ref": ref_entity.normalized
                        }
                    ))

                    reference_count += 1

        self.logger.info(f"Linked {reference_count} references")
```

---

## 6. Semantic Linker

### 6.1 Linking Similar Provisions

```python
class SemanticLinker:
    """Link nodes based on semantic similarity."""

    def __init__(self, embedding_model, similarity_threshold: float = 0.75):
        self.embedding_model = embedding_model
        self.threshold = similarity_threshold

    async def link_semantically(self, kg: LegalKnowledgeGraph) -> None:
        """
        Add RELATED_TO edges between semantically similar nodes.

        Only links nodes of similar types (e.g., paragraph to paragraph).
        """
        # Get all provision nodes (paragraphs, articles) with content
        provision_nodes = [
            node for node in kg.node_index.values()
            if node.node_type in [NodeType.PARAGRAPH, NodeType.ARTICLE, NodeType.SUBSECTION]
            and node.content
        ]

        if len(provision_nodes) < 2:
            return

        # Compute embeddings
        contents = [node.content for node in provision_nodes]
        embeddings = await self._compute_embeddings(contents)

        # Compute pairwise similarities
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(embeddings)

        # Add edges for high similarity pairs
        link_count = 0
        for i, node_i in enumerate(provision_nodes):
            for j, node_j in enumerate(provision_nodes):
                if i >= j:  # Skip self and duplicates
                    continue

                similarity = similarity_matrix[i, j]

                if similarity >= self.threshold:
                    # Add bidirectional RELATED_TO edges
                    kg.add_edge(GraphEdge(
                        source_id=node_i.node_id,
                        target_id=node_j.node_id,
                        edge_type=EdgeType.RELATED_TO,
                        weight=similarity
                    ))

                    kg.add_edge(GraphEdge(
                        source_id=node_j.node_id,
                        target_id=node_i.node_id,
                        edge_type=EdgeType.RELATED_TO,
                        weight=similarity
                    ))

                    link_count += 2

        self.logger.info(f"Added {link_count} semantic similarity edges")

    async def _compute_embeddings(self, texts: List[str]):
        """Compute embeddings for texts."""
        # Use embedding model (e.g., BGE-M3)
        return await asyncio.to_thread(
            self.embedding_model.encode,
            texts,
            batch_size=32,
            normalize_embeddings=True
        )
```

---

## 7. Compliance Linker

### 7.1 Adding Compliance Edges

```python
class ComplianceLinker:
    """Add compliance-related edges to graph."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def link_compliance(
        self,
        kg: LegalKnowledgeGraph,
        compliance_report: 'ComplianceReport'
    ) -> None:
        """
        Add edges based on compliance analysis results.

        Args:
            kg: Knowledge graph
            compliance_report: ComplianceReport from ComplianceAnalyzer
        """
        # Add COMPLIES_WITH edges for compliant mappings
        for mapping in compliance_report.clause_mappings:
            contract_node_id = self._find_node_by_chunk(kg, mapping.contract_chunk_id)

            for req in mapping.law_requirements:
                law_node_id = self._find_node_by_chunk(kg, req.law_chunk_id)

                if contract_node_id and law_node_id:
                    kg.add_edge(GraphEdge(
                        source_id=contract_node_id,
                        target_id=law_node_id,
                        edge_type=EdgeType.COMPLIES_WITH,
                        weight=mapping.match_score
                    ))

        # Add CONFLICTS_WITH edges for conflicts
        for issue in compliance_report.all_issues:
            if issue.status == ComplianceStatus.CONFLICT:
                contract_node_id = self._find_node_by_chunk(kg, issue.contract_chunk_id)

                for req in issue.law_requirements:
                    law_node_id = self._find_node_by_chunk(kg, req.law_chunk_id)

                    if contract_node_id and law_node_id:
                        kg.add_edge(GraphEdge(
                            source_id=contract_node_id,
                            target_id=law_node_id,
                            edge_type=EdgeType.CONFLICTS_WITH,
                            weight=issue.risk_score,
                            metadata={
                                "severity": issue.severity.value,
                                "description": issue.issue_description
                            }
                        ))

        self.logger.info(f"Added compliance edges to graph")

    def _find_node_by_chunk(self, kg: LegalKnowledgeGraph, chunk_id: str) -> Optional[str]:
        """Find graph node corresponding to chunk ID."""
        # Simple heuristic: look for node with matching chunk_id in metadata
        for node_id, node in kg.node_index.items():
            if node.metadata.get("chunk_id") == chunk_id:
                return node_id
        return None
```

---

## 8. Graph-Based Retrieval

### 8.1 Proximity Search

```python
class GraphRetriever:
    """Retrieve provisions using graph structure."""

    def __init__(self, kg: LegalKnowledgeGraph):
        self.kg = kg

    def get_provisions_by_proximity(
        self,
        anchor_node_ids: List[str],
        max_hops: int = 2,
        edge_types: Optional[List[EdgeType]] = None
    ) -> List[str]:
        """
        Find provisions within N hops of anchor nodes.

        Args:
            anchor_node_ids: Starting nodes
            max_hops: Maximum distance
            edge_types: Filter by edge types (None = all types)

        Returns:
            List of node IDs within proximity
        """
        visited = set()
        current_layer = set(anchor_node_ids)

        for hop in range(max_hops):
            next_layer = set()

            for node_id in current_layer:
                if node_id in visited:
                    continue

                visited.add(node_id)

                # Get neighbors
                for neighbor in self.kg.graph.successors(node_id):
                    edge_data = self.kg.graph[node_id][neighbor]

                    # Filter by edge type
                    if edge_types is not None:
                        edge_type = edge_data.get("edge_type")
                        if edge_type not in [et.value for et in edge_types]:
                            continue

                    next_layer.add(neighbor)

            current_layer = next_layer

        return list(visited)

    def get_referenced_provisions(
        self,
        node_id: str,
        recursive: bool = False
    ) -> List[str]:
        """
        Get all provisions referenced by this node.

        Args:
            node_id: Source node
            recursive: Follow references recursively

        Returns:
            List of referenced node IDs
        """
        if not recursive:
            return self.kg.get_neighbors(node_id, EdgeType.REFERENCES)

        # Recursive: DFS traversal
        visited = set()
        stack = [node_id]

        while stack:
            current = stack.pop()

            if current in visited:
                continue

            visited.add(current)

            # Get direct references
            refs = self.kg.get_neighbors(current, EdgeType.REFERENCES)
            stack.extend(refs)

        visited.discard(node_id)  # Remove starting node
        return list(visited)

    def find_path(
        self,
        source_id: str,
        target_id: str,
        max_length: int = 5
    ) -> Optional[List[str]]:
        """
        Find shortest path between two nodes.

        Returns:
            List of node IDs forming path, or None if no path exists
        """
        try:
            path = nx.shortest_path(
                self.kg.graph,
                source=source_id,
                target=target_id
            )

            if len(path) <= max_length:
                return path
            else:
                return None

        except nx.NetworkXNoPath:
            return None
```

---

## 9. Multi-Hop Reasoning

### 9.1 Transitive Reasoning

```python
class MultiHopReasoner:
    """Perform multi-hop reasoning over knowledge graph."""

    def __init__(self, kg: LegalKnowledgeGraph):
        self.kg = kg

    def find_indirect_requirements(
        self,
        contract_node_id: str,
        max_hops: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Find law requirements indirectly referenced by contract clause.

        Example:
        Contract clause → references Článek 5 → which references §89 → §89 has requirements

        Returns:
            List of {requirement_node_id, path, hop_count}
        """
        results = []

        # BFS traversal
        visited = {contract_node_id}
        queue = [(contract_node_id, [contract_node_id], 0)]

        while queue:
            current_id, path, hops = queue.pop(0)

            if hops >= max_hops:
                continue

            # Get neighbors via REFERENCES edges
            for neighbor_id in self.kg.get_neighbors(current_id, EdgeType.REFERENCES):
                if neighbor_id in visited:
                    continue

                visited.add(neighbor_id)
                new_path = path + [neighbor_id]

                # Check if neighbor is a law provision
                neighbor_node = self.kg.get_node(neighbor_id)
                if neighbor_node and neighbor_node.node_type in [NodeType.PARAGRAPH, NodeType.SUBSECTION]:
                    # Found a law requirement
                    results.append({
                        "requirement_node_id": neighbor_id,
                        "path": new_path,
                        "hop_count": hops + 1
                    })

                # Continue searching
                queue.append((neighbor_id, new_path, hops + 1))

        return results

    def explain_relationship(
        self,
        source_id: str,
        target_id: str
    ) -> Optional[str]:
        """
        Generate natural language explanation of relationship between two nodes.

        Returns:
            Explanation string or None if no relationship found
        """
        # Find path
        path = nx.shortest_path(self.kg.graph, source=source_id, target=target_id)

        if not path:
            return None

        # Build explanation
        explanation_parts = []

        for i in range(len(path) - 1):
            src = path[i]
            tgt = path[i + 1]

            src_node = self.kg.get_node(src)
            tgt_node = self.kg.get_node(tgt)

            edge_data = self.kg.graph[src][tgt]
            edge_type = edge_data.get("edge_type")

            # Generate hop description
            if edge_type == EdgeType.REFERENCES.value:
                hop_desc = f"{src_node.label} references {tgt_node.label}"
            elif edge_type == EdgeType.PART_OF.value:
                hop_desc = f"{src_node.label} is part of {tgt_node.label}"
            elif edge_type == EdgeType.CONFLICTS_WITH.value:
                hop_desc = f"{src_node.label} conflicts with {tgt_node.label}"
            elif edge_type == EdgeType.RELATED_TO.value:
                hop_desc = f"{src_node.label} is semantically related to {tgt_node.label}"
            else:
                hop_desc = f"{src_node.label} → {tgt_node.label}"

            explanation_parts.append(hop_desc)

        explanation = " → ".join(explanation_parts)
        return explanation
```

---

## 10. Graph Analysis

### 10.1 Centrality and Importance

```python
class GraphAnalyzer:
    """Analyze graph structure for insights."""

    def __init__(self, kg: LegalKnowledgeGraph):
        self.kg = kg

    def compute_centrality(self) -> Dict[str, float]:
        """
        Compute betweenness centrality for all nodes.

        Returns:
            Dict mapping node_id to centrality score
        """
        centrality = nx.betweenness_centrality(self.kg.graph)
        return centrality

    def find_hub_provisions(self, top_k: int = 10) -> List[tuple[str, float]]:
        """
        Find most important provisions (high centrality + many references).

        Returns:
            List of (node_id, importance_score)
        """
        # Combine degree centrality and betweenness
        degree_cent = nx.degree_centrality(self.kg.graph)
        between_cent = nx.betweenness_centrality(self.kg.graph)

        # Weighted combination
        importance = {}
        for node_id in self.kg.node_index:
            importance[node_id] = 0.5 * degree_cent.get(node_id, 0) + 0.5 * between_cent.get(node_id, 0)

        # Sort and return top-K
        sorted_nodes = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        return sorted_nodes[:top_k]

    def detect_communities(self) -> Dict[str, int]:
        """
        Detect communities (clusters) in graph.

        Returns:
            Dict mapping node_id to community_id
        """
        # Convert to undirected for community detection
        undirected = self.kg.graph.to_undirected()

        # Use Louvain algorithm
        import networkx.algorithms.community as nx_comm
        communities = nx_comm.louvain_communities(undirected)

        # Map nodes to community IDs
        node_to_community = {}
        for comm_id, community in enumerate(communities):
            for node_id in community:
                node_to_community[node_id] = comm_id

        return node_to_community
```

---

## 11. Graph Visualization (Optional)

### 11.1 Export for Visualization

```python
class GraphVisualizer:
    """Export graph for visualization."""

    def export_to_json(self, kg: LegalKnowledgeGraph, output_path: str) -> None:
        """Export graph to JSON format for D3.js visualization."""
        # Convert to node-link format
        data = nx.node_link_data(kg.graph)

        # Add labels for visualization
        for node in data["nodes"]:
            node_obj = kg.get_node(node["id"])
            if node_obj:
                node["label"] = node_obj.label
                node["type"] = node_obj.node_type.value

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def export_to_graphml(self, kg: LegalKnowledgeGraph, output_path: str) -> None:
        """Export to GraphML for Gephi/Cytoscape."""
        nx.write_graphml(kg.graph, output_path)
```

---

## 12. Usage Examples

### 12.1 Building Graph

```python
from src.knowledge_graph import GraphBuilder, ReferenceLinker, SemanticLinker

# Build from parsed documents
builder = GraphBuilder()
kg = builder.build_from_documents([contract_doc, law_doc])

# Link references
ref_linker = ReferenceLinker()
ref_linker.link_references(kg)

# Link semantically similar provisions
semantic_linker = SemanticLinker(embedding_model, threshold=0.75)
await semantic_linker.link_semantically(kg)

# Save graph
kg.save("indexes/knowledge_graph.pkl")
```

### 12.2 Graph-Based Retrieval

```python
# Load graph
kg = LegalKnowledgeGraph.load("indexes/knowledge_graph.pkl")

# Find provisions by reference
para_nodes = kg.get_nodes_by_reference("§89")
print(f"Found {len(para_nodes)} nodes for §89")

# Find provisions within 2 hops
retriever = GraphRetriever(kg)
nearby = retriever.get_provisions_by_proximity(
    anchor_node_ids=para_nodes,
    max_hops=2,
    edge_types=[EdgeType.REFERENCES, EdgeType.RELATED_TO]
)

# Find path between contract and law
path = retriever.find_path(source_id="contract_article_5", target_id="law_para_89")
if path:
    print(f"Path: {' → '.join(path)}")
```

### 12.3 Multi-Hop Reasoning

```python
reasoner = MultiHopReasoner(kg)

# Find indirect requirements
indirect = reasoner.find_indirect_requirements(
    contract_node_id="contract_article_5",
    max_hops=3
)

for result in indirect:
    print(f"Requirement: {result['requirement_node_id']}")
    print(f"Hop count: {result['hop_count']}")
    print(f"Path: {result['path']}")

# Explain relationship
explanation = reasoner.explain_relationship(
    source_id="contract_article_5",
    target_id="law_para_89"
)
print(explanation)
```

---

## 13. Performance Optimization

### 13.1 Index Caching

```python
# Precompute and cache expensive graph metrics
class CachedGraphAnalyzer(GraphAnalyzer):
    def __init__(self, kg: LegalKnowledgeGraph):
        super().__init__(kg)
        self._centrality_cache = None
        self._pagerank_cache = None

    def get_centrality(self, node_id: str) -> float:
        if self._centrality_cache is None:
            self._centrality_cache = self.compute_centrality()
        return self._centrality_cache.get(node_id, 0.0)
```

### 13.2 Subgraph Extraction

```python
# Work with smaller subgraphs for performance
def extract_document_subgraph(kg: LegalKnowledgeGraph, document_id: str) -> LegalKnowledgeGraph:
    """Extract subgraph for a single document."""
    node_ids = kg.get_nodes_by_document(document_id)

    subgraph = kg.graph.subgraph(node_ids).copy()

    sub_kg = LegalKnowledgeGraph()
    sub_kg.graph = subgraph
    # Rebuild indexes
    # ...

    return sub_kg
```

---

## 14. Summary

Knowledge Graph enables:

1. **Structural Understanding** - Represents legal document hierarchy as graph
2. **Reference Tracking** - Captures explicit citations between provisions
3. **Semantic Relationships** - Links similar provisions
4. **Compliance Mapping** - Tracks compliance and conflicts as edges
5. **Multi-Hop Reasoning** - Discovers indirect dependencies
6. **Graph-Enhanced Retrieval** - Uses graph structure to boost relevance

**Integration**:
- Built from DocumentReader output
- Used by GraphAwareReranker for proximity scoring
- Updated by ComplianceAnalyzer with compliance edges
- Supports multi-hop reasoning for complex queries

**Next Steps**:
- See [11. API Interfaces](11_api_interfaces.md) for final integration

---

**Page Count**: ~20 pages
**Last Updated**: 2025-10-08
**Status**: Complete ✅
