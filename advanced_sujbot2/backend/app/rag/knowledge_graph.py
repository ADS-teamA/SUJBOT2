"""
Knowledge Graph for Legal Document Analysis

This module implements a legal knowledge graph system using NetworkX to model
document structure, legal references, semantic relationships, and compliance mappings.

Key Features:
- Hierarchical structure modeling (Document → Part → Chapter → Paragraph)
- Reference tracking with explicit citation resolution
- Semantic similarity linking
- Multi-hop reasoning and path finding
- Graph-based retrieval with proximity search
- Compliance and conflict relationship modeling
"""

import logging
import pickle
import json
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Tuple
from enum import Enum
from pathlib import Path

import networkx as nx


# ============================================================================
# Enums and Data Classes
# ============================================================================

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


# ============================================================================
# Knowledge Graph Core
# ============================================================================

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

    def get_chunks_by_reference(self, legal_reference: str) -> List[Any]:
        """
        Get chunk objects by legal reference.

        This method is used by cross-document retrieval and reranking
        to find target chunks by their legal reference (e.g., "§89", "Článek 5").

        Args:
            legal_reference: Legal reference string (e.g., "§89 odst. 2")

        Returns:
            List of chunk-like dictionaries with matching legal reference
        """
        # Get node IDs with this reference
        node_ids = self.get_nodes_by_reference(legal_reference)

        chunks = []
        for node_id in node_ids:
            node = self.get_node(node_id)
            if node and node.content:
                # Convert GraphNode to chunk-like dictionary
                # This format is compatible with cross_doc_retrieval and reranker
                chunk_data = {
                    'chunk_id': node.metadata.get('chunk_id', node_id),
                    'content': node.content,
                    'title': node.label,
                    'document_id': node.document_id,
                    'legal_reference': node.legal_reference,
                    'hierarchy_path': node.hierarchy_path,
                    'metadata': node.metadata,
                    'node_type': node.node_type.value
                }
                chunks.append(chunk_data)

        return chunks

    def get_subgraph_by_edge_type(self, edge_type: EdgeType) -> nx.DiGraph:
        """Extract subgraph with only specific edge type."""
        edges = [
            (u, v) for u, v, data in self.graph.edges(data=True)
            if data.get("edge_type") == edge_type.value
        ]
        return self.graph.edge_subgraph(edges)

    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        stats = {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "node_types": {},
            "edge_types": {},
            "documents": len(self.document_index)
        }

        # Count node types
        for node in self.node_index.values():
            node_type = node.node_type.value
            stats["node_types"][node_type] = stats["node_types"].get(node_type, 0) + 1

        # Count edge types
        for _, _, data in self.graph.edges(data=True):
            edge_type = data.get("edge_type", "unknown")
            stats["edge_types"][edge_type] = stats["edge_types"].get(edge_type, 0) + 1

        return stats

    def save(self, path: str) -> None:
        """Persist graph to disk."""
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump({
                "graph": self.graph,
                "node_index": self.node_index,
                "reference_index": self.reference_index,
                "document_index": self.document_index
            }, f)

        self.logger.info(f"Saved knowledge graph to {path}")

    @classmethod
    def load(cls, path: str) -> 'LegalKnowledgeGraph':
        """Load graph from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        kg = cls()
        kg.graph = data["graph"]
        kg.node_index = data["node_index"]
        kg.reference_index = data["reference_index"]
        kg.document_index = data["document_index"]

        return kg


# ============================================================================
# Graph Builder
# ============================================================================

class GraphBuilder:
    """Build knowledge graph from document structure."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def build_from_documents(
        self,
        documents: List[Any]
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
        document: Any,
        kg: LegalKnowledgeGraph
    ) -> None:
        """Build subgraph for a single document."""
        # Add document node
        doc_node = GraphNode(
            node_id=f"doc_{document.document_id}",
            node_type=NodeType.DOCUMENT,
            label=getattr(document, 'title', document.document_id),
            document_id=document.document_id,
            metadata={
                "document_type": getattr(document, 'document_type', 'unknown'),
                "total_chunks": len(document.chunks) if hasattr(document, 'chunks') else 0
            }
        )
        kg.add_node(doc_node)

        # Add structural elements based on document type
        document_type = getattr(document, 'document_type', 'unknown')
        if document_type == "law_code":
            self._build_law_structure(document, kg, doc_node.node_id)
        elif document_type == "contract":
            self._build_contract_structure(document, kg, doc_node.node_id)
        else:
            # Generic structure - just link chunks to document
            self._build_generic_structure(document, kg, doc_node.node_id)

    def _build_law_structure(
        self,
        document: Any,
        kg: LegalKnowledgeGraph,
        parent_id: str
    ) -> None:
        """Build structure for law document."""
        if not hasattr(document, 'parts'):
            return

        for part in document.parts:
            # Use element_id to match chunker.py chunk_id format
            part_element_id = getattr(part, 'element_id', f"part_{part.number}")

            # Add Part node
            part_node = GraphNode(
                node_id=f"{document.document_id}_part_{part.number}",
                node_type=NodeType.PART,
                label=f"Část {part.number}: {part.title}",
                document_id=document.document_id,
                legal_reference=f"Část {part.number}",
                hierarchy_path=f"Část {part.number}",
                hierarchy_level=1,
                metadata={
                    "chunk_id": f"chunk_{part_element_id}"  # Match chunker.py format
                }
            )
            kg.add_node(part_node)

            # Add PART_OF edge to document
            kg.add_edge(GraphEdge(
                source_id=part_node.node_id,
                target_id=parent_id,
                edge_type=EdgeType.PART_OF
            ))

            # Add chapters
            if hasattr(part, 'chapters'):
                for chapter in part.chapters:
                    self._add_chapter(chapter, part, document, kg, part_node)

    def _add_chapter(self, chapter, part, document, kg, part_node):
        """Add chapter node and its children."""
        # Use element_id to match chunker.py chunk_id format
        chapter_element_id = getattr(chapter, 'element_id', f"chapter_{chapter.number}")

        chapter_node = GraphNode(
            node_id=f"{document.document_id}_chapter_{part.number}_{chapter.number}",
            node_type=NodeType.CHAPTER,
            label=f"Hlava {chapter.number}: {chapter.title}",
            document_id=document.document_id,
            legal_reference=f"Hlava {chapter.number}",
            hierarchy_path=f"{part_node.hierarchy_path} > Hlava {chapter.number}",
            hierarchy_level=2,
            metadata={
                "chunk_id": f"chunk_{chapter_element_id}"  # Match chunker.py format
            }
        )
        kg.add_node(chapter_node)

        kg.add_edge(GraphEdge(
            source_id=chapter_node.node_id,
            target_id=part_node.node_id,
            edge_type=EdgeType.PART_OF
        ))

        # Add paragraphs
        if hasattr(chapter, 'paragraphs'):
            for paragraph in chapter.paragraphs:
                self._add_paragraph(paragraph, chapter, document, kg, chapter_node)

    def _add_paragraph(self, paragraph, chapter, document, kg, chapter_node):
        """Add paragraph node and its children."""
        # Use element_id to match chunker.py chunk_id format: chunk_{element_id}
        element_id = getattr(paragraph, 'element_id', f"para_{paragraph.number}")

        para_node = GraphNode(
            node_id=f"{document.document_id}_para_{paragraph.number}",
            node_type=NodeType.PARAGRAPH,
            label=f"§{paragraph.number}",
            document_id=document.document_id,
            content=getattr(paragraph, 'content', ''),
            legal_reference=f"§{paragraph.number}",
            hierarchy_path=f"{chapter_node.hierarchy_path} > §{paragraph.number}",
            hierarchy_level=3,
            metadata={
                "chunk_id": f"chunk_{element_id}",  # Match chunker.py format
                "contains_obligation": getattr(paragraph, 'contains_obligation', False),
                "contains_prohibition": getattr(paragraph, 'contains_prohibition', False)
            }
        )
        kg.add_node(para_node)

        kg.add_edge(GraphEdge(
            source_id=para_node.node_id,
            target_id=chapter_node.node_id,
            edge_type=EdgeType.PART_OF
        ))

        # Add subsections
        if hasattr(paragraph, 'subsections'):
            for subsec in paragraph.subsections:
                # Use element_id to match chunker.py chunk_id format
                subsec_element_id = getattr(subsec, 'element_id', f"subsec_{subsec.number}")

                subsec_node = GraphNode(
                    node_id=f"{para_node.node_id}_subsec_{subsec.number}",
                    node_type=NodeType.SUBSECTION,
                    label=f"§{paragraph.number} odst. {subsec.number}",
                    document_id=document.document_id,
                    content=getattr(subsec, 'content', ''),
                    legal_reference=f"§{paragraph.number} odst. {subsec.number}",
                    hierarchy_path=f"{para_node.hierarchy_path} > odst. {subsec.number}",
                    hierarchy_level=4,
                    metadata={
                        "chunk_id": f"chunk_{subsec_element_id}"  # Match chunker.py format
                    }
                )
                kg.add_node(subsec_node)

                kg.add_edge(GraphEdge(
                    source_id=subsec_node.node_id,
                    target_id=para_node.node_id,
                    edge_type=EdgeType.PART_OF
                ))

    def _build_contract_structure(
        self,
        document: Any,
        kg: LegalKnowledgeGraph,
        parent_id: str
    ) -> None:
        """Build structure for contract document."""
        if not hasattr(document, 'articles'):
            return

        for article in document.articles:
            # Use element_id to match chunker.py chunk_id format
            article_element_id = getattr(article, 'element_id', f"article_{article.number}")

            article_node = GraphNode(
                node_id=f"{document.document_id}_article_{article.number}",
                node_type=NodeType.ARTICLE,
                label=f"Článek {article.number}: {article.title}",
                document_id=document.document_id,
                content=getattr(article, 'content', ''),
                legal_reference=f"Článek {article.number}",
                hierarchy_path=f"Článek {article.number}",
                hierarchy_level=1,
                metadata={
                    "chunk_id": f"chunk_{article_element_id}"  # Match chunker.py format
                }
            )
            kg.add_node(article_node)

            kg.add_edge(GraphEdge(
                source_id=article_node.node_id,
                target_id=parent_id,
                edge_type=EdgeType.PART_OF
            ))

            # Add points
            if hasattr(article, 'points'):
                for point in article.points:
                    # Use element_id to match chunker.py chunk_id format
                    point_element_id = getattr(point, 'element_id', f"point_{point.number}")

                    point_node = GraphNode(
                        node_id=f"{article_node.node_id}_point_{point.number}",
                        node_type=NodeType.POINT,
                        label=f"Článek {article.number}.{point.number}",
                        document_id=document.document_id,
                        content=getattr(point, 'content', ''),
                        legal_reference=f"Článek {article.number}.{point.number}",
                        hierarchy_path=f"{article_node.hierarchy_path} > {point.number}",
                        hierarchy_level=2,
                        metadata={
                            "chunk_id": f"chunk_{point_element_id}"  # Match chunker.py format
                        }
                    )
                    kg.add_node(point_node)

                    kg.add_edge(GraphEdge(
                        source_id=point_node.node_id,
                        target_id=article_node.node_id,
                        edge_type=EdgeType.PART_OF
                    ))

    def _build_generic_structure(
        self,
        document: Any,
        kg: LegalKnowledgeGraph,
        parent_id: str
    ) -> None:
        """Build generic structure for documents without specific structure."""
        if hasattr(document, 'chunks'):
            for i, chunk in enumerate(document.chunks):
                chunk_node = GraphNode(
                    node_id=f"{document.document_id}_chunk_{i}",
                    node_type=NodeType.CHUNK,
                    label=f"Chunk {i}",
                    document_id=document.document_id,
                    content=getattr(chunk, 'content', ''),
                    hierarchy_path=f"Chunk {i}",
                    hierarchy_level=1,
                    metadata={
                        "chunk_id": getattr(chunk, 'chunk_id', f"chunk_{i}"),
                        "page": getattr(chunk, 'page', None)
                    }
                )
                kg.add_node(chunk_node)

                kg.add_edge(GraphEdge(
                    source_id=chunk_node.node_id,
                    target_id=parent_id,
                    edge_type=EdgeType.PART_OF
                ))


# ============================================================================
# Reference Linker
# ============================================================================

class ReferenceLinker:
    """Link nodes based on explicit legal references."""

    def __init__(self, reference_extractor=None):
        self.reference_extractor = reference_extractor
        self.logger = logging.getLogger(__name__)

    def link_references(self, kg: LegalKnowledgeGraph) -> None:
        """
        Find and link references between nodes.

        Modifies kg in-place.
        """
        if self.reference_extractor is None:
            self.logger.warning("No reference extractor provided, skipping reference linking")
            return

        reference_count = 0

        # Iterate over all nodes with content
        for node_id, node in kg.node_index.items():
            if not node.content:
                continue

            # Extract references from content
            references = self.reference_extractor.extract(node.content)

            for ref_entity in references:
                # Resolve reference to target node(s)
                normalized = getattr(ref_entity, 'normalized', str(ref_entity))
                target_nodes = kg.get_nodes_by_reference(normalized)

                for target_id in target_nodes:
                    # Don't self-reference
                    if target_id == node_id:
                        continue

                    # Add REFERENCES edge
                    kg.add_edge(GraphEdge(
                        source_id=node_id,
                        target_id=target_id,
                        edge_type=EdgeType.REFERENCES,
                        weight=getattr(ref_entity, 'confidence', 1.0),
                        metadata={
                            "reference_text": getattr(ref_entity, 'value', str(ref_entity)),
                            "normalized_ref": normalized
                        }
                    ))

                    reference_count += 1

        self.logger.info(f"Linked {reference_count} references")


# ============================================================================
# Semantic Linker
# ============================================================================

class SemanticLinker:
    """Link nodes based on semantic similarity."""

    def __init__(self, embedding_model, similarity_threshold: float = 0.75):
        self.embedding_model = embedding_model
        self.threshold = similarity_threshold
        self.logger = logging.getLogger(__name__)

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
            self.logger.info("Not enough provision nodes for semantic linking")
            return

        # Compute embeddings
        contents = [node.content for node in provision_nodes]
        embeddings = await self._compute_embeddings(contents)

        # Compute pairwise similarities
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            similarity_matrix = cosine_similarity(embeddings)
        except ImportError:
            self.logger.warning("sklearn not available, skipping semantic linking")
            return

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
                        weight=float(similarity)
                    ))

                    kg.add_edge(GraphEdge(
                        source_id=node_j.node_id,
                        target_id=node_i.node_id,
                        edge_type=EdgeType.RELATED_TO,
                        weight=float(similarity)
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


# ============================================================================
# Compliance Linker
# ============================================================================

class ComplianceLinker:
    """Add compliance-related edges to graph."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def link_compliance(
        self,
        kg: LegalKnowledgeGraph,
        compliance_report: Any
    ) -> None:
        """
        Add edges based on compliance analysis results.

        Args:
            kg: Knowledge graph
            compliance_report: ComplianceReport from ComplianceAnalyzer
        """
        # Add COMPLIES_WITH edges for compliant mappings
        if hasattr(compliance_report, 'clause_mappings'):
            for mapping in compliance_report.clause_mappings:
                contract_node_id = self._find_node_by_chunk(kg, mapping.contract_chunk_id)

                if hasattr(mapping, 'law_requirements'):
                    for req in mapping.law_requirements:
                        law_node_id = self._find_node_by_chunk(kg, req.law_chunk_id)

                        if contract_node_id and law_node_id:
                            kg.add_edge(GraphEdge(
                                source_id=contract_node_id,
                                target_id=law_node_id,
                                edge_type=EdgeType.COMPLIES_WITH,
                                weight=getattr(mapping, 'match_score', 1.0)
                            ))

        # Add CONFLICTS_WITH edges for conflicts
        if hasattr(compliance_report, 'all_issues'):
            for issue in compliance_report.all_issues:
                if getattr(issue, 'status', None) == 'CONFLICT':
                    contract_node_id = self._find_node_by_chunk(kg, issue.contract_chunk_id)

                    if hasattr(issue, 'law_requirements'):
                        for req in issue.law_requirements:
                            law_node_id = self._find_node_by_chunk(kg, req.law_chunk_id)

                            if contract_node_id and law_node_id:
                                kg.add_edge(GraphEdge(
                                    source_id=contract_node_id,
                                    target_id=law_node_id,
                                    edge_type=EdgeType.CONFLICTS_WITH,
                                    weight=getattr(issue, 'risk_score', 0.5),
                                    metadata={
                                        "severity": getattr(issue, 'severity', 'MEDIUM'),
                                        "description": getattr(issue, 'issue_description', '')
                                    }
                                ))

        self.logger.info("Added compliance edges to graph")

    def _find_node_by_chunk(self, kg: LegalKnowledgeGraph, chunk_id: str) -> Optional[str]:
        """Find graph node corresponding to chunk ID."""
        # Simple heuristic: look for node with matching chunk_id in metadata
        for node_id, node in kg.node_index.items():
            if node.metadata.get("chunk_id") == chunk_id:
                return node_id
        return None


# ============================================================================
# Graph Retriever
# ============================================================================

class GraphRetriever:
    """Retrieve provisions using graph structure."""

    def __init__(self, kg: LegalKnowledgeGraph):
        self.kg = kg
        self.logger = logging.getLogger(__name__)

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

    def find_all_paths(
        self,
        source_id: str,
        target_id: str,
        max_length: int = 5
    ) -> List[List[str]]:
        """
        Find all simple paths between two nodes.

        Returns:
            List of paths, each path is a list of node IDs
        """
        try:
            paths = list(nx.all_simple_paths(
                self.kg.graph,
                source=source_id,
                target=target_id,
                cutoff=max_length
            ))
            return paths
        except nx.NetworkXNoPath:
            return []


# ============================================================================
# Multi-Hop Reasoner
# ============================================================================

class MultiHopReasoner:
    """Perform multi-hop reasoning over knowledge graph."""

    def __init__(self, kg: LegalKnowledgeGraph):
        self.kg = kg
        self.logger = logging.getLogger(__name__)

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
        try:
            path = nx.shortest_path(self.kg.graph, source=source_id, target=target_id)
        except nx.NetworkXNoPath:
            return None

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

    def find_transitive_conflicts(
        self,
        node_id: str,
        max_hops: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Find conflicts that are transitively related to a node.

        Returns:
            List of conflict information dicts
        """
        conflicts = []

        # Find all nodes within max_hops
        visited = set()
        queue = [(node_id, 0)]

        while queue:
            current_id, hops = queue.pop(0)

            if current_id in visited or hops > max_hops:
                continue

            visited.add(current_id)

            # Check for CONFLICTS_WITH edges from this node
            for neighbor_id in self.kg.get_neighbors(current_id, EdgeType.CONFLICTS_WITH):
                edge_data = self.kg.graph[current_id][neighbor_id]

                conflicts.append({
                    "source_node_id": current_id,
                    "target_node_id": neighbor_id,
                    "hop_distance": hops,
                    "severity": edge_data.get("metadata", {}).get("severity", "UNKNOWN"),
                    "description": edge_data.get("metadata", {}).get("description", "")
                })

            # Continue traversal via all edge types
            for neighbor_id in self.kg.graph.successors(current_id):
                if neighbor_id not in visited:
                    queue.append((neighbor_id, hops + 1))

        return conflicts


# ============================================================================
# Graph Analyzer
# ============================================================================

class GraphAnalyzer:
    """Analyze graph structure for insights."""

    def __init__(self, kg: LegalKnowledgeGraph):
        self.kg = kg
        self.logger = logging.getLogger(__name__)

    def compute_centrality(self) -> Dict[str, float]:
        """
        Compute betweenness centrality for all nodes.

        Returns:
            Dict mapping node_id to centrality score
        """
        centrality = nx.betweenness_centrality(self.kg.graph)
        return centrality

    def find_hub_provisions(self, top_k: int = 10) -> List[Tuple[str, float]]:
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
        try:
            import networkx.algorithms.community as nx_comm
            communities = nx_comm.louvain_communities(undirected)
        except (ImportError, AttributeError):
            self.logger.warning("Community detection not available in this NetworkX version")
            return {}

        # Map nodes to community IDs
        node_to_community = {}
        for comm_id, community in enumerate(communities):
            for node_id in community:
                node_to_community[node_id] = comm_id

        return node_to_community

    def compute_pagerank(self) -> Dict[str, float]:
        """
        Compute PageRank scores for all nodes.

        Returns:
            Dict mapping node_id to PageRank score
        """
        pagerank = nx.pagerank(self.kg.graph)
        return pagerank


# ============================================================================
# Graph Visualizer
# ============================================================================

class GraphVisualizer:
    """Export graph for visualization."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

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

        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        self.logger.info(f"Exported graph to JSON: {output_path}")

    def export_to_graphml(self, kg: LegalKnowledgeGraph, output_path: str) -> None:
        """Export to GraphML for Gephi/Cytoscape."""
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        nx.write_graphml(kg.graph, output_path)
        self.logger.info(f"Exported graph to GraphML: {output_path}")

    def generate_summary_report(self, kg: LegalKnowledgeGraph) -> str:
        """Generate a text summary report of the graph."""
        stats = kg.get_statistics()

        report = []
        report.append("=" * 60)
        report.append("Legal Knowledge Graph Summary Report")
        report.append("=" * 60)
        report.append("")
        report.append(f"Total Nodes: {stats['total_nodes']}")
        report.append(f"Total Edges: {stats['total_edges']}")
        report.append(f"Documents: {stats['documents']}")
        report.append("")
        report.append("Node Types:")
        for node_type, count in sorted(stats['node_types'].items()):
            report.append(f"  {node_type}: {count}")
        report.append("")
        report.append("Edge Types:")
        for edge_type, count in sorted(stats['edge_types'].items()):
            report.append(f"  {edge_type}: {count}")
        report.append("")
        report.append("=" * 60)

        return "\n".join(report)
