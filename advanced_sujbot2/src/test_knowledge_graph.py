"""
Test and Example Usage for Legal Knowledge Graph

This module demonstrates how to use the knowledge graph system
for legal document analysis.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import List

from knowledge_graph import (
    LegalKnowledgeGraph,
    GraphBuilder,
    ReferenceLinker,
    SemanticLinker,
    ComplianceLinker,
    GraphRetriever,
    MultiHopReasoner,
    GraphAnalyzer,
    GraphVisualizer,
    NodeType,
    EdgeType
)


# ============================================================================
# Mock Document Classes for Testing
# ============================================================================

@dataclass
class MockSubsection:
    """Mock subsection for testing."""
    number: int
    content: str


@dataclass
class MockParagraph:
    """Mock paragraph for testing."""
    number: int
    content: str
    contains_obligation: bool = False
    contains_prohibition: bool = False
    subsections: List[MockSubsection] = None

    def __post_init__(self):
        if self.subsections is None:
            self.subsections = []


@dataclass
class MockChapter:
    """Mock chapter for testing."""
    number: int
    title: str
    paragraphs: List[MockParagraph] = None

    def __post_init__(self):
        if self.paragraphs is None:
            self.paragraphs = []


@dataclass
class MockPart:
    """Mock part for testing."""
    number: int
    title: str
    chapters: List[MockChapter] = None

    def __post_init__(self):
        if self.chapters is None:
            self.chapters = []


@dataclass
class MockLawDocument:
    """Mock law document for testing."""
    document_id: str
    title: str
    document_type: str = "law_code"
    parts: List[MockPart] = None

    def __post_init__(self):
        if self.parts is None:
            self.parts = []


@dataclass
class MockPoint:
    """Mock point for testing."""
    number: int
    content: str


@dataclass
class MockArticle:
    """Mock article for testing."""
    number: int
    title: str
    content: str
    points: List[MockPoint] = None

    def __post_init__(self):
        if self.points is None:
            self.points = []


@dataclass
class MockContractDocument:
    """Mock contract document for testing."""
    document_id: str
    title: str
    document_type: str = "contract"
    articles: List[MockArticle] = None

    def __post_init__(self):
        if self.articles is None:
            self.articles = []


# ============================================================================
# Test Functions
# ============================================================================

def create_sample_law_document() -> MockLawDocument:
    """Create a sample law document for testing."""
    # Create subsections
    subsec1 = MockSubsection(
        number=1,
        content="Každý má právo na ochranu osobních údajů."
    )
    subsec2 = MockSubsection(
        number=2,
        content="Zpracování osobních údajů musí být zákonné."
    )

    # Create paragraphs
    para89 = MockParagraph(
        number=89,
        content="Ochrana osobních údajů je základním právem.",
        contains_obligation=True,
        subsections=[subsec1, subsec2]
    )

    para90 = MockParagraph(
        number=90,
        content="Správce dat je povinen zajistit bezpečnost údajů. "
                "Tato povinnost se řídí podle §89 odst. 1.",
        contains_obligation=True,
        subsections=[]
    )

    # Create chapters
    chapter1 = MockChapter(
        number=1,
        title="Základní ustanovení",
        paragraphs=[para89, para90]
    )

    # Create parts
    part1 = MockPart(
        number=1,
        title="Obecná ustanovení",
        chapters=[chapter1]
    )

    # Create document
    law = MockLawDocument(
        document_id="law_gdpr_cz",
        title="Zákon o ochraně osobních údajů",
        parts=[part1]
    )

    return law


def create_sample_contract_document() -> MockContractDocument:
    """Create a sample contract document for testing."""
    # Create points
    point1 = MockPoint(
        number=1,
        content="Dodavatel je povinen dodržovat zákon o ochraně osobních údajů."
    )
    point2 = MockPoint(
        number=2,
        content="Zpracování osobních údajů musí být v souladu s §89 a §90."
    )

    # Create articles
    article5 = MockArticle(
        number=5,
        title="Ochrana osobních údajů",
        content="Smluvní strany se zavazují dodržovat příslušné právní předpisy.",
        points=[point1, point2]
    )

    article6 = MockArticle(
        number=6,
        title="Zabezpečení dat",
        content="Dodavatel musí implementovat technická a organizační opatření "
                "podle Článku 5.",
        points=[]
    )

    # Create contract
    contract = MockContractDocument(
        document_id="contract_supplier_001",
        title="Smlouva o zpracování osobních údajů",
        articles=[article5, article6]
    )

    return contract


def test_graph_building():
    """Test building knowledge graph from documents."""
    print("\n" + "=" * 60)
    print("TEST: Graph Building")
    print("=" * 60)

    # Create sample documents
    law = create_sample_law_document()
    contract = create_sample_contract_document()

    # Build graph
    builder = GraphBuilder()
    kg = builder.build_from_documents([law, contract])

    # Print statistics
    stats = kg.get_statistics()
    print(f"\nGraph Statistics:")
    print(f"  Total Nodes: {stats['total_nodes']}")
    print(f"  Total Edges: {stats['total_edges']}")
    print(f"  Documents: {stats['documents']}")
    print(f"\nNode Types:")
    for node_type, count in stats['node_types'].items():
        print(f"    {node_type}: {count}")
    print(f"\nEdge Types:")
    for edge_type, count in stats['edge_types'].items():
        print(f"    {edge_type}: {count}")

    # Test node retrieval
    print(f"\n\nNode Retrieval:")
    para_nodes = kg.get_nodes_by_reference("§89")
    print(f"  Nodes with reference '§89': {len(para_nodes)}")
    for node_id in para_nodes:
        node = kg.get_node(node_id)
        print(f"    - {node.label}: {node.hierarchy_path}")

    # Test document-based retrieval
    law_nodes = kg.get_nodes_by_document(law.document_id)
    print(f"\n  Nodes in law document: {len(law_nodes)}")

    contract_nodes = kg.get_nodes_by_document(contract.document_id)
    print(f"  Nodes in contract document: {len(contract_nodes)}")

    return kg


def test_reference_linking(kg: LegalKnowledgeGraph):
    """Test reference linking (simplified without actual extractor)."""
    print("\n" + "=" * 60)
    print("TEST: Reference Linking")
    print("=" * 60)

    # Manually add some reference edges for testing
    # In real implementation, this would use ReferenceLinker with LegalReferenceExtractor

    # Find contract article 5 and law paragraph 89
    contract_article = None
    law_para = None

    for node_id, node in kg.node_index.items():
        if node.legal_reference == "Článek 5":
            contract_article = node_id
        if node.legal_reference == "§89":
            law_para = node_id

    if contract_article and law_para:
        from knowledge_graph import GraphEdge
        kg.add_edge(GraphEdge(
            source_id=contract_article,
            target_id=law_para,
            edge_type=EdgeType.REFERENCES,
            weight=0.9,
            metadata={
                "reference_text": "§89",
                "normalized_ref": "§89"
            }
        ))
        print(f"\nAdded reference edge: {contract_article} → {law_para}")

    # Test retrieval
    retriever = GraphRetriever(kg)
    if contract_article:
        refs = retriever.get_referenced_provisions(contract_article)
        print(f"\nProvisions referenced by Článek 5: {len(refs)}")
        for ref_id in refs:
            ref_node = kg.get_node(ref_id)
            print(f"  - {ref_node.label}")


def test_proximity_search(kg: LegalKnowledgeGraph):
    """Test proximity-based search."""
    print("\n" + "=" * 60)
    print("TEST: Proximity Search")
    print("=" * 60)

    retriever = GraphRetriever(kg)

    # Find all nodes for §89
    para_nodes = kg.get_nodes_by_reference("§89")

    if para_nodes:
        print(f"\nSearching for provisions within 2 hops of §89...")

        nearby = retriever.get_provisions_by_proximity(
            anchor_node_ids=para_nodes,
            max_hops=2,
            edge_types=[EdgeType.PART_OF, EdgeType.REFERENCES]
        )

        print(f"Found {len(nearby)} nodes within proximity:")
        for node_id in nearby[:10]:  # Limit output
            node = kg.get_node(node_id)
            print(f"  - {node.label} ({node.node_type.value})")


def test_path_finding(kg: LegalKnowledgeGraph):
    """Test path finding between nodes."""
    print("\n" + "=" * 60)
    print("TEST: Path Finding")
    print("=" * 60)

    retriever = GraphRetriever(kg)
    reasoner = MultiHopReasoner(kg)

    # Find contract article and law paragraph
    contract_article = None
    law_para = None

    for node_id, node in kg.node_index.items():
        if node.legal_reference == "Článek 5":
            contract_article = node_id
        if node.legal_reference == "§89":
            law_para = node_id

    if contract_article and law_para:
        print(f"\nFinding path from Článek 5 to §89...")

        path = retriever.find_path(contract_article, law_para, max_length=5)

        if path:
            print(f"Found path with {len(path)} nodes:")
            for i, node_id in enumerate(path):
                node = kg.get_node(node_id)
                print(f"  {i+1}. {node.label} ({node.node_type.value})")

            # Get explanation
            explanation = reasoner.explain_relationship(contract_article, law_para)
            if explanation:
                print(f"\nRelationship explanation:")
                print(f"  {explanation}")
        else:
            print("No path found")


def test_multi_hop_reasoning(kg: LegalKnowledgeGraph):
    """Test multi-hop reasoning."""
    print("\n" + "=" * 60)
    print("TEST: Multi-Hop Reasoning")
    print("=" * 60)

    reasoner = MultiHopReasoner(kg)

    # Find contract article
    contract_article = None
    for node_id, node in kg.node_index.items():
        if node.legal_reference == "Článek 5":
            contract_article = node_id
            break

    if contract_article:
        print(f"\nFinding indirect requirements for Článek 5...")

        indirect = reasoner.find_indirect_requirements(
            contract_node_id=contract_article,
            max_hops=3
        )

        if indirect:
            print(f"Found {len(indirect)} indirect requirements:")
            for result in indirect:
                req_node = kg.get_node(result['requirement_node_id'])
                print(f"\n  Requirement: {req_node.label}")
                print(f"  Hop count: {result['hop_count']}")
                print(f"  Path: {' → '.join([kg.get_node(nid).label for nid in result['path']])}")
        else:
            print("No indirect requirements found")


def test_graph_analysis(kg: LegalKnowledgeGraph):
    """Test graph analysis."""
    print("\n" + "=" * 60)
    print("TEST: Graph Analysis")
    print("=" * 60)

    analyzer = GraphAnalyzer(kg)

    # Compute centrality
    print("\nComputing node centrality...")
    centrality = analyzer.compute_centrality()

    print("\nTop 5 nodes by betweenness centrality:")
    sorted_cent = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    for node_id, score in sorted_cent[:5]:
        node = kg.get_node(node_id)
        print(f"  {node.label}: {score:.4f}")

    # Find hub provisions
    print("\nFinding hub provisions...")
    hubs = analyzer.find_hub_provisions(top_k=5)

    print("\nTop 5 hub provisions:")
    for node_id, importance in hubs:
        node = kg.get_node(node_id)
        print(f"  {node.label}: {importance:.4f}")

    # Compute PageRank
    print("\nComputing PageRank...")
    pagerank = analyzer.compute_pagerank()

    print("\nTop 5 nodes by PageRank:")
    sorted_pr = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
    for node_id, score in sorted_pr[:5]:
        node = kg.get_node(node_id)
        print(f"  {node.label}: {score:.4f}")


def test_graph_export(kg: LegalKnowledgeGraph):
    """Test graph export and visualization."""
    print("\n" + "=" * 60)
    print("TEST: Graph Export")
    print("=" * 60)

    visualizer = GraphVisualizer()

    # Generate summary report
    print("\n" + visualizer.generate_summary_report(kg))

    # Export to JSON (commented out to avoid file creation in test)
    # print("\nExporting to JSON...")
    # visualizer.export_to_json(kg, "/tmp/legal_graph.json")
    # print("  Exported to: /tmp/legal_graph.json")

    # Export to GraphML (commented out to avoid file creation in test)
    # print("\nExporting to GraphML...")
    # visualizer.export_to_graphml(kg, "/tmp/legal_graph.graphml")
    # print("  Exported to: /tmp/legal_graph.graphml")


def test_persistence(kg: LegalKnowledgeGraph):
    """Test saving and loading graph."""
    print("\n" + "=" * 60)
    print("TEST: Persistence")
    print("=" * 60)

    # Save graph
    print("\nSaving graph to /tmp/test_knowledge_graph.pkl...")
    kg.save("/tmp/test_knowledge_graph.pkl")

    # Load graph
    print("Loading graph from /tmp/test_knowledge_graph.pkl...")
    kg_loaded = LegalKnowledgeGraph.load("/tmp/test_knowledge_graph.pkl")

    # Verify
    original_stats = kg.get_statistics()
    loaded_stats = kg_loaded.get_statistics()

    print("\nVerification:")
    print(f"  Original nodes: {original_stats['total_nodes']}")
    print(f"  Loaded nodes: {loaded_stats['total_nodes']}")
    print(f"  Original edges: {original_stats['total_edges']}")
    print(f"  Loaded edges: {loaded_stats['total_edges']}")

    if original_stats == loaded_stats:
        print("\n  ✓ Persistence test PASSED")
    else:
        print("\n  ✗ Persistence test FAILED")


# ============================================================================
# Main Test Runner
# ============================================================================

def main():
    """Run all tests."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("\n" + "=" * 60)
    print("Legal Knowledge Graph - Test Suite")
    print("=" * 60)

    # Run tests
    kg = test_graph_building()
    test_reference_linking(kg)
    test_proximity_search(kg)
    test_path_finding(kg)
    test_multi_hop_reasoning(kg)
    test_graph_analysis(kg)
    test_graph_export(kg)
    test_persistence(kg)

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
