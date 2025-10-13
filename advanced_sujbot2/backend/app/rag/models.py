"""
Data models for legal document structures.

This module defines the data structures used throughout the LegalDocumentReader system,
including documents, structural elements, references, and metadata.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from enum import Enum


class DocumentType(Enum):
    """Types of legal documents"""
    CONTRACT = "contract"
    LAW_CODE = "law_code"
    REGULATION = "regulation"
    UNKNOWN = "unknown"


class ElementType(Enum):
    """Types of structural elements"""
    PART = "part"
    CHAPTER = "chapter"
    SECTION = "section"
    PARAGRAPH = "paragraph"
    SUBSECTION = "subsection"
    LETTER = "letter"
    ARTICLE = "article"
    POINT = "point"


@dataclass
class DocumentMetadata:
    """Document-level metadata"""

    # File info
    file_path: str
    file_format: str  # 'pdf' | 'docx' | 'xml' | 'txt'
    file_size_bytes: int

    # Document info
    title: str
    document_number: Optional[str] = None  # e.g., "č. 89/2012 Sb."
    effective_date: Optional[datetime] = None

    # For contracts
    parties: Optional[List[str]] = None
    contract_type: Optional[str] = None  # e.g., "Smlouva o dílo"

    # For laws
    law_type: Optional[str] = None  # e.g., "Zákon"
    issuing_authority: Optional[str] = None

    # Statistics
    total_pages: int = 0
    total_words: int = 0
    total_sections: int = 0


@dataclass
class StructuralElement:
    """Base class for structural elements"""

    element_id: str
    element_type: str  # 'part' | 'chapter' | 'paragraph' | 'article' | 'subsection'

    # Content
    title: Optional[str]
    content: str

    # Hierarchy
    level: int  # 0 = top-level
    parent_id: Optional[str]
    children_ids: List[str] = field(default_factory=list)

    # Position
    start_char: int = 0
    end_char: int = 0
    start_line: int = 0
    end_line: int = 0

    # Legal reference
    legal_reference: str = ""  # e.g., "§89", "Článek 5"

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Part(StructuralElement):
    """Part (Část) - top-level division in laws"""

    number: str = ""  # Roman numeral: "I", "II", etc.
    chapters: List['Chapter'] = field(default_factory=list)


@dataclass
class Chapter(StructuralElement):
    """Chapter (Hlava) - second-level division"""

    number: str = ""  # Roman numeral: "I", "II", etc.
    part: Optional[Part] = None
    sections: List['Section'] = field(default_factory=list)


@dataclass
class Section(StructuralElement):
    """Section (Díl) - third-level division in laws between Chapter and Paragraph"""

    number: str = ""  # Roman numeral: "I", "II", etc.
    chapter: Optional[Chapter] = None
    paragraphs: List['Paragraph'] = field(default_factory=list)


@dataclass
class Paragraph(StructuralElement):
    """Paragraph (§) - basic unit in laws"""

    number: int = 0  # Arabic numeral: 1, 2, 89, etc.
    chapter: Optional[Chapter] = None
    section: Optional[Section] = None  # Parent Section (Díl) if exists
    subsections: List['Subsection'] = field(default_factory=list)

    # Content classification
    contains_obligation: bool = False
    contains_prohibition: bool = False
    contains_definition: bool = False


@dataclass
class Subsection(StructuralElement):
    """Subsection (odstavec) - subdivision of paragraph"""

    number: int = 0  # (1), (2), etc.
    paragraph: Optional[Paragraph] = None
    letters: List['Letter'] = field(default_factory=list)


@dataclass
class Letter(StructuralElement):
    """Letter (písmeno) - subdivision of subsection"""

    letter: str = ""  # 'a', 'b', 'c', etc.
    subsection: Optional[Subsection] = None


@dataclass
class Article(StructuralElement):
    """Article (Článek) - basic unit in contracts"""

    number: int = 0
    points: List['Point'] = field(default_factory=list)


@dataclass
class Point(StructuralElement):
    """Point (Bod) - subdivision of article"""

    number: str = ""  # "5.1", "5.2", etc.
    article: Optional[Article] = None


@dataclass
class LegalReference:
    """Reference to another legal provision"""

    reference_id: str

    # Source (where the reference appears)
    source_element_id: str
    source_position: Tuple[int, int]  # (start_char, end_char)

    # Target (what is being referenced)
    target_type: str  # 'paragraph' | 'article' | 'law' | 'regulation'
    target_reference: str  # e.g., "§89", "§89 odst. 2"
    target_document: Optional[str] = None  # e.g., "Zákon č. 89/2012 Sb."

    # Context
    context: str = ""  # Surrounding text
    reference_type: str = "direct"  # 'direct' | 'implicit' | 'comparison'

    # Parsed reference components
    components: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentStructure:
    """Hierarchical structure representation"""

    hierarchy: List[StructuralElement] = field(default_factory=list)

    # Quick access indices
    parts: List[Part] = field(default_factory=list)
    chapters: List[Chapter] = field(default_factory=list)
    sections: List[Section] = field(default_factory=list)  # Paragraphs or Articles

    # Flat list of all elements
    all_elements: List[StructuralElement] = field(default_factory=list)

    def get_element_by_reference(self, ref: str) -> Optional[StructuralElement]:
        """Get element by legal reference, e.g., '§89'"""
        for element in self.all_elements:
            if element.legal_reference == ref:
                return element
        return None

    def get_children(self, element: StructuralElement) -> List[StructuralElement]:
        """Get child elements"""
        children = []
        for elem in self.all_elements:
            if elem.parent_id == element.element_id:
                children.append(elem)
        return children

    def get_path(self, element: StructuralElement) -> str:
        """Get full hierarchy path, e.g., 'Část II > Hlava III > §89'"""
        path_parts = [element.legal_reference]
        current = element

        while current.parent_id:
            parent = next((e for e in self.all_elements if e.element_id == current.parent_id), None)
            if parent:
                path_parts.insert(0, parent.legal_reference)
                current = parent
            else:
                break

        return " > ".join(path_parts)


@dataclass
class LegalDocument:
    """Main document container"""

    # Identity
    document_id: str
    document_type: str  # 'contract' | 'law_code' | 'regulation'

    # Content
    raw_text: str
    cleaned_text: str

    # Structure
    structure: DocumentStructure

    # Metadata
    metadata: DocumentMetadata

    # References
    references: List[LegalReference] = field(default_factory=list)

    # Parsed at
    parsed_at: datetime = field(default_factory=datetime.now)


# API Models for Compliance Checking

class ComplianceMode(str, Enum):
    """Compliance check mode."""
    EXHAUSTIVE = "exhaustive"  # Check all provisions thoroughly
    SAMPLE = "sample"  # Sample-based quick check


class IndexingStage(str, Enum):
    """Stages of document indexing."""
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    COMPLETE = "complete"


class AnalysisStage(str, Enum):
    """Stages of compliance analysis."""
    MAPPING = "mapping"
    CHECKING = "checking"
    SCORING = "scoring"
    REPORTING = "reporting"
    GRAPH_BUILDING = "graph_building"
    ANALYSIS = "analysis"
    COMPLETE = "complete"


class SeverityLevel(str, Enum):
    """Issue severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class RetrievalStrategy(str, Enum):
    """Query retrieval strategy."""
    HYBRID = "hybrid"
    CROSS_DOCUMENT = "cross_document"
    SEMANTIC = "semantic"
    KEYWORD = "keyword"


@dataclass
class ComplianceCheckRequest:
    """Request for compliance check."""
    contract_path: str
    law_paths: List[str]
    mode: ComplianceMode = ComplianceMode.EXHAUSTIVE
    generate_graph: bool = True
    enable_query_interface: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate request after initialization."""
        if not self.contract_path:
            raise ValueError("contract_path is required")
        if not self.law_paths:
            raise ValueError("At least one law_path is required")
        if isinstance(self.mode, str):
            self.mode = ComplianceMode(self.mode)


@dataclass
class IndexingProgress:
    """Progress update during document indexing."""
    document_id: str
    document_name: str
    stage: IndexingStage
    progress: float  # 0.0 to 1.0
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate progress after initialization."""
        if isinstance(self.stage, str):
            self.stage = IndexingStage(self.stage)
        if not (0.0 <= self.progress <= 1.0):
            raise ValueError(f"progress must be in [0.0, 1.0], got {self.progress}")


@dataclass
class AnalysisProgress:
    """Progress update during compliance analysis."""
    stage: AnalysisStage
    progress: float  # 0.0 to 1.0
    message: str
    issues_found: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate progress after initialization."""
        if isinstance(self.stage, str):
            self.stage = AnalysisStage(self.stage)
        if not (0.0 <= self.progress <= 1.0):
            raise ValueError(f"progress must be in [0.0, 1.0], got {self.progress}")


@dataclass
class Source:
    """Source citation for an answer or finding."""
    legal_reference: str
    content: str
    document_id: str
    document_type: Optional[str] = None
    confidence: float = 0.0
    page_number: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessedQuery:
    """Processed query with metadata."""
    original_query: str
    normalized_query: str
    retrieval_strategy: RetrievalStrategy
    sub_queries: List[str] = field(default_factory=list)
    query_type: Optional[str] = None
    expanded_terms: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResponse:
    """Response from query API."""
    query: str
    answer: str
    sources: List[Source]
    processed_query: ProcessedQuery
    confidence: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplianceIssue:
    """Single compliance issue found during analysis."""
    issue_id: str
    severity: SeverityLevel
    title: str
    description: str
    law_provision: Source
    contract_provision: Optional[Source] = None
    gap_analysis: str = ""
    recommendations: List[str] = field(default_factory=list)
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplianceReport:
    """Complete compliance check report."""
    report_id: str
    contract_id: str
    law_ids: List[str]
    mode: ComplianceMode
    overall_compliance_score: float
    total_issues: int
    issues: List[ComplianceIssue]
    summary: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def critical_issues(self) -> List[ComplianceIssue]:
        """Get critical severity issues."""
        return [i for i in self.issues if i.severity == SeverityLevel.CRITICAL]

    @property
    def high_issues(self) -> List[ComplianceIssue]:
        """Get high severity issues."""
        return [i for i in self.issues if i.severity == SeverityLevel.HIGH]

    @property
    def medium_issues(self) -> List[ComplianceIssue]:
        """Get medium severity issues."""
        return [i for i in self.issues if i.severity == SeverityLevel.MEDIUM]

    @property
    def low_issues(self) -> List[ComplianceIssue]:
        """Get low severity issues."""
        return [i for i in self.issues if i.severity == SeverityLevel.LOW]

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "report_id": self.report_id,
            "contract_id": self.contract_id,
            "law_ids": self.law_ids,
            "mode": self.mode.value,
            "overall_compliance_score": self.overall_compliance_score,
            "total_issues": self.total_issues,
            "issues": [
                {
                    "issue_id": issue.issue_id,
                    "severity": issue.severity.value,
                    "title": issue.title,
                    "description": issue.description,
                    "recommendations": issue.recommendations,
                    "confidence": issue.confidence,
                }
                for issue in self.issues
            ],
            "summary": self.summary,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class BatchResult:
    """Result of batch processing."""
    batch_id: str
    total_requests: int
    successful: int
    failed: int
    reports: List[ComplianceReport]
    errors: List[Dict[str, str]]
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> Optional[float]:
        """Get batch processing duration in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    @property
    def success_rate(self) -> float:
        """Get success rate as percentage."""
        if self.total_requests == 0:
            return 0.0
        return self.successful / self.total_requests
