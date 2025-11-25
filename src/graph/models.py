"""
Data models for Knowledge Graph entities and relationships.

Schema v2.1 (Extended for Compliance Checking + Definition Alignment):
- 32 Entity Types: Core (8) + Regulatory (6) + Authorization (2) + Nuclear (9) + Events (4) +
  Liability (1) + Legal Terminology (2)
- 41 Relationship Types: Compliance (5) + Regulatory (5) + Structure (4) + Citations (4) +
  Authorization (5) + Nuclear (8) + Temporal (4) + Content (2) + Legal Terminology (1) + Provenance (3)

Key Hierarchies:
- Regulatory: TREATY → REGULATION → DECREE → LEGAL_PROVISION → REQUIREMENT
- Compliance: REQUIREMENT ← COMPLIES_WITH/CONTRADICTS → CLAUSE
- Nuclear: FACILITY → REACTOR → SYSTEM → SAFETY_FUNCTION
- Authorization: AUTHORITY → GRANTED_BY → PERMIT → LICENSE_CONDITION
- Legal Terminology: LEGAL_TERM ← DEFINITION_OF ← DEFINITION (ontology mapping)

Complete graph structure with entities, relationships, compliance verification, and definition alignment support.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime
import json
import logging
import uuid

logger = logging.getLogger(__name__)


class EntityType(Enum):
    """Types of entities extracted from legal documents."""

    # ==================== CORE ENTITIES (existing) ====================
    STANDARD = "standard"
    ORGANIZATION = "organization"  # Includes authorities (use metadata['org_type'])
    DATE = "date"
    CLAUSE = "clause"  # Clause in contracts/documents being analyzed
    TOPIC = "topic"
    PERSON = "person"
    LOCATION = "location"
    CONTRACT = "contract"

    # ==================== REGULATORY HIERARCHY ====================
    # Hierarchical structure: TREATY → REGULATION → DECREE → LEGAL_PROVISION → REQUIREMENT
    REGULATION = "regulation"  # National laws (Atomový zákon, GDPR)
    DECREE = "decree"  # Vyhlášky, implementing regulations (378/2016 Sb.)
    DIRECTIVE = "directive"  # EU directives (Euratom)
    TREATY = "treaty"  # International treaties (ADR, ADN, Vienna Convention)
    LEGAL_PROVISION = "legal_provision"  # Specific § / article / paragraph in law
    REQUIREMENT = "requirement"  # Extracted compliance requirement from LEGAL_PROVISION

    # ==================== AUTHORIZATION & COMPLIANCE ====================
    PERMIT = "permit"  # Formal authorization (e.g., §9 stages: siting, construction, operation)
    LICENSE_CONDITION = "license_condition"  # Specific condition in a permit/license

    # ==================== NUCLEAR TECHNICAL ENTITIES ====================
    REACTOR = "reactor"  # Nuclear reactor unit (VR-1, type, power rating)
    FACILITY = "facility"  # Nuclear facility/site hosting reactors/systems
    SYSTEM = "system"  # Technical systems (I&C, cooling, protection, monitoring)
    SAFETY_FUNCTION = "safety_function"  # Safety functions (shutdown, heat removal, confinement)
    FUEL_TYPE = "fuel_type"  # Nuclear fuel (IRT-4M, enrichment, cladding)
    ISOTOPE = "isotope"  # Radioactive isotopes (U-235, Cs-137, Pu-239)
    RADIATION_SOURCE = "radiation_source"  # Sealed sources, radiation equipment
    WASTE_CATEGORY = "waste_category"  # Radioactive waste categories (low/medium/high level)
    DOSE_LIMIT = "dose_limit"  # Radiation dose limits (mSv/year for workers/public)

    # ==================== EVENTS & PROCESSES ====================
    INCIDENT = "incident"  # Specific safety/security events
    EMERGENCY_CLASSIFICATION = "emergency_classification"  # Event classification levels
    INSPECTION = "inspection"  # Regulatory inspections, audits
    DECOMMISSIONING_PHASE = "decommissioning_phase"  # Phases of decommissioning

    # ==================== LIABILITY & INSURANCE ====================
    LIABILITY_REGIME = "liability_regime"  # Operator liability framework

    # ==================== LEGAL TERMINOLOGY (Definition Alignment) ====================
    LEGAL_TERM = "legal_term"  # Legal terminology requiring alignment (e.g., "Consumer", "Data Controller")
    DEFINITION = "definition"  # Legal definition text from authoritative source


class RelationshipType(Enum):
    """Types of relationships between entities."""

    # ==================== COMPLIANCE CORE (bidirectional checking) ====================
    # Contract → Law direction (compliance verification)
    COMPLIES_WITH = "complies_with"  # Clause → Requirement (with confidence score)
    CONTRADICTS = "contradicts"  # Clause → Requirement (non-compliance)
    PARTIALLY_SATISFIES = "partially_satisfies"  # Clause → Requirement (partial compliance)

    # Law → Contract direction (completeness auditing)
    SPECIFIES_REQUIREMENT = "specifies_requirement"  # Regulation/Provision → Requirement
    REQUIRES_CLAUSE = "requires_clause"  # Requirement → Required clause type

    # ==================== REGULATORY HIERARCHY ====================
    IMPLEMENTS = "implements"  # Decree → Law (national implementation)
    TRANSPOSES = "transposes"  # National law → EU directive (EU law transposition)
    SUPERSEDED_BY = "superseded_by"  # Old regulation → New regulation
    SUPERSEDES = "supersedes"  # New regulation → Old regulation
    AMENDS = "amends"  # Amendment → Original regulation

    # ==================== DOCUMENT STRUCTURE ====================
    CONTAINS_CLAUSE = "contains_clause"  # Contract/Document → Clause
    CONTAINS_PROVISION = "contains_provision"  # Regulation → Legal provision
    CONTAINS = "contains"  # Document → Section (generic hierarchy)
    PART_OF = "part_of"  # Section → Document (reverse hierarchy)

    # ==================== CITATIONS & REFERENCES ====================
    REFERENCES = "references"  # Document A → Document B (generic reference)
    REFERENCED_BY = "referenced_by"  # Document B → Document A (reverse)
    CITES_PROVISION = "cites_provision"  # Document → Legal provision (specific citation)
    BASED_ON = "based_on"  # Analysis/Decision → Evidence/Source

    # ==================== AUTHORIZATION & ENFORCEMENT ====================
    ISSUED_BY = "issued_by"  # Permit/Regulation → Authority/Organization
    GRANTED_BY = "granted_by"  # Permit → Authority (specific authorization)
    ENFORCED_BY = "enforced_by"  # Regulation → Authority (enforcement responsibility)
    SUBJECT_TO_INSPECTION = "subject_to_inspection"  # Facility → Inspection type
    SUPERVISES = "supervises"  # Authority → Facility/Operator (oversight)

    # ==================== NUCLEAR TECHNICAL RELATIONSHIPS ====================
    REGULATED_BY = "regulated_by"  # Facility/Reactor/System → Regulation
    OPERATED_BY = "operated_by"  # Facility/Reactor → Organization
    HAS_SYSTEM = "has_system"  # Facility/Reactor → System
    PERFORMS_FUNCTION = "performs_function"  # System → Safety function
    USES_FUEL = "uses_fuel"  # Reactor → Fuel type
    CONTAINS_ISOTOPE = "contains_isotope"  # Fuel/Source → Isotope
    PRODUCES_WASTE = "produces_waste"  # Reactor/Process → Waste category
    HAS_DOSE_LIMIT = "has_dose_limit"  # Worker category/Area → Dose limit

    # ==================== TEMPORAL RELATIONSHIPS ====================
    EFFECTIVE_DATE = "effective_date"  # Regulation/Contract → Date
    EXPIRY_DATE = "expiry_date"  # Permit/Contract → Date
    SIGNED_ON = "signed_on"  # Contract → Date
    DECOMMISSIONED_ON = "decommissioned_on"  # Facility → Date

    # ==================== CONTENT & TOPICS ====================
    COVERS_TOPIC = "covers_topic"  # Document → Topic
    APPLIES_TO = "applies_to"  # Regulation → Location/Jurisdiction

    # ==================== LEGAL TERMINOLOGY (Definition Alignment) ====================
    DEFINITION_OF = "definition_of"  # Definition → Legal term (links term to its authoritative definition)

    # ==================== PROVENANCE (entity → chunk) ====================
    MENTIONED_IN = "mentioned_in"  # Entity → Chunk (all occurrences)
    DEFINED_IN = "defined_in"  # Entity → Chunk (first occurrence/definition)
    DOCUMENTED_IN = "documented_in"  # Activity/Decision → Report/Document



@dataclass
class Entity:
    """
    Represents an entity extracted from legal documents.

    Entities are the nodes in the knowledge graph.
    Each entity has a unique ID, type, and normalized value.

    Metadata Guidelines for Compliance Use Case
    ============================================

    REQUIREMENT (extracted from legal provisions):
        {
            "mandatory": bool,  # True if legally required
            "risk_level": "high|medium|low",  # Severity if missing
            "jurisdiction": str,  # CZ, EU, international
            "category": str,  # safety, environmental, operational, etc.
            "applicable_to": List[str],  # Entity types this applies to
            "remediation": str,  # How to fix if missing
            "source_provision": str  # § reference in law
        }

    CLAUSE (in analyzed contracts/documents):
        {
            "clause_type": str,  # indemnification, liability, safety, etc.
            "page": int,
            "section": str,
            "risk_level": "high|medium|low",
            "standard": bool,  # True if standard template clause
            "reviewed_by_human": bool
        }

    ORGANIZATION:
        {
            "org_type": "authority|operator|research|commercial",
            "jurisdiction": str,  # CZ, EU, international
            "acronym": str,  # SÚJB, IAEA, etc.
            "regulatory_role": bool  # True if regulatory authority
        }

    DOSE_LIMIT:
        {
            "value": float,  # Numeric value
            "unit": str,  # mSv, Gy, etc.
            "period": str,  # year, hour, single event
            "category": str,  # worker_A, worker_B, public, etc.
            "source_provision": str  # § reference
        }

    ISOTOPE:
        {
            "element": str,  # uranium, plutonium, cesium
            "mass_number": int,  # 235, 239, 137
            "enrichment_percent": float,  # For fuel
            "half_life_years": float
        }

    PERMIT:
        {
            "stage": int,  # §9 stage number (1-5)
            "type": str,  # siting, construction, commissioning, operation, decommissioning
            "paragraph": str,  # § reference
            "requires_documentation": List[str]  # Required supporting documents
        }

    LEGAL_PROVISION:
        {
            "paragraph": str,  # § number
            "article": str,  # Article number (EU directives)
            "parent_regulation": str,  # Parent law/regulation ID
            "text": str  # Full text of provision
        }
    """

    id: str  # Unique identifier (auto-generated)
    type: EntityType  # Entity type
    value: str  # Original text (e.g., "GRI 306: Effluents and Waste 2016")
    normalized_value: str  # Normalized form (e.g., "GRI 306")
    confidence: float  # Extraction confidence (0-1)

    # Provenance: where this entity was found
    source_chunk_ids: List[str] = field(default_factory=list)  # All chunks mentioning this entity
    first_mention_chunk_id: Optional[str] = None  # First chunk where entity appears

    # Context from source document
    document_id: Optional[str] = None
    section_path: Optional[str] = None

    # Type-specific metadata (see docstring for guidelines)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Extraction metadata
    extraction_method: str = "llm"  # "llm", "regex", "spacy"
    extracted_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "type": self.type.value,
            "value": self.value,
            "normalized_value": self.normalized_value,
            "confidence": self.confidence,
            "source_chunk_ids": self.source_chunk_ids,
            "first_mention_chunk_id": self.first_mention_chunk_id,
            "document_id": self.document_id,
            "section_path": self.section_path,
            "metadata": self.metadata,
            "extraction_method": self.extraction_method,
            "extracted_at": self.extracted_at.isoformat() if self.extracted_at else None,
        }

    @classmethod
    def create_legal_term(
        cls,
        term_value: str,
        confidence: float,
        source_chunk_id: str,
        document_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> "Entity":
        """
        Create LEGAL_TERM entity with validation.

        Invariants enforced:
        - term_value must be non-empty
        - confidence in [0.0, 1.0]
        - Must have source provenance

        Args:
            term_value: Legal term text (e.g., "Consumer", "Data Controller")
            confidence: Extraction confidence (0.0-1.0)
            source_chunk_id: Chunk ID where term was found
            document_id: Document ID containing the term
            metadata: Optional metadata dict (related_terms, jurisdiction, category)
            **kwargs: Additional Entity fields

        Returns:
            Entity with type LEGAL_TERM

        Raises:
            ValueError: If invariants violated
        """
        if not term_value or not term_value.strip():
            raise ValueError("Legal term value cannot be empty")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0.0, 1.0], got {confidence}")

        return cls(
            id=str(uuid.uuid4()),
            type=EntityType.LEGAL_TERM,
            value=term_value,
            normalized_value=term_value.lower().strip(),
            confidence=confidence,
            source_chunk_ids=[source_chunk_id],
            first_mention_chunk_id=source_chunk_id,
            document_id=document_id,
            metadata=metadata or {},
            **kwargs
        )

    @classmethod
    def create_definition(
        cls,
        definition_text: str,
        confidence: float,
        source_chunk_id: str,
        source_document_id: str,
        source_provision: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> "Entity":
        """
        Create DEFINITION entity with validation.

        Invariants enforced:
        - definition_text must be non-empty and substantive (min 10 chars)
        - Must reference authoritative source (law/regulation)
        - Source provision recommended for traceability

        Args:
            definition_text: Definition text from law/regulation
            confidence: Extraction confidence (0.0-1.0)
            source_chunk_id: Chunk ID where definition was found
            source_document_id: Document ID (should be law/regulation)
            source_provision: § reference or article number (recommended)
            metadata: Optional metadata dict (parent_regulation, definition_type)
            **kwargs: Additional Entity fields

        Returns:
            Entity with type DEFINITION

        Raises:
            ValueError: If invariants violated
        """
        if not definition_text or len(definition_text.strip()) < 10:
            raise ValueError("Definition text must be non-empty and substantive (min 10 chars)")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0.0, 1.0], got {confidence}")

        metadata_copy = metadata.copy() if metadata else {}
        if source_provision:
            metadata_copy["source_provision"] = source_provision

        return cls(
            id=str(uuid.uuid4()),
            type=EntityType.DEFINITION,
            value=definition_text,
            normalized_value=definition_text[:100].lower().strip(),  # First 100 chars for matching
            confidence=confidence,
            source_chunk_ids=[source_chunk_id],
            first_mention_chunk_id=source_chunk_id,
            document_id=source_document_id,
            metadata=metadata_copy,
            **kwargs
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        """Create Entity from dictionary."""
        data_copy = data.copy()
        data_copy["type"] = EntityType(data_copy["type"])
        if data_copy.get("extracted_at"):
            data_copy["extracted_at"] = datetime.fromisoformat(data_copy["extracted_at"])
        return cls(**data_copy)

    def __hash__(self):
        """Hash based on normalized value and type for deduplication."""
        return hash((self.type, self.normalized_value))

    def __eq__(self, other):
        """Equality based on normalized value and type."""
        if not isinstance(other, Entity):
            return False
        return self.type == other.type and self.normalized_value == other.normalized_value


@dataclass
class Relationship:
    """
    Represents a relationship between two entities.

    Relationships are the edges in the knowledge graph.
    Each relationship has a source entity, target entity, and type.

    Properties Guidelines for Compliance Use Case
    ==============================================

    COMPLIES_WITH (Clause → Requirement):
        {
            "compliance_status": "full|partial|non_compliant",
            "confidence": float,  # LLM confidence in assessment
            "gaps": List[str],  # What's missing if partial
            "evidence_citation": str,  # Quote from clause
            "reasoning": str,  # Legal reasoning chain
            "reviewed_by_human": bool,
            "review_date": str  # ISO date
        }

    CONTRADICTS (Clause → Requirement):
        {
            "severity": "critical|high|medium|low",
            "reason": str,  # Why it contradicts
            "recommended_action": str,
            "legal_risk": str  # Description of legal risk
        }

    SPECIFIES_REQUIREMENT (Regulation/Provision → Requirement):
        {
            "extraction_confidence": float,
            "requirement_text": str,  # Original text from regulation
            "interpretation_notes": str  # Any clarifications
        }

    IMPLEMENTS (Decree → Law):
        {
            "implementing_paragraph": str,  # § in decree
            "implemented_paragraph": str,  # § in parent law
            "effective_date": str  # ISO date
        }

    SUPERVISES (Authority → Facility):
        {
            "supervision_type": "continuous|periodic|event_based",
            "frequency": str,  # "quarterly", "annually", etc.
            "scope": str  # What aspects are supervised
        }

    GRANTED_BY (Permit → Authority):
        {
            "grant_date": str,  # ISO date
            "permit_number": str,
            "validity_period": str,  # Duration or expiry date
            "conditions_count": int  # Number of license conditions
        }
    """

    id: str  # Unique identifier (auto-generated)
    type: RelationshipType  # Relationship type
    source_entity_id: str  # Source entity ID
    target_entity_id: str  # Target entity ID
    confidence: float  # Extraction confidence (0-1)

    # Provenance: where this relationship was extracted
    source_chunk_id: str  # Chunk where relationship was found
    evidence_text: str  # Supporting text snippet

    # Relationship-specific properties (see docstring for guidelines)
    properties: Dict[str, Any] = field(default_factory=dict)

    # Extraction metadata
    extraction_method: str = "llm"  # "llm", "pattern", "heuristic"
    extracted_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "type": self.type.value,
            "source_entity_id": self.source_entity_id,
            "target_entity_id": self.target_entity_id,
            "confidence": self.confidence,
            "source_chunk_id": self.source_chunk_id,
            "evidence_text": self.evidence_text,
            "properties": self.properties,
            "extraction_method": self.extraction_method,
            "extracted_at": self.extracted_at.isoformat() if self.extracted_at else None,
        }

    @classmethod
    def create_definition_of(
        cls,
        definition_entity: "Entity",
        term_entity: "Entity",
        confidence: float,
        source_chunk_id: str,
        evidence_text: str,
        properties: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> "Relationship":
        """
        Create DEFINITION_OF relationship with type validation.

        Invariants enforced:
        - source must be DEFINITION entity
        - target must be LEGAL_TERM entity
        - Prevents type mismatches at construction

        Args:
            definition_entity: DEFINITION entity (source)
            term_entity: LEGAL_TERM entity (target)
            confidence: Relationship confidence (0.0-1.0)
            source_chunk_id: Chunk ID where relationship was found
            evidence_text: Supporting text snippet
            properties: Optional relationship properties
            **kwargs: Additional Relationship fields

        Returns:
            Relationship with type DEFINITION_OF

        Raises:
            ValueError: If entity types are wrong or invariants violated
        """
        if definition_entity.type != EntityType.DEFINITION:
            raise ValueError(
                f"Source must be DEFINITION entity, got {definition_entity.type.value}"
            )
        if term_entity.type != EntityType.LEGAL_TERM:
            raise ValueError(
                f"Target must be LEGAL_TERM entity, got {term_entity.type.value}"
            )
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0.0, 1.0], got {confidence}")

        return cls(
            id=str(uuid.uuid4()),
            type=RelationshipType.DEFINITION_OF,
            source_entity_id=definition_entity.id,
            target_entity_id=term_entity.id,
            confidence=confidence,
            source_chunk_id=source_chunk_id,
            evidence_text=evidence_text,
            properties=properties or {},
            **kwargs
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Relationship":
        """Create Relationship from dictionary."""
        data_copy = data.copy()
        data_copy["type"] = RelationshipType(data_copy["type"])
        if data_copy.get("extracted_at"):
            data_copy["extracted_at"] = datetime.fromisoformat(data_copy["extracted_at"])
        return cls(**data_copy)


@dataclass
class KnowledgeGraph:
    """
    Complete knowledge graph structure.

    Contains all entities, relationships, and metadata.
    Can be serialized to JSON or loaded into Neo4j/SimpleGraphStore.
    """

    # Graph content
    entities: List[Entity] = field(default_factory=list)
    relationships: List[Relationship] = field(default_factory=list)

    # Source metadata
    source_document_id: Optional[str] = None
    source_chunks_file: Optional[str] = None  # Path to phase3_chunks.json

    # Statistics
    stats: Dict[str, Any] = field(default_factory=dict)

    # Graph metadata
    created_at: Optional[datetime] = None
    extraction_config: Dict[str, Any] = field(default_factory=dict)

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        for entity in self.entities:
            if entity.id == entity_id:
                return entity
        return None

    def get_entity_by_value(
        self, normalized_value: str, entity_type: EntityType
    ) -> Optional[Entity]:
        """Get entity by normalized value and type."""
        for entity in self.entities:
            if entity.normalized_value == normalized_value and entity.type == entity_type:
                return entity
        return None

    def get_relationships_for_entity(self, entity_id: str) -> List[Relationship]:
        """Get all relationships involving an entity (as source or target)."""
        return [
            rel
            for rel in self.relationships
            if rel.source_entity_id == entity_id or rel.target_entity_id == entity_id
        ]

    def get_outgoing_relationships(self, entity_id: str) -> List[Relationship]:
        """Get relationships where entity is the source."""
        return [rel for rel in self.relationships if rel.source_entity_id == entity_id]

    def get_incoming_relationships(self, entity_id: str) -> List[Relationship]:
        """Get relationships where entity is the target."""
        return [rel for rel in self.relationships if rel.target_entity_id == entity_id]

    def find_entities(
        self,
        entity_type: Optional[str] = None,
        value_contains: Optional[str] = None,
        min_confidence: float = 0.0,
    ) -> List[Entity]:
        """
        Find entities matching filter criteria.

        Args:
            entity_type: Filter by entity type (e.g., 'organization', 'regulation')
            value_contains: Filter by substring in value (case-insensitive)
            min_confidence: Minimum confidence threshold

        Returns:
            List of matching Entity objects
        """
        results = []
        for entity in self.entities:
            # Filter by type
            # FIX: Convert string entity_type to EntityType enum for comparison (was always failing)
            if entity_type:
                # Handle both string and EntityType enum inputs
                if isinstance(entity_type, str):
                    try:
                        entity_type_enum = EntityType(entity_type)
                    except (ValueError, KeyError):
                        valid_types = [t.value for t in EntityType]
                        logger.error(f"Unknown entity type '{entity_type}'. Valid types: {valid_types}")
                        # Return empty list - no matches for invalid type
                        return []
                else:
                    entity_type_enum = entity_type

                if entity.type != entity_type_enum:
                    continue

            # Filter by value substring
            if value_contains and value_contains.lower() not in entity.value.lower():
                continue

            # Filter by confidence
            if entity.confidence < min_confidence:
                continue

            results.append(entity)

        return results

    def compute_stats(self) -> Dict[str, Any]:
        """Compute statistics about the graph."""
        entity_type_counts = {}
        for entity in self.entities:
            # Convert EntityType enum to string for JSON serialization
            type_key = entity.type.value if hasattr(entity.type, 'value') else str(entity.type)
            entity_type_counts[type_key] = entity_type_counts.get(type_key, 0) + 1

        relationship_type_counts = {}
        for rel in self.relationships:
            # rel.type should be a string
            rel_key = rel.type.value if hasattr(rel.type, 'value') else str(rel.type)
            relationship_type_counts[rel_key] = (
                relationship_type_counts.get(rel_key, 0) + 1
            )

        # Average confidence
        avg_entity_confidence = (
            sum(e.confidence for e in self.entities) / len(self.entities) if self.entities else 0
        )
        avg_rel_confidence = (
            sum(r.confidence for r in self.relationships) / len(self.relationships)
            if self.relationships
            else 0
        )

        self.stats = {
            "total_entities": len(self.entities),
            "total_relationships": len(self.relationships),
            "entity_type_counts": entity_type_counts,
            "relationship_type_counts": relationship_type_counts,
            "avg_entity_confidence": avg_entity_confidence,
            "avg_relationship_confidence": avg_rel_confidence,
        }

        return self.stats

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "entities": [e.to_dict() for e in self.entities],
            "relationships": [r.to_dict() for r in self.relationships],
            "source_document_id": self.source_document_id,
            "source_chunks_file": self.source_chunks_file,
            "stats": self.stats,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "extraction_config": self.extraction_config,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeGraph":
        """Create KnowledgeGraph from dictionary."""
        entities = [Entity.from_dict(e) for e in data.get("entities", [])]
        relationships = [Relationship.from_dict(r) for r in data.get("relationships", [])]

        created_at = None
        if data.get("created_at"):
            created_at = datetime.fromisoformat(data["created_at"])

        return cls(
            entities=entities,
            relationships=relationships,
            source_document_id=data.get("source_document_id"),
            source_chunks_file=data.get("source_chunks_file"),
            stats=data.get("stats", {}),
            created_at=created_at,
            extraction_config=data.get("extraction_config", {}),
        )

    def save_json(self, output_path: str):
        """Save knowledge graph to JSON file."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load_json(cls, input_path: str) -> "KnowledgeGraph":
        """Load knowledge graph from JSON file."""
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
