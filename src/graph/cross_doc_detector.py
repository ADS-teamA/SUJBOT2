"""
Cross-Document Relationship Detector.

Detects relationships between entities across different documents:
- Pattern-based detection (fast, ~80% recall)
- Optional LLM validation (higher quality, more expensive)

Supported cross-document relationship types:
- REFERENCES: Document A references Document B
- SUPERSEDES: Document A supersedes Document B
- ISSUED_BY: Document issued by Organization
- DEVELOPED_BY: Standard developed by Organization
"""

import logging
import re
from typing import List, Dict, Set, Optional, Tuple
from datetime import datetime
from collections import defaultdict

from .models import Entity, Relationship, KnowledgeGraph, EntityType, RelationshipType

logger = logging.getLogger(__name__)


class CrossDocumentRelationshipDetector:
    """
    Detects relationships between entities across different documents.

    Two detection modes:
    1. Pattern-based (default): Fast heuristic matching
    2. LLM-based (optional): Higher quality, more expensive

    Usage:
        detector = CrossDocumentRelationshipDetector(
            use_llm_validation=False,  # Fast mode
            confidence_threshold=0.7
        )

        unified_kg = manager.load_or_create()
        new_rels = detector.detect_cross_document_relationships(
            unified_kg,
            document_id="BZ_VR1"
        )

        for rel in new_rels:
            unified_kg.relationships.append(rel)
    """

    # Pattern definitions for cross-document relationships
    REFERENCE_PATTERNS = [
        # Legal references
        r"(?:podle|dle|§|čl\.|článku?)\s+(\d+)",  # podle § 5, čl. 3
        r"(?:zákon[ua]?|vyhlášk[ay]?|nařízení)\s+(?:č\.?)?\s*(\d+/\d+)",  # zákon č. 263/2016
        r"(?:standard|norma)\s+([A-Z]+\s*\d+)",  # standard ISO 14001
        # Version references
        r"(?:verze|vydání|revision)\s+(\d+\.?\d*)",  # verze 2.0
        # Date references
        r"(?:ze dne|z\s+(?:19|20)\d{2})",  # ze dne 2024-01-01
    ]

    SUPERSEDE_PATTERNS = [
        r"(?:nahrazuje|ruší|mění)\s+(.+?)(?:\s+ze\s+dne|\s+č\.|\s*$)",
        r"(?:nový|aktualizovan[ýá]|revidovan[ýá])\s+(.+?)(?:\s+ze\s+dne|\s+č\.|\s*$)",
        r"(?:předchozí|staré)\s+(?:znění|verze)\s+(.+?)(?:\s+ze\s+dne|\s*$)",
    ]

    ORGANIZATION_PATTERNS = [
        r"(?:vydal[ao]?|schválil[ao]?|připravil[ao]?)\s+([A-ZŠČŘŽÝÁÍÉÚŮ][a-zščřžýáíéúů\s]+)",
        r"(?:organizace|orgán|instituce)\s+([A-ZŠČŘŽÝÁÍÉÚŮ][a-zščřžýáíéúů\s]+)",
    ]

    def __init__(
        self,
        use_llm_validation: bool = False,
        confidence_threshold: float = 0.7,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
    ):
        """
        Initialize cross-document relationship detector.

        Args:
            use_llm_validation: Whether to use LLM for validation (default: False)
            confidence_threshold: Minimum confidence for detected relationships
            llm_provider: LLM provider for validation (openai/anthropic)
            llm_model: LLM model for validation (gpt-4o-mini/haiku)
        """
        self.use_llm_validation = use_llm_validation
        self.confidence_threshold = confidence_threshold
        self.llm_provider = llm_provider
        self.llm_model = llm_model

        logger.info(
            f"CrossDocumentRelationshipDetector initialized: "
            f"llm_validation={use_llm_validation}, threshold={confidence_threshold}"
        )

    def detect_cross_document_relationships(
        self, unified_kg: KnowledgeGraph, new_document_id: str
    ) -> List[Relationship]:
        """
        Detect cross-document relationships for newly added document.

        Detects relationships between:
        - New document entities <-> Existing document entities

        Args:
            unified_kg: Unified knowledge graph
            new_document_id: ID of newly added document

        Returns:
            List of detected cross-document relationships
        """
        logger.info(f"Detecting cross-document relationships for '{new_document_id}'")

        # Partition entities by document
        new_doc_entities = []
        existing_doc_entities = []

        for entity in unified_kg.entities:
            doc_ids = entity.metadata.get("document_ids", [])

            if new_document_id in doc_ids:
                new_doc_entities.append(entity)

            # Check if entity appears in OTHER documents
            other_doc_ids = [doc_id for doc_id in doc_ids if doc_id != new_document_id]
            if other_doc_ids:
                existing_doc_entities.append(entity)

        logger.info(
            f"Partitioned entities: {len(new_doc_entities)} new, "
            f"{len(existing_doc_entities)} existing"
        )

        # Detect relationships
        cross_doc_rels = []

        # 1. REFERENCES relationships (document → document)
        references_rels = self._detect_references(
            new_doc_entities, existing_doc_entities, unified_kg
        )
        cross_doc_rels.extend(references_rels)

        # 2. SUPERSEDES relationships (new standard → old standard)
        supersedes_rels = self._detect_supersedes(
            new_doc_entities, existing_doc_entities, unified_kg
        )
        cross_doc_rels.extend(supersedes_rels)

        # 3. ISSUED_BY relationships (document → organization)
        issued_by_rels = self._detect_issued_by(
            new_doc_entities, existing_doc_entities, unified_kg
        )
        cross_doc_rels.extend(issued_by_rels)

        # Optional: LLM validation
        if self.use_llm_validation:
            logger.info(f"Running LLM validation on {len(cross_doc_rels)} candidates...")
            cross_doc_rels = self._llm_validate_relationships(cross_doc_rels, unified_kg)

        logger.info(f"Detected {len(cross_doc_rels)} cross-document relationships")

        return cross_doc_rels

    def _detect_references(
        self,
        new_entities: List[Entity],
        existing_entities: List[Entity],
        kg: KnowledgeGraph,
    ) -> List[Relationship]:
        """
        Detect REFERENCES relationships using pattern matching.

        Strategy:
        - Extract reference patterns from entity values
        - Match against existing entity normalized values
        - Create REFERENCES relationship if match found

        Args:
            new_entities: Entities from new document
            existing_entities: Entities from existing documents
            kg: Knowledge graph for context

        Returns:
            List of REFERENCES relationships
        """
        relationships = []

        # Build lookup index: normalized_value -> entity
        existing_by_value: Dict[str, List[Entity]] = defaultdict(list)
        for entity in existing_entities:
            # Skip entities with None normalized_value (should be rare - indicates extraction issue)
            if entity.normalized_value is None:
                logger.debug(
                    f"Skipping entity with None normalized_value: {entity.value} "
                    f"(type={entity.type}, source={entity.source_chunk_ids})"
                )
                continue
            existing_by_value[entity.normalized_value.lower()].append(entity)

        # Scan new entities for reference patterns
        for new_entity in new_entities:
            if new_entity.type not in [
                EntityType.STANDARD,
                EntityType.REGULATION,
                EntityType.CLAUSE,
            ]:
                continue

            # Extract references from entity value
            referenced_values = self._extract_references(new_entity.value)

            for ref_value in referenced_values:
                ref_normalized = ref_value.lower()

                # Check if referenced value exists in existing entities
                matches = existing_by_value.get(ref_normalized, [])

                for existing_entity in matches:
                    # Skip self-references (same document)
                    if self._same_document(new_entity, existing_entity):
                        continue

                    # Create REFERENCES relationship
                    rel = Relationship(
                        id=f"cross_doc_ref_{new_entity.id}_{existing_entity.id}",
                        type=RelationshipType.REFERENCES,
                        source_entity_id=new_entity.id,
                        target_entity_id=existing_entity.id,
                        confidence=0.8,  # Pattern-based confidence
                        source_chunk_id=new_entity.first_mention_chunk_id or "",
                        evidence_text=f"{new_entity.value} references {existing_entity.value}",
                        properties={
                            "cross_document": True,
                            "source_document": new_entity.document_id,
                            "target_document": existing_entity.document_id,
                        },
                        extraction_method="pattern",
                        extracted_at=datetime.now(),
                    )

                    relationships.append(rel)

        logger.info(f"Detected {len(relationships)} REFERENCES relationships")
        return relationships

    def _detect_supersedes(
        self,
        new_entities: List[Entity],
        existing_entities: List[Entity],
        kg: KnowledgeGraph,
    ) -> List[Relationship]:
        """
        Detect SUPERSEDES relationships (new standard → old standard).

        Strategy:
        - Match standards/regulations with similar names but different versions
        - Create SUPERSEDES relationship if new version > old version

        Args:
            new_entities: Entities from new document
            existing_entities: Entities from existing documents
            kg: Knowledge graph for context

        Returns:
            List of SUPERSEDES relationships
        """
        relationships = []

        # Filter to standards and regulations
        new_standards = [
            e for e in new_entities if e.type in [EntityType.STANDARD, EntityType.REGULATION]
        ]

        existing_standards = [
            e
            for e in existing_entities
            if e.type in [EntityType.STANDARD, EntityType.REGULATION]
        ]

        # Build index by base name (without version)
        existing_by_base: Dict[str, List[Entity]] = defaultdict(list)
        for entity in existing_standards:
            base_name = self._extract_base_name(entity.normalized_value)
            existing_by_base[base_name].append(entity)

        # Check for superseding relationships
        for new_std in new_standards:
            new_base = self._extract_base_name(new_std.normalized_value)
            new_version = self._extract_version(new_std.normalized_value)

            # Find matching base names
            matches = existing_by_base.get(new_base, [])

            for existing_std in matches:
                # Skip if same document
                if self._same_document(new_std, existing_std):
                    continue

                existing_version = self._extract_version(existing_std.normalized_value)

                # Check if new version > old version
                if self._is_newer_version(new_version, existing_version):
                    rel = Relationship(
                        id=f"cross_doc_supersedes_{new_std.id}_{existing_std.id}",
                        type=RelationshipType.SUPERSEDES,
                        source_entity_id=new_std.id,
                        target_entity_id=existing_std.id,
                        confidence=0.75,  # Slightly lower confidence (version comparison)
                        source_chunk_id=new_std.first_mention_chunk_id or "",
                        evidence_text=f"{new_std.value} supersedes {existing_std.value}",
                        properties={
                            "cross_document": True,
                            "source_document": new_std.document_id,
                            "target_document": existing_std.document_id,
                            "new_version": new_version,
                            "old_version": existing_version,
                        },
                        extraction_method="pattern",
                        extracted_at=datetime.now(),
                    )

                    relationships.append(rel)

        logger.info(f"Detected {len(relationships)} SUPERSEDES relationships")
        return relationships

    def _detect_issued_by(
        self,
        new_entities: List[Entity],
        existing_entities: List[Entity],
        kg: KnowledgeGraph,
    ) -> List[Relationship]:
        """
        Detect ISSUED_BY relationships (document → organization).

        Strategy:
        - Match organizations that appear in multiple documents
        - Create ISSUED_BY if organization is mentioned in document context

        Args:
            new_entities: Entities from new document
            existing_entities: Entities from existing documents
            kg: Knowledge graph for context

        Returns:
            List of ISSUED_BY relationships
        """
        relationships = []

        # Filter entities
        new_docs = [e for e in new_entities if e.type in [EntityType.STANDARD, EntityType.REGULATION]]
        existing_orgs = [e for e in existing_entities if e.type == EntityType.ORGANIZATION]

        # Build organization lookup
        org_by_name: Dict[str, Entity] = {}
        for org in existing_orgs:
            org_by_name[org.normalized_value.lower()] = org

        # Check if organizations are mentioned in new documents
        for doc_entity in new_docs:
            # Extract organization mentions from value
            mentioned_orgs = self._extract_organizations(doc_entity.value)

            for org_name in mentioned_orgs:
                org_normalized = org_name.lower()

                if org_normalized in org_by_name:
                    org_entity = org_by_name[org_normalized]

                    # Skip if same document
                    if self._same_document(doc_entity, org_entity):
                        continue

                    rel = Relationship(
                        id=f"cross_doc_issued_by_{doc_entity.id}_{org_entity.id}",
                        type=RelationshipType.ISSUED_BY,
                        source_entity_id=doc_entity.id,
                        target_entity_id=org_entity.id,
                        confidence=0.7,
                        source_chunk_id=doc_entity.first_mention_chunk_id or "",
                        evidence_text=f"{doc_entity.value} issued by {org_entity.value}",
                        properties={
                            "cross_document": True,
                            "source_document": doc_entity.document_id,
                            "target_document": org_entity.document_id,
                        },
                        extraction_method="pattern",
                        extracted_at=datetime.now(),
                    )

                    relationships.append(rel)

        logger.info(f"Detected {len(relationships)} ISSUED_BY relationships")
        return relationships

    def _extract_references(self, text: str) -> List[str]:
        """Extract reference patterns from text."""
        references = []

        for pattern in self.REFERENCE_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if match.group(1):
                    references.append(match.group(1))

        return references

    def _extract_organizations(self, text: str) -> List[str]:
        """Extract organization names from text."""
        organizations = []

        for pattern in self.ORGANIZATION_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if match.group(1):
                    organizations.append(match.group(1).strip())

        return organizations

    def _extract_base_name(self, name: str) -> str:
        """
        Extract base name without version/date.

        Examples:
            "Zákon 263/2016 Sb." -> "Zákon 263 Sb."
            "ISO 14001:2015" -> "ISO 14001"
        """
        # Remove dates (YYYY-MM-DD, DD.MM.YYYY)
        name = re.sub(r"\d{4}-\d{2}-\d{2}", "", name)
        name = re.sub(r"\d{2}\.\d{2}\.\d{4}", "", name)

        # Remove version numbers (v1.0, version 2, :2015)
        name = re.sub(r"v\d+\.?\d*", "", name, flags=re.IGNORECASE)
        name = re.sub(r"version\s+\d+\.?\d*", "", name, flags=re.IGNORECASE)
        name = re.sub(r":\d{4}", "", name)

        return name.strip()

    def _extract_version(self, name: str) -> Optional[str]:
        """
        Extract version from name.

        Examples:
            "ISO 14001:2015" -> "2015"
            "Zákon 263/2016" -> "2016"
            "v2.0" -> "2.0"
        """
        # Try version patterns
        patterns = [
            r":\s*(\d{4})",  # ISO 14001:2015
            r"/(\d{4})",  # Zákon 263/2016
            r"v(\d+\.?\d*)",  # v2.0
            r"version\s+(\d+\.?\d*)",  # version 2.0
        ]

        for pattern in patterns:
            match = re.search(pattern, name, re.IGNORECASE)
            if match:
                return match.group(1)

        return None

    def _is_newer_version(
        self, new_version: Optional[str], old_version: Optional[str]
    ) -> bool:
        """
        Check if new_version > old_version.

        Args:
            new_version: New version string (e.g., "2016", "2.0")
            old_version: Old version string (e.g., "2015", "1.0")

        Returns:
            True if new_version is newer
        """
        if not new_version or not old_version:
            return False

        try:
            # Try numeric comparison
            new_num = float(new_version)
            old_num = float(old_version)
            return new_num > old_num
        except ValueError:
            # Fallback to string comparison
            return new_version > old_version

    def _same_document(self, entity1: Entity, entity2: Entity) -> bool:
        """Check if two entities belong to the same document."""
        docs1 = set(entity1.metadata.get("document_ids", []))
        docs2 = set(entity2.metadata.get("document_ids", []))

        # If they share ANY document, consider them same-document
        return bool(docs1 & docs2)

    def _llm_validate_relationships(
        self, candidate_rels: List[Relationship], kg: KnowledgeGraph
    ) -> List[Relationship]:
        """
        Validate relationships using LLM.

        (Placeholder for LLM validation - not implemented yet)

        Args:
            candidate_rels: Candidate relationships to validate
            kg: Knowledge graph for context

        Returns:
            Validated relationships
        """
        logger.warning("LLM validation not implemented yet - returning all candidates")
        return candidate_rels
