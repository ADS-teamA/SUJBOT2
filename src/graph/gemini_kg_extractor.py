"""
Knowledge Graph extraction using Gemini 2.5 Pro.

Extracts entities and relationships from phase1_extraction.json
using a single Gemini API call with compact markdown representation.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import google.generativeai as genai

from .models import Entity, EntityType, Relationship, RelationshipType, KnowledgeGraph

logger = logging.getLogger(__name__)

# Gemini 2.5 Flash for KG extraction (fast, high quota)
KG_MODEL = "gemini-2.5-flash"

KG_EXTRACTION_PROMPT = """Jsi expertní analyzátor dokumentů pro extrakci znalostního grafu. Analyzuj dokument a extrahuj entity a vztahy.

## TYPY ENTIT:

### Právní entity:
- **regulation**: Zákony, vyhlášky, nařízení (např. "zákon č. 263/2016 Sb.", "atomový zákon")
- **legal_provision**: Konkrétní ustanovení (§, článek, odstavec)
- **requirement**: Požadavky vyplývající z právních předpisů
- **permit**: Povolení, licence, oprávnění

### Organizace a osoby:
- **organization**: Úřady, instituce, firmy (SÚJB, ČEZ, MAAE)
- **person**: Osoby zmíněné v dokumentu

### Technické entity (pro jaderné/technické dokumenty):
- **facility**: Zařízení, objekty (jaderná elektrárna, reaktor VR-1)
- **system**: Technické systémy (chladicí systém, I&C)
- **safety_function**: Bezpečnostní funkce
- **isotope**: Radioaktivní izotopy (U-235, Cs-137)
- **dose_limit**: Limity dávek záření

### Ostatní:
- **topic**: Hlavní témata dokumentu
- **date**: Důležité datumy (účinnost, lhůty)
- **location**: Místa

## TYPY VZTAHŮ:

- **implements**: Vyhláška implementuje zákon
- **amends**: Předpis mění jiný předpis
- **supersedes**: Předpis nahrazuje starší
- **references**: Odkaz na jiný dokument/ustanovení
- **issued_by**: Předpis/povolení vydáno autoritou
- **regulated_by**: Zařízení regulováno předpisem
- **operated_by**: Zařízení provozováno organizací
- **contains**: Dokument obsahuje ustanovení
- **part_of**: Část je součástí celku

## PRAVIDLA:

1. Extrahuj VŠECHNY významné entity z dokumentu
2. Pro každou entitu uveď:
   - **id**: Unikátní identifikátor (snake_case)
   - **name**: Název entity
   - **type**: Typ z výše uvedených
   - **description**: Krátký popis (max 100 znaků)

3. Pro každý vztah uveď:
   - **source_id**: ID zdrojové entity
   - **target_id**: ID cílové entity
   - **type**: Typ vztahu
   - **evidence**: Krátká citace z dokumentu (max 100 znaků) podporující tento vztah

## PŘÍKLAD VÝSTUPU:

```json
{
  "entities": [
    {"id": "zakon_18_1997", "name": "Zákon č. 18/1997 Sb.", "type": "regulation", "description": "Atomový zákon"},
    {"id": "sujb", "name": "SÚJB", "type": "organization", "description": "Státní úřad pro jadernou bezpečnost"},
    {"id": "par_32", "name": "§ 32", "type": "legal_provision", "description": "Občanskoprávní odpovědnost za jaderné škody"}
  ],
  "relationships": [
    {"source_id": "par_32", "target_id": "zakon_18_1997", "type": "part_of", "evidence": "§ 32 zákona č. 18/1997 Sb."},
    {"source_id": "zakon_18_1997", "target_id": "sujb", "type": "issued_by", "evidence": "SÚJB vykonává státní správu"}
  ]
}
```

Vrať POUZE validní JSON. Žádný markdown, žádné komentáře.

## DOKUMENT K ANALÝZE:

"""


class GeminiKGExtractor:
    """
    Extract Knowledge Graph from phase1_extraction.json using Gemini 2.5 Pro.

    Features:
    - Converts phase1 JSON to compact markdown (saves tokens)
    - Single API call for entire document
    - Returns KnowledgeGraph object compatible with existing pipeline
    """

    def __init__(self, model: str = KG_MODEL):
        """Initialize with Gemini API key."""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable required")

        genai.configure(api_key=api_key)
        self.model_id = model
        logger.info(f"GeminiKGExtractor initialized with model={model}")

    def extract_from_phase1(self, phase1_path: Path) -> KnowledgeGraph:
        """
        Extract KG from phase1_extraction.json file with chunk provenance.

        Args:
            phase1_path: Path to phase1_extraction.json

        Returns:
            KnowledgeGraph with entities and relationships (with source_chunk_ids if phase3 available)

        Raises:
            FileNotFoundError: If phase1_path does not exist
            ValueError: If phase1_path is not a valid JSON file
        """
        # Validate file exists
        if not phase1_path.exists():
            raise FileNotFoundError(
                f"Phase 1 extraction file not found: {phase1_path}. "
                f"Ensure document extraction (Phase 1) completed successfully."
            )

        # Load phase1 extraction with proper error handling
        try:
            with open(phase1_path, "r", encoding="utf-8") as f:
                phase1_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Phase 1 extraction file is corrupted or invalid JSON: {phase1_path}. "
                f"Error at position {e.pos}: {e.msg}"
            )
        except PermissionError:
            raise PermissionError(
                f"Cannot read Phase 1 extraction file (permission denied): {phase1_path}"
            )

        document_id = phase1_data.get("document_id", phase1_path.stem)

        # Try to load phase3_chunks for entity-to-chunk mapping
        phase3_chunks: Optional[List[Dict[str, Any]]] = None
        phase3_path = phase1_path.parent / "phase3_chunks.json"
        if phase3_path.exists():
            try:
                with open(phase3_path, "r", encoding="utf-8") as f:
                    phase3_data = json.load(f)
                    phase3_chunks = phase3_data.get("chunks", [])
                    if not phase3_chunks:
                        logger.warning("phase3_chunks.json exists but 'chunks' array is empty")
                    else:
                        logger.info(f"Loaded {len(phase3_chunks)} chunks for entity mapping")
            except (json.JSONDecodeError, IOError, OSError) as e:
                logger.warning(
                    f"Could not load phase3_chunks for entity-to-chunk mapping: "
                    f"{type(e).__name__}: {e}. Entities will not have chunk provenance."
                )

        # Convert to compact markdown
        compact_text = self._to_compact_markdown(phase1_data)
        logger.info(f"Compact markdown: {len(compact_text)} chars from {phase1_data.get('num_sections', 0)} sections")

        # Extract KG with Gemini
        raw_kg = self._extract_with_gemini(compact_text)

        # Convert to KnowledgeGraph with chunk provenance
        return self._convert_to_knowledge_graph(raw_kg, document_id, phase3_chunks)

    def _to_compact_markdown(self, phase1_data: Dict[str, Any]) -> str:
        """
        Convert phase1 extraction to compact markdown.

        Includes sections with content OR meaningful titles/paths.
        For documents where Gemini extracted structure without content,
        uses title and path as content substitute for KG extraction.
        """
        lines = []

        # Document header
        doc_id = phase1_data.get("document_id", "Unknown")
        lines.append(f"# {doc_id}")
        lines.append("")

        for section in phase1_data.get("sections", []):
            content = section.get("content", "").strip()
            path = section.get("path", "")
            title = section.get("title", "")
            summary = section.get("summary", "")

            # Skip empty sections without any meaningful info
            if not content and not title and not path:
                continue

            # Format: ## path (title if different)
            header = f"## {path}" if path else f"## {title}"
            if title and path and title not in path:
                header += f" - {title}"

            lines.append(header)

            # Use content if available, otherwise use title/path as fallback
            if content:
                lines.append(content)
            elif summary:
                lines.append(f"[Shrnutí: {summary}]")
            # For structure-only extractions, the path/title serves as content
            lines.append("")

        return "\n".join(lines)

    def _extract_with_gemini(self, document_text: str) -> Dict[str, Any]:
        """
        Run KG extraction with Gemini 2.5 Pro.

        Args:
            document_text: Compact markdown representation of document

        Returns:
            Dict with 'entities' and 'relationships' lists

        Raises:
            ValueError: If Gemini returns invalid JSON that cannot be repaired
        """
        model = genai.GenerativeModel(
            self.model_id,
            generation_config={
                "temperature": 0.1,
                "max_output_tokens": 32768,
                "response_mime_type": "application/json",
            },
        )

        prompt = KG_EXTRACTION_PROMPT + document_text

        logger.info(f"Extracting KG with {self.model_id}...")
        response = model.generate_content(prompt)

        # Log token usage
        if hasattr(response, "usage_metadata"):
            logger.info(
                f"Tokens: prompt={response.usage_metadata.prompt_token_count}, "
                f"output={response.usage_metadata.candidates_token_count}"
            )

        # Parse JSON with error handling and repair
        try:
            return json.loads(response.text)
        except json.JSONDecodeError as e:
            logger.warning(f"Gemini KG response not valid JSON at position {e.pos}: {e.msg}")
            # Try json_repair library (same approach as gemini_extractor.py)
            try:
                from json_repair import repair_json
                repaired = repair_json(response.text, return_objects=True)
                logger.info("KG JSON repaired successfully using json_repair library")
                if isinstance(repaired, dict):
                    return repaired
                else:
                    raise ValueError(f"Repaired JSON is not a dict: {type(repaired).__name__}")
            except ImportError:
                logger.error("json_repair library not available for KG JSON repair")
                raise ValueError(f"Gemini returned invalid JSON for KG extraction: {e}") from e
            except Exception as repair_err:
                logger.error(
                    f"Failed to repair KG JSON response: {type(repair_err).__name__}: {repair_err}"
                )
                raise ValueError(f"Gemini returned invalid JSON for KG extraction: {e}") from e

    def _find_chunks_for_entity(
        self, entity_name: str, phase3_chunks: List[Dict[str, Any]]
    ) -> Tuple[List[str], Optional[str]]:
        """
        Find chunks that mention the entity (post-extraction chunk mapping).

        Args:
            entity_name: Name of the entity to search for
            phase3_chunks: List of phase3 chunks with chunk_id, raw_content, context

        Returns:
            Tuple of (matching_chunk_ids, first_mention_chunk_id)
        """
        matching_chunk_ids = []
        first_mention = None
        entity_lower = entity_name.lower()

        for chunk in phase3_chunks:
            # Search in both raw_content and context
            content = (chunk.get("raw_content", "") or "") + " " + (chunk.get("context", "") or "")
            if entity_lower in content.lower():
                chunk_id = chunk.get("chunk_id", "")
                if chunk_id:
                    matching_chunk_ids.append(chunk_id)
                    if first_mention is None:
                        first_mention = chunk_id

        return matching_chunk_ids, first_mention

    def _convert_to_knowledge_graph(
        self,
        raw_kg: Dict[str, Any],
        document_id: str,
        phase3_chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> KnowledgeGraph:
        """
        Convert raw JSON to KnowledgeGraph object with chunk provenance.

        Args:
            raw_kg: Raw KG data from Gemini
            document_id: Document identifier
            phase3_chunks: Optional phase3 chunks for entity-to-chunk mapping
        """
        import uuid

        extraction_time = datetime.now()
        entities = []
        relationships = []

        # Map string types to enums
        entity_type_map = {
            "regulation": EntityType.REGULATION,
            "legal_provision": EntityType.LEGAL_PROVISION,
            "requirement": EntityType.REQUIREMENT,
            "permit": EntityType.PERMIT,
            "organization": EntityType.ORGANIZATION,
            "person": EntityType.PERSON,
            "facility": EntityType.FACILITY,
            "system": EntityType.SYSTEM,
            "safety_function": EntityType.SAFETY_FUNCTION,
            "isotope": EntityType.ISOTOPE,
            "dose_limit": EntityType.DOSE_LIMIT,
            "topic": EntityType.TOPIC,
            "date": EntityType.DATE,
            "location": EntityType.LOCATION,
        }

        rel_type_map = {
            "implements": RelationshipType.IMPLEMENTS,
            "amends": RelationshipType.AMENDS,
            "supersedes": RelationshipType.SUPERSEDES,
            "references": RelationshipType.REFERENCES,
            "issued_by": RelationshipType.ISSUED_BY,
            "regulated_by": RelationshipType.REGULATED_BY,
            "operated_by": RelationshipType.OPERATED_BY,
            "contains": RelationshipType.CONTAINS,
            "part_of": RelationshipType.PART_OF,
        }

        # Map Gemini entity IDs to UUID IDs for relationships
        id_mapping: Dict[str, str] = {}

        # Convert entities with chunk mapping
        for raw_entity in raw_kg.get("entities", []):
            entity_type_str = raw_entity.get("type", "topic")
            entity_type = entity_type_map.get(entity_type_str, EntityType.TOPIC)

            gemini_id = raw_entity.get("id", "")
            entity_uuid = str(uuid.uuid4())
            id_mapping[gemini_id] = entity_uuid

            entity_name = raw_entity.get("name", "")

            # Post-extraction chunk mapping (SOTA: retroactive provenance)
            source_chunk_ids: List[str] = []
            first_mention_chunk_id: Optional[str] = None
            if phase3_chunks and entity_name:
                source_chunk_ids, first_mention_chunk_id = self._find_chunks_for_entity(
                    entity_name, phase3_chunks
                )

            entity = Entity(
                id=entity_uuid,
                type=entity_type,
                value=entity_name,  # Original text
                normalized_value=entity_name.lower().strip(),  # For deduplication
                confidence=0.9,  # High confidence from Gemini
                source_chunk_ids=source_chunk_ids,
                first_mention_chunk_id=first_mention_chunk_id,
                document_id=document_id,
                metadata={"description": raw_entity.get("description", "")},
                extraction_method="gemini",
                extracted_at=extraction_time,
            )
            entities.append(entity)

        # Convert relationships with evidence text
        skipped_relationships: List[Tuple[str, str]] = []
        raw_relationships = raw_kg.get("relationships", [])

        for raw_rel in raw_relationships:
            rel_type_str = raw_rel.get("type", "references")
            rel_type = rel_type_map.get(rel_type_str, RelationshipType.REFERENCES)

            # Log unknown relationship types
            if rel_type_str not in rel_type_map:
                logger.debug(f"Unknown relationship type '{rel_type_str}', defaulting to REFERENCES")

            # Map Gemini IDs to UUIDs
            source_gemini_id = raw_rel.get("source_id", "")
            target_gemini_id = raw_rel.get("target_id", "")

            source_uuid = id_mapping.get(source_gemini_id, "")
            target_uuid = id_mapping.get(target_gemini_id, "")

            # Track and skip relationships with missing entities
            if not source_uuid or not target_uuid:
                skipped_relationships.append((source_gemini_id, target_gemini_id))
                continue

            # Use 'evidence' field from prompt, fall back to 'description'
            evidence_text = raw_rel.get("evidence", "") or raw_rel.get("description", "")

            relationship = Relationship(
                id=str(uuid.uuid4()),
                type=rel_type,
                source_entity_id=source_uuid,
                target_entity_id=target_uuid,
                confidence=0.9,
                source_chunk_id="",  # Relationship provenance TBD
                evidence_text=evidence_text,
                extraction_method="gemini",
                extracted_at=extraction_time,
            )
            relationships.append(relationship)

        # Log aggregate stats for skipped relationships
        if skipped_relationships:
            logger.warning(
                f"Skipped {len(skipped_relationships)}/{len(raw_relationships)} relationships "
                f"due to missing entity references. First 3: {skipped_relationships[:3]}"
            )

        kg = KnowledgeGraph(
            entities=entities,
            relationships=relationships,
            source_document_id=document_id,
            created_at=extraction_time,
        )

        # Log stats
        chunks_mapped = sum(1 for e in entities if e.source_chunk_ids)
        logger.info(
            f"Extracted KG: {len(entities)} entities ({chunks_mapped} with chunk mapping), "
            f"{len(relationships)} relationships"
        )
        return kg


def extract_kg_from_phase1(phase1_path: Path, model: str = KG_MODEL) -> KnowledgeGraph:
    """
    Convenience function to extract KG from phase1_extraction.json.

    Args:
        phase1_path: Path to phase1_extraction.json
        model: Gemini model to use (default: gemini-2.5-pro)

    Returns:
        KnowledgeGraph with entities and relationships
    """
    extractor = GeminiKGExtractor(model=model)
    return extractor.extract_from_phase1(phase1_path)
